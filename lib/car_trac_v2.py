import cv2
import numpy as np
import time
import os
from enum import Enum
import requests

from video_utils import preprocess, get_point
from logger import Logger, prGreen, prRed, prYellow

dirname = os.path.dirname(__file__)
def relpath(path): return os.path.join(dirname, path)

################
### SETTINGS ###
################
debug = False

### Initial counts
parking_lot = 10
faculty = 5

#############
### FILES ###
#############

## Video stream to analyze
video_name = "GOPR1201.m4v"
video_file_path = relpath("../videos/" + video_name)

## MobileNet-SSD detection network
detection_network_path = relpath("../ssd/MobileNetSSD_deploy.prototxt.txt")
caffemodel_path = relpath("../ssd/MobileNetSSD_deploy.caffemodel")

###########################
### BEGINNING OF SCRIPT ###
###########################

class Location(Enum):
    ROAD = 0
    LOT = 1
    KCI = 2

LocationName = ["Road", "Parking Lot", "KCI"]
logger = Logger()

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

class EventManager:
    def handle_event(self, car, start, end):
        global parking_lot
        global faculty

        moved = start != end

        logger.object("{} - Moved: {} - {} to {}".format(
            car,
            prGreen(moved) if moved == True else prYellow(moved),
            LocationName[start],
            LocationName[end]
        ))
        if start == Location.ROAD.value and end == Location.LOT.value:
            self.update_lot_count(True)
        elif start == Location.ROAD.value and end == Location.KCI.value:
            faculty += 1
        elif start == Location.KCI.value and end == Location.LOT.value:
            faculty -= 1
            self.update_lot_count(True)
        elif start == Location.KCI.value and end == Location.ROAD.value:
            faculty -= 1
        elif start == Location.LOT.value and end == Location.ROAD.value:
            self.update_lot_count(False)

    def update_lot_count(self, enter):
        global parking_lot
        global faculty

        parking_lot += 1 if enter else -1
        logger.event("Vehicle {}".format(prGreen("ENTER") if enter else prRed("EXIT")))

        # data = {"timestamp": np.round(time.time() * 1000), "type": "IN" if enter else "OUT"}
        # r = requests.post('http://192.168.56.101:3000/event', json=data)
        # logger.info("Response from webserver: " + r.text)

class VehicleTracker:
    (w, h) = (None, None)
    writer = None
    vs = None
    net = None
    multiTracker = None
    total = -1
    eventManager = EventManager()

    def run(self):
        try:
            self.start()
            self.detect()
        except NameError as e:
            logger.error(str(e))
        finally:
            self.finish()

    def start(self):
        global total

        self.net = cv2.dnn.readNetFromCaffe(detection_network_path, caffemodel_path)
        self.vs = cv2.VideoCapture(video_file_path)

        if not self.vs.isOpened():
            raise NameError('Unable to open the video file.')
            return

        self.multiTracker = cv2.MultiTracker_create()

        try:
            prop = cv2.CAP_PROP_FRAME_COUNT
            total = int(self.vs.get(prop))
            logger.info("{} total frames in video".format(total))

        # an error occurred while trying to determine the total
        # number of frames in the video file
        except:
            logger.warn("could not determine # of frames in video")
            logger.warn("no approx. completion time can be provided")
            total = -1


    def detect(self):
        w = self.w
        h = self.h
        global total

        multiTrackerList = []
        current_centers = {}
        paths = {}
        counter = {}

        i = 0
        while True:
            i += 1

            ### READ and PREPROCESS next frame ###
            # read the next frame from the file
            (grabbed, frame) = self.vs.read()
            # frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
            # if the frame was not grabbed, then we have reached the end
            # of the stream / video
            if frame is None:
                logger.info("Video / Stream finished")
                break

            processed, padhw, shavedim, resized = preprocess(frame)
            frame = resized
            # print(frame.shape)


            ### UPDATE - get updated location of detected objects ###

            toDelete = []
            for oneTracker in multiTrackerList:
                if counter[oneTracker] < 30:
                    (success, box) = oneTracker.update(frame)

                    # check to see if the tracking was a success
                    if success:
                        (x, y, W, H) = [int(v) for v in box]
                        current_centers[oneTracker] = (x, y, x+W, y+H)
                        if( len(paths[oneTracker]) >= 1 and (paths[oneTracker][-1][0] - (2 * x + W) / 2)**2 + (paths[oneTracker][-1][1] == (2 * y + H) / 2)**2 < 4 ):
                            counter[oneTracker] += 1
                        else:
                            counter[oneTracker] = 0
                            paths[oneTracker].append( ((2 * x + W) / 2, (2 * y + H) / 2))
                        cv2.rectangle(frame, (x, y), (x + W, y + H),
                            (0, 255, 0), 2)
                    else:
                        counter[oneTracker] = 100

            for eachCar, path in paths.items():
                if(counter[eachCar] < 10):
                    for eachPoint in path[-20:]:
                        frame = cv2.circle(frame, (int(eachPoint[0]), int(eachPoint[1])), 3, (255, 0, 0), 3)
                    frame = cv2.circle(frame, (int(path[0][0]), int(path[0][1])), 3, (255,0,0),3)
                if(counter[eachCar] == 100 or counter[eachCar] == 30):
                    start = get_point(path[0])
                    end = get_point(path[-1])
                    self.eventManager.handle_event(eachCar, start, end)
                    counter[eachCar] = 1000


            ### INITIALIZE - on initial run ###

            # if the frame dimensions are empty, grab them
            if w is None or h is None:
                (w, h) = frame.shape[:2]
                logger.info("Frame dimensions: ({}, {})".format(w, h))

            # initialize our video writer
            if self.writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                if not debug:
                    self.writer = cv2.VideoWriter(relpath("../test_counting/" + video_name + "-ssd_version3.avi"), fourcc, 30, (frame.shape[1], frame.shape[0]), True)


            ### PROCESS source video through AI ###

            if(i % 8 == 1):
                blob = cv2.dnn.blobFromImage(frame, 0.007843, (448, 448), 127.5)
                self.net.setInput(blob)
                start = time.time()
                detections = self.net.forward()
                end = time.time()

                # loop over each of the layer outputs
                for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                    confidence = detections[0, 0, i, 2]

                    # filter out weak detections by ensuring the `confidence` is
                    # greater than the minimum confidence
                    if confidence > 0.2:
                        # extract the index of the class label from the `detections`,
                        # then compute the (x, y)-coordinates of the bounding box for
                        # the object
                        idx = int(detections[0, 0, i, 1])
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # display the prediction
                        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                        if CLASSES[idx] == "car":
                            tracker = cv2.TrackerCSRT_create()
                            tracker.init(frame, (startX, startY, endX - startX, endY - startY))
                            toAdd = True
                            for tr, cen in current_centers.items():
                                cen_x_1 = (startX + endX) / 2
                                cen_y_1 = (startY + endY) / 2
                                cen_x_2 = (cen[0] + cen[2]) / 2
                                cen_y_2 = (cen[1] + cen[3]) / 2
                                if (cen_x_2  >= startX and cen_y_2 >= startY and cen_x_2 <= endX and cen_y_2 <= endY) or ((cen_x_1  >= cen[0] - 3 and cen_y_1 >= cen[1] - 3 and cen_x_1 <= cen[2] + 3 and cen_y_1 <= cen[3] + 3) or (cen_x_1 - cen_x_2)**2 + (cen_y_1 - cen_y_2)**2 < 400 ):
                                    toAdd = False
                            if toAdd:
                                multiTrackerList.append(tracker)
                                current_centers[tracker] = (startX, startY, endX, endY)
                                paths[tracker] = []
                                counter[tracker] = 0
                            # cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                            # y = startY - 15 if startY - 15 > 15 else startY + 15
                            # cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)


            ### DEBUG INFO ###
            # some information on processing single frame
            if i == 24 and total > 0:
                elap = (end - start)
                logger.info("single frame took {:.4f} seconds".format(elap))
                logger.info("estimated total time to finish: {:.4f}".format(elap * total))

            ### UPDATE GUI ###
            #cv2.line(frame, (25, 0), (25, 400), (0,255,0), 2)
            cv2.line(frame, (100,200), (200, 260), (0, 255, 0), 2)
            cv2.line(frame, (200, 320), (400, 250), (0,255,0), 2)
            cv2.putText(frame, "parking_lot - " + str(parking_lot) + " | faculty - " + str(faculty) , (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (255,0,0), 2)
            cv2.imshow("Frame", frame)

            ### CHECK for EXIT ###
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            ### WRITE output frame ###
            if not debug:
                self.writer.write(frame)


    def finish(self):
        logger.info("Vehicle tracking finished", "\n")
        # Release video reader and writer
        if not debug and self.writer is not None:
            self.writer.release()
        if self.vs is not None:
            self.vs.release()

vehicleTracker = VehicleTracker()
vehicleTracker.run()
