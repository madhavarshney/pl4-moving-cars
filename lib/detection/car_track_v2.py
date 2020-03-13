import cv2
import numpy as np
import time
import requests

from ..settings import *
from ..logger import logger
# from ..api.api_client import ApiClient
from ..api.event_manager import EventManager
from ..video.video_utils import preprocess, get_point
from ..video.goprostream import wake_gopro_on_lan, gopro_live

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(125, 255, size=(len(CLASSES), 3))

class VehicleTracker:
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

        self.eventManager.initialize()

        if USE_GOPRO:
            wake_gopro_on_lan()
            # signal.signal(signal.SIGINT, quit_gopro)
            gopro_live()
            self.vs = cv2.VideoCapture('udp://10.5.5.100:8554', cv2.CAP_FFMPEG)
        else:
            self.vs = cv2.VideoCapture(VIDEO_FILE_PATH)
            # self.vs = cv2.VideoCapture(0)

        if not self.vs.isOpened():
            # TODO: make custom error
            raise NameError('Unable to open the video file.')
            return

        self.net = cv2.dnn.readNetFromCaffe(DETECTION_NETWORK_PATH, CAFFEMODEL_PATH)
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
        global total

        (w, h) = (None, None)
        multiTrackerList = []
        current_centers = {}
        paths = {}
        counter = {}
        fps = [0, 0]

        currentFrame = 0
        while True:
            loopStart = time.time()

            ### READ and PREPROCESS next frame ###
            # self.vs.read()
            # self.vs.read()
            # self.vs.read()
            # read the next frame from the file
            (grabbed, frame) = self.vs.read()
            # self.vs.set(cv2.CAP_PROP_POS_FRAMES, currentFrame * 3)
            # frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
            # if the frame was not grabbed, then we have reached the end
            # of the stream / video
            if frame is None:
                logger.info("Video / Stream finished")
                break

            processed, padhw, shavedim, resized = preprocess(frame)
            frame = resized
            # print(frame.shape)
            if ENABLE_PROFILING:
                frameLoadEnd = time.time()


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

            if ENABLE_PROFILING:
                updateTimeEnd = time.time()

            ### INITIALIZE - (usually) on initial run ###

            # if the frame dimensions are empty, grab them
            if w is None or h is None:
                (w, h) = frame.shape[:2]
                logger.info("Frame dimensions: ({}, {})".format(w, h))

            # initialize our video writer
            if self.writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                if not DEBUG:
                    self.writer = cv2.VideoWriter(relpath("../test_counting/" + VIDEO_NAME + "-ssd_version3.avi"), fourcc, 30, (frame.shape[1], frame.shape[0]), True)


            ### PROCESS source video through AI ###

            if (currentFrame % 8 == 0):
                start = time.time()
                blob = cv2.dnn.blobFromImage(frame, 0.007843, (448, 448), 127.5)
                self.net.setInput(blob)
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

                        if SHOW_OBJECT_LABELS:
                            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                            y = startY - 15 if startY - 15 > 15 else startY + 15
                            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            if ENABLE_PROFILING:
                processTimeEnd = time.time()

            ### DEBUG INFO ###
            # some information on processing single frame
            elap = (end - start)
            if currentFrame == 0 and total > 0:
                logger.info("single frame took {:.4f} seconds".format(elap))
                logger.info("estimated total time to finish: {:.4f} seconds".format(elap * total))

            loopEnd = time.time()

            if ENABLE_PROFILING:
                totalTime = loopEnd - loopStart
                frameLoadTime = np.round((frameLoadEnd - loopStart) * 1000)
                updateTime = np.round((updateTimeEnd - frameLoadEnd) * 1000)
                if currentFrame % 8 == 0:
                    processTime = np.round((elap) * 1000)
                cv2.putText(frame, "Frame Load: {}ms, Loc Update: {}ms, Process: {}ms".format(frameLoadTime, updateTime, processTime), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4,  (255,0,0), 1)

                # frameLoadTime = np.round(((frameLoadEnd - loopStart) / totalTime) * 100)
                # updateTime = np.round(((updateTimeEnd - frameLoadEnd) / totalTime) * 100)
                # processTime = np.round(((processTimeEnd - updateTimeEnd) / totalTime) * 100)
                # print("Frame Load: {}%, Loc Update: {}%, Process: {}%".format(frameLoadTime, updateTime, processTime))
                # cv2.putText(frame, "Frame Load: {}%, Loc Update: {}%, Process: {}%".format(frameLoadTime, updateTime, processTime), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (255,0,0), 2)

            # Calculate FPS
            fps.append(1 / (loopEnd - loopStart))
            if len(fps) > 30:
                fps.pop(0)
            curFps = round(np.mean(fps))

            ### UPDATE GUI ###
            #cv2.line(frame, (25, 0), (25, 400), (0,255,0), 2)
            cv2.line(frame, (100, 200), (200, 260), (0, 255, 0), 2)
            cv2.line(frame, (200, 320), (400, 250), (0, 255, 0), 2)
            cv2.putText(frame, "Parking Lot - " + str(self.eventManager.parking_lot) + " | KCI - " + str(self.eventManager.kci) + " | FPS - " + str(curFps), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (255,0,0), 2)
            cv2.imshow("Frame", frame)

            ### CHECK for EXIT ###
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            ### WRITE output frame ###
            if not DEBUG:
                self.writer.write(frame)

            currentFrame += 1


    def finish(self):
        logger.info("Vehicle tracking finished", "\n")

        # Release video reader and writer
        if not DEBUG and self.writer is not None:
            self.writer.release()
        if self.vs is not None:
            self.vs.release()

        if USE_GOPRO:
            quit_gopro()
