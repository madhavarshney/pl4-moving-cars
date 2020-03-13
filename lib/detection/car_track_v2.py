import cv2
import numpy as np
import time
import requests

from lib.settings import TrackerOptions
from lib.logger import logger
from lib.api.event_manager import EventManager
from lib.video.video_utils import preprocess, get_point
from lib.video.goprostream import wake_gopro_on_lan, gopro_live

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(125, 255, size=(len(CLASSES), 3))

class VehicleTracker:
    options: TrackerOptions    = None
    eventManager: EventManager = None
    vs  = None
    net = None
    writer        = None
    multiTracker  = None
    totalFrames   = -1

    def run(self, options: TrackerOptions = TrackerOptions()):
        self.options = options
        self.eventManager = EventManager(options)

        try:
            self.start()
            self.detect()
        except NameError as e:
            logger.error(str(e))
        finally:
            self.finish()

    def start(self):
        self.eventManager.initialize()

        if self.options.USE_GOPRO:
            wake_gopro_on_lan()
            # signal.signal(signal.SIGINT, quit_gopro)
            gopro_live()
            self.vs = cv2.VideoCapture('udp://10.5.5.100:8554', cv2.CAP_FFMPEG)
        else:
            self.vs = cv2.VideoCapture(self.options.VIDEO_FILE_PATH)
            # self.vs = cv2.VideoCapture(0)

        if not self.vs.isOpened():
            # TODO: make custom error
            raise NameError('Unable to open the video file.')
            return

        self.net = cv2.dnn.readNetFromCaffe(self.options.DETECTION_NETWORK_PATH, self.options.CAFFEMODEL_PATH)
        self.multiTracker = cv2.MultiTracker_create()

        if not self.options.USE_GOPRO:
            try:
                prop = cv2.CAP_PROP_FRAME_COUNT
                self.totalFrames = int(self.vs.get(prop))
                logger.info("{} total frames in video".format(self.totalFrames))

            # an error occurred while trying to determine the total
            # number of frames in the video file
            except Exception as err:
                if self.options.debug:
                    logger.warn(str(err))
                logger.warn("Could not determine # of frames in video")
                logger.warn("No approx. completion time can be provided")
                self.totalFrames = -1


    def detect(self):
        (w, h) = (None, None)
        multiTrackerList = []
        current_centers = {}
        paths = {}
        counter = {}
        fps = [0, 0]
        uifps = [0, 0]
        currentFrame = 0
        loopStart = time.time() - 100

        while True:
            prevLoopStart = loopStart
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
            if self.options.PROFILE:
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

            if self.options.PROFILE:
                updateTimeEnd = time.time()


            ### INITIALIZE - (usually) on initial run ###

            # if the frame dimensions are empty, grab them
            if w is None or h is None:
                (w, h) = frame.shape[:2]
                logger.info("Frame dimensions: ({}, {})".format(w, h))

            # initialize our video writer
            if self.writer is None:
                if self.options.RECORD:
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    self.writer = cv2.VideoWriter(self.options.OUTPUT_FILE_PATH, fourcc, 30, (frame.shape[1], frame.shape[0]), True)


            ### PROCESS source video through AI ###

            if (currentFrame % 8 == 0):
                blob = cv2.dnn.blobFromImage(frame, 0.007843, (448, 448), 127.5)
                self.net.setInput(blob)
                detections = self.net.forward()

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

                        if self.options.SHOW_OBJECT_LABELS:
                            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                            y = startY - 15 if startY - 15 > 15 else startY + 15
                            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            if self.options.PROFILE:
                processTimeEnd = time.time()

            ### DEBUG INFO ###
            loopEnd = time.time()
            timeElapsed = loopEnd - loopStart

            # TODO: since object detections are only done on every nth frame,
            #       the estimate here is often very innacurate
            if self.options.DEBUG and currentFrame == 0 and self.totalFrames > 0:
                # some information on processing single frame
                logger.info("Single frame took {:.4f} seconds".format(timeElapsed))
                logger.info("Estimated total time to finish: {:.4f} seconds (inaccurate)".format(timeElapsed * self.totalFrames))

            if self.options.PROFILE:
                frameLoadTime = np.round((frameLoadEnd - loopStart) * 1000)
                updateTime = np.round((updateTimeEnd - frameLoadEnd) * 1000)
                if currentFrame % 8 == 0:
                    processTime = np.round((loopEnd - updateTimeEnd) * 1000)
                cv2.putText(frame, "Frame Load: {}ms, Loc Update: {}ms, Process: {}ms".format(frameLoadTime, updateTime, processTime), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4,  (255,0,0), 1)

            # Calculate FPS
            if self.options.DEBUG or self.options.PROFILE:
                fps.append(1 / timeElapsed)
                if len(fps) > 30:
                    fps.pop(0)
                curFps = round(np.mean(fps))

            if self.options.DEBUG or self.options.PROFILE:
                uifps.append(1 / (loopStart - prevLoopStart))
                if len(uifps) > 30:
                    uifps.pop(0)
                curUiFps = round(np.mean(uifps))

            ### UPDATE GUI ###
            mainText = "Parking Lot - {} | KCI - {}".format(
                str(self.eventManager.parking_lot), str(self.eventManager.kci))

            if self.options.DEBUG or self.options.PROFILE:
                fpsText = "AI FPS - {} | UI FPS - {}".format(str(curFps), str(curUiFps))
                cv2.putText(frame, fpsText, (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (255,0,0), 2)

            #cv2.line(frame, (25, 0), (25, 400), (0,255,0), 2)
            cv2.line(frame, (100, 200), (200, 260), (0, 255, 0), 2)
            cv2.line(frame, (200, 320), (400, 250), (0, 255, 0), 2)
            cv2.putText(frame, mainText, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (255,0,0), 2)
            cv2.imshow("Frame", frame)

            ### CHECK for EXIT ###
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            ### WRITE output frame ###
            if self.options.RECORD:
                self.writer.write(frame)

            currentFrame += 1


    def finish(self):
        logger.info("Vehicle tracking finished")

        # Release video reader and writer
        if self.writer is not None:
            self.writer.release()
        if self.vs is not None:
            self.vs.release()

        if self.options.USE_GOPRO:
            quit_gopro()
