import os
from enum import Enum

dirname = os.path.dirname(__file__)
def relpath(path): return os.path.join(dirname, path)

####################
### Dev Settings ###
####################

DEBUG = True
PROFILE = True


#####################
### Configuration ###
#####################

class TrackerOptions:
    DEBUG = DEBUG
    PROFILE = PROFILE

    # --- General Config --- #
    RECORD = False
    SHOW_OBJECT_LABELS = False

    # --- Webserver Config --- #
    NOTIFY_WEBSERVER = False
    DEVICE_ID = 'cam-a1'
    WEBSERVER_URL = 'http://192.168.56.101:3000'

    # --- Initial Counts --- #
    INITIAL_PARKING_LOT_COUNT = 10
    INITIAL_KCI_COUNT = 5

    # --- Video Config --- #
    USE_GOPRO = False
    VIDEO_NAME = "GOPR1201.m4v"
    VIDEO_FILE_PATH = relpath("../videos/" + VIDEO_NAME)
    OUTPUT_FILE_PATH = relpath("../test_counting/" + VIDEO_NAME + "-ssd_version3.avi")

    # --- AI/ML Config --- #
    DETECTION_NETWORK_PATH = relpath("../ssd/MobileNetSSD_deploy.prototxt.txt")
    CAFFEMODEL_PATH = relpath("../ssd/MobileNetSSD_deploy.caffemodel")

    def __init__(self):
        # TODO: allow modifying default options
        return


#####################
### LOCATION DATA ###
#####################

class Location(Enum):
    ROAD = 0
    LOT = 1
    KCI = 2

LocationName = ["Road", "Parking Lot", "KCI"]
