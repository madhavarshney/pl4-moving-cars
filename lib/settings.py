import os
from enum import Enum

dirname = os.path.dirname(__file__)
def relpath(path): return os.path.join(dirname, path)

################
### SETTINGS ###
################
DEBUG = True
ENABLE_PROFILING = False
SHOW_OBJECT_LABELS = False

USE_GOPRO = False

DEVICE_ID = 'cam-a1'
SEND_WEBSERVER_EVENTS = False
WEBSERVER_URL = 'http://192.168.56.101:3000'

### Initial counts
INITIAL_PARKING_LOT_COUNT = 10
INITIAL_KCI_COUNT = 5

#############
### FILES ###
#############

## Video stream to analyze
VIDEO_NAME = "GOPR1201.m4v"
VIDEO_FILE_PATH = relpath("../videos/" + VIDEO_NAME)

## MobileNet-SSD detection network
DETECTION_NETWORK_PATH = relpath("../ssd/MobileNetSSD_deploy.prototxt.txt")
CAFFEMODEL_PATH = relpath("../ssd/MobileNetSSD_deploy.caffemodel")

###################
### SHARED CODE ###
###################

class Location(Enum):
    ROAD = 0
    LOT = 1
    KCI = 2

LocationName = ["Road", "Parking Lot", "KCI"]
