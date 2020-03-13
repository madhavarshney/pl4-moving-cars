from lib.settings import TrackerOptions
from lib.detection.car_track_v2 import VehicleTracker

vehicleTracker = VehicleTracker()
vehicleTracker.run(TrackerOptions())
