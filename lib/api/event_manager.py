from ..settings import *
from ..logger import logger, prGreen, prRed, prYellow
from ..api.api_client import ApiClient

class EventManager:
    apiClient = ApiClient()
    parking_lot = INITIAL_PARKING_LOT_COUNT
    kci = INITIAL_KCI_COUNT

    def initialize(self):
        if SEND_WEBSERVER_EVENTS:
            self.apiClient.handshake()
            self.parking_lot = self.apiClient.get_count()

    def handle_event(self, car, start, end):
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
            self.update_kci_count(True)
        elif start == Location.KCI.value and end == Location.LOT.value:
            self.update_kci_count(False)
            self.update_lot_count(True)
        elif start == Location.KCI.value and end == Location.ROAD.value:
            self.update_kci_count(False)
        elif start == Location.LOT.value and end == Location.ROAD.value:
            self.update_lot_count(False)

    def update_kci_count(self, enter):
        self.kci += 1 if enter else -1

    def update_lot_count(self, enter):
        self.parking_lot += 1 if enter else -1
        logger.event("Vehicle {}".format(prGreen("ENTER") if enter else prRed("EXIT")))

        if SEND_WEBSERVER_EVENTS:
            r = self.apiClient.send_event(enter)
