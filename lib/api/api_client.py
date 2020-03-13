# TODO: move somewhere else?
# For testing the API client, run `pipenv run python ./lib/api/api_client.py`
if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append(path.join(path.dirname(__file__), '../..'))


import jwt
import requests

from time import time
from cryptography.hazmat.primitives import serialization as crypto_serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend as crypto_default_backend

from lib.logger import logger
from lib.settings import TrackerOptions

def generate_key_pair():
    key = rsa.generate_private_key(
        backend=crypto_default_backend(),
        public_exponent=65537,
        key_size=2048
    )
    private_key = key.private_bytes(
        crypto_serialization.Encoding.PEM,
        crypto_serialization.PrivateFormat.PKCS8,
        crypto_serialization.NoEncryption())
    public_key = key.public_key().public_bytes(
        crypto_serialization.Encoding.PEM,
        crypto_serialization.PublicFormat.PKCS1
    )
    return [private_key, public_key]


def generate_jwt(private_key, data):
    return jwt.encode(data, private_key, algorithm='RS256').decode('utf-8')


def decode_jwt(public_key, token):
    return jwt.decode(token, public_key, algorithms=['RS256'])

class ApiClient:
    options: TrackerOptions = None
    private_key = None

    def __init__(self, options = TrackerOptions()):
        self.options = options

    def handshake(self):
        [private_key, public_key] = generate_key_pair()
        r = requests.post(self.options.WEBSERVER_URL + '/handshake', json={
            'device_id': self.options.DEVICE_ID,
            'public_key': public_key.decode('utf-8')
        })
        logger.api(r)
        self.private_key = private_key

    def get_count(self):
        r = requests.get(self.options.WEBSERVER_URL + '/count')
        count = r.json()['count']
        logger.api(r)
        logger.info("Initial parking lot count is " + str(count))
        return count

    def send_event(self, inward):
        if self.private_key is None:
            # TODO: make custom error
            raise NameError('ApiClient.handshake() was not called')
        # data = {
        #     "timestamp": np.round(time.time() * 1000),
        #     "type": "IN" if enter else "OUT",
        # }
        # r = requests.post(self.options.WEBSERVER_URL + '/admin/event', json=data)
        token = generate_jwt(self.private_key, {
            'exp': time() + 120,
            'device_id': self.options.DEVICE_ID,
            'timestamp': time() * 1000,
            'type': 'IN' if inward else 'OUT'
        })
        r = requests.post(self.options.WEBSERVER_URL + '/event', json={
            'device_id': self.options.DEVICE_ID,
            'token': token
        })
        logger.api(r)
        # print(r.text)
        return r


# TODO: move somewhere else?
# This is for testing the API client
if __name__ == '__main__':
    apiClient = ApiClient(TrackerOptions())
    apiClient.handshake()
    r = apiClient.send_event(True)
