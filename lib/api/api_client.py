import jwt
import requests

from time import time
from cryptography.hazmat.primitives import serialization as crypto_serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend as crypto_default_backend

from ..logger import logger
from ..settings import *

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
    private_key = None

    def handshake(self):
        [private_key, public_key] = generate_key_pair()
        r = requests.post(WEBSERVER_URL + '/handshake', json={
            'device_id': DEVICE_ID,
            'public_key': public_key.decode('utf-8')
        })
        logger.api(r)
        self.private_key = private_key

    def get_count(self):
        r = requests.get(WEBSERVER_URL + '/count')
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
        # r = requests.post(WEBSERVER_URL + '/admin/event', json=data)
        token = generate_jwt(self.private_key, {
            'exp': time() + 120,
            'device_id': DEVICE_ID,
            'timestamp': time() * 1000,
            'type': 'IN' if inward else 'OUT'
        })
        r = requests.post(WEBSERVER_URL + '/event', json={
            'device_id': DEVICE_ID,
            'token': token
        })
        logger.api(r)
        # print(r.text)
        return r


if __name__ == '__main__':
    apiClient = ApiClient()
    apiClient.handshake()
    r = apiClient.send_event(True)
