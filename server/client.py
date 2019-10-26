import base64
import numpy as np
import cv2
import socketio
from time import sleep
cam = cv2.VideoCapture(0)

sio = socketio.Client()

@sio.event
def connect():
    print('connection established')

@sio.event
def requestPredictMouth(data):
    print('message received with ', data['data'])
    # sio.emit('my response', {'response': 'my response'})

@sio.event
def disconnect():
    print('disconnected from server')

sio.connect('http://localhost')
np.set_printoptions(threshold=10000000)

while True:
    flag, img = cam.read()
    
    #バイトデータ→ndarray変換
    height, width = img.shape[:2]
    img = cv2.resize(img,(160, 80))

    cv2.imshow("", img)
    img = base64.b64encode(img)
    # img = img.tostring()
    
    k = cv2.waitKey(1)
    
    if k == 13:
        break

    sio.emit('sendImage', img)
    sleep(0.01)

sio.emit('predictMouth')

# cam.release()
# cv2.destroyAllWindows()