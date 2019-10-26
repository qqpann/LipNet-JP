import eventlet
import socketio
import numpy as np
import logging
import cv2

sio = socketio.Server()
app = socketio.WSGIApp(sio, static_files={
    '/': {'content_type': 'text/html', 'filename': 'index.html'}
})

buffer = {}
np.set_printoptions(threshold=10000000)

@sio.event
def connect(sid, environ):
    logging.debug('connect ', sid)
    buffer[sid] = []

@sio.event
def my_message(sid, data):
    print('message ', data)

@sio.event
def sendImage(sid, data):
    data = np.frombuffer(data, dtype=np.uint8)
    # print(data)
    data = np.reshape(data, (180,80,3))
    cv2.imshow("", data)
    k = cv2.waitKey(1)
    buffer[sid].append(data)

# 画像配列を変換候補メソッドへ渡す
def predictMouth(sid, data):
    bufferImage = np.array(data[sid])
    # 変換候補メソッドへ投げる
    
    # 返ってきた値を返す
    sio.emit('requestPredictMouth', {'data': ["まいたけ"]}, room=sid)

@sio.event
def disconnect(sid):
    logging.debug('disconnect ', sid)

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)