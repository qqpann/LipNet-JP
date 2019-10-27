import base64
import json
import subprocess

import cv2
import eventlet
import numpy as np
import socketio


sio = socketio.Server()
app = socketio.WSGIApp(sio, static_files={
    '/': {'content_type': 'text/html', 'filename': 'index.html'}
})

buffer = {}
np.set_printoptions(threshold=10000000)

# 予測モデルで予測を行うメソッド
def requestPrediction(bufferImage):
    
    # response = subprocess.check_output("コマンド")
    # return response

    for index, image in enumerate(bufferImage):
        retval = cv2.imwrite(f"images/{str(index)}.jpg", image)
        print(retval)
        if index > 5:
            break

    return ["まいたけ"]

@sio.event
def connect(sid, environ):
    print('connect ', sid)
    # sio.emit('requestPrediction', json.dumps({'data': ["まいたけ"]}), room=sid)
    buffer[sid] = []

@sio.event
def disconnect(sid):
    print('disconnect ', sid)

@sio.event
def message(sid, data):
    print('message ', data)

@sio.event
def sendImage(sid, data):
    data = base64.b64decode(data)
    
    data = np.frombuffer(data, dtype=np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)

    # 要修正
    data = np.reshape(data, (80,160,3))
    buffer[sid].append(data)

# 画像配列を変換候補メソッドへ渡す
@sio.event
def predictMouth(sid):
    bufferImage = np.array(buffer[sid])
    
    # 予測メソッドへ投げる
    response = requestPrediction(bufferImage)

    # 返ってきた値を返す
    sio.emit('requestPrediction', json.dumps({'data': response}), room=sid)
    
    buffer[sid] = []

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('', 80)), app)