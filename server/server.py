import eventlet
import socketio
import numpy as np
import cv2

sio = socketio.Server()
app = socketio.WSGIApp(sio, static_files={
    '/': {'content_type': 'text/html', 'filename': 'index.html'}
})

buffer = {}
np.set_printoptions(threshold=10000000)

# 予測モデルで予測を行うメソッド
def requestPrediction(bufferImage):
    return ["まいたけ"]

@sio.event
def connect(sid, environ):
    print('connect ', sid)
    buffer[sid] = []

@sio.event
def disconnect(sid):
    print('disconnect ', sid)

@sio.event
def message(sid, data):
    print('message ', data)

@sio.event
def sendImage(sid, data):
    data = np.frombuffer(data, dtype=np.uint8)
    data = np.reshape(data, (160,80,3))
    buffer[sid].append(data)

# 画像配列を変換候補メソッドへ渡す
@sio.event
def predictMouth(sid, data):
    bufferImage = np.array(buffer[sid])
    
    # 予測メソッドへ投げる
    response = requestPrediction(bufferImage)

    # 返ってきた値を返す
    sio.emit('requestPredictMouth', {'data': response}, room=sid)
    
    buffer[sid] = []

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('', 80)), app)