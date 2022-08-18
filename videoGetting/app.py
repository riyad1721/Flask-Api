from flask import Flask,render_template,Response
import cv2
import cv2 as cv
import numpy as np
import time
import imutils
from imutils.video import VideoStream

app=Flask(__name__)



def generate_frames():
    while True:
        rtsp_url = ""
        camera = VideoStream(rtsp_url).start()
            
        frame = camera.read()
        if frame is None:
            continue
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_detected_frames():#
    Conf_threshold = 0.3
    NMS_threshold = 0.4
    COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)]


    class_name = []
    with open('./configuration/obj.names', 'r') as f:
        class_name = [cname.strip() for cname in f.readlines()]
    # print(class_name)
    net = cv.dnn.readNet('./weights/custom-yolov4-detector_final.weights', './configuration/custom-yolov4-detector.cfg')
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

    model = cv.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

    rtsp_url = "rtsp://admin:test1234@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
    cap = VideoStream(rtsp_url).start()
    #cap = cv.VideoCapture('rtsp://admin:test%40123%23@103.103.88.178:8256/cam/realmonitor?channel=1&subtype=0')
    starting_time = time.time()
    frame_counter = 0
    while True:
        frame = cap.read()
        
        frame_counter += 1
        if frame is None:
            continue
        classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
        for (classid, score, box) in zip(classes, scores, boxes):
            color = COLORS[int(classid) % len(COLORS)]
            label = "%s : %f" % (class_name[classid], score)
            cv.rectangle(frame, box, color, 1)
            cv.putText(frame, label, (box[0], box[1]-10),
                    cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
        endingTime = time.time() - starting_time
        fps = frame_counter/endingTime
        # print(fps)
        cv.putText(frame, f'FPS: {fps}', (20, 50),
                cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')       


@app.route('/',methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/video',methods=['GET', 'POST'])
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detectvideo',methods=['GET', 'POST'])
def detectvideo():
    return Response(generate_detected_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)

