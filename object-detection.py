import urllib.request
import cv2
import numpy as np

thresold_value = 0.55
# image = cv2.imread('sample.JPG')

file_ref = open('coco.names', 'rt')
classes_names = file_ref.read().rstrip('\n').split('\n')

confilg_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen, confilg_file)
model.setInputSize(320, 320)
model.setInputScale(1.0/ 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)
while True:
    URL = "http://192.168.1.101:8080/shot.jpg"
    img_arr = np.array(bytearray(urllib.request.urlopen(URL).read()))
    img = cv2.imdecode(img_arr, -1)

    class_ids, confidence, bbox = model.detect(img, confThreshold=thresold_value)
    if len(class_ids)!=0:
        for class_id, confis, box in zip(class_ids.flatten(), confidence.flatten(), bbox):
            try:
                cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                cv2.putText(img,classes_names[class_id-1].upper(),(box[0]-5,box[1]-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),1)
                cv2.imshow('IPWebcam', img)
                cv2.waitKey(1)
            except:
                print('Not Recognize')

# cv2.imshow('Image',image)
# cv2.waitKey(1)

