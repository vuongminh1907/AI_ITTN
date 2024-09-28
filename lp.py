from PIL import Image
import cv2
import torch
import math 
import function.utils_rotate as utils_rotate
#from IPython.display import display
import os
import time
import argparse
import function.helper as helper

# load model
yolo_LP_detect = torch.hub.load('ultralytics/yolov5', 'custom', path='model/LPdetect.pt')
yolo_license_plate = torch.hub.load('ultralytics/yolov5', 'custom', path='model/LPOCR.pt')
yolo_license_plate.conf = 0.60

license_db = {}
with open("bsx.txt", "r") as f:
    lines = f.readlines()
    for line in lines[1:]:
        line = line.strip()
        parts = line.split(",")
        license_db[parts[0]] = [parts[1], None, None]



vid = cv2.VideoCapture(0)
# vid = cv2.VideoCapture("1.mp4")
while vid.isOpened():
    ret, frame = vid.read()
    
    plates = yolo_LP_detect(frame, size=640)
    list_plates = plates.pandas().xyxy[0].values.tolist()
    list_read_plates = set()
    for plate in list_plates:
        flag = 0
        x = int(plate[0]) # xmin
        y = int(plate[1]) # ymin
        w = int(plate[2] - plate[0]) # xmax - xmin
        h = int(plate[3] - plate[1]) # ymax - ymin  
        crop_img = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (0,0,225), thickness = 2)
        #cv2.imwrite("crop.jpg", crop_img)
        lp = ""
        for cc in range(0,2):
            for ct in range(0,2):
                lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                if lp != "unknown":
                    list_read_plates.add(lp)
                    if lp in license_db:
                        license_db[lp][1] = time.strftime("%Y-%m-%d %H:%M:%S")
                        print("License plate: ", lp, " Owner: ", license_db[lp][0], " Checkin time: ", license_db[lp][1])
                        cv2.putText(frame, license_db[lp][0], (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    else:
                        print("License plate: ", lp, " Owner: Unknown")
                        cv2.putText(frame, "Unknown", (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    flag = 1
                    break
            if flag == 1:
                break
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) < 1:
        break

vid.release()
cv2.destroyAllWindows()