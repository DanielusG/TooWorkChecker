import torch
from models.experimental import attempt_load
from utils.torch_utils import select_device,load_classifier
import cv2
from utils.general import non_max_suppression,scale_coords
from utils.datasets import letterbox
from utils.plots import plot_one_box
import numpy as np
import sqlite3
import datetime
from time import sleep
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
width = 640
height = 480
video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

img_size = 640
half = False

def print_log(info: str):
    print("[INFO][" + str(datetime.datetime.now()) + "] " + info)

def init_db():
    con = sqlite3.connect("log_history.db")
    cur = con.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS log_history (id INTEGER PRIMARY KEY, datetime TEXT, event INTEGER)")
    con.commit()
    return con
def add_event_person(isTherePerson):
    con = init_db()
    cur = con.cursor()
    if isTherePerson:
        cur.execute("INSERT INTO log_history (datetime, event) VALUES (datetime('now', 'localtime'), 1)")
        print_log("Person detected")
    else:
        cur.execute("INSERT INTO log_history (datetime, event) VALUES (datetime('now', 'localtime'), 0)")
        print_log("Person not detected")

with torch.no_grad():
    device = select_device('cpu')
    model = attempt_load('yolov7.pt', map_location=device)
    stride = int(model.stride.max())  # model stride
    #model.half()
    model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters()))) 
    names = model.module.names if hasattr(model, 'module') else model.names
    while True:
        img0 = video.read()[1]
        img = letterbox(img0, img_size, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]
        pred = non_max_suppression(pred)
        isTherePerson = False
        for i, det in enumerate(pred):
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                if label == "person":
                    isTherePerson = True
        add_event_person(isTherePerson)
        sleep(1)
        if cv2.waitKey(1) == ord('q'):
            break