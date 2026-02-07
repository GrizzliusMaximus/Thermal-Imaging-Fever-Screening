"""This example is for Raspberry Pi (Linux) only!
   It will not work on microcontrollers running CircuitPython!"""
 
import os
import math
import time
 
import busio
import board
 
import numpy as np

from scipy.interpolate import griddata
 
from colour import Color
 
import Adafruit_AMG88xx

from imutils.video import VideoStream
from imutils.video import FPS
from multiprocessing import Pool, Manager
import cv2
import imutils
from micropython import const
 
i2c_bus = busio.I2C(board.SCL, board.SDA)
 
#low range of the sensor (this will be blue on the screen)
MINTEMP = 26.
 
#high range of the sensor (this will be red on the screen)
MAXTEMP = 32.
 
#how many color values we can have
COLORDEPTH = 1024
 

#initialize the sensor
sensor = Adafruit_AMG88xx.Adafruit_AMG88xx()
 
# pylint: disable=invalid-slice-index
points = [(math.floor(ix / 8), (ix % 8)) for ix in range(0, 64)]
grid_x, grid_y = np.mgrid[0:7:64j, 0:7:64j]
# pylint: enable=invalid-slice-index
 
#sensor is an 8x8 grid so lets do a square
height = 320   
width = 320
 
#the list of colors we can choose from
blue = Color("indigo")
colors = list(blue.range_to(Color("red"), COLORDEPTH))
 
#create the array of colors
colors = [(int(c.blue * 255), int(c.green * 255), int(c.red * 255)) for c in colors]


#some utility functions
def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


manager = Manager()
mboxes = manager.list()
def f(frame):
    # convert the input frame from (1) BGR to grayscale (for face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    global mboxes

    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
        minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

    tboxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
    mboxes[:] = []
    mboxes += tboxes


detector = cv2.CascadeClassifier("haarcascade.xml")

vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)



proc = 1
p = Pool(processes = proc)
boxes = []
encodings = []
pp = []
blank  = 0;

frame = vs.read()
for i in range(proc):
    pp.append(p.apply_async(f, [frame]))


show_cam = True
show_temp = True

# sensor._fps = const(0x01);

write_cnt = -1;
file_cnt = 1;
 
while True:
    # print(sensor.readThermistor())
    # print(sensor._ave.MAMOD)
    # print(sensor._fpsc.FPS)
    #read the pixels
    pixels = np.flip(np.array(sensor.readPixels()).reshape((8,8)),1).T
    # print(np.around(pixels,1))
    # print("")
    n_pixels = map_value(pixels, MINTEMP, MAXTEMP, 0, COLORDEPTH - 1).reshape(-1)
    #perform interpolation
    bicubic = griddata(points, n_pixels, (grid_x, grid_y), method='cubic')

    frame2 = vs.read()
    frame2 = imutils.resize(frame2, width=500)

    for i in range(proc):
        if pp[i].ready() == True:
            pp[i] = p.apply_async(f, [frame2])
            if mboxes[:] != [] or blank == 2: #tolerance to fast movement fail detection
                boxes = []
                boxes += mboxes;
                blank = 0
            else:
                blank += 1
            break

    # cv2.imshow('Frame2',frame2)
    #frame = np.array([[colors[np.clip(np.int(bicubic[i,j]), 0, COLORDEPTH- 1)] for j in range(bicubic.shape[1])] for i in range(bicubic.shape[0])])/255
    bicubic = np.uint8(np.clip(bicubic,0,COLORDEPTH- 1)/(COLORDEPTH- 1)*255)
    frame = cv2.applyColorMap(bicubic, cv2.COLORMAP_JET)
    frame = imutils.resize(frame, width=500)
    bicubic = cv2.cvtColor(bicubic, cv2.COLOR_GRAY2RGB)
    bicubic = imutils.resize(bicubic, width=500)
    half = int(frame2.shape[0]/2)
    if show_cam:
        #frame[250-half:250+half+1] =  frame[250-half:250+half+1]*(bicubic[250-half:250+half+1]/255)*0.5 + frame2*(1- (bicubic[250-half:250+half+1]/255)*0.5)
        frame[250-half:250+half+1] =  frame[250-half:250+half+1]*0.5 + frame2*0.5

    for (top, right, bottom, left) in boxes:
        # rescale the face coordinates
        top = int(top + 250-half)
        right = int(right)
        bottom = int(bottom + 250-half)
        left = int(left)

        # draw the predicted face name on the image
        cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15

    if show_temp:
        for i in range(pixels.shape[0]):
            for j in range(pixels.shape[1]):
                cv2.putText(frame,str(np.around(pixels[j,i],1).T), (int(500/8*i), int(500/8*j+500/16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('Frame',frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    elif key == ord("c"):
        show_cam = 1-show_cam
    elif key == ord("t"):
        show_temp = 1-show_temp
    elif key == ord("f"):
        sensor.setFPS()
    elif key == ord("m"):
        sensor.setMovingAverageMode(1-sensor._ave.MAMOD)
    elif key == ord("z"):
        if write_cnt == -1:
            show_cam = 0
            write_cnt = 0
            f = open('output/' + str(file_cnt) + '.txt', 'w')
    if write_cnt != -1:
        for pixel_y in pixels:
            for pixel in pixel_y:
                f.write("%s " % pixel)
            f.write("\n")
        f.write("\n")
        print(write_cnt)
        write_cnt += 1
    if write_cnt >= 288000:
        show_cam = 1
        write_cnt = -1
        file_cnt += 1


cv2.destroyAllWindows()
vs.stop()