
# import libraries
import numpy as np
import cv2
from picamera2 import Picamera2
import time
import math
import os
import board
import neopixel
import multiprocessing
from rpi_ws281x import *
try:
    from worker_comm import stop_program
except ImportError:
    from irobot_edu_sdk.utils import stop_program

from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import event, hand_over, Color, Robot, Root, Create3
from irobot_edu_sdk.music import Note
print("Initialized libraries")

name = "CapstoneRobot1"


pixels = neopixel.NeoPixel(board.D12, 150) 
pixels.show()

robot = Create3(Bluetooth(name))
###### CONST DATA ########
# Parameters of the program that are static
width = 300 
height = 240
IMG_WIDTH = 300
MAX_IR_VALUE = 255
LED_COUNT = 90
CAM_FOV_OVER_IR_FOV = 0.5
CAM_LED_COUNT = int(LED_COUNT * CAM_FOV_OVER_IR_FOV)
CAM_LED_OFFSET = int((LED_COUNT - CAM_LED_COUNT) / 2)
DEBUG = True
SPEED = 10
TH = 150
##### ON OFF DATA #######
num_bumps = 0
MAX_BUMPS = 100
###### STATE DATA ########
sensor_data = [0] * 7
people_data = [0] * CAM_LED_COUNT
bump_data = [0, 0]
###### LIGHTS SETUP ##########


values = [0, 0, 0] * 90 # RGB
def resetLights(left = 0, right = LED_COUNT - 1):
    for i in range(left, right):
        pixels[i] = [0, 0, 0]    
    pixels.show()
### LOADING LIGHT FLASH
for i in range(89):
    pixels[i] = [50, 100, 100]
pixels.show()
time.sleep(2)
for i in range(89):
    pixels[i] = [0, 0, 0]
pixels.show()

# resetLights()
#for x in range (0, 5):
    #strip.setPixelColor(x, Color(0,255,0))
def updateLights():
    global pixels, bump_data
    sensorIndexScale = (7 + 1) / CAM_LED_COUNT
    for i in range(CAM_LED_OFFSET, LED_COUNT - CAM_LED_OFFSET - 1):
        sensorIndex = int((i - CAM_LED_OFFSET) * sensorIndexScale)
        if (sensorIndex > 6):
            continue
        personIndex = i - CAM_LED_OFFSET
        
        sensorValue = sensor_data[sensorIndex]
        sensorLEDValue = 0
        personLEDValue = 0
        if (sensorValue > 1000):
            sensorLEDValue = 200 
        elif (sensorValue > 500):
            sensorLEDValue = 150
        elif (sensorValue > 100):
            sensorLEDValue = 50
        else:    
            if (sensorValue > 50):
                sensorLEDValue = 20
            if (sensorValue > 30):
                sensorLEDValue = 10
            if (personIndex > 0 and personIndex < len(people_data)):
                personValue = people_data[personIndex]
                if (personValue > 0):
                    personLEDValue = 150

        pixels[i] = [sensorLEDValue, 0, personLEDValue]
    if(bump_data[0] > 0):

        brightness = (bump_data[0] - 1) * 70
        for i in range(0, CAM_LED_OFFSET):
            pixels[i] = [brightness, 0, 0]
        bump_data[0] -= 1
        
    if(bump_data[1] > 0):
        brightness = (bump_data[1] - 1) * 70
        for i in range(LED_COUNT - CAM_LED_OFFSET, LED_COUNT):
            pixels[i] = [brightness, 0, 0]
        bump_data[1] -= 1



    

###### PI CAMERA CONFIG ######
# initialize the camera and grab a reference to the raw camera capture
camera = Picamera2()
config = camera.create_preview_configuration(main={"format": 'XRGB8888', "size": (320, 240)})
#config = camera.create_preview_configuration({'format': 'BGR888'})
#sets the camera to RGB instead of RGBW


###### DEBUG PRINT METHODS
def printIRdata():    
    print(sensor_data)
    blocks = [chr(0x2591), chr(0x2592), chr(0x2593), chr(0x2588)]
    scale = math.log2(MAX_IR_VALUE) / len(blocks)
    printstr = ""
    for num in sensor_data:
        num = min(num, MAX_IR_VALUE - 1)
        if (num > 1):
            num = math.log2(num)
        char = blocks[int(num / scale)]
        printstr += char * 12
    print(printstr)
def printHumansData():
    #blocks = [chr(0x2591), chr(0x2592), chr(0x2593), chr(0x2588)]
    #scale = MAX_IR_VALUE / len(blocks)
    printstr = "           "
    printWidth = CAM_LED_COUNT
    printVals = [0] * printWidth
    for i in range(CAM_LED_COUNT):
        charI = int(i * printWidth / CAM_LED_COUNT)
        printVals[charI] += people_data[i]

    for i in range(printWidth):
        if (printVals[i] > 0):
            printstr += chr(0x2588)
        else:
            printstr += "x"#chr(0x2591)
    print(printstr)
camera.configure(config)
camera.start()

# allow the camera to warmup
time.sleep(0.1)
# grab an image from the camera


### ROBOT EVENT METHODS #####

@event(robot.when_bumped, [False, True])
async def bumped(robot):
    global bump_data, num_bumps
    print('Right bump sensor hit')
    bump_data = [bump_data[0], 4]
    if (num_bumps > MAX_BUMPS):
        num_bumps = 0
    
@event(robot.when_bumped, [True, False])
async def bumped(robot):
    global bump_data, num_bumps
    print('Left bump sensor hit')
    bump_data = [4, bump_data[1]]
    num_bumps += 1
    if (num_bumps > MAX_BUMPS):
        num_bumps = 0
@event(robot.when_bumped, [True, True])
async def bumped(robot):
    global bump_data, num_bumps
    print('Both bump sensor hit')
    bump_data = [4, 4]
    num_bumps += 1
    if (num_bumps > MAX_BUMPS):
        num_bumps = 0
    

### TEST ROBOT MOVEMENT ###
async def forward(robot):
    await robot.set_lights_on_rgb(0, 255, 0)
    await robot.set_wheel_speeds(SPEED, SPEED)
    
async def backoff(robot, direction):
    await robot.set_lights_on_rgb(255, 80, 0)
    await robot.move(-20)
    await robot.turn_left(direction)


def front_obstacle():
    return sensor_data[3] > TH
    
def turnDirection(): #negative left positive right
    leftSum = sensor_data[0] + sensor_data[1] + sensor_data[2]
    rightSum = sensor_data[4] + sensor_data[5] + sensor_data[6]
    if (leftSum < rightSum):
        return -45
    return 45
    
### ROBOT LOOP METHODS
'''
@event(robot.when_play)
async def play(robot):
    await forward(robot)
    while True:
        if front_obstacle():
            direction = turnDirection()
            print("Turning: ", direction, "degrees")
            await backoff(robot, direction)
            await forward(robot)
     '''   
     
########
# INPUT 
########
def doInputIR():
    global sensors, sensorData
    sensors = (await robot.get_ir_proximity()).sensors
    sensor_data[0] = sensors[0]
    sensor_data[1] = sensors[1]
    sensor_data[2] = sensors[2]
    sensor_data[3] = sensors[3]
    sensor_data[4] = sensors[4]
    sensor_data[5] = sensors[5]
    sensor_data[6] = sensors[6]
def doInputCamera():
    # print("Ir sensor input:", sensors)
    img = camera.capture_array()
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    for i in range(CAM_LED_COUNT):
        people_data[i] = 0
    # people_data = [0] * CAM_LED_COUNT
    
    # display the image on screen and wait for a keypress.
    # I only turn this on when debugging, I
    cv2.imshow("Image", img)
    # cv2.waitKey(0)
    
    process_frame(img)

def doInput():
    global sensors, sensor_data
    doInputIR()
    doInputCamera()
    
######## 
# STATE
########
# In this case, the only state needing to be modified is the lights, because that 
# was our only output we were able to do
    
def updateState():
    updateFromDetections()
    updateLights()  

########
# OUTPUT
########    
def doOutput():
    pixels.show()
    if DEBUG:
        printIRdata()
        printHumansData()

# When play: the methods that are called on start. This is where we hold the main loop
@event(robot.when_play)
async def play(robot):
    global num_bumps
    while True:
        if (num_bumps >= MAX_BUMPS):
            time.sleep(1)
            continue
        doInput()
        updateState()
        doOutput()     


        await hand_over()
    camera.release()
    cv2.destroyAllWindows()
#### EXPERIMENTAL SEPARATE LIGHTS LOOP
#@event(robot.when_play)
#async def play(robot):
 #   while True:
 #       updateLights()
 #       await hand_over()
##### UPDATING STATE BASED ON SEEN OBJECT
 
    
#### UPDATE STATE BASED ON DETECTIONS
def updateFromDetections(detections):
    for i in range(detections.shape[2]):
        # confidence of prediction
        confidence = detections[0, 0, i, 2]
        # set confidence level threshold to filter weak predictions
        if confidence > 0.5:
            # get class id
            class_id = int(detections[0, 0, i, 1])
            # scale to the frame
            x_top_left = int(detections[0, 0, i, 3] * width) 
            y_top_left = int(detections[0, 0, i, 4] * height)
            x_bottom_right   = int(detections[0, 0, i, 5] * width)
            y_bottom_right   = int(detections[0, 0, i, 6] * height)
            
            # draw bounding box around the detected object
            
            
            if class_id == 15:
                print("Found person")
                leftPixel = x_top_left * CAM_LED_COUNT / IMG_WIDTH# + CAM_LED_OFFSET
                leftPixel = int(leftPixel)
                rightPixel = x_bottom_right * CAM_LED_COUNT / IMG_WIDTH# + CAM_LED_OFFSET
                rightPixel = int(rightPixel)
                # print(leftPixel, rightPixel)
                for i in range(leftPixel, rightPixel):
                    people_data[i] = 1
                #print(leftPixel, rightPixel)
                
                # cv2.rectangle(frame, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right), (0, 255, 0))
                # detect only people
                # get class label
                # label = classNames[class_id] + ": " + str(confidence)
                '''
                # get width and text of the label string
                (w, h),t = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                y_top_left = max(y_top_left, h)
                # draw bounding box around the text
                cv2.rectangle(frame, (x_top_left, y_top_left - h), 
                                   (x_top_left + w, y_top_left + t), (0, 0, 0), cv2.FILLED)
                cv2.putText(frame, label, (x_top_left, y_top_left),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                '''
'''
@event(robot.when_play)
async def play(robot):
    time.sleep(1)
    while True:
        img = camera.capture_array()
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # display the image on screen and wait for a keypress
        #cv2.imshow("Image", img)
        #cv2.waitKey(0)
        process_frame(img)
        #if cv2.waitKey(1) >= 0:  # Break with ESC 
        #    break
        await hand_over()
    camera.release()
    cv2.destroyAllWindows()
'''

### DEFAULT CODE FROM THIS POINT ON
# download the model as plain text as a PROTOTXT file and the trained model as a CAFFEMODEL file from  here: https://github.com/djmv/MobilNet_SSD_opencv

# path to the prototxt file with text description of the network architecture
prototxt = "MobileNetSSD_deploy.prototxt"
# path to the .caffemodel file with learned network
caffe_model = "MobileNetSSD_deploy.caffemodel"

# read a network model (pre-trained) stored in Caffe framework's format
net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)

# dictionary with the object class id and names on which the model is trained
classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

# capture the webcam feed
# cap = rawCapture

def process_frame(frame):
    #print("Processing frame")
    #print("Current IR values: ", sensor_data)
    # ret, frame = cap.read()
    
    cv2.imshow("Image", frame)
    
    # size of image
    width = frame.shape[1] 
    height = frame.shape[0]
    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(frame, scalefactor = 1/127.5, size = (300, 240), mean = (127.5, 127.5, 127.5), swapRB=True, crop=False)
    # blob object is passed as input to the object
    net.setInput(blob)
    
    # network prediction
    detections = net.forward()
    # detections array is in the format 1,1,N,7, where N is the #detected bounding boxes
    # for each detection, the description (7) contains : [image_id, label, conf, x_min, y_min, x_max, y_max]
    
    updateFromDetections(detections)
    updateLights()
    #cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    #cv2.imshow("frame", frame)

def camera_input():
    while True:
        img = camera.capture_array()
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # display the image on screen and wait for a keypress
        #cv2.imshow("Image", img)
        #cv2.waitKey(0)
        process_frame(img)
        if cv2.waitKey(1) >= 0:  # Break with ESC 
            break
        #await hand_over()
    camera.release()
    cv2.destroyAllWindows()

#camera_input()


'''
while True:
    ret, frame = cap.read()
    
    # size of image
    width = frame.shape[1] 
    height = frame.shape[0]
    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(frame, scalefactor = 1/127.5, size = (300, 300), mean = (127.5, 127.5, 127.5), swapRB=True, crop=False)
    # blob object is passed as input to the object
    net.setInput(blob)
    # network prediction
    detections = net.forward()
    # detections array is in the format 1,1,N,7, where N is the #detected bounding boxes
    # for each detection, the description (7) contains : [image_id, label, conf, x_min, y_min, x_max, y_max]
    for i in range(detections.shape[2]):
        # confidence of prediction
        confidence = detections[0, 0, i, 2]
        # set confidence level threshold to filter weak predictions
        if confidence > 0.5:
            # get class id
            class_id = int(detections[0, 0, i, 1])
            # scale to the frame
            x_top_left = int(detections[0, 0, i, 3] * width) 
            y_top_left = int(detections[0, 0, i, 4] * height)
            x_bottom_right   = int(detections[0, 0, i, 5] * width)
            y_bottom_right   = int(detections[0, 0, i, 6] * height)
            
            # draw bounding box around the detected object
            cv2.rectangle(frame, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right),
                          (0, 255, 0))
            
            if class_id in classNames:
                # get class label
                label = classNames[class_id] + ": " + str(confidence)
                # get width and text of the label string
                (w, h),t = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                y_top_left = max(y_top_left, h)
                # draw bounding box around the text
                cv2.rectangle(frame, (x_top_left, y_top_left - h), 
                                   (x_top_left + w, y_top_left + t), (0, 0, 0), cv2.FILLED)
                cv2.putText(frame, label, (x_top_left, y_top_left),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                                
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) >= 0:  # Break with ESC 
        break

'''

#lightsProcess = multiprocessing.Process(target=updateLights, args=())
#lightsProcess.start()
#robotProcess = multiprocessing.Process(target=robot.play)
#robotProcess.start()
robot.play()
