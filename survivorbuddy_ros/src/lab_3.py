#!/usr/bin/env python3
import rospy
from std_msgs.msg import Int16
from sensor_msgs.msg import CompressedImage

# Python 2/3 compatibility imports
import sys
import rospy
import moveit_commander

import os
import time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = os.path.abspath(os.path.dirname(__file__)) + '/gesture_recognizer.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
mpDrawStyle = mp.solutions.drawing_styles

class DetectGesture(object):
    def __init__(self,imgOrient='CW'):
        self.imgOrient = imgOrient

        self.pub = rospy.Publisher("/gesture_commands", Int16, queue_size=20)
        self.sub = rospy.Subscriber("/camera/image/compressed", CompressedImage, callback=self.callback, queue_size=1)
        rospy.loginfo("Node started.")

        # Initialize variables
        self.frame = 1
        # self.t0 = 0
        # self.tf = 0

    def callback(self,data):
        # Image data stream
        global recognizer

        if (self.frame % 2) == 0:            
            # Process the image
            img_raw = np.frombuffer(data.data, np.uint8)
            img = cv2.imdecode(img_raw, 1)

            # Orient image
            if self.imgOrient == 'CW':
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif self.imgOrient == 'CCW':
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Gesture detection and ROS publisher
            img_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            recognition_result = recognizer.recognize(img_mp)
            if recognition_result.gestures:
                top_gesture = recognition_result.gestures[0][0]
                if top_gesture.category_name == "None":
                    gesture_index = 0
                    command = 'None'
                elif top_gesture.category_name == "Closed_Fist":
                    gesture_index = 1
                    command = 'pause video'
                elif top_gesture.category_name == "Open_Palm":
                    gesture_index = 2
                    command = 'play video'
                elif top_gesture.category_name == "Pointing_Up":
                    gesture_index = 3
                    command = 'unmute video'
                elif top_gesture.category_name == "Thumb_Down":
                    gesture_index = 4
                    command = 'volume down'
                elif top_gesture.category_name == "Thumb_Up":
                    gesture_index = 5
                    command = 'volume up'
                elif top_gesture.category_name == "Victory":
                    gesture_index = 6
                    command = 'mute video'
                elif top_gesture.category_name == "ILoveYou":
                    gesture_index = 7
                    command = 'exit video'
                print(f"{top_gesture.category_name} ({top_gesture.score:.2f})")
                print(command,'\n')
                self.pub.publish(gesture_index)
            else:
                print('No gesture recognized')
                command = 'null'

            # Display hand and gesture detection
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hands_detect = hands.process(img_rgb)
            if hands_detect.multi_hand_landmarks:
                for hand_landmarks in hands_detect.multi_hand_landmarks:
                    mpDraw.draw_landmarks(img,
                                          hand_landmarks,
                                          mpHands.HAND_CONNECTIONS,
                                          mpDrawStyle.get_default_hand_landmarks_style(),
                                          mpDrawStyle.get_default_hand_connections_style())
                      
            # self.tf = time.time()
            # fps = int(1/(self.tf-self.t0))
            # self.t0 = self.tf

            # Display image
            cv2.namedWindow("Gesture Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Gesture Detection", 800, 500)
            cv2.putText(img, command, (10,70), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 3)
            cv2.imshow("Gesture Detection", img)
            cv2.waitKey(1)

        # For image sample rate
        self.frame += 1

if __name__ == '__main__':
    rospy.init_node("lab_3_node",anonymous=True)
    moveit_commander.roscpp_initialize(sys.argv)

    DetectGesture(imgOrient='None')
    
    rospy.spin()