#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import CompressedImage, Image
from moveit_msgs.msg import DisplayTrajectory
from geometry_msgs.msg import TwistStamped
from moveit_msgs.msg import ExecuteTrajectoryActionGoal

# Python 2/3 compatibility imports
import sys
import copy
import rospy
import geometry_msgs.msg
import moveit_commander

import cv2
import os
import time
import numpy as np

try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))


from std_msgs.msg import String

deg2rad = 3.141592653589793 / 180
rad2deg = 1 / deg2rad

class GenericBehavior(object):
    """
    Generic behavior class.rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        pub.publish(twist)
        rate.sleep()
    """
    def __init__(self,accelerate=False,imgOrientCW=True):
        global rad2deg
        self.accelerate = accelerate
        self.imgOrientCW = imgOrientCW

        if self.accelerate:
            self.twist = TwistStamped()
            self.pub = rospy.Publisher(
                "/sb_cmd_state", TwistStamped, queue_size=10
                )
            self.rate = rospy.Rate(10) # 10hz
            # while not rospy.is_shutdown():
            # self.pub.publish(self.twist)
            # self.rate.sleep()

            self.currentState = np.array([0,0,0,0])
        else:
            self.pub = rospy.Publisher(
                "/move_group/display_planned_path", DisplayTrajectory, queue_size=20 # to speed up, publish to rostopic in sb_interface.py
            )
            self.robot = moveit_commander.RobotCommander()
            self.scene = moveit_commander.PlanningSceneInterface()

            self.group_name = "survivor_buddy_head"
            self.move_group = moveit_commander.MoveGroupCommander(self.group_name)

            self.currentState = np.array(self.move_group.get_current_joint_values()) * rad2deg

        self.audio_sub = rospy.Subscriber("/audio", Float32MultiArray, callback=self.callback_1, queue_size=1)
        self.camera_sub = rospy.Subscriber("/camera/image/compressed", CompressedImage, callback=self.callback_2, queue_size=1)
        rospy.loginfo("Node started.")

        # Initialize variables
        self.attention = False
        self.isLost = False
        self.logFaceHist = False
        self.lims = [0,0]
        self.face_vec = np.array([0,0])
        self.box_dims = np.array([[0,0],[0,0]])
        self.face_capture_hist = [0,[0,0,0,0,0,0,0,0,0,0,0,0]]
        self.frame = 1
        self.counter = 1

        # Initalize with robot sleep behavior
        self.behavior_sleep()

    def callback_1(self, data):
        # Audio data stream
        if (max(data.data) > 0.8) & (not self.attention):
            # Awaken robot with loud noise (clap) and begin search
            self.attention = True # Cannot execute again until asleep
            self.behavior_wake()
            self.behavior_engage()
            print('There was no human to be found :(')
            self.behavior_sleep()

    def callback_2(self,data):
        # Image data stream
        if (self.frame % 3) == 0:
            
            # Process the image
            np_arr = np.frombuffer(data.data, np.uint8)
            image_np = cv2.imdecode(np_arr, 1)
            if self.imgOrientCW:
                image_np = cv2.rotate(image_np, cv2.ROTATE_90_CLOCKWISE)
            else:
                image_np = cv2.rotate(image_np, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Image dimensions
            image_dims = np.flip(np.array(image_np.shape))
            image_center = np.flip(np.array(image_np.shape[:2])//2) # Image center
            min_image_dim = min(image_dims[1:]) # Smallest dimension of image
            self.lims = [min_image_dim // 2, min_image_dim//10] # Upper and lower face detection limits
            
            # Import .xml file
            file_path = os.path.abspath(os.path.dirname(__file__))
            face_cascade = cv2.CascadeClassifier(file_path + "/haarcascade_frontalface_default.xml")

            # Draw circles indicating upper and lower sensing limits
            cv2.circle(image_np, (image_center[0],image_center[1]), self.lims[0], (255, 0, 0), 1) # Upper limit
            cv2.circle(image_np, (image_center[0],image_center[1]), self.lims[1], (255, 0, 0), 1) # Lower limit

            # Applying the face detection method on the grayscale image
            gray_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            faces_rect = face_cascade.detectMultiScale(
                image=gray_img,
                scaleFactor=1.3,
                minNeighbors=3,
                minSize = (self.lims[1]*2, self.lims[1]*2)) # tune these parameters

            # Iterating through rectangles of detected faces
            for (x, y, w, h) in faces_rect:
                cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 255, 0), 1)

            # Compute and visualize the location of the nearest face
            if len(faces_rect) > 0:
                # Store the position of the closest face
                idx_max_face = np.argmax(faces_rect[:,2]*faces_rect[:,3])
                face_rect = faces_rect[idx_max_face,:]
                # Face center
                face_center = [face_rect[0]+face_rect[2]//2, face_rect[1]+face_rect[3]//2]
                # Box dimensions
                box_dim_old = self.box_dims[:,0]
                self.box_dims = np.vstack((face_rect[2:],box_dim_old))
                # self.box_dims[:,0] = face_rect[2:]
                # Compute vector from center of image to center of face
                face_vec_flip = face_center - image_center
                self.face_vec = np.array([face_vec_flip[0], -face_vec_flip[1]]) # move origin from top left to bottom left
                # Draw compute vector from center of image to center of face and draw the line
                cv2.line(image_np,(image_center[0],image_center[1]), (face_center[0],face_center[1]), (0,0,255))

            # Log face detection buffer
            if self.logFaceHist:
                if len(faces_rect) > 0:
                    # Detect a new face
                    if self.face_capture_hist[0] == 0:
                        print('New face detected.')
                        self.face_capture_hist[0] = 1
                        self.isLost = False
                    
                    # Indicate if lost face has been found again
                    elif self.isLost:
                        print('Found you!')
                        self.isLost = False

                    # Cycle face detection history
                    self.face_capture_hist[1].pop(0)
                    self.face_capture_hist[1].append(1)

                else:
                    # Cycle face detection history
                    self.face_capture_hist[1].pop(0)
                    self.face_capture_hist[1].append(0)

                    # Detect if a face has been lost
                    if not (1 in self.face_capture_hist[1]):
                        print('Cannot find face.')
                        self.face_capture_hist[0] = 0

                    # Indicate that face has been temporarily lost
                    else:
                        print(f'Get back in frame.')
                        self.isLost = True

            # Display image
            cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Face Detection", 800, 500)
            cv2.imshow("Face Detection", image_np)
            cv2.waitKey(3)

            # For synchronizing while loop in mirroring behavior
            self.counter += 1

        # For image sample rate
        self.frame += 1
    
    def behavior_sleep(self):
        # Robot falls asleep
        # if self.accelerate: # Too slow for MoveIt
        # Get the robot ready for sleep
        print('Getting sleepy....')
        goalState = [20, 0, 0, 10]
        self.move_robot(goalState)
        rospy.sleep(2)

        # Move the robot to sleep position
        print('Goodnight...Zzzzzzz.....')
        goalState = [40, 0, 0, 20]
        self.move_robot(goalState)

        self.attention = False # Robot will now be responsive to loud noises again
        self.logFaceHist = False # Stop logging face detection info

    def behavior_wake(self):
        # Robot awakens
        print('Clap detected!')

        # Wake robot and set to default position
        goalState = [0, 0, 0, 0]
        self.move_robot(goalState)
        rospy.sleep(2)

    def behavior_engage(self):
        # Robot searches for person
        print('Seeking human...')
        
        maxSearch = 10
        for i in range(maxSearch):
            print(f'Search attempt {i+1} of {maxSearch}')

            goalState = [0, 0, 0, 0]
            self.move_robot(goalState)

            if self.accelerate:
                omega1 = np.random.randint(-2,2)
                omega2 = np.random.randint(-8,8)
                omega3 = np.random.randint(-3,3)
                omega4 = np.random.randint(-2,2)
                omega = [omega1,omega2,omega3,omega4]
                goalState = self.currentState + omega
                self.move_robot(goalState)
                rospy.sleep(1)

                nMoves = np.random.randint(1,8)
                for j in range(nMoves):
                    goalState = self.currentState + omega
                    self.move_robot(goalState)
                    if j == nMoves-1:
                        rospy.sleep(2)
                    else:
                        rospy.sleep(0.2)
            else:

                # Move the robot in a random search motion
                j1 = np.random.randint(-15,5)
                j2 = np.random.randint(-60,60)
                j3 = np.random.randint(-30,30)
                j4 = np.random.randint(-10,10)
                goalState = [j1,j2,j3,j4]
                self.move_robot(goalState)

            # Maintain position for a (random) fixed time to identify a face
            tStart = time.time()
            tElapsed = time.time() - tStart
            tSearch = np.float(np.random.randint(2,4))/100
            self.logFaceHist = True # Log face detection history
            while (tElapsed < tSearch) & (self.face_capture_hist[0]==0):
                tElapsed = time.time() - tStart
            
            if self.face_capture_hist[0] == 1:
                # Engage mirroring behavior
                self.behavior_mirror()
                break

    def behavior_mirror(self):
        # Robot mirrors human behavior
        global deg2rad
        print('Human acquired. Now mirroring movements.')

        # Straighten head tilt and nod
        goalState = [self.currentState[0], self.currentState[1], 0, 0]
        self.move_robot(goalState)

        # Set robot arm displacement scaling limits
        upper_lim_xy = self.lims[0]
        lower_lim_xy = self.lims[1]

        # Loop while a face is detected
        counter_old = self.counter
        while self.face_capture_hist[0]==1:
            if self.counter > counter_old: # Synchronize with image capture rate to not over-process
                # print(self.face_capture_hist)
                # Vector magnitudes
                vec_mag_xy = np.linalg.norm(self.face_vec) # In-plane positional displacement
                vec_mag_z = np.linalg.norm(self.box_dims[0,:]) - np.linalg.norm(self.box_dims[1,:]) # Out-of-plane velocity
                
                # Cower away if the face is too close
                if np.linalg.norm(self.box_dims[:,0]) > upper_lim_xy * 1.5:
                    self.behavior_cower(triggerClose=True)
                    break

                # Check if vector length is greater than lower limit
                if vec_mag_xy > lower_lim_xy:
                    if vec_mag_xy > upper_lim_xy:
                        mag_xy = 1
                    else:
                        mag_xy = (vec_mag_xy - lower_lim_xy) / (upper_lim_xy - lower_lim_xy) # scale max displacement from 0 to 1
                else:
                    mag_xy = 0

                # Check if face is moving too close too quickly
                mag_z = np.sign(vec_mag_z)*np.sqrt(abs(vec_mag_z))
                if mag_z > 12:
                    self.behavior_cower(triggerClose=False)
                    break
                elif mag_z < 2:
                    scale_z = 0
                else:
                    scale_z = mag_z / 12

                # Scale each axis accordingly
                angle = np.arctan2(self.face_vec[1],self.face_vec[0])
                scale = [mag_xy*np.cos(angle),mag_xy*np.sin(angle),scale_z]

                # Angular rates
                # omega = [-20*scale[1], -20*scale[0], 0, 0] # Current working form # Negative due to robot angle definitions 
                omega = [-10*scale[1] + 10*scale[2], -20*scale[0], 0, 10*scale[1] - 10*scale[2]] # Negative due to robot angle definitions
                # omega = np.array([-20*scale[2], -20*scale[0], 0, -10*scale[1]]) # Negative due to robot angle definitions

                # Move joints to mirror human motion (X = X0 + dt*Xdot)
                goalState = self.currentState + omega
                self.move_robot(goalState)

            counter_old = self.counter
        
        # If face is lost, search for another
        print('Human has been lost.')
        self.behavior_engage()

    def behavior_cower(self,triggerClose):
        # Cower in fear
        if triggerClose:
            print('Eeeeeek!! Too close!')
        else:
            print('Yikes! Slow down!')

        self.logFaceHist = False # Stop logging face detection history

        # Move robot in a scared, timid motion
        goalState = [-40, self.currentState[1], self.currentState[2], 40]
        self.move_robot(goalState)
        rospy.sleep(3)

        goalState = [-20, self.currentState[1], self.currentState[2], 20]
        self.move_robot(goalState)
        rospy.sleep(2)

        goalState = [0, self.currentState[1], self.currentState[2], 0]
        self.move_robot(goalState)
        rospy.sleep(1)

    def move_robot(self, goalState):
        # Use MoveIt to move the robot to the joint goal positions
        global deg2rad

        if self.accelerate:
            # Bypass MoveIt to accelerate robot motion
            self.twist.twist.linear.x =  -goalState[0]
            self.twist.twist.linear.y =  -goalState[1]
            self.twist.twist.linear.z =   goalState[2]
            self.twist.twist.angular.x = -goalState[3]

            self.pub.publish(self.twist)
            # self.rate.sleep()
            self.currentState = np.array(goalState)

        else:
            # Joint goal states
            joint_goal = self.move_group.get_current_joint_values()

            joint_goal[0] = goalState[0] * deg2rad
            joint_goal[1] = goalState[1] * deg2rad
            joint_goal[2] = goalState[2] * deg2rad
            joint_goal[3] = goalState[3] * deg2rad

            # Move joints
            self.move_group.go(joint_goal, wait=True)
            plan = self.move_group.plan()
            self.move_group.stop()

            display_trajectory = DisplayTrajectory()
            display_trajectory.trajectory_start = self.robot.get_current_state()
            self.pub.publish(display_trajectory)

            # Execute plan
            self.move_group.execute(plan[1], wait=True)


if __name__ == '__main__':
    rospy.init_node("lab_2_node",anonymous=True)
    moveit_commander.roscpp_initialize(sys.argv)

    GenericBehavior(accelerate=True,imgOrientCW=True)
    
    rospy.spin()