#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class CameraStreamer:
    def __init__(self):
        rospy.init_node('camera_streamer', anonymous=True)
        
        # Parameters
        self.video_source = rospy.get_param('~video_source', 0)  # Default to webcam
        self.frame_rate = rospy.get_param('~frame_rate', 30)
        
        # Publishers
        self.image_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Video capture
        self.cap = None
        self.connect_video_source()
        
    def connect_video_source(self):
        """Connect to the video source (file or camera)"""
        try:
            # Check if video_source is a file path or camera index
            if isinstance(self.video_source, str) and (self.video_source.endswith('.mp4') or 
                                                       self.video_source.endswith('.avi')):
                self.cap = cv2.VideoCapture(self.video_source)
            else:
                # Treat as camera index
                self.cap = cv2.VideoCapture(int(self.video_source))
            
            if not self.cap.isOpened():
                rospy.logerr("Failed to open video source: {}".format(self.video_source))
                return False
                
            rospy.loginfo("Successfully connected to video source: {}".format(self.video_source))
            return True
            
        except Exception as e:
            rospy.logerr("Error connecting to video source: {}".format(str(e)))
            return False
    
    def stream(self):
        """Main loop for streaming video frames"""
        rate = rospy.Rate(self.frame_rate)
        
        while not rospy.is_shutdown():
            if self.cap is None or not self.cap.isOpened():
                if not self.connect_video_source():
                    rospy.sleep(1)
                    continue
                    
            ret, frame = self.cap.read()
            
            if not ret:
                rospy.logwarn("Failed to capture frame, reconnecting...")
                self.cap.release()
                self.cap = None
                continue
                
            try:
                # Convert frame to ROS Image message
                img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                img_msg.header.stamp = rospy.Time.now()
                
                # Publish the image
                self.image_pub.publish(img_msg)
                
            except CvBridgeError as e:
                rospy.logerr("CV Bridge error: {}".format(str(e)))
                
            rate.sleep()
            
    def shutdown(self):
        """Clean up on shutdown"""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        rospy.loginfo("Camera streamer node shutdown")
            
if __name__ == '__main__':
    try:
        streamer = CameraStreamer()
        rospy.on_shutdown(streamer.shutdown)
        streamer.stream()
    except rospy.ROSInterruptException:
        pass
