#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from perception_stack.msg import TrackedObjectList, TrackedObject

class OpticalFlowEstimator:
    def __init__(self):
        rospy.init_node('optical_flow_estimator', anonymous=True)
        
        # Parameters
        self.prev_gray = None
        self.prev_timestamp = None
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Publisher and Subscriber
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        self.tracked_objects_sub = rospy.Subscriber('/perception/tracked_objects', TrackedObjectList, 
                                                   self.tracked_objects_callback)
        self.flow_image_pub = rospy.Publisher('/perception/optical_flow', Image, queue_size=10)
        self.refined_objects_pub = rospy.Publisher('/perception/refined_objects', TrackedObjectList, queue_size=10)
        
        self.flow_data = None
        self.tracked_objects = None
        
        rospy.loginfo("Optical flow estimator node initialized")
    
    def image_callback(self, msg):
        """Process incoming images for optical flow"""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            timestamp = msg.header.stamp.to_sec()
            
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Create a copy for visualization
            flow_vis = cv_image.copy()
            
            # If we have a previous frame, calculate optical flow
            if self.prev_gray is not None and self.prev_timestamp is not None:
                # Calculate dense optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                
                # Calculate time difference
                dt = timestamp - self.prev_timestamp
                if dt <= 0:
                    dt = 0.033  # Default to 30 fps
                
                # Save flow data for object velocity refinement
                self.flow_data = {
                    'flow': flow,
                    'dt': dt,
                    'timestamp': timestamp
                }
                
                # Create visualization
                # Convert flow to polar coordinates (magnitude and angle)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                # Create HSV image for visualization
                hsv = np.zeros_like(cv_image)
                hsv[..., 0] = ang * 180 / np.pi / 2  # Hue based on angle
                hsv[..., 1] = 255  # Full saturation
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value based on magnitude
                
                # Convert HSV to BGR for display
                flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                
                # Refine tracked objects with flow data if available
                if self.tracked_objects is not None:
                    self.refine_object_velocities()
            
            # Store current frame for next iteration
            self.prev_gray = gray
            self.prev_timestamp = timestamp
            
            # Publish flow visualization
            flow_msg = self.bridge.cv2_to_imgmsg(flow_vis, "bgr8")
            flow_msg.header = msg.header
            self.flow_image_pub.publish(flow_msg)
            
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {str(e)}")
        except Exception as e:
            rospy.logerr(f"Error in optical flow: {str(e)}")
    
    def tracked_objects_callback(self, msg):
        """Store tracked objects for velocity refinement"""
        self.tracked_objects = msg
    
    def refine_object_velocities(self):
        """Refine object velocities using optical flow data"""
        if self.flow_data is None or self.tracked_objects is None:
            return
            
        flow = self.flow_data['flow']
        dt = self.flow_data['dt']
        
        refined_objects = TrackedObjectList()
        refined_objects.header = self.tracked_objects.header
        
        for obj in self.tracked_objects.objects:
            # Get object bounding box
            x, y, w, h = int(obj.x), int(obj.y), int(obj.width), int(obj.height)
            
            # Ensure box is within image bounds
            if x < 0 or y < 0 or x + w >= flow.shape[1] or y + h >= flow.shape[0]:
                refined_objects.objects.append(obj)
                continue
            
            # Extract flow in object region
            obj_flow = flow[y:y+h, x:x+w]
            
            # Calculate mean flow in object region
            mean_flow_x = np.mean(obj_flow[..., 0])
            mean_flow_y = np.mean(obj_flow[..., 1])
            
            # Convert flow to velocity (pixels per second)
            vel_x = mean_flow_x / dt
            vel_y = mean_flow_y / dt
            speed = np.sqrt(vel_x**2 + vel_y**2)
            
            # Create refined object with updated velocity
            refined_obj = TrackedObject()
            refined_obj.id = obj.id
            refined_obj.class_label = obj.class_label
            refined_obj.confidence = obj.confidence
            refined_obj.x = obj.x
            refined_obj.y = obj.y
            refined_obj.width = obj.width
            refined_obj.height = obj.height
            refined_obj.vel_x = vel_x
            refined_obj.vel_y = vel_y
            refined_obj.speed = speed
            
            refined_objects.objects.append(refined_obj)
        
        # Publish refined objects
        self.refined_objects_pub.publish(refined_objects)

if __name__ == '__main__':
    try:
        flow_estimator = OpticalFlowEstimator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
