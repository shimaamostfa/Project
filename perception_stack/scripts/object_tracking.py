#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from perception_stack.msg import TrackedObject, TrackedObjectList

class ObjectTracker:
    def __init__(self):
        rospy.init_node('object_tracker', anonymous=True)
        
        # Parameters
        self.max_disappeared = rospy.get_param('~max_disappeared', 30)  # Max frames before object is considered gone
        self.min_object_area = rospy.get_param('~min_object_area', 500)  # Min area to track an object
        self.next_object_id = 0
        self.objects = {}  # Dictionary to store tracked objects {ID: centroid}
        self.disappeared = {}  # Dictionary to track disappeared frames {ID: count}
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Subscribers and Publishers
        self.image_sub = message_filters.Subscriber('/camera/image_raw', Image)
        self.segmentation_sub = message_filters.Subscriber('/perception/segmentation', Image)
        
        # Synchronize the subscribers
        self.ts = message_filters.TimeSynchronizer([self.image_sub, self.segmentation_sub], 10)
        self.ts.registerCallback(self.tracking_callback)
        
        # Publisher for tracked objects
        self.tracked_objects_pub = rospy.Publisher('/perception/tracked_objects', TrackedObjectList, queue_size=10)
        self.tracked_image_pub = rospy.Publisher('/perception/tracked_image', Image, queue_size=10)
        
        # Class/color mapping from segmentation
        self.class_names = [
            'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        rospy.loginfo("Object tracker node initialized")
    
    def register(self, centroid, bbox, class_id):
        """Register a new object with a unique ID"""
        self.objects[self.next_object_id] = {
            'centroid': centroid,
            'bbox': bbox,
            'class_id': class_id,
            'history': [centroid],  # Track position history for velocity
            'last_update': rospy.Time.now().to_sec()
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """Deregister an object that has disappeared for too long"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, input_centroids, input_bboxes, input_classes):
        """Update tracked objects with new detections"""
        # If we have no objects, register all centroids
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_bboxes[i], input_classes[i])
        
        # Otherwise, try to match input centroids to existing objects
        else:
            # Get IDs and centroids of current objects
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[obj_id]['centroid'] for obj_id in object_ids]
            
            # Compute distances between each pair of object centroids and input centroids
            distances = np.zeros((len(object_centroids), len(input_centroids)))
            for i in range(len(object_centroids)):
                for j in range(len(input_centroids)):
                    distances[i, j] = np.linalg.norm(np.array(object_centroids[i]) - np.array(input_centroids[j]))
            
            # Find the smallest distance for each row (existing object)
            # and sort by distance
            rows = distances.min(axis=1).argsort()
            
            # Find the smallest distance for each column (input centroid)
            # and sort by the ordered rows
            cols = distances.argmin(axis=1)[rows]
            
            # Keep track of which rows and columns we've already examined
            used_rows = set()
            used_cols = set()
            
            # Loop through the sorted row indices
            for (row, col) in zip(rows, cols):
                # If this row or column has already been used, skip it
                if row in used_rows or col in used_cols:
                    continue
                
                # Otherwise, update the object with the new centroid
                object_id = object_ids[row]
                self.objects[object_id]['centroid'] = input_centroids[col]
                self.objects[object_id]['bbox'] = input_bboxes[col]
                self.objects[object_id]['class_id'] = input_classes[col]
                
                # Add to history (limiting to last 10 positions for velocity calculation)
                self.objects[object_id]['history'].append(input_centroids[col])
                if len(self.objects[object_id]['history']) > 10:
                    self.objects[object_id]['history'] = self.objects[object_id]['history'][-10:]
                
                self.objects[object_id]['last_update'] = rospy.Time.now().to_sec()
                self.disappeared[object_id] = 0
                
                # Mark this row and column as used
                used_rows.add(row)
                used_cols.add(col)
            
            # Compute the unused rows and columns
            unused_rows = set(range(distances.shape[0])).difference(used_rows)
            unused_cols = set(range(distances.shape[1])).difference(used_cols)
            
            # If we have more existing objects than input centroids,
            # check if any objects have disappeared
            if distances.shape[0] >= distances.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    # If the object has been missing for too many frames, deregister it
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            
            # Otherwise, register each new input centroid as a new object
            else:
                for col in unused_cols:
                    self.register(input_centroids[col], input_bboxes[col], input_classes[col])
                    
        # Return the updated set of tracked objects
        return self.objects
    
    def extract_objects_from_segmentation(self, segmentation_image):
        """Extract object info from segmentation image"""
        # Convert segmentation to grayscale for contour detection
        gray = cv2.cvtColor(segmentation_image, cv2.COLOR_BGR2GRAY)
        
        # Find unique classes in the segmentation
        unique_classes = np.unique(gray)
        
        # Skip background class (usually 0)
        unique_classes = unique_classes[unique_classes > 0]
        
        centroids = []
        bboxes = []
        class_ids = []
        
        # Process each class separately
        for class_id in unique_classes:
            # Create a binary mask for this class
            mask = np.zeros_like(gray)
            mask[gray == class_id] = 255
            
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process each contour (object instance)
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Skip small objects
                if area < self.min_object_area:
                    continue
                
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    centroids.append((cX, cY))
                    bboxes.append((x, y, w, h))
                    class_ids.append(class_id)
        
        return centroids, bboxes, class_ids
    
    def calculate_velocity(self, history, time_window=0.5):
        """Calculate velocity from position history"""
        if len(history) < 2:
            return 0, 0  # No velocity if not enough history
            
        # Calculate time-based velocity using the last two positions
        pos1 = history[-2]
        pos2 = history[-1]
        
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        
        # Simple velocity estimation (pixels per frame)
        # This could be improved with proper time-based calculations
        vel_x = dx
        vel_y = dy
        
        return vel_x, vel_y
    
    def tracking_callback(self, image_msg, segmentation_msg):
        """Process synchronized image and segmentation data"""
        try:
            # Convert ROS messages to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            segmentation_image = self.bridge.imgmsg_to_cv2(segmentation_msg, "bgr8")
            
            # Extract objects from segmentation
            centroids, bboxes, class_ids = self.extract_objects_from_segmentation(segmentation_image)
            
            # Update tracked objects
            objects = self.update(centroids, bboxes, class_ids)
            
            # Create tracked objects message
            tracked_objects_msg = TrackedObjectList()
            tracked_objects_msg.header = image_msg.header
            
            # Create a visualization image
            vis_image = cv_image.copy()
            
            # Process each tracked object
            for object_id, obj in objects.items():
                centroid = obj['centroid']
                bbox = obj['bbox']
                class_id = obj['class_id']
                
                # Calculate velocity
                vel_x, vel_y = self.calculate_velocity(obj['history'])
                speed = np.sqrt(vel_x**2 + vel_y**2)
                
                # Get class name
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
                
                # Create a tracked object message
                tracked_obj = TrackedObject()
                tracked_obj.id = object_id
                tracked_obj.class_label = class_name
                tracked_obj.confidence = 1.0  # Placeholder, would come from detector
                tracked_obj.x = float(bbox[0])
                tracked_obj.y = float(bbox[1])
                tracked_obj.width = float(bbox[2])
                tracked_obj.height = float(bbox[3])
                tracked_obj.vel_x = float(vel_x)
                tracked_obj.vel_y = float(vel_y)
                tracked_obj.speed = float(speed)
                
                tracked_objects_msg.objects.append(tracked_obj)
                
                # Draw on visualization image
                x, y, w, h = bbox
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(vis_image, centroid, 4, (0, 0, 255), -1)
                
                # Display object ID and class
                text = f"ID: {object_id}, {class_name}"
                cv2.putText(vis_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw velocity vector
                if speed > 1:  # Only show if there's significant movement
                    end_point = (int(centroid[0] + vel_x * 3), int(centroid[1] + vel_y * 3))
                    cv2.arrowedLine(vis_image, centroid, end_point, (255, 0, 0), 2)
            
            # Publish tracked objects message
            self.tracked_objects_pub.publish(tracked_objects_msg)
            
            # Publish visualization image
            vis_msg = self.bridge.cv2_to_imgmsg(vis_image, "bgr8")
            vis_msg.header = image_msg.header
            self.tracked_image_pub.publish(vis_msg)
            
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {str(e)}")
        except Exception as e:
            rospy.logerr(f"Error in tracking: {str(e)}")
            
if __name__ == '__main__':
    try:
        tracker = ObjectTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
