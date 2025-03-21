#!/usr/bin/env python3

import rospy
from perception_stack.msg import TrackedObjectList, TrackedObject
import message_filters

class FusionNode:
    def __init__(self):
        rospy.init_node('fusion_node', anonymous=True)
        
        # Subscribe to tracked objects and refined objects with optical flow
        self.objects_sub = message_filters.Subscriber('/perception/tracked_objects', TrackedObjectList)
        self.refined_sub = message_filters.Subscriber('/perception/refined_objects', TrackedObjectList)
        
        # Synchronize the subscribers
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.objects_sub, self.refined_sub], 10, 0.1)
        self.ts.registerCallback(self.fusion_callback)
        
        # Publisher for final object list
        self.fused_objects_pub = rospy.Publisher('/perception/fused_objects', TrackedObjectList, queue_size=10)
        
        rospy.loginfo("Fusion node initialized")
    
    def fusion_callback(self, objects_msg, refined_msg):
        """Fuse tracking and optical flow data"""
        # Create a dictionary of refined objects for quick lookup
        refined_dict = {obj.id: obj for obj in refined_msg.objects}
        
        # Create fused objects message
        fused_objects = TrackedObjectList()
        fused_objects.header = objects_msg.header
        
        for obj in objects_msg.objects:
            # Check if we have refined data for this object
            if obj.id in refined_dict:
                refined = refined_dict[obj.id]
                
                # Create a fused object
                fused_obj = TrackedObject()
                fused_obj.id = obj.id
                fused_obj.class_label = obj.class_label
                fused_obj.confidence = obj.confidence
                fused_obj.x = obj.x
                fused_obj.y = obj.y
                fused_obj.width = obj.width
                fused_obj.height = obj.height
                
                # Use refined velocity if available and not too different from tracking
                # Simple outlier rejection
                if abs(refined.vel_x - obj.vel_x) < 20 and abs(refined.vel_y - obj.vel_y) < 20:
                    # Use weighted average (70% optical flow, 30% tracking)
                    fused_obj.vel_x = 0.7 * refined.vel_x + 0.3 * obj.vel_x
                    fused_obj.vel_y = 0.7 * refined.vel_y + 0.3 * obj.vel_y
                else:
                    # Fallback to tracking velocity
                    fused_obj.vel_x = obj.vel_x
                    fused_obj.vel_y = obj.vel_y
                
                # Calculate speed from velocity components
                fused_obj.speed = (fused_obj.vel_x**2 + fused_obj.vel_y**2)**0.5
                
                fused_objects.objects.append(fused_obj)
            else:
                # Just use the tracking data if no refined data
                fused_objects.objects.append(obj)
        
        # Publish fused objects
        self.fused_objects_pub.publish(fused_objects)

if __name__ == '__main__':
    try:
        fusion = FusionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
