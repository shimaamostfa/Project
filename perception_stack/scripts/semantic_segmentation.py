#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import torch
import torchvision
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class SemanticSegmentation:
    def __init__(self):
        rospy.init_node('semantic_segmentation', anonymous=True)
        
        # Parameters
        self.model_name = rospy.get_param('~model', 'fcn_resnet50')  # Default model
        self.device = rospy.get_param('~device', 'cpu')  # Default to CPU
        self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.5)
        
        # Publishers and Subscribers
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        self.segmentation_pub = rospy.Publisher('/perception/segmentation', Image, queue_size=10)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Load model
        self.model = self.load_model()
        self.class_names = self.get_class_names()
        
        rospy.loginfo("Semantic segmentation node initialized")
        
    def load_model(self):
        """Load the semantic segmentation model"""
        try:
            rospy.loginfo(f"Loading {self.model_name} on {self.device}")
            
            # Load a pre-trained segmentation model
            if self.model_name == 'fcn_resnet50':
                model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
            elif self.model_name == 'fcn_resnet101':
                model = torchvision.models.segmentation.fcn_resnet101(pretrained=True)
            elif self.model_name == 'deeplabv3_resnet50':
                model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
            elif self.model_name == 'deeplabv3_resnet101':
                model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
            else:
                rospy.logerr(f"Unknown model: {self.model_name}, defaulting to fcn_resnet50")
                model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
            
            # Set model to evaluation mode and move to device
            model.eval()
            model.to(self.device)
            
            return model
            
        except Exception as e:
            rospy.logerr(f"Error loading model: {str(e)}")
            return None
            
    def get_class_names(self):
        """Get the class names for the COCO dataset"""
        return [
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
    
    def preprocess_image(self, cv_image):
        """Preprocess the image for the model"""
        # Resize if needed
        resized = cv2.resize(cv_image, (520, 520))
        
        # Convert to RGB (from BGR)
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize and convert to tensor
        input_tensor = torch.from_numpy(rgb_image.transpose((2, 0, 1))).float() / 255.0
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        
        return input_tensor.to(self.device)
        
    def image_callback(self, msg):
        """Process incoming image messages"""
        if self.model is None:
            rospy.logerr("Model not loaded, skipping frame")
            return
            
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            original_height, original_width = cv_image.shape[:2]
            
            # Preprocess the image
            input_tensor = self.preprocess_image(cv_image)
            
            # Perform inference
            with torch.no_grad():
                output = self.model(input_tensor)["out"][0]
                
            # Process the output
            output_predictions = output.argmax(0).byte().cpu().numpy()
            
            # Resize back to original dimensions
            segmentation_map = cv2.resize(output_predictions, (original_width, original_height), 
                                          interpolation=cv2.INTER_NEAREST)
            
            # Create a colored segmentation image for visualization
            segmentation_colors = np.zeros((original_height, original_width, 3), dtype=np.uint8)
            
            # Assign different colors to different classes
            for class_idx in range(len(self.class_names)):
                # Random color for each class (but consistent)
                color = np.array([hash(self.class_names[class_idx]) % 256, 
                                 (hash(self.class_names[class_idx]) * 2) % 256,
                                 (hash(self.class_names[class_idx]) * 3) % 256], dtype=np.uint8)
                segmentation_colors[segmentation_map == class_idx] = color
            
            # Publish the segmentation result
            segmentation_msg = self.bridge.cv2_to_imgmsg(segmentation_colors, "bgr8")
            segmentation_msg.header = msg.header
            self.segmentation_pub.publish(segmentation_msg)
            
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {str(e)}")
        except Exception as e:
            rospy.logerr(f"Error in segmentation: {str(e)}")
            
if __name__ == '__main__':
    try:
        segmentation = SemanticSegmentation()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
