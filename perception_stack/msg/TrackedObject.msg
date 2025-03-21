# Message definition for a tracked object

uint32 id              # Unique ID of the tracked object
string class_label     # Class label from semantic segmentation
float32 confidence     # Detection confidence

# Bounding box coordinates
float32 x              # x-coordinate of top-left corner
float32 y              # y-coordinate of top-left corner
float32 width          # Width of bounding box
float32 height         # Height of bounding box

# Motion information
float32 vel_x          # Velocity in x direction (pixels/sec)
float32 vel_y          # Velocity in y direction (pixels/sec)
float32 speed          # Overall speed magnitude
