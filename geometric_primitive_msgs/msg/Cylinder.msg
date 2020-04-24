# Message that contains cylinder as a geometric primitive.

# ID of the cylinder
uint32 id

# Pose of the cylinder in 3D
# Position originates in the geometric centre of the cylinder
# Orientation Z-axis is coincidental with the cylinder's axis of symmetry
geometry_msgs/Pose pose

# Radius of the cylinder (in XY plane)
float64 radius
# Height of the cylinder (along Z axis)
float64 height
