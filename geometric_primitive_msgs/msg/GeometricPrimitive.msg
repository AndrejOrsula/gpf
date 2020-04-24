# Message that contains a single geometric primitive of a specific type

uint8 PLANE=0
uint8 SPHERE=1
uint8 CYLINDER=2

# The type of the shape
uint8 type

# Plane
Plane[<=1] plane
# Sphere
Sphere[<=1] sphere
# Cylinder
Cylinder[<=1] cylinder
