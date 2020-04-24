cmake_minimum_required(VERSION 3.5)
project(geometric_primitive_msgs)

# ament_cmake
find_package(ament_cmake REQUIRED)
# rosidl
find_package(rosidl_default_generators REQUIRED)
# ROS2 interfaces
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Interface dependencies
set(INTERFACE_DEPENDENCIES std_msgs geometry_msgs)

# New interfaces
set(MSG_FILES
    "msg/Plane.msg"
    "msg/Sphere.msg"
    "msg/Cylinder.msg"
    "msg/PlaneStamped.msg"
    "msg/SphereStamped.msg"
    "msg/CylinderStamped.msg"
    "msg/PlanesStamped.msg"
    "msg/SpheresStamped.msg"
    "msg/CylindersStamped.msg"
    "msg/GeometricPrimitive.msg"
    "msg/GeometricPrimitiveStamped.msg"
    "msg/GeometricPrimitiveList.msg"
    "msg/GeometricPrimitiveListStamped.msg")
set(SRV_FILES)
set(ACTION_FILES)

# Generate interfaces
rosidl_generate_interfaces(
  ${PROJECT_NAME}
  ${MSG_FILES}
  ${SRV_FILES}
  ${ACTION_FILES}
  DEPENDENCIES
  ${INTERFACE_DEPENDENCIES}
  ADD_LINTER_TESTS)

# Export interfaces
ament_export_dependencies(rosidl_default_runtime)

# Setup the project
ament_package()
