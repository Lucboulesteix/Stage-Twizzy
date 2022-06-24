# Stage-Twizzy
Content pertaining to IUT ORSAY internship.

Covered the study of multi-sensor perception systems and the implementation of algos for sensor fusion for vehicular applications

Sensors onboard: 
    * 2x Intel RealSense D435 RGBD Camera
    * 1x Velodyne Puck16/VLP16 LiDAR module

Base vehicle platform: Renault Twizy Series 80

Contains:
    * Documentation and other presentation material
    * Code for system calibration and other "offline" work
    * Code for real-time sensor fusion on the vehicle
Uses ROS software layer to interface all sensors, display on RVIZ/RQT
Uses MobileNet object detector wth custom msg format for object detection and camera->LiDAR fusion 

   
