###IMPORTS
import rospy
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

import message_filters

import pyrealsense2 as rs
import numpy as np
import cv2

from cv_bridge import CvBridge, CvBridgeError

import argparse
import numpy as np
import imutils

import math
import csv
from math import cos, sin, tan, pi, sqrt
import time

from sensor_msgs.msg import PointCloud2
import struct

###EXTERNAL PARAMETERS AND DATASETS

#used when mapping lidar data to image (first elements are closest to car)
color_map = [(250,10,10),(220,30,10),(180,100,60),(170,140,60),(160,180,60),(130,190,70),(100,210,80),(50,230,150),(10,240,220),(10,200,230),(10,140,240),(5,100,250),(5,50,250),(0,0,255),(10,10,255)]

#contains LIDAR-CAMERA transformation matrix
data_file = 'data.csv'
#CONTROLS
plane_slack = 7 #pixels, controls how arbitrarily close lidar data needs to be to best fit line in order to be drawn
sidelen = 1000 #pixels, width of array for hough transform
default_thresh = 200 #number of votes needed for Hough transform to detect positives (set low for best performance)
min_distance = 1 #m, distance from which points are considered for Hough transform
max_distance = 12#m, distance max for which points are considered for Hough transform (far away points are likely not road)

### ROS FUNCTIONS
def run_detection(): #starting loop, called from app manager (main)
    global hough_pub, img_pub #making publisher objects available to anyone
    global azimuth, matrix #making datafile info available
    #loading data
    params = load(data_file)
    azimuth = params[1]
    matrix = params[0]

    rospy.init_node('velodyne_hough', anonymous=True)#create ROS node
    #create ROS publisher objects
    hough_pub = rospy.Publisher('/cam_p/color/hough_line_detector', Image , queue_size=10) #plane space data
    img_pub = rospy.Publisher('/cam_p/color/LIDAR_road_projection', Image , queue_size=10) #final (RGB video) image
    try:
        listener()#launch main loop
    except rospy.ROSInterruptException: #some error
        print("Something goes wrong with Velodyne image projector")
        cv2.destroyAllWindows()
        quit()

def listener():#waits for new data and sets up main loop
    #Create subscriber objects to collect data
    lidar_sub = message_filters.Subscriber('/velodyne_points', PointCloud2) #LIDAR
    image_sub = message_filters.Subscriber('/cam_p/color/image_raw', Image) #MobileNet DNNimg

    #define synchronisation parameters
    delay_in_ms_NeverWakeUp = 50
    delay_in_ms_WakeUp = 0.001
    delay_in_s_WakeUp = 0.001 * delay_in_ms_WakeUp
    delay_in_s_NeverWakeUp = 0.001 * delay_in_ms_NeverWakeUp

    ts = message_filters.ApproximateTimeSynchronizer([lidar_sub, image_sub], 100, delay_in_s_NeverWakeUp) #synchroniser
    ts.registerCallback(callback)#calls callback routine when synchronised data is found
    rospy.spin() #loop

def callback(lidar, image): #routine when new synchronised data is found
    #creating images for working purposes
    img_frame = CvBridge().imgmsg_to_cv2(image, "bgr8") #RGBD camera image

    empty_array = np.zeros((int(sidelen/4),sidelen,3), dtype = np.uint8)#image for hough transform (plane space)
    img = cv2.cvtColor(empty_array,cv2.COLOR_BGR2RGB) #initializing image for plane space

    # Checking LIDAR data structure
    if lidar.point_step==22:
        struct_chain = '<ffffHf'
    elif lidar.point_step==32:
        struct_chain = '<ffffHfffH' #4 4 4 4
    else:
        print('Unknown point step structure')
    velo_bytes_iter = struct.iter_unpack(struct_chain, lidar.data)

    #finding parameters of line in plane space
    for i in velo_bytes_iter: #scan all lidar points
        velodyne_yz = (i[0],i[2]) #coords of lidata data projected onto yz plane (x coord squashed)
        img_coords= fit_for_hough(velodyne_yz, sidelen) #find coords in plane space
        sph_coords = convert_c2s(i[0],i[1],i[2]) #calculate distance, azimuth,elevation (spherical coords)in lidar space
        distance = sph_coords[0] #find distance of point to determine whether or not to cull

        if (azimuth[0]< sph_coords[1] < azimuth[1] and distance < max_distance and distance > min_distance): #check if point is in camera FOV and in range
            color = (255,255,255) #white, default color for plane space (easy to spot in black bgd)
            cv2.circle(img,img_coords, 3, color, thickness = -1 ) #draw points in plane space

    #plane space now initialized, finding best fit for road surface via Hough transform
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #convert to greyscale for hough transform
    lines = cv2.HoughLines(gray,1,np.pi/180,default_thresh) #apply hough transform

    for rho,theta in lines[0]: #most voted on line (likely road)
        #getting line params
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + sidelen*(-b))
        y1 = int(y0 + sidelen*(a))
        x2 = int(x0 - sidelen*(-b))
        y2 = int(y0 - sidelen*(a))
        line_vect = np.matrix([[x2-x1],[y2-y1]])

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)#drawing resulting line in plane space for illustration purposes

    #correct hough transform found, fitting datapoints to it
    velo_bytes_iter = struct.iter_unpack(struct_chain, lidar.data)#reloading lidar data again for projection
    for i in velo_bytes_iter: #scan all points
        sph_coords = convert_c2s(i[0],i[1],i[2]) #calculate distance, azimuth,elevation (spherical coords)
        #print(azimuth)
        if (azimuth[0]< sph_coords[1] < azimuth[1]): #check if point is in camera FOV
            plane_coords = (i[0], i[2]) #find coords of point in plane space
            img_coords = fit_for_hough(plane_coords, sidelen)#find coords of point in plane space
            x = img_coords[0]
            y = img_coords[1]
            #find distance of lidar point in plane space from previously-determined road line
            #FORMULA: adapted from hyperplane distance formuma (1D hyperplane)    #print(xmax,ymax)
            distance_num = abs((x2-x1)*(y1-y)-(x1-x)*(y2-y1))
            distance_den = sqrt((x2-x1)**2+(y2-y1)**2)
            map_distance = distance_num/distance_den #distance of point from best fit line in plane space

            if (map_distance < plane_slack): #point is "close enough" to predefined plane/line (see params) = corresponds to road surface and should be drawn
                img_coords = apply_transform((i[0],i[1],i[2]), matrix) #coordinates on actual video image
                color = pixel_color(sph_coords[0])  #find color as function of distance
                cv2.circle(img_frame, img_coords,2,color, thickness = -1) #draw point on image

    reproj_img = CvBridge().cv2_to_imgmsg(img_frame, "bgr8") #convert video image back into ROS compatible format
    hough_img = CvBridge().cv2_to_imgmsg(img, "bgr8")#convert plane space image for publishing (illustration purposes)
    hough_pub.publish(hough_img) #publish plane space image (can be disabled if needed)
    img_pub.publish(reproj_img) #publish vide image

### MISC FUNCTIONS

def fit_for_hough(plane_coords, buffer_size): #fits 3D LIDAR points to 2D "plane space" (see doc)
    img_coords= (int(plane_coords[0]*150-buffer_size/2), int(((buffer_size/8)-plane_coords[1]*40))) ##fitting points to plane space array for best detection
    return img_coords

def convert_c2s(x,y,z): #converts cartesian coordinates (x,y,z) to spherical (r,φ,θ), where θ is the angle formed between r_vector and z_vector, φ the angle formed between x-vector and r-vector , r the norm of vector
    r = math.sqrt(x**2 + y**2 + z**2)
    phi = math.atan2(y,x)
    #theta = math.acos(z/r) #not used
    return [r,phi]

def pixel_color(distance): #associates color from color map to distance for better illustration
    #ASSUMPTION: Not much point in having much detail on far away points, so majority of colors reservered for nearby points
    cmap_length = len(color_map) #how many colors avaiable?
    reduced_len = cmap_length-1 #number of colors used for nearby points
    #define how colormap is cut up
    velodyne_range = 100 #m, expected distance range for datapoints
    half_field = velodyne_range/2 #range with greater color precision

    if (distance < half_field): #use greater precision for color mapping
        quanta = half_field/reduced_len #distance per color bucket
        cbucket = int(distance/quanta) #find which color to use as fct of distance
        try:
            return color_map[cbucket] #return color
        except:
            return (0,0,255) #red as default in case of error
    else: #distance is large
        try:
            return color_map[-1] #use last color
        except:
            return (0,0,255) #red as default

def apply_transform(lidar_point, transform_matrix): #applies transform to transpose lidar data onto image
    lidar_point = np.matrix([lidar_point[0],lidar_point[1],lidar_point[2],1]).T #creates homogeneous collumn vector of xyz coords
    h_points = transform_matrix*lidar_point #coordinates of pixel in homogenous form
    (u,v) = (int(h_points[0]/h_points[2]),int(h_points[1]/h_points[2])) #converted to stadard 2d 'u,v'/'x,y' pixel format
    return (u,v) #coords on image


### FILE SYSTEM

def load(path,delim = ',', newline = '\n'): #loads csv file and constructs matrices
    params = []
    with open(path, newline = newline) as im:
        reader = csv.reader(im, delimiter = delim)
        for row in reader:
            params.append(row) #array of params with description first and value (str!) second

    im.close() #closing file
    #creating Rx matrix
    rx_angle = float(params[0][1])
    rx_matrix = np.array([[1,0,0],[0, cos(rx_angle), - sin(rx_angle)],[0, sin(rx_angle), cos(rx_angle)]])
    rx_matrix = np.matrix(rx_matrix)
    #creating Ry matrix
    ry_angle = float(params[1][1])
    ry_matrix = np.array([[cos(ry_angle),0,sin(ry_angle)],[0, 1, 0],[-sin(ry_angle), 0, cos(ry_angle)]])
    ry_matrix = np.matrix(ry_matrix)
    #creating Rz matrix
    rz_angle = float(params[2][1])
    rz_matrix = np.array([[cos(rz_angle), -sin(rz_angle),0],[sin(rz_angle),cos(rz_angle),0],[0,0,1]])
    rz_matrix = np.matrix(rz_matrix)
    #creating translation matrix
    t_matrix = np.array([[float(params[3][1])],[float(params[4][1])],[float(params[5][1])]])
    t_matrix = np.matrix(t_matrix)
    #creating projection matrix
    cx = int(params[8][1])
    cy = int(params[9][1])
    #print(cx,cy)
    p_matrix = np.array([[float(params[6][1]), 0, cx,0],[0,float(params[7][1]), cy , 0],[0,0,1,0]])
    p_matrix = np.matrix(p_matrix)
    #create final rotation matrix (opt)
    r_matrix = rx_matrix*ry_matrix*rz_matrix
    #aquire azimuthal information
    azi_min = float(params[10][1])
    azi_max = float(params[11][1])
    azimuth = (azi_min, azi_max)

    comp_matrix = np.matrix([0,0,0,1]) #to allow for multiplication
    sptfm_matrix = np.concatenate((r_matrix, t_matrix), axis = 1) #creating matrix for spatial transform (rotation +    translation)
    sptfm_matrix = np.concatenate((sptfm_matrix,comp_matrix), axis = 0 )
    final_matrix = p_matrix*sptfm_matrix#LIDAR-to-image transformation matrix

    return (final_matrix, azimuth)

### START MANUALLY
#run_detection()