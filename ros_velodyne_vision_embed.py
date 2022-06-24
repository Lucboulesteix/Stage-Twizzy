###IMPORTS
#ROS LIBRARIES
import rospy
import message_filters
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import PointCloud2
#image processing
import cv2
import imutils
from cv_bridge import CvBridge, CvBridgeError
#math
from math import cos, sin, tan, pi, sqrt, atan2
import numpy as np
#misc
import argparse
import csv
import time
import struct

###EXTERNAL PARAMETERS AND DATASETS
#directory
data_file = 'data.csv' #contains matrix information

#colors for lidar display, [BGR] from closest to farthest
color_map = [(250,10,10),(220,30,10),(180,100,60),(170,140,60),(160,180,60),(130,190,70),(100,210,80),(50,230,150),(10,240,220),(10,200,230),(10,140,240),(5,100,250),(5,50,250),(0,0,255),(10,10,255)] 

###ROS FUNCTIONS

def run_projection(): #starting loop, called from app manager
    global pub #making publisher object available
    #creating ros node
    rospy.init_node('velodyne_image_projector', anonymous=True)
    #create ros publier object
    pub = rospy.Publisher('/cam_p/color/velodyne_projection', Image , queue_size=10)
    try:
        listener()#launch main loop
    except rospy.ROSInterruptException: #some error
        print("Something goes wrong with Velodyne image projector")
        cv2.destroyAllWindows()
        pass
        
def listener():#waits for new data and sets up main loop
    global transform #make lidar-rgbd matrix available
    global azimuth #make camera FOV params available

    #set up subscriptions to relevant data
    image_sub = message_filters.Subscriber('/cam_p/color/image_raw', Image)
    lidar_sub = message_filters.Subscriber('/velodyne_points', PointCloud2)

    #read transform params
    params = load(data_file)
    transform = params[0] #lidar-image matrix
    azimuth  = params[1] #min and max azimuth (cam FOV)

    #define synchronisation parameters
    delay_in_ms_NeverWakeUp = 50
    delay_in_ms_WakeUp = 0.001
    delay_in_s_WakeUp = 0.001 * delay_in_ms_WakeUp
    delay_in_s_NeverWakeUp = 0.001 * delay_in_ms_NeverWakeUp

    ts = message_filters.ApproximateTimeSynchronizer([image_sub, lidar_sub], 100, delay_in_s_NeverWakeUp) #synchroniser
    ts.registerCallback(callback)#calls callback routine when synchronised data is found
    rospy.spin() #loop

def callback(image, lidar): #routine when new synchronised data is found
    #notification of new data
    #rospy.loginfo(rospy.get_caller_id() + "I heard "+ str(lidar.point_step) )
    
    #convert msg to cv2-compatible format
    img_frame = CvBridge().imgmsg_to_cv2(image, "bgr8")
    
    (ymax,xmax,c) = img_frame.shape #image size to determine if point is in image

    # Checking LIDAR data structure
    if lidar.point_step==22:
        struct_chain = '<ffffHf'
    elif lidar.point_step==32:
        struct_chain = '<ffffHfffH' #4 4 4 4
    else:
        print('Unknown point step structure')
    velo_bytes_iter = struct.iter_unpack(struct_chain, lidar.data)#lidar iterator object for data
    
    j = 0 #frame counter
    for i in velo_bytes_iter: #scan all points
        velodyne_scan_xyz = (i[0],i[1],i[2]) #coords in LIDAR space (for some reason in format (y,x,z)if lidar spec is to be believed)
        sph_coords = convert_c2s(i[0],i[1],i[2]) #calculate distance, azimuth,elevation (spherical coords)
        if (azimuth[0]< sph_coords[1] < azimuth[1]): #check if point is in camera FOV
            point = apply_transform(velodyne_scan_xyz,transform) #find correponding point on image
            if ( 0<point[0] <xmax and 0<point[1]<ymax): #check if pixel is on image
                try:
                    cv2.circle(img_frame, point,1,pixel_color(sph_coords[0]),thickness = -1) #draw circle
                    j +=1
                except:
                    pass
    #rospy.loginfo("handled " +str(j)+ " LIDAR points in frame, framerate is FPS "+ str(fps))
    
    #convert back to msg format
    img_pub = CvBridge().cv2_to_imgmsg(img_frame, "bgr8")
    pub.publish(img_pub) #publish image

###MISC FUNCTIONS

def convert_c2s(x,y,z): #converts cartesian coordinates (x,y,z) to spherical (r,φ,θ), where θ is the angle formed between r_vector and z_vector, φ the angle formed between x-vector and r-vector , r the norm of vector
    r = sqrt(x**2 + y**2 + z**2)
    phi = atan2(y,x)
    #theta = math.acos(z/r) #not used
    return [r,phi]

def apply_transform(lidar_point, transform_matrix): #finds (u,v) of single lidar datapoint
    lidar_point = np.matrix([lidar_point[0],lidar_point[1],lidar_point[2],1]).T #creates collum vector of xyz coords
    h_points = transform_matrix*lidar_point #coordinates of pixel in homogenous form
    (u,v) = (int(h_points[0]/h_points[2]),int(h_points[1]/h_points[2])) #converted to stadard 2d 'u,v'/'x,y' pixel format
    return (u,v)
    
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

### run for standalone operation
#run_projection()