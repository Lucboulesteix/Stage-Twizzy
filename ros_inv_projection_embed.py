### IMPORTS
 #ROS LIBRARIES
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image
from custom_msg.msg import object
from sensor_msgs.msg import PointCloud2
#MATH
from os import getcwd, chdir
from math import cos, sin, sqrt, atan2, acos, pi
import numpy as np
#image processing
import cv2
from cv_bridge import CvBridge, CvBridgeError
#MISC
import csv

### ###EXTERNAL PARAMETERS AND DATASETS
data_file = 'data.csv' #contains matrix info

mkr_queue_size = 20 #number of markers to draw per frame (too low = will leave old markers in space)
max_distance_detect = 30 #will only display predictions below distance to account for detection artifacts for mobilenet (set to high value if disabled)

#Orientation estimation params
max_wh = 2.5 #aspect ratio of dnn box for broadside-facing objects
min_wh = 0.9 #aspect ratio of dnn box for front-facing objects

#SIDE DETECTION
x_bins_nbr = 30
x_range = 15
draw_distance = 30 #distance used for detecting and drawing road side
min_x = -x_range/2
max_x = x_range/2

frequency = [0 for k in range(x_bins_nbr)] #tracks frequency of x coords
bins = [round((((k+1)*x_range/x_bins_nbr)+min_x),2) for k in range(x_bins_nbr)]
max_azi = 1
min_azi = -1



### ROS FUNCTIONS
def run_inv_projection():
  global matrix #making matrix avaiable to all
  global marker_pub #making publisher object available
  global marker_msg

  rospy.init_node('rviz_marker')#create ROS Node
  marker_msg = MarkerArray() #blank marker object for publishing

  marker_pub = rospy.Publisher("/visualization_marker_array", MarkerArray, queue_size = 10) #publisher object
  sub = rospy.Subscriber('/cam_p/color/object_detector', object , report_info_callback) #subcribe to object detector and callback when new data

  matrix = load('data.csv') #load transform matrix (RGBD-LIDAR)

  while not rospy.is_shutdown():#loop
    rospy.rostime.wallsleep(0.1)

def report_info_callback(data): #routine when new data is published

  dnn_info = get_info(data, matrix) #getting xyz coords in lidar space + aux
  while(True):#loop while old data still present
    try:
      marker_msg.markers.pop(0)#get rid of existing markers
    except: #have cleared all markers
      break
  marker_msg = place_markers(dnn_info[0],dnn_info[1], dnn_info[2]) #create marker array from data (supply array of x, y and z coordsà
  #rospy.loginfo("Published RVIZ message")
  marker_pub.publish(marker_msg)#publish marker array


###SIDE DETECTION
def detect_peaks():
    global frequency, bins
    global x_bins_nbr
    #params

    peaks_flag = False #are there enough peaks to work with?

    peaks = signal.find_peaks(frequency, prominence = prominence, height = height, threshold = threshold)
    #for n in peaks[0]
    peak_distance = []
    peaks = peaks[0]

    #THREE POSSIBLE CASES
    if (len(peak_distance)>2):#have detected multiple peaks, probable additionnal surface (wall, tree) detected: keep points closest to mid point
        peaks_flag = True
        leftover_peaks = []#keep closest peaks to midpoint
        mid_idx = int((x_bins_nbr/2)) #bin index for x = ~0m
        for u in range(mid_idx, -1,-1): #find closest peak in the negative coords
            if u in peaks:
                leftover_peaks.append()
                break #have found peak, move on
        for u in range(mid_idx, x_bins_nbr,1): #find closest peak in the positive coords
            if u in peaks:
                leftover_peaks.append()
                break #have found peak, move on
        peaks = leftover_peaks #keep these two peaks
    elif (len(peaks) < 2): #not enough peaks
        peaks_flag = False #throwing away data
    elif (len(peaks) == 2 ):#precisely enough peaks (good !)
        peaks_flag = True

    if (peaks_flag == True):#find associated distance
        right,left = bins[peaks[0]], bins[peaks[1]]

    elif (peaks_flag == False):
        right,left = (0,0) #do not use
    return (left, right, peaks_flag) #return point

def stats(x): #used for tracing histogram of datapints
    global x_bins_nbr,bins, x_range

    quanta = x_range/x_bins_nbr
    adj_x = x + x_range/2
    bucket = int(adj_x/quanta)
    occurences = frequency[bucket]
    frequency[bucket] = occurences + 1

def find_sides(lidar): #returns two markers corresponding to side
  global bins, frequency
  requency = [0 for k in range(x_bins_nbr)] #tracks frequency of x coords

  velo_bytes_iter = iter_velodyne(lidar) #load data
  for i in velo_bytes_iter:
    velodyne_xyz = (i[0], i[1],i[2])
    azimuth = convert_c2s(i[0], i[1],i[2])
    if (min_azi < azimuth < max_azi and min_x < i[1] < max_x): #confine data to small x distance from car) + point in camera FOV
      stats(i[1])

  #a) SIDE DETECTION
  peaks = detect_peaks() #attempting to resolve road sides
  lim_1, lim_2 = peaks[0], peaks[1] #sides of road (may not be valid if noting is detected, see flag)
  side_flag = peaks[2] #has side been detected?

  #constructing payload
  left_coords = (lim_1, 0 , -1.8)
  right_coords = (lim_2, 0 , -1.8)
  side_class = 'side'
  side_orientation = (0,0,0,0)

  return((left_coords, right_coords),(side_class), side_orientation)







### DNN LOGIC

def get_info(msg, matrix): #returns tuples of xyz coords (LIDAR reference) for all detected objects
  #getting coords + class of all elements detected
  x = msg.x_coords
  y = msg.y_coords
  z = msg.z_coords
  classes = msg.object_classes
  orientation = estimate_orientation(msg)#estimate orientation for all detected objects

  coords = [] #storesxyz coords of objects in LIDAR space

  for n in range(len(classes)): #same length is assumed for all lists of coords
    (i,j,k) = (x[n],y[n],z[n])
    distance = convert_c2s(i,j,k)[0]
    type = classes[n]

    if (distance < max_distance_detect): #will only keep te point if distance is low enough, otherwise messes with visualisation
      inv = inverse_transform(matrix, (i,j,k)) #coords in LIDAR space
      if (inv == (0,0,0) or distance < 0.1): #sometimes marker placement fails and prints them at center of scene, need to eliminate
        classes[n] = 'null' #marking object as invisible/filler null type
      coords.append(inv)
  return (coords, classes, orientation) #list of coords and corresponding types

def estimate_orientation(data):#attempts an estimation of objection orientation from box dimensions, takes as argument a MN/DNN message

  nbr_points = len(data.u_coords) #how many points to consider? (2 pts per object)
  #data
  u_coords = data.u_coords
  v_coords = data.v_coords

  i = 0 #counter
  orientation_coords = []

  while(i < nbr_points): #while there are still objects toprocess
    #find width and height of box
    x1 = u_coords[i]
    x2 = u_coords[i+1]
    width = abs(x2-x1)
    y1 = v_coords[i]
    y2 = v_coords[i+1]
    height = abs(y2-y1)
    #aspect ratio
    wh_ratio = width/height
    if(wh_ratio > max_wh): #truncate width height ratio to account for most vehicules
      wh_ratio = max_wh
    if(wh_ratio < min_wh): #truncate width height ratio to account for most vehicules
      wh_ratio = min_wh

    #we consider that a high wh ratio means vehicle broadside is facing us, whereas low ratio means we're looking at back or front
    delta = wh_ratio - min_wh
    max_delta = max_wh - min_wh
    norm_delta = delta/max_delta #normalizing between "front facing" and "broadside" positions
    orientation = (0,0,norm_delta,1) #given RVIZ LIDAR space orientation and marker symetry, only change one axis of rotation
    orientation_coords.append(orientation)
    i += 2 #move on to next object
  return orientation_coords

### MARKER LOGIC

def iter_velodyne(lidar): #loads data from velodyne point cloud unto iterable object
    # Checking LIDAR data structure
    if lidar.point_step==22:
        struct_chain = '<ffffHf'
    elif lidar.point_step==32:
        struct_chain = '<ffffHfffH' #4 4 4 4
    else:
        print('Unknown point step structure')

    velo_iter = struct.iter_unpack(struct_chain, lidar.data) #velodyne pointcloud iterator object
    return velo_iter

def create_marker(id, object_type, xyz_coords, ori_coords):
  #create single marker object
  marker = Marker()
  #information for rviz
  marker.header.frame_id = "velodyne"
  marker.header.stamp = rospy.Time.now()
  #id and type
  marker.id = id
  #rospy.loginfo(object_type)
  if(object_type == 'car'):
    #scale and type
    marker.type = 1 #rectangle
    marker.scale.x = 2
    marker.scale.y = 1
    marker.scale.z = 1
    #color
    marker.color.r = 1
    marker.color.g = 0
    marker.color.b = 0
    marker.color.a = 1
  elif(object_type == 'person'):
  #scale and type
    marker.type = 3 #cylinder
    marker.scale.x = 1
    marker.scale.y = 1
    marker.scale.z = 1
    #color
    marker.color.r = 0
    marker.color.g = 1
    marker.color.b = 0
    marker.color.a = 1
  elif(object_type == 'null'): #transparent marker for marker array fillup
    #scale and type
    marker.type = 1
    marker.scale.x = 0.01
    marker.scale.y = 0.01
    marker.scale.z = 0.01
    #color
    marker.color.r = 0
    marker.color.g = 0
    marker.color.b = 0
    marker.color.a = 0.01 #RVIZ doesn't want fully transparent markers
  elif (object_type == 'side'):
    #scale and type
    marker.type = 0
    marker.scale.x = 1
    marker.scale.y = 1
    marker.scale.z = 1
    #color
    marker.color.r = 0
    marker.color.g = 0
    marker.color.b = 1
    marker.color.a = 1 #RVIZ doesn't want fully transparent markers
  else: #default marker type
    #scale and type
    marker.type = 2 #sphere
    marker.scale.x = 0.5
    marker.scale.y = 0.5
    marker.scale.z = 0.5
    #color
    marker.color.r = 0
    marker.color.g = 0
    marker.color.b = 1
    marker.color.a = 1

  # Set the position of the marker
  marker.pose.position.x = xyz_coords[0]
  marker.pose.position.y = xyz_coords[1]
  marker.pose.position.z = xyz_coords[2]-1 #offset to account for lidar height
  #orientate marker
  marker.pose.orientation.x = ori_coords[0]
  marker.pose.orientation.y = ori_coords[1]
  marker.pose.orientation.z = ori_coords[2]
  marker.pose.orientation.w = ori_coords[3]
  return marker #return fully-initialized marker for marker creator

def place_markers(coords, types, orientation): #places markers on scene
  marker_array = MarkerArray()#place to store all markers (temp)
  for i in range(mkr_queue_size): #combing through all points: has issue where older markers would stay up because the number of points to draw varied. Brute force approach: draw MANY more markers and set them to be invisible
    try:
      new = create_marker(i, types[i], coords[i],orientation[i]) #create marker with detected information
    except: #have drawn all elements, need to fill up array with blanks
      new = create_marker(i, 'null', (0,0,0), (0,0,0,1)) #invisible marker

    marker_array.markers.append(new) #add to temp marker array
  return marker_array #return object


### FILE SYSTEM
def load(path, delim = ',', newline = '\n'): #returns inverse transformation matrix
  params = []
  with open(path, newline = '\n') as im:
    reader = csv.reader(im, delimiter = delim)
    for row in reader:
      params.append(row) #array of params with description first and value (str!) second
  im.close()
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
  #create rotation matrix
  r_matrix = rx_matrix*ry_matrix*rz_matrix
  #inverse translation matrix
  t_inv_matrix = -r_matrix.T*t_matrix
  #final trans-rot matrix and inverse
  comp_matrix = np.matrix([0,0,0,1]) #to allow for multiplication
  mtx = np.concatenate((r_matrix.T, t_inv_matrix), axis = 1) #creating matrix for spatial transform (rotation + translation)
  mtx = np.concatenate((mtx,comp_matrix), axis = 0 ) #final matrix
  return (mtx)

###MISC
def inverse_transform(tf_matrix, coords): #takes xyz coords in camera reference frame and return xyz coords in LIDAR reference frame
  coord_mtx = np.matrix([[coords[0]],[coords[1]],[coords[2]],[1]]) #homogenuous collumn vector
  new = tf_matrix*coord_mtx #applying inverse transform to get points in LIDAR space
  #converting homogenous coords to xyz standard
  x = new[0]/new[-1]
  x = x.item()
  y = new[1]/new[-1]
  y = y.item()
  z = new[2]/new[-1]
  z = z.item()
  return (x,y,z)

def convert_c2s(x,y,z): #converts cartesian coordinates (x,y,z) to spherical (r,φ,θ), where θ is the angle formed between r_vector and z_vector, φ the angle formed between x-vector and r-vector , r the norm of vector
    r = sqrt(x**2 + y**2 + z**2)
    phi = atan2(y,x)
    #theta = acos(z/r) #not used
    return [r,phi]

### Standalone oepration
#run_inv_projection()

