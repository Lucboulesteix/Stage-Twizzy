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

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')

#misc
import argparse
import csv
import time
import struct
from scipy import signal

###EXTERNAL PARAMETERS AND DATASETS
#directory
data_file = 'data.csv' #contains matrix information

#colors for lidar display, [BGR] from closest to farthest
color_map = [(250,10,10),(220,30,10),(180,100,60),(170,140,60),(160,180,60),(130,190,70),(100,210,80),(50,230,150),(10,240,220),(10,200,230),(10,140,240),(5,100,250),(5,50,250),(0,0,255),(10,10,255)] 

road_color = (0,255,0)
lidar_color = (255,0,255)
side_color = (0,0,255)

#SIDE DETECTION
x_bins_nbr = 30
x_range = 15
draw_distance = 30 #distance used for detecting and drawing road side

prominence = 80
height = 100
threshold = 5
distance = None

#HOUGH CONTROLS
plane_slack = 5 #pixels, controls how arbitrarily close lidar data needs to be to best fit line in order to be drawn
sidelen = 1200 #pixels, width of array for hough transform
default_thresh = 100 #number of votes needed for Hough transform to detect positives (set low for best performance)
min_distance = 1 #m, distance from which points are considered for Hough transform
max_distance = 12#m, distance max for which points are considered for Hough transform (far away points are likely not road)

#ROAD TRACING CONTROLS
stat_limit = 10**3 #number of LIDAR points to consider for stats
std_coef = 0.8 #numbers of std around mean to include
pixel_size = 1 #dictates if pixels should be skipped (higher = more slips), analyzes every pixel if  = 1
pixel_skip = 1 #determines if each pixel needs to be analyzed or every nth pixeldetermining max left and right pixel x values for line fit

display_lidar_proj = False #project LIDAR data points on image ?
display_side_detect = True#display lines for side of road when detected?


### Global variation declaration (do not touch)
array_full = False #has array idx been set back to zero?
stats_available = False #do we have stats? (set to False by default)

#global variables declared in main
#used for color stat information storage
r_array = []
g_array = []
b_array = []
idx = 0

mu, sigma = 0,0

min_x = -x_range/2
max_x = x_range/2
bins = [round((((k+1)*x_range/x_bins_nbr)+min_x),2) for k in range(x_bins_nbr)]
frequency = [0 for k in range(x_bins_nbr)]

initial_ts = time.time()
framecount = 0

###ROS FUNCTIONS

def run_road_detect(): #starting loop, called from app manager
    global pub, stats_pub #making publisher object available
    #creating ros node
    rospy.init_node('velodyne_image_projector', anonymous=True)
    #create ros publier object
    pub = rospy.Publisher('/cam_p/color/road_detection', Image , queue_size=10)
    stats_pub = rospy.Publisher('/cam_p/color/side_statistics', Image, queue_size = 10)
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
    global frequency #for side detection
    global stats_available #color stats
    global initial_ts, framecount #for fps quantify
    global lidar_color, road_color, hough_color 
    global mu, sigma
    
    #framerate()
    
    frequency = [0 for k in range(x_bins_nbr)] #tracks frequency of x coords
    lidar_coords = [[[] for n in range(2)] for k in range(16)] #tracks lidar stats
    
    #convert msg to cv2-compatible format
    img_frame = CvBridge().imgmsg_to_cv2(image, "bgr8")
    (ymax,xmax,c) = img_frame.shape #image shape to determine if point is in image
    
    #create empty image for hough transform
    empty_array = np.zeros((int(sidelen/4),sidelen,3), dtype = np.uint8)#image for hough transform (plane space)
    hough_img = cv2.cvtColor(empty_array,cv2.COLOR_BGR2RGB) #initializing image for plane space
    
    #STEP 1 : FIND PLANE AND SIDES WITH HOUGH + SCIPY PEAK DETECTION
    velo_bytes_iter = iter_velodyne(lidar)
    for i in velo_bytes_iter: #scan all points
        velodyne_scan_yxz = (i[0],i[1],i[2]) #coords in LIDAR space (for some reason in format (y,x,z)if lidar spec is to be believed)
        velodyne_yz = (i[0],i[2]) #coords of lidata data projected onto yz plane (x coord squashed)
        sph_coords = convert_c2s(i[0],i[1],i[2]) #calculate distance, azimuth,elevation (spherical coords)
        distance = sph_coords[0] #find distance of point to determine whether or not to cull
        
        img_coords= fit_for_hough(velodyne_yz, sidelen) #find coords in plane space
        
        if (azimuth[0]< sph_coords[1] < azimuth[1]): #check if point is in camera FOV
            img_coords= fit_for_hough(velodyne_yz, sidelen) #find coords in plane space
            cv2.circle(hough_img,img_coords, 3, (255,255,255), thickness = -1 ) #draw points in plane space
            if ((min_x < i[1] < max_x)): #confine data to small x distance from car
                stats(i[1]) #get stats on distribution of x coords
    
    #a) SIDE DETECTION
    peaks = detect_peaks() #attempting to resolve road sides
    lim_1, lim_2 = peaks[0], peaks[1] #sides of road (may not be valid if noting is detected, see flag)
    side_flag = peaks[2] #has side been detected?
    if (display_side_detect == True and side_flag == True): #have found side and want to draw
        draw_lines(img_frame, transform, lim_1, lim_2, draw_distance)
    
    #b) HOUGH TRANSFORM FOR ROAD SURFACE
    gray = cv2.cvtColor(hough_img,cv2.COLOR_BGR2GRAY) #convert to greyscale for hough transform
    lines = cv2.HoughLines(gray,1,np.pi/180,default_thresh) #apply hough transform on plane space
    n_fail = 1
    while(lines is None):
        thresh = default_thresh - n_fail*50
        n_fail += 1
        lines = cv2.HoughLines(gray,1,np.pi/180,thresh)
        
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
    
    #c) correct hough transform found, fitting datapoints to it
    velo_bytes_iter = iter_velodyne(lidar)
    for i in velo_bytes_iter: #scan all points
        sph_coords = convert_c2s(i[0],i[1],i[2]) #calculate distance, azimuth,elevation (spherical coords)
        
        if (azimuth[0]< sph_coords[1] < azimuth[1]): #check if point is in camera FOV
            ring = i[4]
            plane_coords = (i[0], i[2]) #find coords of point in plane space
            img_coords = fit_for_hough(plane_coords, sidelen)#find coords of point in plane space
            x = img_coords[0]
            y = img_coords[1]
            #find distance of lidar point in plane space from previously-determined road line
            #FORMULA: adapted from hyperplane distance formuma (1D hyperplane)
            distance_num = abs((x2-x1)*(y1-y)-(x1-x)*(y2-y1))
            distance_den = sqrt((x2-x1)**2+(y2-y1)**2)
            map_distance = distance_num/distance_den #distance of point from best fit line in plane space
            
            if (map_distance < plane_slack): #point is "close enough" to predefined plane/line (see params) = corresponds to road surface and should be drawn
                img_coords = apply_transform((i[0],i[1],i[2]), transform) #coordinates on actual video image

                a = img_coords[0] #x coordinate on image
                b = img_coords[1] #y coordinate on image
                #gathering stats on color of road
                lidar_coords[ring][0].append(a)
                lidar_coords[ring][1].append(b)
                color_stats(img_frame,img_coords, stat_limit)
                if (display_lidar_proj == True): #display plane lidar data if requested
                    cv2.circle(img_frame, img_coords,1,lidar_color, thickness = -1) #draw point on image
    
    #now have all information
    rings = get_ring_coords(lidar_coords) #get coordinates of space delimited by LIDAR

    #color statistics
    #adding data into np arrays for stats calculation (erratic behaviour with lists)
    b_arr = np.array(b_array)
    g_arr = np.array(g_array)
    r_arr = np.array(r_array)
    #average value for r, g and b colors
    avg_b = np.mean(b_array)
    avg_g = np.mean(g_array)
    avg_r = np.mean(r_array)
    mu = (avg_b, avg_g, avg_r)
    #stdev of color for r, g and b colors
    b_sigma = np.std(b_arr)
    g_sigma = np.std(g_arr)
    r_sigma = np.std(r_arr)
    sigma = (b_sigma, g_sigma, r_sigma)
    stats_available = True
    
    if stats_available == True:
        #fill_road(img_frame, mu, sigma, rings, side_flag, lim_1, lim_2)
        mnm = 0
    
    #convert back to msg format
    img_pub = CvBridge().cv2_to_imgmsg(img_frame, "bgr8")
    pub.publish(img_pub) #publish image

### HOUGH surface detection
def fit_for_hough(plane_coords, buffer_size): #fits 3D LIDAR points to 2D "plane space" (see doc)
    img_coords= (int(plane_coords[0]*150-buffer_size/2), int(((buffer_size/8)+plane_coords[1]*40))) ##fitting points to plane space array for best detection
    return img_coords
    
def extrapolate_from_rings(point_y, rings, ring_nbr): #attempts to find better fit for pixels than lidar rings (avoids staircase effect when drawing points)
    try:
        yA = rings[ring_nbr][2] #height of top ring
        yB = rings[ring_nbr-1][2] #height of bottom ring
        y_delta = yB-yA #y-distance between rings
        #displacement in x coords on left of image
        xA_left = rings[ring_nbr][0]
        xB_left = rings[ring_nbr-1][0]
        x_delta_left = xB_left - xA_left
        #displacement in x coords on right of image
        xA_right = rings[ring_nbr][1]
        xB_right = rings[ring_nbr-1][1]
        x_delta_right = xB_right - xA_right
    except:
        return (0, 0)
    #slopes
    try:
        slope_left = x_delta_left/y_delta
        slope_right = x_delta_right/y_delta
    except:
        slope_left = 1
        slope_right = 1
    #determining max left and right pixel x values for line fit
    left_lim = xA_left + slope_left*(point_y - yA)
    right_lim = xA_right + slope_right*(point_y - yA)
    left_lim = int(left_lim)
    right_lim = int(right_lim)
    #constructed payload
    payload = (left_lim, right_lim)
    return payload

def get_ring_coords(lidar_ring_coords):#finds [x1,x2,y1,y2] range of points in a ring
    output_array = [[] for k in range(16)] #tracks coord range (x1,x2, y1,y2) of each lidar ring

    for k in range(16): #all rings
        try:
            xlist = lidar_ring_coords[k][0]
            ylist = lidar_ring_coords[k][1]
            x1 = min(xlist)#leftmost point on ring
            x2 = max(xlist)#rightmost point on ring
            y1 = min(ylist)#top-most point on ring
            y2 = max(ylist)#bottom-most point on ring
            output_array[k] = [x1,x2,y1,y2]
        except:
            output_array.remove([])
    return output_array

def color_stats(img, pixel, max_idx):
    global b_array, g_array, r_array
    global idx, array_full

    try:#pixel in image
        color_of_pixel = img[pixel[1],pixel[0]]
        if (array_full == False): #array is being filled for first time
            #print('filling')
            b_array.append(color_of_pixel[0])
            g_array.append(color_of_pixel[1])
            r_array.append(color_of_pixel[2])
            idx += 1 #increment array idx
        elif (array_full == True): #array is fully initialzed
            b_array[idx] = color_of_pixel[0]
            g_array[idx] = color_of_pixel[1]
            r_array[idx] = color_of_pixel[2]
            idx +=1 #increment array idx
        else: #what
            print("something went wrong")

        if idx >= max_idx: #restrict number of entries for stat keeping
            idx = 0 #loop back to beginning of array
            array_full = True #telling loop that array is fully initialised
    except:
        pass #pixel likely outside of image

### SIDE OF ROAD DETECTION
def between_sides(point, left_side, right_side, draw_distance): #given road "corridor", returns whether or not pixel is in corridor or not
    
    #calculating params for drawing lines
    vertical_offset = -1.75 #distance between LIDAR and ground (approx)
    start_offset = 2 #distance from car of lower points
    alpha = 0.12 #angle between lidar and road surface
    h_comp = draw_distance * sqrt(1-cos(alpha)**2)
    far_offset = vertical_offset + h_comp #compensating for lidar tilt vs ground
    #POINTS IN IMAGE (generating points to create lines on image
    left_bottom = (start_offset, left_side, vertical_offset)
    left_bottom = apply_transform(left_bottom, transform )
    
    left_top = (draw_distance, left_side, far_offset)
    left_top = apply_transform(left_top, transform)
    
    right_bottom = (start_offset, right_side, vertical_offset)
    right_bottom = apply_transform(right_bottom, transform)
    
    right_top = (draw_distance, right_side, far_offset)
    right_top = apply_transform(right_top, transform)
    #have pixel corner coordinates of corridor: check if suplied pixel is within 
    try:
        #"height" of corridor on image
        y_left_delta = left_top[1]-left_bottom[1]
        y_right_delta = right_top[1] - right_bottom[1]
        
        #displacement in x coords on each side
        x_left_delta = left_top[0] - left_bottom[0]
        x_right_delta = right_top[0]-right_bottom[0]
    except:
        return False #error
    #SLOPES
    try:
        slope_left = x_left_delta/y_left_delta
        slope_right = x_right_delta/y_right_delta
        
        #print(slope_left, slope_right)
    except Exception as e:
        print(e)
        slope_left = 1
        slope_right = 1
    #determining max left and right pixel values (img) corresponding to corridor
    left_lim = int(left_top[0] + slope_left*(point[1] - left_top[1]))
    right_lim = int(right_top[0] + slope_right*(point[1] - right_top[1]))
    #is supplied pixel wthin these bounds?
    
    if (left_lim < point[0] < right_lim):
        return (True, left_lim, right_lim)
    else:
        return (False, left_lim, right_lim)
        
    

def detect_peaks():
    global frequency, bins
    global x_bins_nbr
    #params

    peaks_flag = False #are there enough peaks to work with?
    
#   DRAWING PLOT
    fig = plt.figure()
    x1 = np.array(bins)
    y1 = np.array(frequency)
    line1, = plt.plot(x1, y1)
    line1.set_ydata(np.array(frequency))
    plt.xlabel("x coordinates (m)")
    plt.ylabel("frequency (a.u.)")
    plt.title("distribution of lateral coordinates")
    
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
        
    if(peaks_flag == True):
        plt.axvline(x = left, color = 'r')
        plt.axvline(x = right, color = 'r')
    
    #rendering plot
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = CvBridge().cv2_to_imgmsg(img, "bgr8")
    stats_pub.publish(img)
    
    plt.clf()
        
    return (left, right, peaks_flag) #return point
    
def stats(x): #used for tracing histogram of datapints
    global x_bins_nbr,bins, x_range
    
    quanta = x_range/x_bins_nbr
    adj_x = x + x_range/2
    bucket = int(adj_x/quanta)
    occurences = frequency[bucket]
    frequency[bucket] = occurences + 1
    
###GRAPHICAL
def fill_road(img, color_avg_bgr, color_std_bgr, rings, mode, left_side, right_side):#find pixels with color matching road surface (modes: True = HOUGH + SIDES, False: HOUGH only)
    
    global road_color
    
    try:
        ymin = rings[-1][2]
    except:
        ymin = 300 #roughly halfway down in order to avoid hangs
     
    #color = (255,255,255)
    b_avg, g_avg, r_avg = color_avg_bgr
    b_sig, g_sig, r_sig = color_std_bgr
    #setting up
    bmin, bmax = (b_avg - std_coef*b_sig),(b_avg + std_coef*b_sig)
    gmin, gmax = (g_avg - std_coef*g_sig),(g_avg + std_coef*g_sig)
    rmin, rmax = (r_avg - std_coef*r_sig),(r_avg + std_coef*r_sig)
    (img_y, img_x,c) = np.shape(img)
    #print(img_x, img_y)

    for v in range(int(img_y/pixel_skip)):
        y = v*pixel_skip #actual coordinate along y axis
        if (y >= img_y):#prevent out of bounds from roudning 
            y = img_y-1
            
        if (mode == True): #use sides and plane
            side_limits = between_sides((1,y), left_side, right_side, draw_distance)[1:]
            #these are the areas to search for that particular vertical line
            ring_nbr = 0
            try:
                while (rings[ring_nbr][2] > y): #finding closest lidar  ring
                    ring_nbr +=1
            except: #have not matched data to a ring, search entire image
                ring_limits = (0, np.shape(img)[1])
            ring_limits = extrapolate_from_rings(y, rings, ring_nbr)
            
            #FIND NEW LIMITS ON IMAGE AS INTERSECTION of both hough-derived limits and side-detection-derived limits
            left, right = seg_intersection(side_limits, ring_limits)
            if (left < 0):
                left = 0
            if (right > img_x):
                right = img_x
            for u in range(int(left/pixel_skip),int((right)/pixel_skip)): #have narrowed search area
                x = u*pixel_skip
                #color of pixel
                b = img[y,x][0]
                g = img[y,x][1]
                r = img[y,x][2]
                #is pixel a color match?
                in_blue = (bmin < b <bmax)
                in_green = (gmin < g < gmax)
                in_red = (rmin < r < rmax)
                if (in_blue and in_green and in_red): #pixel matches description
                    cv2.circle(img, (x,y),pixel_size,road_color, thickness = -1)
            
        elif (mode == False): #use hough transform for road detection
                
            ring_nbr = 0
            try:
                while (rings[ring_nbr][2] > y): #finding closest lidar  ring
                    ring_nbr +=1
            except: #have not matched data to a ring, search entire image
                ring_limits = (0, np.shape(img)[1])
                
            #find lmits from rings
            left, right = extrapolate_from_rings(y, rings, ring_nbr)
            if (left < 0):
                left = 0
            if (right > img_x):
                right = img_x
            
            for u in range(int(left/pixel_skip),int((right)/pixel_skip)): #have narrowed search area
                x = u*pixel_skip
                #pixel color
                b = img[y,x][0]
                g = img[y,x][1]
                r = img[y,x][2]
                in_blue = (bmin < b <bmax)
                in_green = (gmin < g < gmax)
                in_red = (rmin < r < rmax)
                    
                #print(in_blue, in_green, in_red)
                if (in_blue and in_green and in_red): #pixel matches description
                    cv2.circle(img, (x,y),pixel_size,road_color, thickness = -1)
                    
def draw_lines(img, transform, left_side, right_side, draw_distance): #draw lines on imageleft, 
    #calculating params for drawing lines
    vertical_offset = -1.75 #distance between LIDAR and ground (approx)
    start_offset = 2 #distance from car of lower points
    alpha = 0.12 #angle between lidar and road surface
    h_comp = draw_distance * sqrt(1-cos(alpha)**2)
    far_offset = vertical_offset + h_comp #compensating for lidar tilt vs ground
    
    #POINTS IN IMAGE (generating points to create lines on image
    left_bottom = (start_offset, left_side, vertical_offset)
    left_bottom = apply_transform(left_bottom, transform )
    
    left_top = (draw_distance, left_side, far_offset)
    left_top = apply_transform(left_top, transform)
    
    right_bottom = (start_offset, right_side, vertical_offset)
    right_bottom = apply_transform(right_bottom, transform)
    
    right_top = (draw_distance, right_side, far_offset)
    right_top = apply_transform(right_top, transform)
    
    #draw on frame
    cv2.line(img, left_top, left_bottom, side_color, thickness = 4) 
    cv2.line(img, right_top, right_bottom, side_color, thickness = 4)
    return img
    

###MISC FUNCTIONS

def framerate():#estimates average fps and prints to screen
    global framecount, initial_ts
    
    framecount +=1
    
    now = time.time()
    span = now - initial_ts
    framerate = round((framecount/span),2)
    rospy.loginfo("AVG FPS: {}".format(framerate))


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
    



def seg_intersection(seg_a, seg_b): #find coordinates of segment at intersection of two other (a,b) segments
    #print(seg_a)
    j = max(seg_a[0], seg_b[0])
    k = min(seg_a[1], seg_b[1])
    if (k < j):
        #print("error with segment intersect", (j,k))
        return (0,0)
    else:
        return (j,k)

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
run_road_detect()