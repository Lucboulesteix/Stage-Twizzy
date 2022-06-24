### IMPORTS
import rosbag
import cv2
import struct
import os
import time
import math
import csv
import time

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk

from statistics import mean, stdev
from PIL import Image, ImageDraw, ImageTk
from math import cos, sin, pi, sqrt
from random import randint

### PATHS AND CONTROL
#paths
dataset_dir = 'C:/Users/etudiant/Desktop/dataset/'
matrix_dir = 'C:/Users/etudiant/Desktop/Projet Twizzy STAGE LUC BOULESTEIX/python/Projection LIDAR/matrices/'
sample_file = '_2022-04-20-15-37-57.bag'
matrix_file = 'data.csv'

proj_data = matrix_dir + matrix_file
dataset = dataset_dir  + sample_file

#control

trans_range =  10 # +/- delta to test, meters
rot_range = 0.02 #+/- delta to test, radians

test_nbr = 20
point_nbr = 10

### FUNCTIONS

### MISC

def convert_c2s(x,y,z): #converts cartesian coordinates (x,y,z) to spherical (r,φ,θ), where θ is the angle formed between r_vector and z_vector, φ the angle formed between x-vector and r-vector , r the norm of vector

    r = math.sqrt(x**2 + y**2 + z**2)
    phi = math.atan2(y,x)
    theta = math.acos(z/r)
    return [r,phi,theta]


def load_velodyne(velodyne_msg): #returns iterator object for pointcloud
# Verification
    if velodyne_frames[0].point_step==22: # 22 '<ffffHf' 32 '<ffffHfffH'
        struct_chain = '<ffffHf'
    elif velodyne_frames[0].point_step==32:
        struct_chain = '<ffffHfffH'
    else:
        print('Unknown point step structure')
        quit()
    velo_bytes_iter = struct.iter_unpack(struct_chain, velodyne_msg.data)
    return velo_bytes_iter

def apply_transform(lidar_point, transform_matrix): #finds (u,v) of single lidar datapoint
    lidar_point = np.matrix([[lidar_point[0]],[lidar_point[1]],[lidar_point[2]],[1]]) #creates collum vector of xyz coords
    h_points = transform_matrix*lidar_point #coordinates of pixel in homogenous form
    (u,v) = ((h_points[0]/h_points[2]),(h_points[1]/h_points[2])) #converted to stadard 2d 'u,v'/'x,y' format
    return (u,v)

def point_format(point): #converts floating point coods into integer coords for image dispplay
    (u,v) = (int(point[0]), int(point[1]))
    return (u,v)

def distance(a, b): #calculates distance of two points on image (2D coords)
    x1 = a[0]
    y1 = a[1]
    x2 = b[0]
    y2 = b[1]

    d_squared = (x2-x1)**2 + (y2-y1)**2
    d = sqrt(d_squared)
    return d

### FILE SYSTEM

class projector:

    def __init__(self, path, delim = ',', newline = '\n'):
        params = []
        try:
            with open(path, newline = newline) as im:
                reader = csv.reader(im, delimiter = delim)
                for row in reader:
                    params.append(row) #array of params with description first and value (str!) second
                im.close()
        except:
            print("Data File Error: could not find matrix data !")

    #rotation
        (rx,ry,rz) = (float(params[0][1]),float(params[1][1]),float(params[2][1]))
        #translation
        (tx,ty,tz) = (float(params[3][1]),float(params[4][1]),float(params[5][1]))
        #projection params
        (fx,fy) = (float(params[6][1]),float(params[7][1]))
        (cx,cy) = (float(params[8][1]),float(params[9][1]))
        #azimuth
        azi_min = float(params[10][1])
        azi_max = float(params[11][1])
        #storing params in dict for rapid search
        paramsdict = {"rx": rx, "ry" : ry, "rz": rz, "tx": tx, "ty": ty, "tz" : tz, "fx" : fx, "fy" : fy, "cx": cx, "cy": cy, "azi_min": azi_min, "azi_max": azi_max }

        #making params accessible
        self.rx = rx
        self.ry = ry
        self.rz = rz
        self.tx = tx
        self.ty = ty
        self.tz = tz
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.azi_min = azi_min
        self.azi_max = azi_max

        self.dict = paramsdict

        #rotation matrices
        rx_matrix = np.matrix([[1,0,0],[0, cos(self.rx), - sin(self.rx)],[0, sin(self.rx), cos(self.rx)]])
        ry_matrix = np.matrix([[cos(self.ry),0,sin(self.ry)],[0, 1, 0],[-sin(self.ry), 0, cos(self.ry)]])
        rz_matrix = np.matrix([[cos(self.rz), -sin(self.rz),0],[sin(self.rz),cos(self.rz),0],[0,0,1]])
        rotation_matrix = rx_matrix * ry_matrix * rz_matrix
        #translation matrix
        translation_matrix = np.matrix([[self.tx],[self.ty],[self.tz]])
        #extrinsic matrix

        comp_matrix = np.matrix([0,0,0,1]) #to allow for multiplication
        extrinsic_matrix = np.concatenate((rotation_matrix, translation_matrix), axis = 1)
        extrinsic_matrix = np.concatenate((extrinsic_matrix,comp_matrix), axis = 0 )
        #camera matrix
        proj_matrix = np.array([[self.fx, 0,self.cx,0],[0,self.fy, self.cy, 0],[0,0,1,0]])
        #composite matrix
        transform = proj_matrix * extrinsic_matrix

        self.transform = transform


class dataload: #acquires data from bag file and acquires stats, pairing information, etc

    def __init__(self, path):

        #load data
        bag = rosbag.Bag(path)

        velodyne_frames = []
        cam_color_frames = []


        for topic, msg, t in bag.read_messages():
            if topic == '/velodyne_points':
                velodyne_frames.append(msg)
            elif topic == '/cam_p/color/image_raw':
                cam_color_frames.append(msg)
            elif topic == '/cam_p/color/object_detector':
                object_frames.append(msg)

        #acquiring raw data + info
        self.raw_RGB = cam_color_frames
        self.RGB_count = len(cam_color_frames)

        self.raw_LIDAR = velodyne_frames
        self.LIDAR_count = len(velodyne_frames)



    def pair(self): #pairs disperate LIDAR and RGBD data

        paired_frame_idx = []
        lidar_idx = 0

        for rgb_idx in range(self.RGB_count):
            rgb_timestamp = (self.raw_RGB[rgb_idx].header.stamp.secs)+(self.raw_RGB[rgb_idx].header.stamp.nsecs)*(10**-9)
            #finding most recent lidar frame
            while(True):
                try:
                    lidar_timestamp = (self.raw_LIDAR[lidar_idx].header.stamp.secs)+(self.raw_LIDAR[lidar_idx].header.stamp.nsecs)*(10**-9)
                except: #end of file, don't want to exceed list size
                    break
                if(lidar_timestamp > rgb_timestamp): #we want the previous lidar frame in this case
                    lidar_idx -=1
                    if (lidar_idx < 0): #edge case if first frame is RGB and not lidar
                        lidar_idx = 0
                    paired_frame_idx.append((rgb_idx,lidar_idx))
                    break #go to next rgb frame
                else:
                    lidar_idx +=1 #test next lidar frame

        paired_data = []
        for r,l in paired_frame_idx:
            paired_data.append((self.raw_RGB[r], self.raw_LIDAR[l]))

        #self.pair_idx = paired_frame_idx
        self.pairs = paired_data

def fudge(params_dict, param, offset):
    paramdict = []

    for key in params_dict.keys():

        if (key == param):
            value = params_dict[key] + offset
            #print(params_dict[key], value)
            paramdict.append(value)
        else:
            value = params_dict[key]
            paramdict.append(value)


    rx = paramdict[0]
    ry = paramdict[1]
    rz = paramdict[2]
    tx = paramdict[3]
    ty = paramdict[4]
    tz = paramdict[5]
    fx = paramdict[6]
    fy = paramdict[7]
    cx = paramdict[8]
    cy = paramdict[9]

    #creating matrices
    rx_matrix = np.matrix([[1,0,0],[0, cos(rx), - sin(rx)],[0, sin(rx), cos(rx)]])
    ry_matrix = np.matrix([[cos(ry),0,sin(ry)],[0, 1, 0],[-sin(ry), 0, cos(ry)]])
    rz_matrix = np.matrix([[cos(rz), -sin(rz),0],[sin(rz),cos(rz),0],[0,0,1]])
    rotation_matrix = rx_matrix * ry_matrix * rz_matrix
    translation_matrix = np.matrix([[tx],[ty],[tz]])
    #extrinsic matrix
    comp_matrix = np.matrix([0,0,0,1]) #to allow for multiplication
    extrinsic_matrix = np.concatenate((rotation_matrix, translation_matrix), axis = 1)
    extrinsic_matrix = np.concatenate((extrinsic_matrix,comp_matrix), axis = 0 )
    #camera matrix
    proj_matrix = np.array([[fx, 0,cx,0],[0,fy, cy, 0],[0,0,1,0]])
    transform = proj_matrix * extrinsic_matrix

    return transform

### mainloop

#acquiring information
camera = projector(proj_data)

data  = dataload(dataset)
data.pair()

###

idx = 1000

(rgb, lidar) = data.pairs[idx] #a random datapoint for testing
#velobytes = load_velodyne(lidar)

#creating image to draw on
array = np.frombuffer(rgb.data, dtype=np.uint8)
array = np.reshape(array, (rgb.height,rgb.width,3))
colorImage = cv2.cvtColor(array, cv2.COLOR_BGR2YCR_CB)



for i in range(1,point_nbr+1):

    d_stats = []

    point = (i*5, 0, 1) #coords of point in LIDAR space
    baseline_coords = apply_transform(point, camera.transform) #coords on image (float)
    baseline_coords_img = point_format(baseline_coords)
    #draw first point
    cv2.circle(colorImage, baseline_coords_img, 2, (255, 255, 255), thickness = -1)
    stats = [] #for statistics

    #dtermining color to use on image
    shade = (250/point_nbr)*i
    inv_shade = 255 - shade
    ycbcr_color = (80, shade, inv_shade)

    trans_delta_step = (2*trans_range)/test_nbr #step to use when checking rotation values

    for k in range(test_nbr):
        offset = -trans_range + trans_delta_step*k #offset to use when iterating through possible t vals
        #print(offset)
        fudged_transform = fudge(camera.dict,"cx", offset)

        new_coords = apply_transform(point, fudged_transform)
        new_coords_img = point_format(new_coords)

        cv2.circle(colorImage, new_coords_img, 1, ycbcr_color, thickness = -1)
        deviation = distance(new_coords, baseline_coords)
        d_stats.append(deviation)
        #print(deviation)

    avg = round(mean(d_stats), 3)
    sigma = round(stdev(d_stats),3)
    #print(len(d_stats))
    print("Distance : {}, average pixel disp : {} with stdev of {}".format(i*5, avg, sigma))






colorImage = cv2.cvtColor(colorImage, cv2.COLOR_YCrCb2RGB)
cv2.imwrite("exp.png", colorImage)

#cv2.imshow("test", colorImage)
#cv2.waitKey(0)





