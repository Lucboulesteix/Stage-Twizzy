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

from PIL import Image, ImageDraw, ImageTk
from math import cos, sin, pi
from random import randint

###Control variables
#paths
dataset_dir = 'C:/Users/etudiant/Desktop/dataset/'
matrix_dir = 'C:/Users/etudiant/Desktop/Projet Twizzy STAGE LUC BOULESTEIX/python/Projection LIDAR/matrices/'
sample_file = '_2022-04-20-15-37-57.bag'
matrix_file = 'data.csv'

#operating variables
color_map = [(250,10,10),(220,30,10),(180,100,60),(170,140,60),(160,180,60),(130,190,70),(100,210,80),(50,230,150),(10,240,220),(10,200,230),(10,140,240),(5,100,250),(5,50,250),(0,0,255),(10,10,255)] #colors for lidar display, [BGR] from closest to farthest

#knobs and levers
color_disp = True #est_ce qu'on utilise plusieurs couleurs pour l'affichage?

lidar_optimize = True #forces program to recalculate LIDAR mapping for every frame if False (can cause stutter if true but greater average fps)
frame_ratio = 1 #number of RGB frames to skip per frame (1 = real time) BUGGY
fps_ratio = 1 #multiplication factor for frametime (increase if stutter)

r_inc = 0.01 #increment/decrement for rotation adjustments
t_inc = 0.01

### MISCfunctions

def convert_c2s(x,y,z): #converts cartesian coordinates (x,y,z) to spherical (r,φ,θ), where θ is the angle formed between r_vector and z_vector, φ the angle formed between x-vector and r-vector , r the norm of vector

    r = math.sqrt(x**2 + y**2 + z**2)
    phi = math.atan2(y,x)
    theta = math.acos(z/r)
    return [r,phi,theta]

def apply_transform(lidar_point, transform_matrix): #finds (u,v) of single lidar datapoint
    lidar_point = np.matrix([[lidar_point[0]],[lidar_point[1]],[lidar_point[2]],[1]]) #creates collum vector of xyz coords
    h_points = transform_matrix*lidar_point #coordinates of pixel in homogenous form
    (u,v) = (int(h_points[0]/h_points[2]),int(h_points[1]/h_points[2])) #converted to stadard 2d 'u,v'/'x,y' format
    return (u,v)

def check_velodyne(velodyne_msg): #check for correct data structuring and returns format
# Verification
    if velodyne_frames[0].point_step==22: # 22 '<ffffHf' 32 '<ffffHfffH'
        struct_chain = '<ffffHf'
    elif velodyne_frames[0].point_step==32:
        struct_chain = '<ffffHfffH'
    else:
        print('Unknown point step structure')
        quit()
    return struct_chain

def frametime(cam_color_frames): #determines average framerate of recording and returns average period in ms
    first_ts = cam_color_frames[0].header.stamp.secs + cam_color_frames[0].header.stamp.nsecs /(10**9)
    last_ts = cam_color_frames[-1].header.stamp.secs + cam_color_frames[-1].header.stamp.nsecs /(10**9)
    span = last_ts - first_ts
    nbr_frames = len(cam_color_frames)
    frame_period= int((span/nbr_frames)*1000)
    return frame_period

### FILE SYSTEM

def read_and_initialise(path, delim = ',', newline = '\n'):
    #get transform parameters from csv datafile
    params = []
    try:
        with open(path, newline = '\n') as im:
            reader = csv.reader(im, delimiter = delim)
            for row in reader:
                params.append(row) #array of params with description first and value (str!) second
    except:
        print("Data File Error: could not find matrix data !")
        quit()
    #closing file
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
    #creating projection matrix
    proj_matrix = np.array([[float(params[6][1]), 0, float(params[8][1]),0],[0,float(params[7][1]), float(params[9][1]) , 0],[0,0,1,0]])
    proj_matrix = np.matrix(proj_matrix)
    #create final rotation matrix
    r_matrix = rx_matrix*ry_matrix*rz_matrix
    #aquire azimuthal information
    azi_min = float(params[10][1])
    azi_max = float(params[11][1])

    r_inv = r_matrix.T
    #create inverse translation matrix
    t_inv = -r_inv*t_matrix

    comp_matrix = np.matrix([0,0,0,1]) #to allow for multiplication
    matrix = np.concatenate((r_matrix, t_matrix), axis = 1) #creating matrix for spatial transform (rotation +    translation)
    matrix = np.concatenate((matrix,comp_matrix), axis = 0 )
    #inverse matrix
    inv_matrix = np.concatenate((r_inv, t_inv), axis = 1)
    inv_matrix = np.concatenate((inv_matrix,comp_matrix), axis = 0 )

    #save data
    data = [['rx', rx_angle],['ry', ry_angle],['rz', rz_angle],['tx', float(params[3][1])],['ty', float(params[4][1])],['tz', float(params[5][1])],['fx', float(params[6][1])],['fy', float(params[7][1])],['cx', float(params[8][1])],['cy', float(params[9][1])],['azi_min', azi_min],['azi_max', azi_max]] #for later use

    #results as tuple
    return(rx_matrix, ry_matrix, rz_matrix, r_matrix, t_matrix,proj_matrix,azi_min, azi_max, data)

def load(matrix_dir):
    #chargement des matrices depuis fichiers csv
    matrices = read_and_initialise(matrix_dir, delim = ',')
    #azimuthal range
    azi_min = matrices[-3]
    azi_max = matrices[-2]
    #rotation matrix
    r_matrix = matrices[3]
    #projection matrix
    p_matrix = matrices[5]
    #translation matrix
    t_matrix = matrices[4]

    data = matrices[-1] #paramters for later modification

    comp_matrix = np.matrix([0,0,0,1]) #to allow for multiplication
    sptfm_matrix = np.concatenate((r_matrix, t_matrix), axis = 1) #creating matrix for spatial transform (rotation +    translation)
    sptfm_matrix = np.concatenate((sptfm_matrix,comp_matrix), axis = 0 )
    final_matrix = p_matrix*sptfm_matrix
    return (final_matrix, data)

def save_params(path, data, delim = ',', newline = '\n'):
    with open(path, 'w', newline=newline) as file:
        mywriter = csv.writer(file, delimiter=delim)
        mywriter.writerows(data)
        file.close()

### DATA PREPROCESSING

def pair_frames(RGB_msg, lidar_msg): #from rosbag messages returns tuples corresponding to frame-pairs (rgb,lidar)
    paired_frames = []
    lidar_idx = 0
    for rgb_idx in range(len(RGB_msg)):
        rgb_timestamp = (RGB_msg[rgb_idx].header.stamp.secs)+(RGB_msg[rgb_idx].header.stamp.nsecs)*(10**-9)
        #finding most recent lidar frame
        while(True):
            #print(lidar_idx)
            try:
                lidar_timestamp = (lidar_msg[lidar_idx].header.stamp.secs)+(lidar_msg[lidar_idx].header.stamp.nsecs)*(10**-9)
            except: #end of file, don't want to exceed list size
                break
            if(lidar_timestamp > rgb_timestamp): #we want the previous lidar frame in this case
                lidar_idx -=1
                if (lidar_idx < 0): #edge case if first frame is RGB and not lidar
                    lidar_idx = 0
                paired_frames.append((rgb_idx,lidar_idx))
                break #go to next rgb frame
            else:
                lidar_idx +=1 #test next lidar frame
        #create image associated with rgb data
    return paired_frames


def sanitize(LIDAR_frame, transform_matrix, struct_chain, azi_min = -3.14, azi_max = 3.14): #given two synced lidar and rgb frames, returns coordinates of pixels of lidar data.
    global data_cull
    global lidar_disp
    velo_bytes_iter = struct.iter_unpack(struct_chain, LIDAR_frame.data)

    velodyne_scan_xyz = []
    velodyne_scan_sph = []
    distance_points = [] #distance of points
    image_points = [] #points on screen after transform


    for i in velo_bytes_iter:#iterate through all data points
        try:

            point_xyz = (i[0],i[1],i[2],i[4])
            point_sph = convert_c2s(i[0],i[1],i[2])#conversion en spherique
            if(azi_min < point_sph[1]<azi_max): #cull data out of frame
            #    #velodyne_scan_xyz.append(point_xyz)
            #    #velodyne_scan_sph.append(point_sph)
                distance_points.append(point_sph[0])
                on_image = apply_transform(point_xyz, transform_matrix) #coords of point after transform
                image_points.append(on_image)

        except:
            break
    #print(len(velodyne_scan_sph))
    return(velodyne_scan_xyz, velodyne_scan_sph, image_points,distance_points)


### Graphical operations

def draw_on_image(frame, point_coords, d_coords, radius = 1, thickness = -1): #from specific frame returns image with draw points on it


    array = np.frombuffer(frame.data, dtype=np.uint8)
    array = np.reshape(array, (frame.height,frame.width,3))
    colorImage = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    shape = np.shape(colorImage)
    #print("Image res: x = {} px, y = {} px".format(shape[1], shape[0]))

    (u,v,c) = colorImage.shape
    n = []
    for c_idx in range(len(point_coords)):
        if( 0 < point_coords[c_idx][1] <u and 0 < point_coords[c_idx][0] < v): #pixel in image
            try:
                color = pixel_color(d_coords[c_idx])
                cv2.circle(colorImage, point_coords[c_idx], radius, color, thickness)
                #n.append('&')
            except Exception as e:
                print(e)
    #print(len(n))
    return colorImage

def img_format(image): #takes cv2_formatted image (after processing) and formats into tkinter_compatable format
    b,g,r = cv2.split(image)#converts color space info for correct color
    img = cv2.merge((r,g,b))

    img = Image.fromarray(img) #convertis format PIL
    img = ImageTk.PhotoImage(img) #convertis format tkinter
    return img

def pixel_color(distance):
    global color_map #custom colormap defined in main file
    global color_disp

    #ASSUMPTION: most objects in first half of velodyne range, and few points past half of total range
    if (color_disp == True):
        cmap_length = len(color_map)
        reduced_len = cmap_length-1 #we want to half a fine color range forst nearby objects
        velodyne_range = 100 #m
        half_field = velodyne_range/2 #range with greater color precision
        if (distance < half_field): #use greater precision for color mapping
            quanta = half_field/reduced_len
            cbucket = int(distance/quanta) #find which color to use as fct of distance
            #print(cbucket)
            try:
                return color_map[cbucket]
            except:
                return (0,0,255) #red
        else: #distance is large
            try:
                return color_map[-1]
            except:
                return (0,0,255) #red
    else: #use default
        return (0,0,255)


###GUI functions

def init_display():
    #global params from other functions
    global root
    global rx_txt,ry_txt,rz_txt,tx_txt,ty_txt,tz_txt,fx_txt,fy_txt,cx_txt,cy_txt
    global azimin_txt, azimax_txt
    global imglabel,frameratelabel
    global camera_params
    global cam_color_frames
    global velodyne_frames
    global first_img
    global frame_id, paused
    global framelabel
    global recording_period
    global sanitized_lidar
    global i_ts, framerate

    i_ts = time.time()
    paused = False

    frame_id = 0 #first frame
    #code#
    root = tk.Tk()#main window
    root.title("Reprojection LIDAR" )
    recording_period = frametime(cam_color_frames)*fps_ratio
    #draw first image
    sanitized_lidar = sanitize(velodyne_frames[pairs[frame_id][1]], final_matrix, struct_chain , azi_min = camera_params[-2],       azi_max = camera_params[-1] )
    first_img = draw_on_image(cam_color_frames[pairs[frame_id][0]], sanitized_lidar[2], sanitized_lidar[3]) #computed image
    first_img = img_format(first_img) #recompute image
    #print(first_img)

    #draw GUI
    #frames
    #img_frame = tk.Frame(root, height = cam_color_frames[0].height, width = cam_color_frames[0].width ,relief = 'groove', borderwidth = 5) #for picture
    img_frame = tk.Frame(root, height = 400, width = 848 ,relief = 'groove', borderwidth = 5) #for picture
    btn_frame = tk.Frame(root, height = 650, width = 400, relief = 'groove', borderwidth = 5) #for buttons
    credits_frame = tk.Frame(root, height = 200, width = 800)
    #placing frames
    img_frame.place(x = 10, y = 10)
    btn_frame.place(x = 920 , y = 10)
    credits_frame.place(x = 10, y = 520)
    try:
        #IUT LOGO (very important)
        logo = ImageTk.PhotoImage(Image.open("iut.png"))
    except:
        logo = ''
    try:
        #pause button (very important)
        pause_logo = ImageTk.PhotoImage(Image.open("pause.png"))
    except:
        pause_logo = ''
    try:
        #stop button (very important)
        stop_logo = ImageTk.PhotoImage(Image.open("stop.png"))
    except:
        stop_logo = ''
    #creating buttons
    #x rotation
    rx_plus = tk.Button(btn_frame, text = 'increase x angle', command = increase_angle_rx, width = 12)
    rx_minus = tk.Button(btn_frame, text = 'decrease x angle', command = reduce_angle_rx, width = 12)
    rx_txt = tk.Label(btn_frame, text = "x (rad): {}".format(camera_params[0][1]), bg = 'white', width = 14)
    #y rotation
    ry_plus = tk.Button(btn_frame, text = 'increase y angle', command = increase_angle_ry, width = 12)
    ry_minus = tk.Button(btn_frame, text = 'decrease y angle', command = reduce_angle_ry, width = 12)
    ry_txt = tk.Label(btn_frame, text = "y (rad): {}".format(camera_params[1][1]), bg = 'white', width = 14)
    #z rotation
    rz_plus = tk.Button(btn_frame, text = 'increase z angle', command = increase_angle_rz, width = 12)
    rz_minus = tk.Button(btn_frame, text = 'decrease z angle', command = reduce_angle_rz, width = 12)
    rz_txt = tk.Label(btn_frame, text = "z (rad): {}".format(camera_params[2][1]), bg = 'white',width = 14)

    #x translation
    tx_plus = tk.Button(btn_frame, text = 'increase x trans', command = increase_tx, width = 12)
    tx_minus = tk.Button(btn_frame, text = 'decrease x trans', command = reduce_tx, width = 12)
    tx_txt = tk.Label(btn_frame, text = "x (m): {}".format(camera_params[3][1]), bg = 'white', width = 14)
    #y translation
    ty_plus = tk.Button(btn_frame, text = 'increase y trans', command = increase_ty, width = 12)
    ty_minus = tk.Button(btn_frame, text = 'decrease y trans', command = reduce_ty, width = 12)
    ty_txt = tk.Label(btn_frame, text = "y (m): {}".format(camera_params[4][1]), bg = 'white', width = 14)
    #z translation
    tz_plus = tk.Button(btn_frame, text = 'increase z trans', command = increase_tz, width = 12)
    tz_minus = tk.Button(btn_frame, text = 'decrease z trans', command = reduce_tz, width = 12)
    tz_txt = tk.Label(btn_frame, text = "z (m): {}".format(camera_params[5][1]), bg = 'white',width = 14)
    #fx
    fx_plus = tk.Button(btn_frame, text = 'increase fx', command = increase_fx, width = 12)
    fx_minus = tk.Button(btn_frame, text = 'decrease fx', command = reduce_fx, width = 12)
    fx_txt = tk.Label(btn_frame, text = "fx (px): {}".format(camera_params[6][1]), bg = 'white',width = 14)
    #fy
    fy_plus = tk.Button(btn_frame, text = 'increase fy', command = increase_fy, width = 12)
    fy_minus = tk.Button(btn_frame, text = 'decrease fy', command = reduce_fy, width = 12)
    fy_txt = tk.Label(btn_frame, text = "fy (px): {}".format(camera_params[7][1]), bg = 'white',width = 14)
    #cx
    cx_plus = tk.Button(btn_frame, text = 'increase cx', command = increase_cx, width = 12)
    cx_minus = tk.Button(btn_frame, text = 'decrease cx', command = reduce_cx, width = 12)
    cx_txt = tk.Label(btn_frame, text = "cx (px): {}".format(camera_params[8][1]), bg = 'white',width = 14)
    #cy
    cy_plus = tk.Button(btn_frame, text = 'increase cy', command = increase_cy , width = 12)
    cy_minus = tk.Button(btn_frame, text = 'decrease cy', command = reduce_cy, width = 12)
    cy_txt = tk.Label(btn_frame, text = "cy (px): {}".format(camera_params[9][1]), bg = 'white',width = 14)
    #azimuth min
    azimin_plus = tk.Button(btn_frame, text = 'increase min azi', command = increase_azimin, width = 12)
    azimin_minus = tk.Button(btn_frame, text = 'decrease min azi', command = reduce_azimin, width = 12)
    azimin_txt = tk.Label(btn_frame, text = "azi min (rad): {}".format(camera_params[10][1]), bg = 'white',width = 14)
    #azimuth max
    azimax_plus = tk.Button(btn_frame, text = 'increase max azi', command = increase_azimax , width = 12)
    azimax_minus = tk.Button(btn_frame, text = 'decrease max azi', command = reduce_azimax, width = 12)
    azimax_txt = tk.Label(btn_frame, text = "azi max (rad): {}".format(camera_params[11][1]), bg = 'white',width = 14)
    #QUIT
    quit_btn = tk.Button(btn_frame, text = 'STOP', image = stop_logo, command = stop_prog, borderwidth = 10, relief = 'raise', font =  ("Arial", 20))
    #Pause
    pause_btn = tk.Button(btn_frame, image = pause_logo, command = pause, relief = 'raise', borderwidth = 6)
    #labels
    mainlabel = tk.Label(btn_frame, text = 'Parameters', font =  ("Arial", 20))
    authorlabel = tk.Label(credits_frame, text = "Ecrit par: Luc BOULESTEIX \n DUT MP Orsay \n Tuteur: S. Rodriguez \n Stage S4 2022", font = ("Arial",14 ))
    imglabel = tk.Label(img_frame, image = '')
    logolabel = tk.Label(credits_frame, image = logo)
    framelabel = tk.Label(btn_frame, text = "Frame #{}".format(str(frame_id)))
    #frameratelabel = tk.Label(btn_frame, text = str(framerate))
    #placement
    #x rotation
    rx_minus.place(x = 15,y = 60)
    rx_txt.place(x = 140, y = 60)
    rx_plus.place(x = 270,y = 60)
    #y rotation
    ry_minus.place(x = 15,y = 90)
    ry_txt.place(x = 140, y = 90)
    ry_plus.place(x = 270,y = 90)
    #z rotation
    rz_minus.place(x = 15,y = 120)
    rz_txt.place(x = 140, y = 120)
    rz_plus.place(x = 270,y = 120)
    #x translation
    tx_minus.place(x = 15,y = 160)
    tx_txt.place(x = 140, y = 160)
    tx_plus.place(x = 270,y = 160)
    #y translation
    ty_minus.place(x = 15,y = 190)
    ty_txt.place(x = 140, y = 190)
    ty_plus.place(x = 270,y = 190)
    #z translation
    tz_minus.place(x = 15,y = 220)
    tz_txt.place(x = 140, y = 220)
    tz_plus.place(x = 270,y = 220)
    #fx index
    fx_minus.place(x = 15,y = 300)
    fx_txt.place(x = 140, y = 300)
    fx_plus.place(x = 270,y = 300)
    #fy index
    fy_minus.place(x = 15,y = 330)
    fy_txt.place(x = 140, y = 330)
    fy_plus.place(x = 270,y = 330)
    #cx index
    cx_minus.place(x = 15,y = 370)
    cx_txt.place(x = 140, y = 370)
    cx_plus.place(x = 270,y = 370)
    #cy index
    cy_minus.place(x = 15,y = 400)
    cy_txt.place(x = 140, y = 400)
    cy_plus.place(x = 270,y = 400)
    #min azi
    azimin_minus.place(x = 15,y = 450)
    azimin_txt.place(x = 140, y = 450)
    azimin_plus.place(x = 270,y = 450)
    #max azi
    azimax_minus.place(x = 15,y = 480)
    azimax_txt.place(x = 140, y = 480)
    azimax_plus.place(x = 270,y = 480)
    #quit btn
    quit_btn.place(x = 180, y = 540)
    #main label
    mainlabel.place(x = 110, y = 10)
    #img label
    imglabel.pack()
    #author label
    authorlabel.place(x = 550, y = 20)
    #logo placement
    logolabel.place(x = 10, y = 10)
    #frame id placement
    framelabel.place(x = 20, y = 510)
    pause_btn.place(x = 80, y = 540)
    #run
    update_display()
    #set video loop
    #root.after(100, update_display)
    root.mainloop()

def update_display_btn():
    global cam_color_frames
    global velodyne_frames
    global pairs
    global first_img
    global imglabel
    global final_matrix
    global rx_txt,ry_txt,rz_txt,tx_txt,ty_txt,tz_txt,fx_txt,fy_txt,cx_txt,cy_txt
    global azimin_txt,azimax_txt
    global frame_id
    global framelabel,framerate_label
    global sanitized_lidar
    global i_ts


    #new matrix coefficients = need to recalculate points on image
    sanitized_lidar = sanitize(velodyne_frames[pairs[frame_id][1]], final_matrix, struct_chain , azi_min = camera_params[-2][1], azi_max = camera_params[-1][1])
    newimg = draw_on_image(cam_color_frames[pairs[frame_id][0]], sanitized_lidar[2], sanitized_lidar[3]) #computed image
    newimg = img_format(newimg) #recompute image
    #print(newimg)

    #update text labels
    rx_txt.configure(text = "x angle: {}".format(camera_params[0][1]))
    ry_txt.configure(text = "y angle: {}".format(camera_params[1][1]))
    rz_txt.configure(text = "z angle: {}".format(camera_params[2][1]))
    tx_txt.configure(text = "x trans: {}".format(camera_params[3][1]))
    ty_txt.configure(text = "y trans: {}".format(camera_params[4][1]))
    tz_txt.configure(text = "z trans: {}".format(camera_params[5][1]))
    fx_txt.configure(text = "fx coef: {}".format(camera_params[6][1]))
    fy_txt.configure(text = "fy coef: {}".format(camera_params[7][1]))
    cx_txt.configure(text = "cx coef: {}".format(camera_params[8][1]))
    cy_txt.configure(text = "cy coef: {}".format(camera_params[9][1]))
    azimin_txt.configure(text = "min azimuth :{}".format(camera_params[10][1]))
    azimax_txt.configure(text = "max azimuth: {}".format(camera_params[11][1]))
    #update image
    #framelabel.configure(text = str(frame_id))
    imglabel.configure(image=newimg)
    imglabel.image=newimg


def update_display():
    global cam_color_frames,velodyne_frames
    global final_matrix
    global pairs,sanitized_lidar
    global imglabel,framelabel,frameratelabel
    global final_matrix
    global rx_txt,ry_txt,rz_txt,tx_txt,ty_txt,tz_txt,fx_txt,fy_txt,cx_txt,cy_txt
    global azimin_txt, azimax_txt
    global frame_id, paused, frame_ratio
    global framerate, i_ts

    if paused == False:
        frame_id = (frame_id + frame_ratio)%(len(cam_color_frames)-1) #replays recording at the end
    elif (paused == True):
        frame_id = frame_id

    if (lidar_optimize == True):
        if (pairs[frame_id][1] != pairs[frame_id-1][1]): #if LIDAR information changes, recalculate, otherwise not (optimization)
            sanitized_lidar = sanitize(velodyne_frames[pairs[frame_id][1]], final_matrix, struct_chain , azi_min = camera_params[-2][1], azi_max = camera_params[-1][1])
    else:
        sanitized_lidar = sanitize(velodyne_frames[pairs[frame_id][1]], final_matrix, struct_chain , azi_min = camera_params[-2][1], azi_max = camera_params[-1][1])
    newimg = draw_on_image(cam_color_frames[pairs[frame_id][0]], sanitized_lidar[2], sanitized_lidar[3]) #computed image
    newimg = img_format(newimg) #recompute image
    #print(newimg)
    #frameratelabel.configure(text = str(framerate))

    framelabel.configure(text = "Frame #{}".format(str(frame_id)))
    imglabel.configure(image=newimg)
    imglabel.image=newimg
    i_ts = time.time()
    root.after(recording_period, update_display)

### BUTTONS

def reduce_angle_rx(): #from param_list, decreases angle rx and reloads transformation matrix
    #declarations de variables globales pour que les matrices et données soient accessibles partout
    global camera_params
    global matrix_dir
    global final_matrix
    camera_params[0][1] = round((camera_params[0][1] - r_inc),3) #decrease
    save_params(matrix_dir,camera_params)
    new = load(matrix_dir)
    camera_params = new[1]
    final_matrix = new[0]
    update_display_btn()


def increase_angle_rx(): #from param_list, increases angle rx and reloads transformation matrix
    #declarations de variables globales pour que les matrices et données soient accessibles partout
    global camera_params
    global matrix_dir
    global final_matrix
    camera_params[0][1] = round((camera_params[0][1] + r_inc),3) #increases
    save_params(matrix_dir,camera_params)
    new = load(matrix_dir)
    camera_params = new[1]
    final_matrix = new[0]
    update_display_btn()

def reduce_angle_ry(): #from param_list, decreases angle ry and reloads transformation matrix
    #declarations de variables globales pour que les matrices et données soient accessibles partout
    global camera_params
    global matrix_dir
    global final_matrix
    camera_params[1][1] = round((camera_params[1][1] - r_inc),3) #decrease
    save_params(matrix_dir,camera_params)
    new = load(matrix_dir)
    camera_params = new[1]
    final_matrix = new[0]
    update_display_btn()


def increase_angle_ry(): #from param_list, increases angle ry and reloads transformation matrix
    #declarations de variables globales pour que les matrices et données soient accessibles partout
    global camera_params
    global matrix_dir
    global final_matrix
    camera_params[1][1] = round((camera_params[1][1] + r_inc),3) #increases
    save_params(matrix_dir,camera_params)
    new = load(matrix_dir)
    camera_params = new[1]
    final_matrix = new[0]
    update_display_btn()

def reduce_angle_rz(): #from param_list, decreases angle ry and reloads transformation matrix
    #declarations de variables globales pour que les matrices et données soient accessibles partout
    global camera_params
    global matrix_dir
    global final_matrix
    camera_params[2][1] = round((camera_params[2][1] - r_inc),3) #decrease
    save_params(matrix_dir,camera_params)
    new = load(matrix_dir)
    camera_params = new[1]
    final_matrix = new[0]
    update_display_btn()


def increase_angle_rz(): #from param_list, increases angle rz and reloads transformation matrix
    #declarations de variables globales pour que les matrices et données soient accessibles partout
    global camera_params
    global matrix_dir
    global final_matrix
    camera_params[2][1] = round((camera_params[2][1] + r_inc),3) #increases
    save_params(matrix_dir,camera_params)
    new = load(matrix_dir)
    camera_params = new[1]
    final_matrix = new[0]
    update_display_btn()

def reduce_tx(): #decreases translation indicator along x axis en reloads matrix
    #declarations de variables globales pour que les matrices et données soient accessibles partout
    global camera_params
    global matrix_dir
    global final_matrix
    camera_params[3][1] = round((camera_params[3][1] - t_inc),2) #decrease
    save_params(matrix_dir,camera_params)
    new = load(matrix_dir)
    camera_params = new[1]
    final_matrix = new[0]
    update_display_btn()


def increase_tx(): #increases translation indicator along x axis en reloads matrix
    #declarations de variables globales pour que les matrices et données soient accessibles partout
    global camera_params
    global matrix_dir
    global final_matrix
    camera_params[3][1] = round((camera_params[3][1] + t_inc),2) #increases
    save_params(matrix_dir,camera_params)
    new = load(matrix_dir)
    camera_params = new[1]
    final_matrix = new[0]
    update_display_btn()

def reduce_ty(): #decreases translation indicator along y axis en reloads matrix
    #declarations de variables globales pour que les matrices et données soient accessibles partout
    global camera_params
    global matrix_dir
    global final_matrix
    camera_params[4][1] = round((camera_params[4][1] - t_inc),2) #decrease
    save_params(matrix_dir,camera_params)
    new = load(matrix_dir)
    camera_params = new[1]
    final_matrix = new[0]
    update_display_btn()


def increase_ty(): #increases translation indicator along y axis en reloads matrix
    #declarations de variables globales pour que les matrices et données soient accessibles partout
    global camera_params
    global matrix_dir
    global final_matrix
    camera_params[4][1] = round((camera_params[4][1] + 0.05),2) #increases
    save_params(matrix_dir,camera_params)
    new = load(matrix_dir)
    camera_params = new[1]
    final_matrix = new[0]
    update_display_btn()

def reduce_tz(): #decreases translation indicator along z axis en reloads matrix
    #declarations de variables globales pour que les matrices et données soient accessibles partout
    global camera_params
    global matrix_dir
    global final_matrix
    camera_params[5][1] = round((camera_params[5][1] - t_inc),2) #decrease
    save_params(matrix_dir,camera_params)
    new = load(matrix_dir)
    camera_params = new[1]
    final_matrix = new[0]
    update_display_btn()


def increase_tz(): #increases translation indicator along z axis en reloads matrix
    #declarations de variables globales pour que les matrices et données soient accessibles partout
    global camera_params
    global matrix_dir
    global final_matrix
    camera_params[5][1] = round((camera_params[5][1] + t_inc),2) #increases
    save_params(matrix_dir,camera_params)
    new = load(matrix_dir)
    camera_params = new[1]
    final_matrix = new[0]
    update_display_btn()

def reduce_fx(): #decreases optical index of camera lens for translation along x axis
    #declarations de variables globales pour que les matrices et données soient accessibles partout
    global camera_params
    global matrix_dir
    global final_matrix
    camera_params[6][1] = camera_params[6][1] - 1 #decreases by one pixel
    save_params(matrix_dir,camera_params)
    new = load(matrix_dir)
    camera_params = new[1]
    final_matrix = new[0]
    update_display_btn()


def increase_fx(): #increases optical index of camera lens for translation along x axis
    #declarations de variables globales pour que les matrices et données soient accessibles partout
    global camera_params
    global matrix_dir
    global final_matrix
    camera_params[6][1] = camera_params[6][1] + 1 #increases by one pixel
    save_params(matrix_dir,camera_params)
    new = load(matrix_dir)
    camera_params = new[1]
    final_matrix = new[0]
    update_display_btn()

def reduce_fy(): #decreases optical index of camera lens for translation along y axis
    #declarations de variables globales pour que les matrices et données soient accessibles partout
    global camera_params
    global matrix_dir
    global final_matrix
    camera_params[7][1] = round((camera_params[7][1] - 1),2) #decreases by one pixel
    save_params(matrix_dir,camera_params)
    new = load(matrix_dir)
    camera_params = new[1]
    final_matrix = new[0]
    update_display_btn()


def increase_fy(): #increases optical index of camera lens for translation along y axis
    #declarations de variables globales pour que les matrices et données soient accessibles partout
    global camera_params
    global matrix_dir
    global final_matrix
    camera_params[7][1] = round((camera_params[7][1] + 1),2) #increases by one pixel
    save_params(matrix_dir,camera_params)
    new = load(matrix_dir)
    camera_params = new[1]
    final_matrix = new[0]
    update_display_btn()

def reduce_cx(): #decreases optical center of image along x axis
    #declarations de variables globales pour que les matrices et données soient accessibles partout
    global camera_params
    global matrix_dir
    global final_matrix
    camera_params[8][1] = round((camera_params[8][1] - 10),2) #decreases by 10 pixel
    save_params(matrix_dir,camera_params)
    new = load(matrix_dir)
    camera_params = new[1]
    final_matrix = new[0]
    update_display_btn()


def increase_cx(): #increases optical center of image along x axis
    #declarations de variables globales pour que les matrices et données soient accessibles partout
    global camera_params
    global matrix_dir
    global final_matrix
    camera_params[8][1] = round((camera_params[8][1] + 10),2) #increases by 10 pixel
    save_params(matrix_dir,camera_params)
    new = load(matrix_dir)
    camera_params = new[1]
    final_matrix = new[0]
    update_display_btn()

def reduce_cy(): #decreases optical center of image along y axis
    #declarations de variables globales pour que les matrices et données soient accessibles partout
    global camera_params
    global matrix_dir
    global final_matrix
    camera_params[9][1] = round((camera_params[9][1] - 10),2) #decreases by 10 pixel
    save_params(matrix_dir,camera_params)
    new = load(matrix_dir)
    camera_params = new[1]
    final_matrix = new[0]
    update_display_btn()


def increase_cy(): #increases optical center of image along y axis
    #declarations de variables globales pour que les matrices et données soient accessibles partout
    global camera_params
    global matrix_dir
    global final_matrix
    camera_params[9][1] = round((camera_params[9][1] + 10),2) #increases by 10 pixel
    save_params(matrix_dir,camera_params)
    new = load(matrix_dir)
    camera_params = new[1]
    final_matrix = new[0]
    update_display_btn()

def reduce_azimin(): #decreases minimum azimuth
    #declarations de variables globales pour que les matrices et données soient accessibles partout
    global camera_params
    global matrix_dir
    global final_matrix
    camera_params[10][1] = round((camera_params[10][1] - 0.05),2) #decreases
    save_params(matrix_dir,camera_params)
    new = load(matrix_dir)
    camera_params = new[1]
    final_matrix = new[0]
    update_display_btn()


def increase_azimin(): #increases minimum azimuth
    #declarations de variables globales pour que les matrices et données soient accessibles partout
    global camera_params
    global matrix_dir
    global final_matrix
    camera_params[10][1] = round((camera_params[10][1] + 0.05),2) #increases
    save_params(matrix_dir,camera_params)
    new = load(matrix_dir)
    camera_params = new[1]
    final_matrix = new[0]
    update_display_btn()

def reduce_azimax(): #decreases maximum azimuth
    #declarations de variables globales pour que les matrices et données soient accessibles partout
    global camera_params
    global matrix_dir
    global final_matrix
    camera_params[11][1] = round((camera_params[11][1] - 0.05),2) #decreases
    save_params(matrix_dir,camera_params)
    new = load(matrix_dir)
    camera_params = new[1]
    final_matrix = new[0]
    update_display_btn()


def increase_azimax(): #increases maximum azimuth
    #declarations de variables globales pour que les matrices et données soient accessibles partout
    global camera_params
    global matrix_dir
    global final_matrix
    camera_params[11][1] = round((camera_params[11][1] + 0.05),2) #increases
    save_params(matrix_dir,camera_params)
    new = load(matrix_dir)
    camera_params = new[1]
    final_matrix = new[0]
    update_display_btn()

def pause():
    global paused
    paused = not(paused)


def stop_prog():
    global root
    root.destroy()

###main
global c_flag
global c_auto

dataset_dir += sample_file #chemin complet du fichier
matrix_dir +=matrix_file

#Chargment des données
bag = rosbag.Bag(dataset_dir)

velodyne_frames = []
cam_color_frames = []
count  = 0
load_idx = 0 #counts number of dots to put after loading

for topic, msg, t in bag.read_messages():
    if topic == '/velodyne_points':
        velodyne_frames.append(msg)
    elif topic == '/cam_p/color/image_raw':
        cam_color_frames.append(msg)
    elif topic == '/cam_p/color/object_detector':
        object_frames.append(msg)
    count+=1

    if ((int(count/1000)%5) != load_idx):#simple loading indicator
        progress_indicator = '\rLoading file' + (int((count/1000)) % 5)*'.'
        load_idx = int((count/1000))%5
        print(progress_indicator, end = '')
print('\rSample file loaded ! ')

p = load(matrix_dir)
final_matrix = p[0]
camera_params = p[-1]

#recupération des données LIDAR avec timestamp et correspondance
pairs = pair_frames(cam_color_frames, velodyne_frames)
struct_chain = check_velodyne(velodyne_frames) #check for correct data structuring and returns format
init_display()