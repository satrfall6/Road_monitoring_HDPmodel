# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 11:22:04 2020

@author: satrf
"""
import numpy as np
import cv2
from skimage.segmentation import slic
from fast_slic import Slic


import math
import os 
import fnmatch

from os.path import join, isdir, isfile

LOW_THRESHOLD = 0.65

def cal_distance(x2, y2, x1, y1):

    return  math.sqrt((x2 - x1)**2 + (y2 - y1)**2)




def cal_farneback(prev_gray, gray, mask = None):
    '''
    Objective: calculate the Farneback flow of given 2 frames and return interest points
    input:
        prev_gray: last frame in gray-scale
        gray: current frame in gray-scale
        mask(temp): input the mask for drawing line, but might not use it 
    Output:
        flow: doesn't use it at this moment, might remove it later
        magnitude: the magnitude for the given 2 frames, for thresholding the pixels
        mask[..., 0]: it's actually the angle, for building the descriptor
        rgb: for visualizing the flow, not use at this moment
    '''
    fb_params = dict(pyr_scale  = 0.03, levels  = 1, winsize  = 4,
             iterations = 3, poly_n = 5, poly_sigma  = 1.2, flags = 0)   
    
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **fb_params)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=0)
 
    # Sets image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Sets image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # Converts HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    
    return flow, magnitude, mask[..., 0], rgb
    

def get_interest_pixel(magnitude):
    '''
    Objective: get the pixels within the threshold based on magnitude
    input:
        magnitude: the magnitude calculated from 2 frames

    Output:
        an array of interest points (x, y) in the shape of [n, 1, 2]
    '''
    mag = magnitude.copy()

    high_threshold  = np.percentile(magnitude, 95)
    mag[np.where(mag<LOW_THRESHOLD)] = 0
    mag[np.where(mag>high_threshold)] = 0
    
    non_zero_pixel = np.where(mag != 0)
    
    return np.c_[non_zero_pixel[0].reshape(-1,1), non_zero_pixel[1].reshape(-1,1)].reshape(-1,1,2).astype('float32')
 
def cal_klt(prev_pts, prev_gray, gray, mask):

      
    '''
    Objective: calculate the KLT flow of given 2 frames and return interest points
    input:
        prev_pts: the interest point of last frame
        prev_gray: last frame in gray-scale
        gray: current frame in gray-scale
        mask(temp): input the mask for drawing line, but might not use it 
    Output:
        good_old: the interest point of last frame(good_new from last), for checking the index at current frame 
        good_new: the interest point at current frame
        status: indicating that if find the next interest point for the point
    Note: both good_new and good_old are in shape [n, 2], but 
          the input for "cv2.calcOpticalFlowPyrLK" needs to be [n, 1, 2]
    '''
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize = (30, 24), maxLevel = 3, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.05))

    
    next_pts, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
    # Selects good feature points for previous position
    good_old = prev_pts[status == 1] #.reshape(-1, 2) 
    # Selects good feature points for next position
    good_new = next_pts[status == 1] #reshape(-1, 2) 
    

  
    return good_old, good_new, mask, status[np.where(status==1)].reshape(-1,1)#, prev_pts

def if_inbound(c_min, c_max, bound):

    '''
    Objective: sometimes the interest point might be on boundary, so shift the rectangle
    input:
        c_min: the minimun coordinate of the rectangle
        c_max: the maximum coordinate of the rectangle
        bound: the bound for image size
        
    Output:
        the shifted minimum and maximum of the rectangle (min, max)
    '''
    
    if c_min < 0:
        
        c_max = c_max + (0 - c_min)
        c_min = c_min + (0 - c_min)
        
    elif c_max >= bound:
        c_min -= (c_max - bound +1)
        c_max -= (c_max - bound +1)
        
    return int(c_min), int(c_max)
        
def get_rectangle(x, y, gray, size = 12):
    
    '''
    Objective: get the pixels within the rectangle of a point
    input:
        x: the x coordinate of the point
        y: the y coordinate of the point
        gray: the frame to get interest point from
        size: the length of the rectangle
    Output:
        minimum x, y and maximum x, y 
    '''
    x_min = x - int(size/2)
    x_max = x + int(size/2)
    y_min = y - int(size/2)
    y_max = y + int(size/2)
    y_min, y_max = if_inbound(y_min, y_max, gray.shape[0])
    x_min, x_max = if_inbound(x_min, x_max, gray.shape[1])
    
    rectangle = [x_min, y_min, x_max, y_max]
    
    return rectangle

def cal_hist(rectangle, angle, bins_num = 8):
    '''
    Objective: calculate the histogram of the orientation of the given rectangle region
    input:
        rectangle: output from "get_rectangle"
        angle: the array of angle calculated from Farneback flow "cal_farneback"

    Output:
        (size of rectangle)**2 * (bin size); EX: 12*12*8
    '''
    
    low, high, interval = 0, 180, 180/bins_num
    
    hist = np.array([])
    for x in range(rectangle[0], rectangle[2]):
        for y in range(rectangle[1], rectangle[3]):   
            ang_idx = categorize_by_bin(angle[y, x], low, high, interval)
            if len(hist) < 1:
                hist = np.zeros([1,bins_num])
                hist[...,ang_idx] += 1
            else:
                pts_hist = np.zeros([1,bins_num])
                pts_hist[...,ang_idx] += 1
#                hist = np.c_[hist, pts_hist] , modified in v1
                hist = hist + pts_hist
    n_bin = np.zeros([1, bins_num])
    n_bin[0][np.where(hist==np.amax(hist))[0][0]] +=1
    
    return n_bin
          
def categorize_by_bin(angle, low, high, interval, if_right = False):

    '''
    Objective: sort the orientation of the flow points to 9 direction (0, 180, 20)
    input:
        angle: the angle in degree
        low: the low boundary
        high: the high boundary
        interval: the interval of the range
        if_right: 
    Output:
        the category of the angle of this point (0~7)
    
    '''
    
    
    bins = np.linspace(low, high, int(((high-low) / interval)+1))
    
    return np.clip(np.digitize(angle, bins, right=if_right), 1, 8)-1
   

def intersection(prev_pts, interest_pts): 
    '''
    Objective: check if the elements in the first array also in the second one
    Input:
        2 arrays

    Output:
        the element in both array
    '''
    
    arr = [v for v in prev_pts if v in interest_pts] 
    
    return np.array(arr)

def cal_slic(frame, n, compactness_val):
    
    '''
    Objective: seperate the frames into super pixels which have similar characteristic
    Input:
        the running frames

    Output:
        an array shows different regions in numbers
        
        an arry shows regions in numbers
   
    '''
#    slic_mask = slic(frame, n_segments=n, compactness=compactness_val)
    slic_func = Slic( num_components=n, compactness=compactness_val)
    slic_mask = slic_func.iterate(frame)
    return slic_mask

def cal_region_mask(frame, n, compactness_val):
    
    '''
    Objective: seperate the frames into super pixels which have similar characteristic
    Input:
        the running frames

    Output:
        an array shows different regions in numbers
        
        an arry shows regions in numbers
    Note!!: don't know what is the best way to make sure the region for each
            frame is in the same location, just hard code it for 9 regions at 
            this moment
            
    '''
    region_mask = slic(frame, n_segments=n, compactness = compactness_val)
    while len(set(region_mask.reshape(-1))) < n:
        compactness_val += 2 
        region_mask = slic(frame, n_segments=n, compactness = compactness_val)
#        if len(set(region_mask.reshape(-1))) > n:
#            compactness_val -= 1
#            region_mask = slic(frame, n_segments=n, compactness = compactness_val)
    corner = [np.array([0,0]), 
              np.array([0, frame.shape[1]-1]),
              np.array([frame.shape[0]-1, 0]), 
              np.array([frame.shape[0]-1, frame.shape[1]-1])]
    corner_val = np.array([0, 2, 6, 8])
    for i, c in enumerate(corner):
        region_mask[np.where(region_mask == region_mask[c[0],c[1]])] = corner_val[i]+n

############### need to be re-done ###############

    for x in range(corner[0][1], corner[1][1]):
        if region_mask[0, x] not in corner_val + n:
            region_mask[np.where(region_mask == region_mask[0, x])] = 1 + n
            corner_val =np.append(corner_val, 1)
    
    for y in range(corner[0][0], corner[2][0]):
        if region_mask[y, 0] not in corner_val + n:
            region_mask[np.where(region_mask == region_mask[y, 0])] = 3 + n
            corner_val =np.append(corner_val, 3)
    for y in range(corner[1][0], corner[3][0]):
        if region_mask[y, corner[3][1]] not in corner_val + n:
            region_mask[np.where(region_mask == region_mask[y, corner[3][1]])] = 5 + n
            corner_val =np.append(corner_val, 5)
            
    for x in range(corner[2][1], corner[3][1]):
        if region_mask[frame.shape[0]-1, x] not in corner_val + n:
            region_mask[np.where(region_mask == region_mask[frame.shape[0]-1, x])] = 7 + n
            corner_val = np.append(corner_val, 7)
    for r in set(region_mask.reshape(-1)):
        if r not in corner_val + n:
            region_mask[np.where(region_mask == r)] = 4 + n
            corner_val = np.append(corner_val, 4)

###########################################################################                 
    return region_mask-9

def name_of_superpixel(sp_centers, superpixel):
    '''
    Objective: each superpixel need to be named for later modeling.
                naming the SPs by their center position to make sure 
                for different frames, the SPs represent the same ones
    Input:
        the labels for SPs

    Output:
        the "name" for each superpixel, an integer
   
    '''
    return np.where(sp_centers[:,0]==superpixel)[0][0]
    
    
def find_sp_center(sp_label):
    
    '''
    Objective: for each SP, find the center. since the way of finidng the center
                might change, build independent function
    
    Input:
        the labels for SPs

    Output:
        the form of sp and centers combination, 
        i.e. [[1,(150, 266)], ... [144,(186, 174)]]
        
    note: np.where() return tuple(y, x)
            problem is not visually ordered, test if all the frames in similar order
   
    '''
    sp_center_list = []
    for sp in set(sp_label.reshape(-1)):
        y_max = np.max(np.where(sp_label == sp)[0])
        y_min = np.min(np.where(sp_label == sp)[0])
        x_max = np.max(np.where(sp_label == sp)[1])
        x_min = np.min(np.where(sp_label == sp)[1])
    
        center = (int((y_min+y_max)/2), int((x_min+x_max)))
        sp_center_list.append([sp, center])
    return np.array(sort_by_2nd(sp_center_list))

def sort_by_2nd(sub_li): 
    '''
    sort by the second element
    '''
    
    l = len(sub_li) 
    for i in range(0, l): 
        for j in range(0, l-i-1): 
            if (sub_li[j][1] > sub_li[j + 1][1]): 
                tempo = sub_li[j] 
                sub_li[j]= sub_li[j + 1] 
                sub_li[j + 1]= tempo 
    return sub_li

    
#%%  
    
'''
need to be updated 
'''


def check_active(pts, slic_mask, magnitude):
    
    sp = slic_mask[int(pts[0][1]), int(pts[0][0])]
    total_pixel = len(np.where(slic_mask==sp)[0])
    active_pixel = sum(magnitude[np.where(slic_mask==sp)]>LOW_THRESHOLD)
    if active_pixel/total_pixel >0.4:
        if_active = True
    else: 
        if_active = False
        
    return if_active

def check_rgs(pts, rgs_mask):
    
   return int(rgs_mask[int(pts[1]), int(pts[0])])

def frames_to_avi_overlap(path_in, path_out, name_of_sub_dir = None, fps = 25, 
                          frames_in_video = 250, 
                          frame_overlap = 50):
    '''
    Objective: when the frames of a scene are not enough, make videos with overlapped frames
    input:
        path_in: the root path of the directories of the frames, e.g.: for the same scene, day1, day2,...
        path_out: the path to store the made videos
        name_of_sub_dir: day1, day2,... or train001, train002, ..., etc.
        fps: frame per second
        frames_in_video: max frames in a video
        frame_gap: ex:20, 1~200, 20~220, 40~240,..., etc.
    Output:
        videos made from the frames
    
    '''

    data_path = path_in#join(path_in, name_of_sub_dir)
    files = [f for f in os.listdir(data_path) if isfile(join(data_path, f)) and (fnmatch.fnmatch(f, '*.tif') or fnmatch.fnmatch(f, '*.jpg'))]
    #for sorting the file names properly
    files.sort(key = lambda x: x[:-4])
    files.sort()
#    os.makedirs(join(path_out,name_of_sub_dir),exist_ok=True)
    
    video_count = 0
    for i in range(0, len(files)-frames_in_video, frames_in_video-frame_overlap):
        
        frame_array = []
        for j in range(0, frames_in_video):
            filename = path_in + '/' + files[i+j]
            #reading each files
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)       
            #inserting the frames into an image array
            frame_array.append(img)
        
        file_name = path_out + '/'+str(video_count).zfill(4)+'.avi'
        out = cv2.VideoWriter(file_name,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        for f in range(len(frame_array)):
            # writing to a image array
            out.write(frame_array[f])
        out.release()    
        
        video_count+=1

def video_to_frames(path_in, path_out, name_of_videos, frames_in_video = 400):
    
    '''
    Objective: when the video is very long, e.g. 90000 frames, slice the frames into smaller videos
    input:
        path_in: the path of the video
        path_out: the path to store the made sub-directories of frames, e.g. ./name_of_videos/(0001, 0002, ...)
        name_of_sub_dir: 0001, 0002, ..., etc.
        frames_in_video: max frames in a video

    Output:
        sub-directories with frames inside under the path ./path_out/name_of_videos/(0001, 0002, ...)  
    
    '''


    # create a folder for each video to store its frames
    
    # set a variable to store video data 
    
     
    cap = cv2.VideoCapture(join(path_in, name_of_videos + '.avi'))

    # this indicates that which frame are we looking at
    order_of_frames = 1
    
#    video_count = 0    
    
    gap_between_frames = 1   
    # run the loop to all image direcotry(each video)
    while cap.isOpened(): 
        
        ret, frame = cap.read()
        if not ret:
            break
#        if order_of_frames % frames_in_video ==1:
#            video_count += 1
#            order_of_frames = 1
#            os.makedirs(join(path_out, name_of_videos+'/'+str(video_count).zfill(4)), exist_ok=True)
#            pic_path = join(path_out, name_of_videos+'/'+str(video_count).zfill(4))
#            
        if (order_of_frames % gap_between_frames == 0):  # save frame for every 2 frames
            # print(pic_path + str(c) + '.jpg')

             cv2.imwrite(path_out + '/' +  
                         str(order_of_frames).zfill(5) + '.jpg', frame)  
#            cv2.imwrite(pic_path + '/' +  
#                        str(order_of_frames).zfill(3) + '.jpg', frame)  

        order_of_frames += 1
#        cv2.waitKey(0)
    # after running through each frame, close this video    
    cap.release()