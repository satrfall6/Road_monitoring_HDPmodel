# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 11:25:55 2020

@author: satrf
"""

import os 
os.chdir(r'C:\Users\satrf\LB_stuff\Git\Road_monitoring_HDPmodel')
import cv2 
import pandas as pd
import tqdm
from os.path import join, isdir, isfile
import pickle

import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import built_descriptor as des
import topic_modeling as tm
import tomotopy as tp

data_path = (r'./data/')
frame_path = (r'./data/frames/')
path_of_videos = train_path = (r'./data/train/')
test_path = (r'./data/test/')
vw_path = (r'./data/visual_words/')

'''
# make video clips
video_to_frames(data_path, frame_path, "junction1" , frames_in_video = 400)

frames_to_avi_overlap(frame_path, train_path, fps = 25, 
                          frames_in_video = 50, 
                          frame_overlap = 25)
'''

# the list of the name of videos in the train path, 3598 clips in total
#sub_dir_train = [d[0:-4] for d in os.listdir(train_path) if isfile(join(train_path, d))]

# hyperparams 

TOTAL_REGIONS = 9
x_sp, y_sp, NUM_BINS = 12, 12, 8
TOTAL_WORDS = x_sp* y_sp* NUM_BINS + 1

if_show = False

# for saving and load object
def save_obj(path, obj, name):
    '''
    Objective: For saving the stats of training 
    
    '''
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path, name):
    '''
    Objective: For loaing the stats from training progress
    '''
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)
  

   
#%%
def make_descriptor(path_of_videos):
#%%
    # read the files in the video path 
    sub_files = [d[0:-4] for d in os.listdir(path_of_videos) 
            if isfile(join(path_of_videos, d)) 
            and fnmatch.fnmatch(d, '*.avi')]
    sub_files = sub_files[:]
       
    # Initialize the final descroptor for 9 regions
    final_vw_dict = dict((k, np.array([])) for k in range(TOTAL_REGIONS))  
    
    for videos in tqdm.tqdm(sub_files):

        cap = cv2.VideoCapture(join(path_of_videos, videos + '.avi'))
        ret, frame = cap.read()
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        
        # create mask for visualization
        color = (0, 255, 255)
        mask_klt = np.zeros_like(frame)
        mask_farn = np.zeros_like(frame)
        mask_farn[..., 1] = 255

        # Parameters for Shi-Tomasi corner detection
        feature_params = dict(maxCorners = 200, qualityLevel = 0.001, minDistance = 1, blockSize = 3)
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)

        # for counting which frame is being processing for each video
        frame_count = 0
    
        
        while cap.isOpened():
            
            # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
            ret, frame = cap.read()
            if not ret:
                break
    
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            #cpu
#            _, magnitude, angle, rgb = des.cal_farneback(prev_gray, gray, mask_farn)
            
            #gpu
            magnitude, angle, rgb = des.cal_farneback_gpu(prev_gray, frame, if_show)
            
            slic_sp_labels = des.cal_slic(frame, n=144, compactness_val = 52)
            sp_centers = des.find_sp_center(slic_sp_labels)
            
            if frame_count < 1:
                # for the first frame, calculate the interest points from Farneback 
                interest_pts = des.get_interest_pixel(magnitude) #return interest points coordinate in the form for KLT tracker
                # find the intersection of points extract from goodFeaturesToTrack and interest point
                '''
                interest points might be too many, but will test how it goes later
                '''
                interest_pts = des.intersection(prev_pts, interest_pts)    
                if len(interest_pts)<1:
                    continue
                
                good_old, good_new, mask, status = des.cal_klt(interest_pts, prev_gray, gray, mask_klt)
                # initialize the dict for temporary visual word
                visual_words = np.zeros([good_new.shape[0], TOTAL_WORDS+3])# "+3" here are x, y, if the SP activate; 
                visual_words[0:good_new.shape[0], -2:] = good_old

            else:
                # from the 2nd frame, calculate klt from last frame
                good_old, good_new, mask, status = des.cal_klt(prev_pts, prev_gray
                                                    , gray, mask_klt)                                        
                


                
            for i, pts in enumerate(good_old):


                # if distance is too small, ignore the point
                if des.cal_distance(good_new[i][0],good_new[i][1],good_old[i][0],good_old[i][1]) < 0.25:
                    status[i][0] = 0
                   
                    '''
                    sometimes the KLT tracker returns points out off bound, 
                    not sure why, ignore these points for now
                    '''         
                elif good_new[i][0] >= frame.shape[1]:
                    status[i][0] = 0
                    
                elif good_new[i][1] >= frame.shape[0]:
                    status[i][0] = 0
                    
                else:
                    
                    '''
                    # important! good_new and good_old are in order [x,y]
                    '''
                    
                    # indicating where the good new points are in the visual word; 
                    pts_idx = np.where(np.sum((visual_words[:,-2:]==good_old[i])*1, axis=1) == 2)[0]
                    
                    # sometimes, more than 1 point will the klt point indicates to; bugging 
                    if len(pts_idx) != 1:
                        continue
                    
                    # get the rectangle around interest points
                    # Note: input order is x, y; output in the order [x_min, y_min, x_max, y_max]
                    rectangle = des.get_rectangle(visual_words[pts_idx,-2], visual_words[pts_idx, -1], gray)
                    
                    # find the dominant orientation for ip around 12*12 neighbors
                    dominant_ori = des.cal_hist(rectangle, angle)
                    
                    # find the sp that this ip belongs to                    
                    superpixel = slic_sp_labels[int(visual_words[pts_idx,-1]), 
                                                int(visual_words[pts_idx, -2])]
                    # name the corresponding sp, i.e. 0~143, since we use the index for assigning
                    '''
                    the way of naming might need to be modified. 
                    '''
                    name_of_sp = des.name_of_superpixel(sp_centers, superpixel)
                    
                    # check if the sp that the ip travel throuhg at this frame is activated
                    if_active = des.check_active(visual_words[pts_idx,-2:], slic_sp_labels, magnitude)
                    
                    
                    # keep summing the descriptor for interest points
                    # !! should use the "name_of_sp" instead of the original "superpixel"
                    visual_words[pts_idx, (name_of_sp-1)*NUM_BINS + 
                                             np.where(dominant_ori!=0)[0][0]] += 1
                    # if not activated, the indicator += 1
                    if not if_active:
                        # the 1153th, index is 1152
                        visual_words[pts_idx, -3] += 1
                                 
                    # assign new ip for next tracking                 
                    visual_words[pts_idx, -2:] = good_new[i]
                    
                    
                    if if_show == True:
                        mask_klt = cv2.line(mask_klt, (good_new[i][0], good_new[i][1]), (good_old[i][0], good_old[i][1]), color, 1)
     
            # sometimes videos contain no motion; bugging                
            if len(np.where(status!=0)[0])<1:
                break
            ############################################################################
            
            # assign new previos points for the klt track in the next loop
            prev_pts = good_new.reshape(-1, 1, 2)[status == 1].reshape(-1, 1, 2)
         
            # Updates previous frame
            prev_gray = gray.copy()
            frame_count += 1
      
            if if_show == True:    
                # Opens a new window and displays the input frame
                cv2.imshow("input", frame)
       
                # Overlays the optical flow tracks on the original frame
                output = cv2.add(frame, mask_klt)
                # Opens a new window and displays the output frame
                cv2.imshow("interest points flow" + videos , output) 
    
                # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
 
                      

        # only take interest point appeared more than 20 frames
        visual_words = visual_words[np.where(np.sum(visual_words[:,:-3], axis = 1)>=20)]
        rgs_mask = des.cal_region_mask(prev_gray,  n = 9, compactness_val = 50) 
        ###########################################################
        

        for i in range(visual_words.shape[0]):
            
            # check if the super pixel the point belongs is activated, and the region of this super pixel
            rgs = des.check_rgs(visual_words[i, -2:], rgs_mask)
            
            # keep updating the final visual word dict
            if len(final_vw_dict[rgs]) < 1:
                final_vw_dict[rgs] = visual_words[i,:-2].reshape(1,-1)
            else:
                final_vw_dict[rgs] = np.r_[final_vw_dict[rgs], visual_words[i,:-2].reshape(1,-1)]
                                      

 #%% 
        cap.release()
        cv2.destroyAllWindows()
        
    return final_vw_dict

def cal_clip_score(model, clip_bow_dict, score_dict):
    
    clip_corpus = convert_to_nlp_format(clip_bow_dict, TOTAL_REGIONS)
    s_clip = 0
    w_rj = 0
    for rgs in clip_corpus.keys():
        if clip_corpus[rgs].shape[0] > 0:
            r_tr = model.get_topics()
            s_hist = score_dict[rgs]
            clip_reconstruction = np.zeros([1, TOTAL_WORDS])
            for doc in clip_corpus[rgs]:
                clip_reconstruction += tm.cal_region_reconstruction(doc, model)
            s_rj = tm.cal_confidence_score(clip_reconstruction, r_tr)
            if s_rj > np.percentile(s_hist, 98):
                w_rj = 1
        s_clip += s_rj * w_rj     
    return s_clip
#%%
#def main():
#    
#    if IF_TRAIN:
#        make_descrptor()
#        for rgs in final_vw_dict.keys()
#   
#    else:     
#        test()
#    
#    if IF_VISUALIZE:
#        
#        visualize()