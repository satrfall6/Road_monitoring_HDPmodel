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
train_path = (r'./data/train/')
path_of_videos = (r'./data/total_video/') 
test_path = (r'./data/test/')
vw_path = (r'./data/visual_words/')
vw_dense_path = (r'./data/visual_words/KLTthre0.18_denseSampled_rgMaskRegular/')
vw_sparse_path = (r'./data/visual_words/KLTthre0.25_sparseSampled_rgMaskRegular/')

normal_test_path = (r'./data/normal_test/')
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

IF_TRAIN = False
if_show = True

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
def make_descriptor(path_of_videos, this_video = None):
#%%
    import time
    start = time.time()
    
    if not this_video: 
        pixel_discard = 0.5
        # read the files in the video path 
        sub_files = [d[0:-4] for d in os.listdir(train_path) 
                if isfile(join(train_path, d)) 
                and fnmatch.fnmatch(d, '*.avi')]
        sub_files = sub_files[209:210]
    
    else:
        pixel_discard = 0.85
        sub_files = [this_video]
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
#        feature_params = dict(maxCorners = 200, qualityLevel = 0.0001, minDistance = 3, blockSize = 5)
#        prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)

        # for counting which frame is being processing for each video
        frame_count = 0
        no_motion_count = 0
        
        while cap.isOpened():
            
            # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
            ret, frame = cap.read()
            if not ret:
                break
            last_frame = frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            #cpu
#            _, magnitude, angle, rgb = des.cal_farneback(prev_gray, gray, mask_farn)
            
            #gpu
            magnitude, angle, rgb = des.cal_farneback_gpu(prev_gray, frame, if_show)
            
            slic_sp_labels = des.cal_slic(frame, n=144, compactness_val = 52)
            sp_centers = des.find_sp_center(slic_sp_labels)
            
            if frame_count < 1:
                # for the first frame, calculate the interest points from Farneback 
                interest_pts = des.get_interest_pixel(magnitude, slic_sp_labels,
                                                      ratio_to_discard = pixel_discard) #return interest points coordinate in the form for KLT tracker
                if  len(interest_pts) < 1:
                    no_motion_count +=1
                    prev_gray = gray.copy()

                    if no_motion_count >=25:
                        break
                    continue
                '''
                set the way to calculate interest points
                '''
                # find the intersection of points extract from goodFeaturesToTrack and interest point

#                interest_pts = des.intersection(prev_pts, interest_pts)    
#                if len(interest_pts)<1:
#                    continue
#                
#                good_old, good_new, mask, status = des.cal_klt(interest_pts, prev_gray, gray, mask_klt)
                
#                try:
                good_old, good_new, mask, status = des.cal_klt(interest_pts, prev_gray, gray, mask_klt)
#                except:
#                    good_old, good_new, mask, status = des.cal_klt(prev_pts, prev_gray, gray, mask_klt)

                
                # initialize the dict for temporary visual word
                visual_words = np.zeros([good_new.shape[0], TOTAL_WORDS+2])# "+3" here are x, y, if the SP activate; 
                visual_words[0:good_new.shape[0], -2:] = good_old

            else:
                # from the 2nd frame, calculate klt from last frame
                good_old, good_new, mask, status = des.cal_klt(prev_pts, prev_gray
                                                    , gray, mask_klt)                                        
                
            
            active_sp_seen = []
            for i, pts in enumerate(good_old):


                # if distance is too small, ignore the point
                '''
                threshold of klt tracker
                '''
                if des.cal_distance(good_new[i][0],good_new[i][1],good_old[i][0],good_old[i][1]) < 0.18:
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

                    superpixel = slic_sp_labels[min(int(visual_words[pts_idx,-1]),frame.shape[0]-1), 
                                                min(int(visual_words[pts_idx,-2]),frame.shape[1]-1)]

                    # name the corresponding sp, i.e. 0~143, since we use the index for assigning
                    name_of_sp = des.name_of_superpixel(sp_centers, superpixel)
                    
                    # check if the sp that the ip travel throuhg at this frame is activated
                    if superpixel in active_sp_seen:
                        if_active = True
                    else:
                                        
                        if_active = des.check_active(magnitude, slic_sp_labels, None,
                                                     visual_words[pts_idx,-2:], frame)
                        # if not activated, the indicator += 1
                        if not if_active:
                            # the 1153th, index is 1152
                            visual_words[pts_idx, -3] += 1
                        else:
                            active_sp_seen.append(superpixel)
                    
                    # keep summing the descriptor for interest points
                    # !! should use the "name_of_sp" instead of the original "superpixel"
                    visual_words[pts_idx, (name_of_sp-1)*NUM_BINS + 
                                             np.where(dominant_ori!=0)[0][0]] += 1

                        
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
                cv2.imshow('dense flow', rgb)
                # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
 
                      

        # only take interest point appeared more than 24 frames
        visual_words = visual_words[np.where(np.sum(visual_words[:,:-3], axis = 1)>=24)]
        
        # calculate the region mask for locating the region of IPs
        rgs_mask = des.cal_region_mask(last_frame,  n = 9, compactness_val = 50) 
        ###########################################################
        

        for i in range(visual_words.shape[0]):
            
            # check if the super pixel the point belongs is activated, and the region of this super pixel
            rgs = des.check_rgs(visual_words[i, -2:], rgs_mask)
            
            # keep updating the final visual word dict
            if len(final_vw_dict[rgs]) < 1:
                final_vw_dict[rgs] = visual_words[i,:-2].reshape(1,-1)
            else:
                final_vw_dict[rgs] = np.r_[final_vw_dict[rgs], visual_words[i,:-2].reshape(1,-1)]
                                      

 
        cap.release()
        cv2.destroyAllWindows()
    end = time.time()
    print('total time spend is: ', end-start) 
#%%       
    return final_vw_dict


def visualize_anomalies(path_of_videos, video, anomalous_rgs):


    cap = cv2.VideoCapture(join(path_of_videos, video + '.avi'))
    ret, frame = cap.read()
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    
    # create mask for visualization
    anomaly_mask = np.zeros_like(frame)
    mask_klt = np.zeros_like(frame)
    color = (0, 255, 255)
    mask_farn = np.zeros_like(frame)
    mask_farn[..., 1] = 255

    feature_params = dict(maxCorners = 200, qualityLevel = 0.0001, minDistance = 3, blockSize = 5)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)

    # for counting which frame is being processing for each video
    frame_count = 0
    slic_sp_labels = des.cal_slic(frame, n=144, compactness_val = 52)

    while cap.isOpened():
        
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        if frame_count < 1:
#            _, magnitude, angle, rgb = des.cal_farneback(prev_gray, gray, mask_farn)
            magnitude, angle, rgb = des.cal_farneback_gpu(prev_gray, frame, if_show)
            interest_pts = des.get_interest_pixel(magnitude, slic_sp_labels,
                                                  ratio_to_discard = 0.75) 
#            rgs_mask = des.cal_region_mask(frame,  n = 9, compactness_val = 50) 

            try:
                good_old, good_new, mask, status = des.cal_klt(interest_pts, prev_gray, gray, mask_klt)
            except:
                good_old, good_new, mask, status = des.cal_klt(prev_pts, prev_gray, gray, mask_klt)

        else: 
            good_old, good_new, mask, status = des.cal_klt(prev_pts, prev_gray, gray, mask_klt)

        for i, pts in enumerate(good_old):
            
            if des.cal_distance(good_new[i][0],good_new[i][1],good_old[i][0],good_old[i][1]) > 0.175:

                mask_klt = cv2.line(mask_klt, (good_new[i][0], good_new[i][1]), (good_old[i][0], good_old[i][1]), color, 1)

        
        # assign new previos points for the klt track in the next loop
        prev_pts = good_new.reshape(-1, 1, 2)[status == 1].reshape(-1, 1, 2)
     
        # Updates previous frame
        prev_gray = gray.copy()
        frame_count +=1
        rgs_mask = des.cal_region_mask(gray,  n = 9, compactness_val = 50) 

        try:
            for a in anomalous_rgs:
                anomaly_mask[:,:,2][np.where(rgs_mask==a)] = 150
        except:
            pass

        # Overlays the optical flow tracks on the original frame
        output = cv2.add(frame, anomaly_mask)
        output = cv2.add(output, mask_klt)
        cv2.imshow('dense flow', rgb)
        # Opens a new window and displays the output frame
        cv2.imshow("anomaly region" , output) 
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        

    cap.release()
    cv2.destroyAllWindows()  

def cal_clip_score(model_dict, clip_bow_dict, score_dict, perc = 90):
    '''
    Objective: this is to calculate clip score and compare with the distribution
                that from "cal_region_r_and_s"
        
    '''
    clip_corpus = tm.convert_to_nlp_format(clip_bow_dict, TOTAL_REGIONS)
    s_clip = 0
    w_rj = 0
    anomalous_rgs = []
    for rgs in clip_corpus.keys():
        if len(clip_corpus[rgs]) < 1:
            continue
        r_tr = model_dict[rgs].get_topics()
        clip_reconstruction = np.zeros([1, TOTAL_WORDS])
        for doc in clip_corpus[rgs]:
            clip_reconstruction += tm.cal_ip_reconstruction(doc, model_dict[rgs])
        s_rj = tm.cal_confidence_score(clip_reconstruction, r_tr)
        

        s = score_dict[rgs]
        '''
        anomaly threshold and bin size are set here
        '''
        if s_rj > np.percentile(s, perc):
            s_hist, s_edges = np.histogram(s, bins = int(15))
            try:
                freq = s_hist[np.where(s_edges > s_rj)[0][0]-1]/len(s)
                if freq <= 0.02:
                    w_rj = 1
                    anomalous_rgs.append(rgs)
            except: 
                w_rj = 1
                anomalous_rgs.append(rgs)
        s_clip += s_rj * w_rj   
        
    return s_clip, anomalous_rgs      
#%%
#def main():
    
    if IF_TRAIN:
        
        final_vw_dict = make_descriptor(train_path)
        save_obj(vw_path, final_vw_dict, 'final_vw_dict')
        print('"final_vw_dict" was saved successfully!')
        corpus_dict = tm.convert_to_nlp_format(final_vw_dict, TOTAL_REGIONS)
        save_obj(vw_sparse_path, corpus_dict, 'corpus_dict_3')
        print('"corpus_dict" was saved successfully!')

        model_dict_hdp = tm.topic_modeling(corpus_dict, 'hdp', TOTAL_REGIONS, 
                   total_words = 1153, num_topics = 100)
        save_obj(vw_sparse_path, model_dict_hdp, 'model_dict_hdp_3')
        print('"model_dict_hdp" was saved successfully!')

        reconstruction_dict_hdp, score_dict_hdp = tm.cal_region_r_and_s(corpus_dict, model_dict_hdp, TOTAL_REGIONS)
        save_obj(vw_sparse_path, reconstruction_dict_hdp, 'reconstruction_dict_hdp_3')
        save_obj(vw_sparse_path, score_dict_hdp, 'score_dict_hdp_3')
        print('"reconstruction_dict and score_dict" was saved successfully!')



        #%%
#    else:    
        test_p = normal_test_path
        sub_files = [d[0:-4] for d in os.listdir(test_p) 
                    if isfile(join(test_p, d)) 
                    and fnmatch.fnmatch(d, '*.avi')]
        
        
        temp_dict = dict()
        for video in sub_files[:]:
            if video not in temp_dict.keys():
                clip_bow_dict = make_descriptor(test_p, video)
                temp_dict[video] = clip_bow_dict
        save_obj(test_p, temp_dict, 'normal_temp_dict')

        #%%

        temp_dict = load_obj(test_path, 'dense_temp_dict') 
#        temp_dict = load_obj(normal_test_path, 'normal_temp_dict') 
    
        test_model = 'hdp_1'
        path = vw_dense_path
        model_dict = load_obj(path,'model_dict_'+test_model )
        score_dict = load_obj(path, 'score_dict_'+test_model)        
    
    
        for video in temp_dict.keys():
            
    
            s_clip, anomalous_rgs = cal_clip_score(model_dict, temp_dict[video],
                                                   score_dict, perc=88)
            print('video', video, 'score is: ', s_clip)
            visualize_anomalies(test_path, video, anomalous_rgs)

