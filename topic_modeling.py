# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 14:42:39 2021

@author: satrf
"""
import tqdm
import gensim
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
 
def convert_to_nlp_format(final_vw_dict, total_regions):
    '''
    Objective: the descriptor is in BOW foramt, transfer it to [(word_id, word_count),...]
    input:
        the dictionary for all regions of BOW array (1153-vector)
    output: 
        dictionary contains list of list 
    '''
    corpus_dict = dict((k, []) for k in range(total_regions))
    for key in final_vw_dict.keys():
        for doc in final_vw_dict[key]:
            temp_doc = [(i, int(word_count)) for i, word_count in enumerate(doc)
                                                if word_count>0]
            
            corpus_dict[key].append(temp_doc)
    return corpus_dict

def cal_region_r_and_s(corpus_dict, model_dict, total_regions):
    '''
    Objective: this is to calculate reconstruction and score for the base distribution
                different from "cal_clip_score"
    input:
        
    output: 
        reconstruction_dict: a dictionary of arrays, stores reconstructions for IPs
        score_dict: a 
    '''
    reconstruction_dict = dict((k, np.array([])) for k in range(total_regions))  
    score_dict = dict((k, np.array([])) for k in range(total_regions))
    for rgs in tqdm.tqdm(corpus_dict.keys()):
        
        if len(corpus_dict[rgs]) >0:
    
#            lda_model = gensim.models.LdaModel(corpus=corpus_dict[rgs], id2word=None, 
#                                           num_topics = 150) 
            r_region = np.array([cal_ip_reconstruction(doc, model_dict[rgs]) 
                                for doc in corpus_dict[rgs]
                                ])   
            reconstruction_dict[rgs] = r_region.reshape(r_region.shape[0], -1)
            '''
            r_tr test here
            '''            
            r_tr = model_dict[rgs].get_topics()
            
            score_dict[rgs] = [cal_confidence_score(r_j.reshape(1,-1), r_tr) 
                               for r_j in reconstruction_dict[rgs]
                               ]
    return reconstruction_dict, score_dict

def topic_modeling(corpus_dict, model_type, total_regions, 
                   total_words = 1153, num_topics = 150):
    '''
    Objective: do "lda" and "hdp" modeling 
    input:
        region_corpus: corpus of a region, the output from convert_to_nlp_format
        model_type: a string, either 'lda' or 'hdp'
        num_topics: it is available to assign num_topics for lda model while hdp not 
    Output:
        flow: doesn't use it at this moment, might remove it later
        magnitude: the magnitude for the given 2 frames, for thresholding the pixels
        mask[..., 0]: it's actually the angle, for building the descriptor
        rgb: for visualizing the flow, not use at this moment
    '''
    model_dict = dict((k, None) for k in range(total_regions))
    for rgs in corpus_dict.keys(): 
        if len(corpus_dict[rgs]) > 0:          
            if model_type == 'lda':
                lda_model = gensim.models.LdaModel(corpus=corpus_dict[rgs], id2word=None, 
                                           num_topics = num_topics)
                model_dict[rgs] = lda_model
            elif model_type == 'hdp':
                region_dict = [str(i) for i in range(total_words)] 
                hdp_model = gensim.models.hdpmodel.HdpModel(corpus=corpus_dict[rgs], 
                                                            id2word=region_dict)
                model_dict[rgs] = hdp_model
            else: 
                print("Not an available model type!")
                return None
    return model_dict

def cal_ip_reconstruction(doc, model):
    '''
    Objective: calculate equation(5) for an IP 
    input:
        doc: a document in (word_id, word_cound) format 
        model: gensim model function, either 'lda' or 'hdp'
    Output:
        r_d: reconstruction of "an IP" in a clip (IP in a region)
        (A region might contains multiple IPs)
    '''
    # the total topics 
    try:
        document_topics = model.get_document_topics(doc, minimum_probability = 0.005)
    except:
        document_topics = [t for t in model.__getitem__(doc) if t[1] > 0.005]

    r_d = np.array([])    
    for topic in document_topics:
        try:
            topic_terms = model.get_topic_terms(topic[0])
        except:
            topic_terms_dist = model.get_topics()[topic[0]]
            over_thres = topic_terms_dist[np.where(topic_terms_dist>0.0035)].shape[0]
            top_words = topic_terms_dist.argsort()[-over_thres:][::-1]
            top_words_prob = topic_terms_dist[topic_terms_dist.argsort()[-over_thres:][::-1]]
            topic_terms = [(top_words[i], top_words_prob[i]) for i in range(over_thres)]
        r = topic_terms_to_array(topic_terms) * topic[1]
        if len(r_d) < 1 :
            r_d = r
        else:
            r_d += r
            
    return r_d
 
def topic_terms_to_array(topic_terms, total_words=1153):
    '''
    Objective: topic terms from gensim model is in the format [(word_id, word_prob)...]
                in order to sum the topic terms, convert to array
    input:
        topic_terms: [(0, 0.2), (1, 0.7), (3, 0.05)...]
    Output:
        [0.2, 0.7, 0, 0.05, ...]
    '''
    terms_array = np.zeros([1, total_words])
    for terms in topic_terms:
        terms_array[0, int(terms[0])] += terms[1]
    return terms_array

def cal_confidence_score(r_j, r_tr):
    '''
    Objective: to calculate the confidence score s_rj for a "region of a clip"
               the smaller the s_rj is, more similar r_j to r_tr
    input:
        r_j: 1153-vector, summing all the r_d in a region of a clip
        r_tr: a n*1153 array
              the base reconstruction for a region
              use the 150 topics for now
    Output:
        s_rj: the score for "a region" of a clip, so it's the sum of all IPs
    '''
    return min([cal_cosine_similarity(r_j, r) for r in r_tr])
    
    
def cal_cosine_similarity(r1, r2):
    '''
    Objective: calculate the cosine similarity for 2 given vectors
    '''
    return 1 - (cosine_similarity(r1, r2.reshape(1,-1), dense_output=True))[0][0]
#%%
