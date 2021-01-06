# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 14:42:39 2021

@author: satrf
"""

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
            temp_doc = []
            for i, word_count in enumerate(doc):
                if word_count > 0:
                    temp_doc.append((i, int(word_count)))
            corpus_dict[key].append(temp_doc)
    return corpus_dict

def hdp_modeling(region_corpus, model_type, total_words, num_topics = 100):
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
    if model_type == 'lda':
        lda_model = gensim.models.LdaModel(corpus=region_corpus, id2word=None, 
                                   num_topics = num_topics)
        return lda_model
    elif model_type == 'hdp':
        region_dict = [str(i) for i in range(total_words +1)] 
        Hdp_model = gensim.models.hdpmodel.HdpModel(corpus=region_corpus, id2word=region_dict)
        return Hdp_model
#    else: 

def cal_region_reconstruction(doc, model):
    '''
    Objective: calculate equation(5) for an IP 
    input:
        doc: a document in BOW format (1153-vector)
        model: gensim model function, either 'lda' or 'hdp'
    Output:
        r_d: reconstruction of "an IP" in a clip
    '''
    # the total topics 
    document_topics = model.get_document_topics(doc, minimum_probability = 0.005)
    r_d = np.array([])    
    for topic in document_topics:
        topic_terms = model.get_topic_terms(topic[0])
        r = topic_terms_to_array(topic_terms) * topic[1]
        if len(r_d) < 1 :
            r_d = r
        else:
            r_d += r
            
    return r_d
 
def topic_terms_to_array(topic_terms, total_words):
    '''
    Objective: topic terms from gensim model is in the format [(word_id, word_prob)...]
                in order to sum the topic terms, convert to array
    input:
        topic_terms: [(0, 0.2), (1, 0.7), (3, 0.05)...]
    Output:
        [0.2, 0.7, 0, 0.05, ...]
    '''
    terms_array = np.zeros([1, total_words +1])
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