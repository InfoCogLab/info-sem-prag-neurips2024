"""
Compute complexity of human semantic system
"""

import os

import numpy as np
import pandas as pd
import random
import torch

import src.settings as settings
from src.data_utils.helper_fns import gen_batch
from src.data_utils.read_data import get_feature_data
from src.data_utils.read_data import load_cleaned_results

from src.models.vae import VAE

from src.data_utils.read_data import get_glove_vectors



def run():
    if settings.see_distractors_pragmatics:
        num_imgs = 3 if settings.with_ctx_representation else 2
    else:
        num_imgs = 1 if not settings.see_distractor else (num_distractors + 1)
   
    for seed in settings.seeds:
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
        data = get_feature_data(t_features_filename, excluded_ids=excluded_ids)
        data = data.sample(frac=1, random_state=46) # Shuffle the data.

        # initialize dictionary of human names
        human_names = []
        for i in list(data['responses']):
            for j in i.keys():
                human_names.append(j)
        human_names = list(set(human_names))
        human_probs = {key: [] for key in human_names}
        ids_to_names = dict(enumerate(human_names))

        for targ_idx in list(data.index.values):

            speaker_obs, _, _, _ = gen_batch(data, 1, "topname", p_notseedist=1, glove_data=glove_data, preset_targ_idx=targ_idx)
            
            if speaker_obs != None: # i.e. we have the glove embeds
                
                responses = data['responses'][targ_idx]
                total = sum(list(responses.values()))
                normalized_responses = {key: value/total for key, value in responses.items()}
                for k,v in human_probs.items():
                    if k in normalized_responses.keys():
                        human_probs[k].append(normalized_responses[k])
                    else:
                        human_probs[k].append(0.0)
 
        
        human_matrix = np.empty([len(list(human_probs.values())[0]), len(human_names)])
        for i in range(len(human_names)):
            human_matrix[:, i] = human_probs[ids_to_names[i]]
        
        human_matrix = np.array(human_matrix)
        print(human_matrix.shape)
        
        epsilon = 1e-10

        # prob of objects
        p_o = np.full(human_matrix.shape[0], 1/human_matrix.shape[0])  # each object is equally likely
        
        # calculate marginal probability of each word
        p_w = human_matrix.T @ p_o    

        joint_probs = human_matrix * p_o[:, np.newaxis]

        mi = joint_probs * np.log2((joint_probs + epsilon) / (p_o[:, np.newaxis] * p_w + epsilon))
        total_mi = np.nansum(mi)  # summing up all the MI 

        print("Human complexity:", total_mi)
         # bits --> human responses: 5.647 bits


if __name__ == '__main__':
    feature_len = 512
    settings.see_distractor = False
    settings.see_distractors_pragmatics = True

    settings.with_ctx_representation = False   
    settings.dropout = False
    settings.see_probabilities = True

    settings.eval_someRE = False

    settings.random_init = False
    random_init_dir = "random_init/" if settings.random_init else "anneal/"

    num_distractors = 1
    settings.num_distractors = num_distractors
    
    b_size = 128
    c_dim = 128
    variational = True
   

    settings.num_protos = 3000 
    print("num protos:", settings.num_protos)
    
    c_dirs = os.listdir("src/saved_models/" + str(settings.num_protos) + '_VQ/' + random_init_dir + '/without_ctx/')
    c_dirs = [i for i in c_dirs if "kl_weight" in i]
    c_values = [float(i.split("kl_weight")[1]) for i in c_dirs]
    settings.kl_weights = c_values
    #settings.kl_weights = [0.0001]


    settings.kl_incr = 0.0
    settings.entropy_weight = 0.0
    settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    settings.learned_marginal = False
    settings.max_num_align_data = 1
    settings.distinct_words = False  # FIXME
    with_bbox = False


    t_features_filename = 'src/data/t_features.csv'
    settings.d_features_filename = 'src/data/d_features.csv'
    settings.d_bboxes_filename = 'src/data/d_xyxy.tsv'
    settings.ctx_features_filename = 'src/data/ctx_features.csv'
    manynames = load_cleaned_results(filename="src/data/manynames.tsv")
    someRE = pd.read_csv("src/data/someRE.csv", sep = ";")
    merged_tmp = pd.merge(manynames, someRE, on=['link_vg'])
    excluded_ids = [i for i in merged_tmp['vg_image_id']] 

    vae_model = VAE(512, 32)
    vae_model.load_state_dict(torch.load('src/saved_models/vae0.001.pt'))
    vae_model.to(settings.device)

    settings.embedding_cache = {}
    settings.sample_first = True
    speaker_type = 'vq'  # Options are 'vq', 'cont', or 'onehot'
    settings.seeds = [0]

    glove_data = get_glove_vectors(32)
    run()

