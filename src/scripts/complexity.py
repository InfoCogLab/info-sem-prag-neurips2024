"""
Compute complexity in semantics setting
"""

import os
import json

import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn

import src.settings as settings
from src.data_utils.helper_fns import gen_batch, get_glove_embedding, get_unique_labels, get_entry_for_labels, get_unique_by_field, get_rand_entries
from src.data_utils.read_data import get_feature_data
from src.data_utils.read_data import load_cleaned_results

from src.models.decoder import Decoder
from src.models.listener_pragmatics import ListenerPragmaticsCosines
from src.models.team import Team
from src.models.vae import VAE
from src.models.vq import VQ

from src.data_utils.read_data import get_glove_vectors




def get_complexity(model, dataset, glove_data=None, vae=None):
   
    likelihoods_matrix = []
    
    for targ_idx in list(dataset.index.values):
        speaker_obs, _, _, _ = gen_batch(dataset, 1, "topname", p_notseedist=1, glove_data=glove_data, vae=vae, preset_targ_idx=targ_idx)
        
        if speaker_obs != None: # i.e. we have the glove embeds    
    
            # we repeat the input to get a sense of the topname
            speaker_obs = speaker_obs.repeat(100, 1, 1)
            
            with torch.no_grad(): 
                likelihood = model.speaker.get_token_dist(speaker_obs)
                likelihoods_matrix.append(likelihood)

                
        else:
            pass
    
    likelihoods_matrix = np.array(likelihoods_matrix)

    epsilon = 1e-10

    # prob of objects
    p_o = np.full(likelihoods_matrix.shape[0], 1/likelihoods_matrix.shape[0])  # each object is equally likely
    
    # calculate marginal probability of each word
    p_w = likelihoods_matrix.T @ p_o    

    joint_probs = likelihoods_matrix * p_o[:, np.newaxis]

    mi = joint_probs * np.log2((joint_probs + epsilon) / (p_o[:, np.newaxis] * p_w + epsilon))
    total_mi = np.nansum(mi)  # summing all the MI 

    return total_mi


    



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

        mask = data["swapped_t_d"] == 1
        data.loc[mask, ["t_features", "d_features"]] = data.loc[mask, ["d_features", "t_features"]].values

        speaker = VQ(feature_len, c_dim, num_layers=3, num_protos=settings.num_protos, num_simultaneous_tokens=1, variational=variational, num_imgs=num_imgs)
        listener = ListenerPragmaticsCosines(feature_len)
        decoder = Decoder(c_dim, feature_len, num_layers=3, num_imgs=1)
        model = Team(speaker, listener, decoder)

        folder_ctx = "with_ctx/" if settings.with_ctx_representation else "without_ctx/"
        
        for c in settings.kl_weights:
            settings.kl_weight = c
                
            json_file_path = "src/saved_models/" + str(settings.num_protos) + '_VQ/' + random_init_dir + folder_ctx + 'kl_weight' + str(c) + '/seed' + str(seed) + '/'
            
            for folder_utility in [i for i in os.listdir(json_file_path) if "utility" in i]:

                u = (folder_utility.split("utility")[1].split("/")[0])
                folder_alpha = os.listdir(json_file_path + folder_utility)[0]
                a = (folder_alpha.split("alpha")[1].split("/")[0])
            
                print("u:", u, "i:", a, "c:", c)

                folder_utility = folder_utility + '/'
                folder_alpha = folder_alpha + '/'
                
                try:
                    # get convergence epoch for that model
                    json_file = json_file_path+"objective0.json"
                    print(json_file)
                    with open(json_file, 'r') as f: 
                        existing_params = json.load(f)
                    convergence_epochs = [existing_params["utility"+str(u)]["inf_weight"+str(a)]['convergence epoch']]
                    
                    for convergence_epoch in convergence_epochs:
                        print("check epoch:", convergence_epoch)

                        # load model
                        model_to_eval_path = 'src/saved_models/' + str(settings.num_protos) + '_VQ/' + random_init_dir + folder_ctx + 'kl_weight'+str(c) + '/seed' + str(seed) + '/' + folder_utility + folder_alpha + str(convergence_epoch) + '/'
                        model.load_state_dict(torch.load(model_to_eval_path + '/model.pt'))
                        model.to(settings.device)
                
                except Exception as error:
                    print("model not found", error)

                complexity = get_complexity(model, data, glove_data, vae_model)

                to_add = pd.DataFrame([[u, a, c, complexity]])
                to_add.columns = ["Utility", "Alpha", "Complexity", "complexity_test"]
   
                save_path = "Plots/" + str(settings.num_protos) + "_VQ/" + "simplex/seed" + str(seed) + "/"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_file_name = f'complexity_seed{seed}.csv'
                if os.path.exists(save_path + save_file_name):
                    df = pd.read_csv(save_path + save_file_name, index_col=0)
                    df_new = pd.concat([df, to_add])
                    df_new.to_csv(save_path + save_file_name)
                else:
                    to_add.to_csv(save_path + save_file_name)


if __name__ == '__main__':
    feature_len = 512
    settings.see_distractor = False
    settings.see_distractors_pragmatics = True

    settings.with_ctx_representation = False   
    settings.dropout = False
    settings.see_probabilities = True

    settings.eval_someRE = False

    settings.random_init = True
    random_init_dir = "random_init/" if settings.random_init else "anneal/"

    num_distractors = 1
    settings.num_distractors = num_distractors
    
    b_size = 128
    c_dim = 128
    variational = True

    settings.num_protos = 3000  
    print("num protos:", settings.num_protos)
    
    # c_dirs = os.listdir("src/saved_models/" + str(settings.num_protos) + '_VQd/' + random_init_dir + '/without_ctx/')
    # c_dirs = [i for i in c_dirs if "kl_weight" in i]
    # c_values = [float(i.split("kl_weight")[1]) for i in c_dirs]
    # settings.kl_weights = c_values
    settings.kl_weights = [0.0001]


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

