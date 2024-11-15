"""
Compute NID in semantics setting
"""

import os
import ast
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
import random

import src.settings as settings
from src.data_utils.helper_fns import gen_batch, get_glove_embedding, get_unique_labels, get_entry_for_labels, get_unique_by_field
from src.data_utils.read_data import get_feature_data
from src.data_utils.read_data import load_cleaned_results

from src.models.decoder import Decoder
from src.models.listener_pragmatics import ListenerPragmaticsCosines
from src.models.team import Team
from src.models.vae import VAE
from src.models.vq import VQ

from src.utils.NID import get_NID

from src.data_utils.read_data import get_glove_vectors






def run():
    if settings.see_distractors_pragmatics:
        num_imgs = 3 if settings.with_ctx_representation else 2
    else:
        num_imgs = 1 if not settings.see_distractor else (num_distractors + 1)

    data = get_feature_data(t_features_filename, excluded_ids=excluded_ids)
    data = data.sample(frac=1, random_state=46) # Shuffle the data.

    # TO TEST ON ALL DATA

    print("Len data:", len(data))
    print(len(set(data['topname'])))

    # re-swap target and distractor to judge with the correct human name
    mask = data["swapped_t_d"] == 1
    data.loc[mask, ["t_features", "d_features"]] = data.loc[mask, ["d_features", "t_features"]].values
    check_data = data

    print("context:", settings.with_ctx_representation)

    speaker = VQ(feature_len, c_dim, num_layers=3, num_protos=settings.num_protos, num_simultaneous_tokens=1, variational=variational, num_imgs=num_imgs)
    listener = ListenerPragmaticsCosines(feature_len)
    decoder = Decoder(c_dim, feature_len, num_layers=3, num_imgs=1)
    model = Team(speaker, listener, decoder)
    model.to(settings.device)

   
    print("------------------")
    
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
                folder_ctx = "with_ctx/" if settings.with_ctx_representation else "without_ctx/"
                folder_utility = "utility"+str(u)+"/"
                folder_alpha = "alpha"+str(a)+"/"


                # get convergence epoch for that model
                json_file_path = "src/saved_models/" + str(settings.num_protos) + '_VQ/' + random_init_dir + folder_ctx + 'kl_weight' + str(c) + '/seed' + str(seed) + '/'
                json_file = json_file_path+"objective0.json"
                print(json_file)
                with open(json_file, 'r') as f:
                    existing_params = json.load(f)
                convergence_epoch = existing_params["utility"+str(u)]["inf_weight"+str(a)]['convergence epoch']

                # load model
                folder_ctx = "with_ctx/" if settings.with_ctx_representation else "without_ctx/"
                folder_utility = "utility"+str(u)+"/"
                folder_alpha = "alpha"+str(a)+"/"
                model_to_eval_path = 'src/saved_models/' + str(settings.num_protos) + '_VQ/' + random_init_dir + folder_ctx + 'kl_weight' + str(c) + '/seed' + str(seed) + '/' + folder_utility + folder_alpha + str(convergence_epoch)
                model.load_state_dict(torch.load(model_to_eval_path + '/model.pt'))
                model.to(settings.device)
                
            except Exception as error:
                print(error)
                pass

            model_NID, w_count_top, h_count_top, model_above_1, humans_above_1, H_m, H_h = get_NID(model, check_data, len(check_data), "topname", glove_data=glove_data, vae=vae)
            print("model NID:", model_NID, "I:", a, "U:", u, "C:", c)
            
            to_add = pd.DataFrame([[u, a, c, model_NID, w_count_top, h_count_top, model_above_1, humans_above_1, H_m, H_h]])
            to_add.columns = ["Utility", "Alpha", "Complexity", "NID", "model_topnames", "human_topnames", "model_above1", "humans_above1", "H_m", "H_h"]

            save_path = "Plots/" + str(settings.num_protos) + "_VQ/" + "simplex/seed" + str(seed) + "/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_file_name = f'NID_entropy_count_seed{seed}.csv'
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
    if settings.see_distractors_pragmatics:
        settings.see_distractor = False 
    
    settings.with_ctx_representation = False
    settings.dropout = False
    settings.see_probabilities = True
    
    settings.eval_someRE = False
    
    num_distractors = 1
    settings.num_distractors = num_distractors
    c_dim = 128
    variational = True
    settings.num_protos = 3000 

    settings.random_init = True
    random_init_dir = "random_init/" if settings.random_init else "anneal/"
    
    c_dirs = os.listdir("src/saved_models/" + str(settings.num_protos) + '_VQ/' + random_init_dir + '/without_ctx/')
    c_dirs = [i for i in c_dirs if "kl_weight" in i]
    c_values = [float(i.split("kl_weight")[1]) for i in c_dirs]
    settings.kl_weights = c_values
    #settings.kl_weights = [0.0001]

    settings.entropy_weight = 0.0 
    settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    settings.learned_marginal = False
    settings.max_num_align_data = 1
    settings.distinct_words = False 
    with_bbox = False

    t_features_filename = 'src/data/t_features.csv'
    settings.d_features_filename = 'src/data/d_features.csv'
    settings.d_bboxes_filename = 'src/data/d_xyxy.tsv'
    settings.ctx_features_filename = 'src/data/ctx_features.csv'
    manynames = load_cleaned_results(filename="src/data/manynames.tsv")
    someRE = pd.read_csv("src/data/someRE.csv", sep = ";")
    merged_tmp = pd.merge(manynames, someRE, on=['link_vg'])
    excluded_ids = [str(i) for i in merged_tmp['vg_image_id']] 
    print("excluded ids:", len(excluded_ids))

    vae = VAE(512, 32)
    vae.load_state_dict(torch.load('src/saved_models/vae0.001.pt'))
    vae.to(settings.device)
    settings.embedding_cache = {}
    settings.sample_first = True
    speaker_type = 'vq'  # Options are 'vq', 'cont', or 'onehot'
    
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    glove_data = get_glove_vectors(32)
    run()

