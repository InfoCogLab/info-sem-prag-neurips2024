"""
Train model with lambda_U=1.0.
"""

import os
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
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

from src.utils.plotting import plot_metrics, plot_naming, plot_scatter
from src.utils.performance_metrics import PerformanceMetrics

from src.data_utils.read_data import get_glove_vectors




def evaluate(model, dataset, batch_size, p_notseedist, vae, glove_data, fieldname, num_dist=None):
    model.eval()
    num_test_batches = 10
    num_correct = 0
    total_recons_loss = 0
    num_total = 0
    for _ in range(num_test_batches):
        with torch.no_grad():
            speaker_obs, listener_obs, labels, _ = gen_batch(dataset, batch_size, fieldname, p_notseedist, vae=vae, glove_data=glove_data, see_distractors=settings.see_distractor, num_dist=num_dist) 
            outputs, _, _, recons = model(speaker_obs, listener_obs)
        
        pred_labels = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        num_correct += np.sum(pred_labels == labels.cpu().numpy())
        num_total += pred_labels.size
        total_recons_loss += torch.mean(((speaker_obs[:, 0:1, :] - recons) ** 2)).item()

    acc = num_correct / num_total
    total_recons_loss = total_recons_loss / num_test_batches
    print("Evaluation accuracy", acc)
    print("Evaluation recons loss", total_recons_loss)
    
    return acc, total_recons_loss
    



def eval_model(model, vae, comm_dim, train_data, val_data, viz_data, glove_data, num_cand_to_metrics, savepath,
               epoch, fieldname, p_notseedist, calculate_complexity=False, plot_comms_flag=False, alignment_dataset=None, save_model=True):
    # Create a directory to save information, models, etc.
    basepath = savepath + str(epoch) + '/'
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    # Calculate efficiency values like complexity and informativeness.
    # Can estimate complexity by sampling inputs and measuring communication probabilities.
    # get_probs(model.speaker, train_data)
    # Or we can use MINE to estimate complexity and informativeness.
    if calculate_complexity:
        print("Need for conditional mutual information: not implemented!")
        pass 
    else:
        train_complexity = None
        val_complexity = None

    eval_batch_size = 256
    val_is_train = len(train_data) == len(val_data)  
    if val_is_train:
        print("WARNING: ASSUMING VALIDATION AND TRAIN ARE SAME")


    complexities = [train_complexity, val_complexity]
    for set_distinction in [True]:
        for feature_idx, data in enumerate([train_data, val_data]):
            if feature_idx == 1 and val_is_train:
                pass
            for num_candidates in num_cand_to_metrics.get(set_distinction).keys():
                if feature_idx == 1 and val_is_train:
                    pass  
                else:
                    settings.distinct_words = set_distinction
                    acc, recons = evaluate(model, data, eval_batch_size, p_notseedist, vae, glove_data, fieldname=fieldname, num_dist=num_candidates - 1)
                    
                relevant_metrics = num_cand_to_metrics.get(set_distinction).get(num_candidates)[feature_idx]
                relevant_metrics.add_data(epoch, complexities[feature_idx], -1 * recons, acc, settings.kl_weight)

    # Plot some of the metrics for online visualization
    comm_accs = []
    recons = []
    labels_acc = []
    labels_rec = []
    epoch_idxs = None
    plot_metric_data = num_cand_to_metrics.get(True)
    for feature_idx, label in enumerate(['train', 'val']):
        for num_candidates in sorted(plot_metric_data.keys()):
            recons.append(plot_metric_data.get(num_candidates)[feature_idx].recons)
            comm_accs.append(plot_metric_data.get(num_candidates)[feature_idx].comm_accs)
            labels_rec.append(" ".join([label, str(num_candidates), "recons"]))
            labels_acc.append(" ".join([label, str(num_candidates), "utility"]))
            if epoch_idxs is None:
                epoch_idxs = plot_metric_data.get(num_candidates)[feature_idx].epoch_idxs
    plot_metrics(comm_accs, labels_acc, epoch_idxs, basepath=basepath + 'utility_')
    plot_metrics(recons, labels_rec, epoch_idxs, basepath=basepath + 'recons_')

    # Save the model and metrics to files.
    for feature_idx, label in enumerate(['train', 'val']):
        for set_distinction in num_cand_to_metrics.keys():
            for num_candidates in sorted(num_cand_to_metrics.get(set_distinction).keys()):
                metric = num_cand_to_metrics.get(set_distinction).get(num_candidates)[feature_idx]
                metric.to_file(basepath + "_".join([label, str(set_distinction), str(num_candidates), "metrics"]))
    if not save_model:
        return
    torch.save(model.state_dict(), basepath + 'model.pt')
    torch.save(model, basepath + 'model_obj.pt')



def train(model, lambda_U, lambda_I, train_data, val_data, viz_data, glove_data, p_notseedist, vae, savepath, logs_dir, comm_dim, fieldname, batch_size=128, burnin_epochs=500, val_period=200, plot_comms_flag=False, calculate_complexity=False):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 0.0005 better for decoder with pre-trained VAE; this is better for random init
       
    epoch = 0
    converged = False
    running_acc = 0
    running_mse = 0
    num_cand_to_metrics = {True: {2: []}}
    for set_distinct in [True]:
        for empty_list in num_cand_to_metrics.get(set_distinct).values():
            empty_list.extend([PerformanceMetrics(), PerformanceMetrics()])  # Train metrics, validation metrics        
    while converged == False:
        epoch += 1

        speaker_obs, listener_obs, labels, _ = gen_batch(train_data, batch_size, fieldname, p_notseedist, vae=vae, glove_data=glove_data, see_distractors=settings.see_distractor)
        
        optimizer.zero_grad()        
        outputs, speaker_loss, info, recons = model(speaker_obs, listener_obs)
            
        error = criterion(outputs, labels)
        utility_loss = lambda_U * error 
        loss = lambda_U * error

        if len(speaker_obs.shape) == 2:
            speaker_obs = torch.unsqueeze(speaker_obs, 1)
        
        # target reconstruction
        recons_loss = torch.mean(((speaker_obs[:, 0:1, :] - recons) ** 2))

        loss += lambda_I * recons_loss
        loss += speaker_loss

        loss.backward()
        optimizer.step()

        # Metrics
        pred_labels = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        num_correct = np.sum(pred_labels == labels.cpu().numpy())
        num_total = pred_labels.size
        running_acc = running_acc * 0.95 + 0.05 * num_correct / num_total
        running_mse = running_mse * 0.95 + 0.05 * recons_loss.item()

        if epoch % val_period == val_period - 1: # if it's a validation epoch
            eval_model(model, vae, comm_dim, train_data, val_data, viz_data, glove_data, num_cand_to_metrics, savepath, epoch, fieldname, p_notseedist, calculate_complexity=calculate_complexity and epoch == n_epochs - 1, plot_comms_flag=plot_comms_flag)
            
        if epoch == n_epochs-1: # if we are done with the training
            converged = True
            eval_model(model, vae, comm_dim, train_data, val_data, viz_data, glove_data, num_cand_to_metrics, savepath, epoch, fieldname, p_notseedist, calculate_complexity=calculate_complexity and epoch == n_epochs - 1, plot_comms_flag=plot_comms_flag)
            # save model info
            json_file = logs_dir+"objective" + str(idx_job) + ".json"
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    existing_params = json.load(f)
            else:
                existing_params = {}

            dic = {}
            dic["inf_weight"+str(lambda_I)] = {"objective": loss.item(),
                                                 "speaker loss": speaker_loss.item(), # weighted already
                                                 "recons loss": lambda_I * recons_loss.item(),
                                                 "unw recons loss": recons_loss.item(),
                                                 "utility loss": utility_loss.item(), # weighted already
                                                 "unw utility loss": error.item(),
                                                 "convergence epoch": epoch} 

            if "utility"+str(lambda_U) in existing_params.keys():
                existing_params["utility"+str(lambda_U)].update(dic)
            else:
                existing_params["utility"+str(lambda_U)] = dic

            with open(json_file, 'w') as f:
                json.dump(existing_params, f, indent=4)
                        
        else: # not validation epoch and not done with the training
            continue 




def run():
    if settings.see_distractors_pragmatics:
        num_imgs = 3 if settings.with_ctx_representation else 2
    else:
        num_imgs = 1 if not settings.see_distractor else (num_distractors + 1)

    for seed in seeds:
        print("seed:", seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        data = get_feature_data(t_features_filename, excluded_ids=excluded_ids)
        data = data.sample(frac=1, random_state=46) # Shuffle the data.
        train_data, test_data, val_data = np.split(data.sample(frac=1, random_state=46), 
                                            [int(.7*len(data)), int(.9*len(data))])
        train_data, test_data, val_data = train_data.reset_index(), test_data.reset_index(), val_data.reset_index()
        train_data, test_data, val_data = train_data.sample(frac=1, random_state=46).reset_index(), test_data.sample(frac=1, random_state=46).reset_index(), val_data.sample(frac=1, random_state=46).reset_index() 
        print("Len train set:",len(train_data), "Len val set:", len(val_data), "Len test set:", len(test_data))
        viz_data = train_data  # For debugging, it's faster to just reuse datasets
        
        print("context:", settings.with_ctx_representation)
        
        for u in settings.utilities:
                
            # directories
            folder_ctx = "with_ctx/" if settings.with_ctx_representation else "without_ctx/"
            folder_utility_type = "utility"+str(u)+"/"
            save_loc = 'src/saved_models/' + str(settings.num_protos) + "_VQ/random_init/"+ folder_ctx + 'kl_weight' + str(settings.kl_weight) + '/seed' + str(seed) + '/' + folder_utility_type
            json_file_path = "src/saved_models/" + str(settings.num_protos) + "_VQ/random_init/" + folder_ctx + 'kl_weight' + str(settings.kl_weight) + '/seed' + str(seed) + '/'
        
            trained_weights_dir = json_file_path + "done_weights.json"

            idx = 0
            while idx != len(settings.alphas):
                alpha = settings.alphas[idx] # we take the new alpha
                savepath_new = save_loc + "alpha"+str(alpha) + '/'

                print("complexity:", settings.kl_weight) # complexity weight (normalized)
                informativeness_weight = alpha
                print("informativeness:", informativeness_weight)
                utility_weight = u
                print("utility:", utility_weight)
                weights = [settings.kl_weight, informativeness_weight, utility_weight]

                try:
                    with open(trained_weights_dir, "r") as file:
                        done = json.load(file)
                        done_triplets = list(done.values())
                except FileNotFoundError:
                    done_triplets = []
                    done = {}
                    os.makedirs(json_file_path, exist_ok=True)
            
                if str(weights) in done_triplets: # this triplet has been trained already
                    idx += 1 # new alpha
                    print("have already:", str(weights))
                else: # if we have to train this triplet
                    key = str(weights)
                    done[key] = str(weights)
                    print("new", str(weights))
                    with open(trained_weights_dir, 'w') as f:
                        json.dump(done, f, indent=4)
            
                    speaker = VQ(feature_len, c_dim, num_layers=3, num_protos=settings.num_protos, num_simultaneous_tokens=1, variational=variational, num_imgs=num_imgs)
                    listener = ListenerPragmaticsCosines(feature_len)
                    decoder = Decoder(c_dim, feature_len, num_layers=3, num_imgs=1)
                    model = Team(speaker, listener, decoder)
                    model.to(settings.device)
                    
                    train(model, utility_weight, informativeness_weight, train_data, val_data, viz_data, glove_data=glove_data, p_notseedist=0, vae=vae_model, savepath=savepath_new, logs_dir=json_file_path, comm_dim=c_dim, fieldname='topname', batch_size=b_size, val_period=v_period, plot_comms_flag=do_plot_comms, calculate_complexity=do_calc_complexity)
            

                    idx += 1 # new alpha, if applicable



if __name__ == '__main__':
    feature_len = 512
    settings.see_distractor = False # training as in Tucker et al., 2023
    settings.see_distractors_pragmatics = True # pragmatics setup (not Tucker's setting) 
    if settings.see_distractors_pragmatics:
        settings.see_distractor = False 
    
    settings.with_ctx_representation = False 
    settings.dropout = False # allows for playing with speaker's probability of seeing the distractor
    settings.see_probabilities = True # sets pragmatics vs semantics setting in a binary fashion: either the speaker sees the distractor or not
    
    settings.eval_someRE = False # if we want to evaluate on the ManyNames subset from Mädebach et al., 2022
    
    num_distractors = 1
    settings.num_distractors = num_distractors
    v_period = 1000 # How often to test on the validation set and calculate various info metrics.
    b_size = 128
    c_dim = 128
    variational = True
    settings.num_protos = 3000 # VQ prototypes (for reference, 442 is the number of topnames in the whole MN)
    
    # Measuring complexity takes a lot of time. For debugging other features, set to false.
    do_calc_complexity = False
    do_plot_comms = False

    # Parameters of our random initialization
    settings.alphas = [0.0]
    settings.utilities = [0.9999] 
    settings.kl_weight = 0.0001

    idx_job = 0
    n_epochs = 20000 
   
    settings.kl_incr = 0.0 # complexity increase (we can ignore this in our annealing path)
    settings.entropy_weight = 0.01 * settings.kl_weight
    settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    settings.learned_marginal = False
    settings.max_num_align_data = 1
    settings.distinct_words = False  
    with_bbox = False
    
    t_features_filename = 'src/data/t_features.csv'
    settings.d_features_filename = 'src/data/d_features.csv'
    settings.d_bboxes_filename = 'src/data/d_xyxy.tsv'
    settings.ctx_features_filename = 'src/data/ctx_features.csv'

    # We load ManyNames excluding images from Mädebach et al., 2022 from training: they can be useful at test time for evaluating pragmatics on unseen data
    manynames = load_cleaned_results(filename="src/data/manynames.tsv")
    someRE = pd.read_csv("src/data/someRE.csv", sep = ";")
    merged_tmp = pd.merge(manynames, someRE, on=['link_vg'])
    excluded_ids = [str(i) for i in merged_tmp['vg_image_id']]
    print("excluded: ", len(excluded_ids)) 
            
    vae_model = VAE(512, 32)
    vae_model.load_state_dict(torch.load('src/saved_models/vae0.001.pt'))
    vae_model.to(settings.device)
    settings.embedding_cache = {}
    settings.sample_first = True
    speaker_type = 'vq'  # Options are 'vq', 'cont', or 'onehot'
    
    seeds = [0]
    
    glove_data = get_glove_vectors(32)
    run()

