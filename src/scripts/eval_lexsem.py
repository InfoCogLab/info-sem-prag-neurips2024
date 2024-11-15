"""
Run model evaluation in semantics setting
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
from src.data_utils.helper_fns import gen_batch, get_glove_embedding, get_unique_labels, get_entry_for_labels, get_unique_by_field, get_rand_entries
from src.data_utils.read_data import get_feature_data
from src.data_utils.read_data import load_cleaned_results

from src.models.decoder import Decoder
from src.models.listener_pragmatics import ListenerPragmaticsCosines
from src.models.team import Team
from src.models.vae import VAE
from src.models.vq import VQ

from src.utils.mine_pragmatics import get_info_lexsem
from src.utils.plotting import plot_metrics, plot_naming, plot_scatter
from src.utils.performance_metrics import PerformanceMetrics

from src.data_utils.read_data import get_glove_vectors



def evaluate_lexsem(model, dataset, batch_size, vae, glove_data, fieldname, num_dist=None):
    model.eval()
    num_test_batches = 10
    num_correct = 0
    total_recons_loss = 0
    num_total = 0
    for _ in range(num_test_batches):
        with torch.no_grad():
            # p_notseedist=1 regulates the semantics setting
            speaker_obs, listener_obs, labels, _ = gen_batch(dataset, batch_size, fieldname, p_notseedist=1, vae=vae, glove_data=glove_data, see_distractors=settings.see_distractor, num_dist=num_dist)
            outputs, _, _, recons = model(speaker_obs, listener_obs)
            
        pred_labels = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        num_correct += np.sum(pred_labels == labels.cpu().numpy())
        num_total += pred_labels.size
        total_recons_loss += torch.mean(((speaker_obs[:, 0:1, :] - recons) ** 2)).item()

    acc = num_correct / num_total
    total_recons_loss = total_recons_loss / num_test_batches
 
    print("Lex sem")
    print("Evaluation on test set accuracy", acc)
    print("Evaluation on test set recons loss", total_recons_loss)
    
    return acc, total_recons_loss


# used in Tucker et al., 2023 only
def plot_comms(model, dataset, basepath):
    num_tests = 1000  # Generate lots of samples for the same input because it's not deterministic.
    labels = []
    if settings.with_ctx_representation:
        for f, f_d, f_ctx in zip(dataset['t_features'], dataset['d_features'], dataset['ctx_features']):
            speaker_obs = np.expand_dims(np.vstack([f] + [f_d] + [f_ctx]), axis=0)
            speaker_obs = torch.Tensor(np.vstack(speaker_obs).astype(np.float)).to(settings.device)   
            speaker_obs = speaker_obs.unsqueeze(0)
            speaker_obs = speaker_obs.repeat(num_tests, 1, 1)
            speaker_obs = speaker_obs.view(3000, -1)

            likelihoods = model.speaker.get_token_dist(speaker_obs)
            top_comm_idx = np.argmax(likelihoods)
            top_likelihood = likelihoods[top_comm_idx]
            label = top_comm_idx if top_likelihood > 0.4 else -1
            labels.append(label)
    features = np.vstack(dataset)
    label_np = np.reshape(np.array(labels), (-1, 1))
    all_np = np.hstack([label_np, features])
    regrouped_data = []
    plot_labels = []
    plot_mean = False
    for c in np.unique(labels):
        ix = np.where(all_np[:, 0] == c)
        matching_features = np.vstack(all_np[ix, 1:])
        averaged = np.mean(matching_features, axis=0, keepdims=True)
        plot_features = averaged if plot_mean else matching_features
        regrouped_data.append(plot_features)
        plot_labels.append(c)
    plot_naming(regrouped_data, viz_method='mds', labels=plot_labels, savepath=basepath + 'training_mds')
    plot_naming(regrouped_data, viz_method='tsne', labels=plot_labels, savepath=basepath + 'training_tsne')




def eval_model_lexsem(model, vae, comm_dim, data, glove_data, num_cand_to_metrics, save_eval_path,
               fieldname, calculate_complexity=False, plot_comms_flag=False, alignment_dataset=None, save_model=True):
        
    if not os.path.exists(save_eval_path + 'lexsem/'):
        os.makedirs(save_eval_path + 'lexsem/')
    if calculate_complexity:
        test_complexity = get_info_lexsem(model, data, targ_dim=comm_dim, p_notseedist=1, glove_data=glove_data, num_epochs=200)
        print("Test complexity", test_complexity)
    else:
        test_complexity = None
        val_complexity = None
           
    eval_batch_size = 256
    complexities = [test_complexity]
    for set_distinction in [True]:
        for feature_idx, data in enumerate([data]):
            for num_candidates in num_cand_to_metrics.get(set_distinction).keys():
                settings.distinct_words = set_distinction
                acc, recons = evaluate_lexsem(model, data, eval_batch_size, vae, glove_data, fieldname=fieldname, num_dist=num_candidates - 1)
            relevant_metrics = num_cand_to_metrics.get(set_distinction).get(num_candidates)[feature_idx]
            relevant_metrics.add_data("eval_epoch", complexities[feature_idx], -1 * recons, acc, settings.kl_weight)

    # Plot some of the metrics for online visualization
    comm_accs = []
    labels = []
    epoch_idxs = None
    plot_metric_data = num_cand_to_metrics.get(True)
    for feature_idx, label in enumerate(['test']):
        for num_candidates in sorted(plot_metric_data.keys()):
            comm_accs.append(plot_metric_data.get(num_candidates)[feature_idx].comm_accs)
            labels.append(" ".join([label, str(num_candidates), "utility"]))
            if epoch_idxs is None:
                epoch_idxs = plot_metric_data.get(num_candidates)[feature_idx].epoch_idxs
    plot_metrics(comm_accs, labels, epoch_idxs, save_eval_path + 'lexsem/')

    # Save the model and metrics to files.
    for feature_idx, label in enumerate(['test']):
        for set_distinction in num_cand_to_metrics.keys():
            for num_candidates in sorted(num_cand_to_metrics.get(set_distinction).keys()):
                metric = num_cand_to_metrics.get(set_distinction).get(num_candidates)[feature_idx]
                metric.to_file(save_eval_path + 'lexsem/' + "_".join([label, str(set_distinction), str(num_candidates), "metrics"]))
    if not save_model:
        return
    torch.save(model.state_dict(), save_eval_path + 'lexsem/model.pt')



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

        train_data, test_data, val_data = np.split(data.sample(frac=1, random_state=46), 
                                        [int(.7*len(data)), int(.9*len(data))])
        train_data, test_data, val_data = train_data.reset_index(), test_data.reset_index(), val_data.reset_index()
        train_data, test_data, val_data = train_data.sample(frac=1, random_state=46).reset_index(), test_data.sample(frac=1, random_state=46).reset_index(), val_data.sample(frac=1, random_state=46).reset_index()

        print("Len test set:", len(test_data))
        print("Len val set:", len(val_data))

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
                        model_to_eval_path = 'src/saved_models/' + str(settings.num_protos) + '_VQ/' + random_init_dir + folder_ctx + 'kl_weight'+str(c) + '/seed' + str(seed) + '/' + folder_utility + folder_alpha + str(convergence_epoch)
                        save_eval_path = model_to_eval_path + '/evaluation/'
                        model.load_state_dict(torch.load(model_to_eval_path + '/model.pt'))
                        model.to(settings.device)
                
                        print("Lexsem task") 
                        num_cand_to_metrics = {True: {2: []}}
                        for empty_list in num_cand_to_metrics.get(True).values():
                            empty_list.extend([PerformanceMetrics()])
                        # we evaluate lexsem on all the data
                        eval_model_lexsem(model, vae_model, c_dim, data, glove_data, num_cand_to_metrics, save_eval_path, fieldname='topname', calculate_complexity=do_calc_complexity, plot_comms_flag=do_plot_comms)
                
                except Exception as error:
                    print("model not found", error)
                
                



if __name__ == '__main__':
    feature_len = 512
    settings.see_distractor = False
    settings.see_distractors_pragmatics = True

    settings.with_ctx_representation = False   
    settings.dropout = False
    settings.see_probabilities = True

    settings.eval_someRE = False

    settings.random_init = False  # False when evaluating annealed models, else True
    random_init_dir = "random_init/" if settings.random_init else "anneal/"

    num_distractors = 1
    settings.num_distractors = num_distractors
    
    b_size = 128
    c_dim = 128
    variational = True
    # Measuring complexity takes a lot of time. For debugging other features, set to false.
    do_calc_complexity = False
    do_plot_comms = False

    settings.num_protos = 3000 # 442 is the number of topnames in MN 
    print("num protos:", settings.num_protos)
    
    c_dirs = os.listdir("src/saved_models/" + str(settings.num_protos) + '_VQ/' + random_init_dir + '/without_ctx/')
    c_dirs = [i for i in c_dirs if "kl_weight" in i]
    c_values = [float(i.split("kl_weight")[1]) for i in c_dirs]
    settings.kl_weights = c_values
    #settings.kl_weights = [0.0001]


    settings.kl_incr = 0.0 # complexity increase (ignore for this annealing path)
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

