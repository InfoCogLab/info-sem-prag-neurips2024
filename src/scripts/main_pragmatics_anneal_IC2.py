"fill the simplex setting a value for U, and gradually descreasing I while increasing C"

import os
import json

import numpy as np
import shutil
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
   

def train(model, lambda_U, lambda_I, train_data, val_data, viz_data, glove_data, p_notseedist, vae, savepath, logs_dir, comm_dim, fieldname, batch_size=1024, burnin_epochs=500, val_period=200, plot_comms_flag=False, calculate_complexity=False):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epoch = 0
    stored_objectives = []
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

        if epoch % val_period == val_period - 1 and epoch != n_epochs - 1: # if it's a validation epoch
            # we check if converged
            stored_objectives.append(loss.item())
            eval_model(model, vae, comm_dim, train_data, val_data, viz_data, glove_data, num_cand_to_metrics, savepath, epoch, fieldname, p_notseedist, calculate_complexity=calculate_complexity and epoch == n_epochs - 1, plot_comms_flag=plot_comms_flag)

            metrics_path = savepath + str(epoch) + '/val_True_2_metrics'
            if os.path.exists(metrics_path):
                acc_history = PerformanceMetrics.from_file(metrics_path).comm_accs
                recent_acc = acc_history[-look_back:]
                rec_history = PerformanceMetrics.from_file(metrics_path).recons
                recent_rec = rec_history[-look_back:]
                recent_objectives = stored_objectives[-look_back:]

                # if converged
                if len(recent_acc) == look_back and np.var(recent_objectives) < epsilon_convergence:
 
                    converged = True
                    print(np.var(recent_acc))
                    print(np.var(recent_rec))
                    print(np.var(recent_objectives))
                    print("CONVERGED!!!")
                    
                    # delete previous epochs
                    trained_epochs = os.listdir(savepath)
                    for del_epoch in trained_epochs:
                        if del_epoch != str(epoch):
                            shutil.rmtree(savepath + del_epoch)

                    # save model info
                    json_file = logs_dir+"objective" + str(idx_job) +".json"
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

                    # go to following model
                    break

                else: # not converged
                    continue

            else:
                continue
        
        elif epoch == n_epochs-1: # if we are done with the training
            converged = True
            eval_model(model, vae, comm_dim, train_data, val_data, viz_data, glove_data, num_cand_to_metrics, savepath, epoch, fieldname, p_notseedist, calculate_complexity=calculate_complexity and epoch == n_epochs - 1, plot_comms_flag=plot_comms_flag)
            
            # delete previous epochs
            trained_epochs = os.listdir(savepath)
            for del_epoch in trained_epochs:
                if del_epoch != str(epoch):
                    shutil.rmtree(savepath + del_epoch)

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

    print("context:", settings.with_ctx_representation)

    #### TO ANNEAL one of the IC DIAGONALS (we set U fixed, and gradually descrese I and increase C)
    
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

        first_trained = False
        for idx_ in range(1, len(settings.alphas)):
            speaker = VQ(feature_len, c_dim, num_layers=3, num_protos=settings.num_protos, num_simultaneous_tokens=1, variational=variational, num_imgs=num_imgs)
            listener = ListenerPragmaticsCosines(feature_len)
            decoder = Decoder(c_dim, feature_len, num_layers=3, num_imgs=1)
            model = Team(speaker, listener, decoder)
            
            # here we retrieve the info about original model 
            # we keep the ulility value fixed, and gradually descrease I and increase C
            u_set = 0.9799   
            # complete list of utilities to fix in u_set, one by one 
            # [0.9799, 0.9499, 0.8999, 0.7999, 0.6999, 0.5999, 0.4999, 0.3999, 0.2999, 0.1999, 0.1499, 0.1249, 
            # 0.0999, 0.0749, 0.0499, 0.0199, 0.0149, 0.0099, 0.0079, 0.0049, 0.0029, 0.0]
            # either iterate (one long code run) or launch different scripts  
            c_original = 0.0001
            a_original = round(1.0 - (c_original + u_set), 4)

            # info current model
            a = settings.alphas[idx_]
            c = round(1.0 - (u_set + a), 4)
            settings.kl_weight = c
            settings.entropy_weight = 0.01 * settings.kl_weight

            if c < 0:
                print("I am not training this model!")
                pass
            elif c == c_original and a == a_original:
                pass

            else:
                if first_trained == True:

                    # info previous model
                    a_previous = settings.alphas[idx_-1]
                    c_previous = round(1.0 - (u_set + a_previous), 4)
                    print("annealing from... u:", u_set, " a:", a_previous, " c:", c_previous)
                    print("u:", u_set, " a:", a, " c:", c)

                    # get convergence epoch of previous model
                    folder_ctx = "with_ctx/" if settings.with_ctx_representation else "without_ctx/"
                    json_file_path = "src/saved_models/" + str(settings.num_protos) + '_VQ/anneal/' + folder_ctx + 'kl_weight' + str(c_previous) + '/seed' + str(seed) + '/'
                    json_file = json_file_path+"objective0.json"
                    with open(json_file, 'r') as f:
                        existing_params = json.load(f)
                        convergence_epoch = existing_params["utility"+str(u_set)]["inf_weight"+str(a_previous)]['convergence epoch']
                        print(convergence_epoch)
                    
                    model_to_cont = 'src/saved_models/' + str(settings.num_protos) + '_VQ/anneal/' + folder_ctx + 'kl_weight' + str(c_previous) + '/seed' + str(seed) + '/utility'+str(u_set) +'/alpha'+str(a_previous) +'/'+ str(convergence_epoch)
                    
                    model.load_state_dict(torch.load(model_to_cont + '/model.pt'))
                    model.to(settings.device)
                    
                    folder_utility_type = "utility"+str(u_set)+"/"
                    folder_alpha_type = "alpha"+str(a)+"/"
                    save_loc = 'src/saved_models/' + str(settings.num_protos) + "_VQ/anneal/" + folder_ctx + 'kl_weight' + str(c) + '/seed' + str(seed) + '/' + folder_utility_type + folder_alpha_type
                    json_file_path = "src/saved_models/" + str(settings.num_protos) + "_VQ/anneal/" + folder_ctx + 'kl_weight' + str(c) + '/seed' + str(seed) + '/'
                    print("save loc:", save_loc)
                    
                    train(model, u_set, a, train_data, val_data, viz_data, glove_data=glove_data, p_notseedist=0, vae=vae_model, savepath=save_loc, logs_dir=json_file_path, comm_dim=c_dim, fieldname='topname', batch_size=b_size, burnin_epochs=num_burnin, val_period=v_period, plot_comms_flag=do_plot_comms, calculate_complexity=do_calc_complexity)
                    
                else:
                    print("annealing from... u:", u_set, " a:", a_original, " c:", c_original)
                    print("u:", u_set, " a:", a, " c:", c)

                    # get convergence epoch of original model
                    folder_ctx = "with_ctx/" if settings.with_ctx_representation else "without_ctx/"
                    json_file_path = "src/saved_models/" + str(settings.num_protos) + '_VQ/anneal/' + folder_ctx + 'kl_weight' + str(c_original) + '/seed' + str(seed) + '/'
                    json_file = json_file_path+"objective0.json"
                    with open(json_file, 'r') as f:
                        existing_params = json.load(f)
                        convergence_epoch = existing_params["utility"+str(u_set)]["inf_weight"+str(a_original)]['convergence epoch']
                        print(convergence_epoch)
                    
                    model_to_cont = 'src/saved_models/' + str(settings.num_protos) + '_VQ/anneal/' + folder_ctx + 'kl_weight' + str(c_original) + '/seed' + str(seed) + '/utility'+str(u_set) +'/alpha'+str(a_original) +'/'+ str(convergence_epoch)
                    
                    model.load_state_dict(torch.load(model_to_cont + '/model.pt'))
                    model.to(settings.device)
                    
                    folder_utility_type = "utility"+str(u_set)+"/"
                    folder_alpha_type = "alpha"+str(a)+"/"
                    save_loc = 'src/saved_models/' + str(settings.num_protos) + "_VQ/anneal/" + folder_ctx + 'kl_weight' + str(c) + '/seed' + str(seed) + '/' + folder_utility_type + folder_alpha_type
                    json_file_path = "src/saved_models/" + str(settings.num_protos) + "_VQ/anneal/" + folder_ctx + 'kl_weight' + str(c) + '/seed' + str(seed) + '/'
                    print("save loc:", save_loc)
                    
                    train(model, u_set, a, train_data, val_data, viz_data, glove_data=glove_data, p_notseedist=0, vae=vae_model, savepath=save_loc, logs_dir=json_file_path, comm_dim=c_dim, fieldname='topname', batch_size=b_size, burnin_epochs=num_burnin, val_period=v_period, plot_comms_flag=do_plot_comms, calculate_complexity=do_calc_complexity)
                    first_trained = True

            


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
    v_period = 200  
    num_burnin = 500
    b_size = 128
    c_dim = 128
    variational = True
    settings.num_protos = 3000 

    do_calc_complexity = False
    do_plot_comms = False

    look_back = 5
    epsilon_convergence = 0.0001
    n_epochs = 10000
    
    settings.alphas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.985, 0.99, 0.992, 0.995, 0.997, 0.998, 1.0]
    settings.alphas.reverse()

    settings.kl_incr = 0.0 # complexity increase (ignore for this annealing path)
    
    settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    settings.learned_marginal = False
    settings.max_num_align_data = 1
    settings.distinct_words = False  
    
    
    t_features_filename = 'src/data/t_features.csv'
    settings.d_features_filename = 'src/data/d_features.csv'
    settings.d_bboxes_filename = 'src/data/d_xyxy.tsv'
    settings.ctx_features_filename = 'src/data/ctx_features.csv'
    manynames = load_cleaned_results(filename="src/data/manynames.tsv")
    someRE = pd.read_csv("src/data/someRE.csv", sep = ";")
    merged_tmp = pd.merge(manynames, someRE, on=['link_vg'])
    excluded_ids = [str(i) for i in merged_tmp['vg_image_id']] 
    print("excluded ids:", len(excluded_ids))

    vae_model = VAE(512, 32)
    vae_model.load_state_dict(torch.load('src/saved_models/vae0.001.pt'))
    vae_model.to(settings.device)
    settings.embedding_cache = {}
    settings.sample_first = True
    speaker_type = 'vq'  # Options are 'vq', 'cont', or 'onehot'
    
    seeds = [0]
    idx_job = 0

    glove_data = get_glove_vectors(32)
    run()

