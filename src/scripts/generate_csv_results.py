"""
Group results of evaluations in a csv
"""

import os
import pandas as pd
import json
import numpy as np
import ternary
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from src.utils.performance_metrics import PerformanceMetrics
import src.settings as settings


   
def generate_csv_metric(models_path_anneal, models_path_rand, eval_type, metric, savepath):
    
    # ANNEALED MODELS
    plot_data = []
    for c in settings.kl_weights:
        settings.kl_weight = c
        updated_dir = models_path_anneal + "kl_weight" + str(c) + '/seed' + str(settings.seed) + '/'

        folder_ctx = "with_ctx/" if settings.with_ctx_representation else "without_ctx/"
        json_file_path = "src/saved_models/" + str(settings.num_protos) + '_VQ/anneal/' + folder_ctx + 'kl_weight' + str(c) + '/seed' + str(settings.seed) + '/'
        
        for folder_utility in [i for i in os.listdir(json_file_path) if "utility" in i]:
            updated_dir2 = updated_dir + folder_utility + "/"
            
            u = (folder_utility.split("utility")[1].split("/")[0])
            folder_alpha = os.listdir(json_file_path + folder_utility)[0]
            a = (folder_alpha.split("alpha")[1].split("/")[0])
            updated_dir3 = updated_dir2 + folder_alpha + "/"

            print("u:", u, "i:", a, "c:", c)

            # get convergence epoch for that model
            json_file = json_file_path+"objective0.json"
            print(json_file)
            with open(json_file, 'r') as f: 
                existing_params = json.load(f)
            convergence_epoch = existing_params["utility"+str(u)]["inf_weight"+str(a)]['convergence epoch']
            
            if eval_type != "training":
                path = updated_dir3 + str(convergence_epoch) + '/evaluation/'+ eval_type +'/test_True_2_metrics'
                metrics = PerformanceMetrics.from_file(path)
                if metric == "accuracy":
                    to_append = metrics.comm_accs[-1]
                elif metric == "informativeness":
                    to_append = metrics.recons[-1]
                plot_data.append((u, a, c, to_append))
            else:
                metrics = PerformanceMetrics.from_file(updated_dir3 + str(convergence_epoch) + '/train_True_2_metrics')
                if metric == "accuracy":
                    to_append = metrics.comm_accs[-1]
                elif metric == "informativeness":
                    to_append = metrics.recons[-1]
                plot_data.append((u, a, c, to_append))           

    df_anneal = pd.DataFrame(plot_data, columns=["Utility", "Alpha", "Complexity", metric])


    # RANDOMLY INITIALIZED MODELS
    
    plot_data = []
    utility_value, alpha_value, c_value = 0.9999, 0.0, 0.0001
    path = models_path_rand + "kl_weight"+str(c_value) +"/seed" + str(settings.seed) + "/utility" + str(utility_value) + "/alpha" + str(alpha_value) + '/'
    
    if eval_type != "training":
        metrics = PerformanceMetrics.from_file(path + str(19999) + '/evaluation/'+ eval_type +'/test_True_2_metrics')
        if metric == "accuracy":
            to_append = metrics.comm_accs[-1]
        elif metric == "informativeness":
            to_append = metrics.recons[-1]
        plot_data.append((utility_value, alpha_value, c_value, to_append))
    else:
        metrics = PerformanceMetrics.from_file(path + str(19999) + '/train_True_2_metrics')
        if metric == "accuracy":
            to_append = metrics.comm_accs[-1]
        elif metric == "informativeness":
            to_append = metrics.recons[-1]
        plot_data.append((utility_value, alpha_value, c_value, to_append))

    df_rand = pd.DataFrame(plot_data, columns=["Utility", "Alpha", "Complexity", metric])
    
    df = pd.concat([df_anneal, df_rand])

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    df.to_csv(savepath + eval_type + "_" + metric + ".csv")

    

    
def run():
    save_path =  "Plots/" + str(settings.num_protos) + "_VQ/simplex/seed" + str(settings.seed) + "/"
    # Metrics' csvs
    basedir_anneal = "src/saved_models/"+ str(settings.num_protos) + '_VQ/anneal/' + settings.folder_ctx
    basedir_rand = "src/saved_models/"+ str(settings.num_protos) + '_VQ/random_init/' + settings.folder_ctx 
    generate_csv_metric(basedir_anneal, basedir_rand, "pragmatics", "accuracy", save_path)
    generate_csv_metric(basedir_anneal, basedir_rand, "pragmatics", "informativeness", save_path)
    generate_csv_metric(basedir_anneal, basedir_rand, "lexsem", "informativeness", save_path)



if __name__ == '__main__':
   
    settings.with_ctx_representation = False
    settings.kl_weight = 1.0
    settings.folder_ctx = "with_ctx/" if settings.with_ctx_representation else "without_ctx/"
    
    settings.num_protos = 3000

    c_dirs = os.listdir("src/saved_models/" + str(settings.num_protos) + '_VQ/anneal/without_ctx/')
    c_dirs = [i for i in c_dirs if "kl_weight" in i]
    c_values = [float(i.split("kl_weight")[1]) for i in c_dirs]
    settings.kl_weights = c_values

    settings.seed = 0
    
    run()

