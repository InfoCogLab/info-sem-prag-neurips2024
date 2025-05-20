# Bridging semantics and pragmatics in information-theoretic emergent communication

Repository associated with the NeurIPS 2024 paper by Eleonora Gualdoni, Mycal Tucker, Roger P. Levy, and Noga Zaslavsky




### **Demo**
The best way to get a sense of our task and data of interest is to use our **demo** **`demo_target_distractor_selection.ipynb`**, which allows you to visualize ManyNames images, with annotated target - distractor pairs. 

### **External resources**

1. Download the file **manynames.tsv** from the official webpage at [this link](https://amore-upf.github.io/manynames/) and store it in the **data** folder.
2. Download Glove embeddings (**glove.6B.100d.txt**) from [the official release](https://nlp.stanford.edu/projects/glove/).
3. To choose distractor objects, we passed [Anderson et al., 2018](https://arxiv.org/abs/1707.07998)'s object detection model on ManyNames images. We used the Detectron implementation of the model, available [here](https://github.com/airsplay/py-bottom-up-attention), and the corresponding [script](https://github.com/airsplay/py-bottom-up-attention/blob/master/demo/demo_feature_extraction_attr.ipynb) released by the authors. You can skip this step: in the **data** folder, we provide the file **manynames_detections.tsv** with all the information relevant for the next steps.

### **Targets' and distractors' features**
The script **src/data_utils/read_data.py** allows to download ManyNames images and extract the visual features of target and distractor objects. The resulting features are saved in the **data** folder as csv files. To facilitate the process, we provide the resulting files **t_features.csv** and **d_features.csv**, together with a file storing the x1-y1-x2-y2 coordinates of the distractors: **d_xyxy.csv**.

(The files marked with "someRE" correspond to features extracted from the a subset of images from the ManyNames dataset, i.e. those used in [Mädebach et al., 2022](https://escholarship.org/uc/item/7cs7204s)'s study, which we excluded during our agents' training phases in view of future analyses) 


### **Model training**

All the following scripts are located in the folder **src/scripts**.

1. As a first step, train a VAE model that will generate speaker's mental representations $m$ with the script **`train_vae.py`**
2. Then, train the first model (randomly initialized) in the pragmatics setting, with **`main_pragmatics_rand.py`**
3. Anneal models from the randomly initialized one to fill in the simplex, by running:
    - **`main_pragmatics_anneal_UI.py`**, that anneals models going from $\lambda_U=1$ to $\lambda_I=1$ 
    - **`main_pragmatics_anneal_IC1.py`**, that anneals models going from $\lambda_I=1$ to $\lambda_C=1$ 
    - **`main_pragmatics_anneal_IC2.py`**, that anneals models by setting a value of $\lambda_U$, and gradually descreasing $\lambda_I$ while increasing $\lambda_C$, filling the entire simplex.

### **Evaluations and Analyses**

All the following scripts are located in the folder **src/scripts**.

1. Compute utility and informativeness at test time, with the scripts **`eval_pragmatics.py`** and **`eval_lexsem.py`**
2. Compute complexity of the lexicon in the semantics setting, with the script **`complexity.py`**. The script **`complexity_humans.py`** computes the complexity of the human lexicon for the ManyNames domain, to use as reference.
3. Compute NID of the lexicon and its size in the semantics setting, with the script **`system_NID.py`** (the script **`system_NID_restricted.py`** computes NID considering only on images with high probability for their human name)
4. Generate csv files of the results with **`generate_csv_results.py`**
5. Generate simplex plots with the R script **`simplex_plots.R`**

### Cite

```bibtex
@inproceedings{Gualdoni2024bridging,
 author = {Gualdoni, Eleonora and Tucker, Mycal and Levy, Roger and Zaslavsky, Noga},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {21059--21078},
 publisher = {Curran Associates, Inc.},
 title = {Bridging semantics and pragmatics in information-theoretic emergent communication},
 volume = {37},
 year = {2024}
```
}
