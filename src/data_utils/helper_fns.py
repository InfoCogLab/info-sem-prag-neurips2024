import numpy as np
import src.settings as settings
import torch


def gen_batch(all_data, batch_size, fieldname, p_notseedist=0, vae=None, glove_data=None, see_distractors=False, num_dist=None, preset_targ_idx=None):
    assert glove_data is not None, "Relic argument allowed no glove data, but here we do want it."
    # Given the dataset, creates a batch of inputs.
    # That's:
    # 1) The speaker's observation
    # 2) The listener's observation
    # 3) The label (which is the index of the speaker's observation).
    # 4) Word embeddings for the target, if glove_data is not None

    speaker_obs = []
    listener_obs = []
    labels = []
    embeddings = []
    all_words = all_data[fieldname]

    if num_dist is None:
        num_dist = settings.num_distractors 
    if not settings.see_distractors_pragmatics: 
        all_features = all_data['features'] 
    else:
        all_features = all_data['t_features']
        all_features_dist = all_data['d_features']
        all_features_ctx = all_data['ctx_features']

    while len(labels) < batch_size:
        targ_idx = int(np.random.random() * len(all_features)) if preset_targ_idx is None else preset_targ_idx
        # Get the word embedding
        word = all_words[targ_idx]
        emb = get_glove_embedding(glove_data, word)
        if emb is None:
            if preset_targ_idx is not None:  # No embedding for the target id you want.
                return None, None, None, [None]
            continue
        emb = emb.to_numpy()
        embeddings.append(emb)
        targ_features = all_features[targ_idx]
        distractor_features = []
        candidate_words = set()
        candidate_words.add(word)
        
        if not settings.see_distractors_pragmatics:
            while len(distractor_features) < num_dist:
                dist_id = int(np.random.random() * len(all_features))
                dist_word = all_words[dist_id]
                if dist_word in candidate_words and settings.distinct_words:
                    continue  # Already exists, so skip
                candidate_words.add(dist_word)
                distractor_features.append(all_features[dist_id]) 
        else:
            distractor_features = all_features_dist[targ_idx]
            ctx_features = all_features_ctx[targ_idx]
        
        obs_targ_idx = int(np.random.random() * (num_dist + 1)) 
        # if settings.distinct_words:
        #     print("Words", candidate_words)
        # distractor_features = [all_features[int(np.random.random() * len(all_features))] for _ in range(num_dist)]
        if settings.see_distractors_pragmatics: 
            if settings.with_ctx_representation:
                s_obs = np.expand_dims(np.vstack([targ_features] + [distractor_features] + [ctx_features]), axis=0)
                speaker_obs.append(s_obs)
                l_obs = np.expand_dims(np.vstack([distractor_features][:obs_targ_idx] + [targ_features] + [distractor_features][obs_targ_idx:] + [ctx_features]), axis=0)
                listener_obs.append(l_obs)
                labels.append(obs_targ_idx)
            else:
                s_obs = np.expand_dims(np.vstack([targ_features] + [distractor_features]), axis=0)
                speaker_obs.append(s_obs)
                l_obs = np.expand_dims(np.vstack([distractor_features][:obs_targ_idx] + [targ_features] + [distractor_features][obs_targ_idx:]), axis=0)
                listener_obs.append(l_obs)
                labels.append(obs_targ_idx)
        else:
            s_obs = targ_features if not see_distractors else np.expand_dims(np.vstack([targ_features] + distractor_features), axis=0)
            speaker_obs.append(s_obs)
            l_obs = np.expand_dims(np.vstack(distractor_features[:obs_targ_idx] + [targ_features] + distractor_features[obs_targ_idx:]), axis=0)
            listener_obs.append(l_obs)
            labels.append(obs_targ_idx)            
        
    speaker_tensor = torch.Tensor(np.vstack(speaker_obs).astype(np.float)).to(settings.device)
    listener_tensor = torch.Tensor(np.vstack(listener_obs).astype(np.float)).to(settings.device)
    

    if vae is not None:
        with torch.no_grad():
            speaker_tensor, _ = vae(speaker_tensor)
            pass
    label_tensor = torch.Tensor(labels).long().to(settings.device)
    
    if settings.see_distractors_pragmatics:
        
        # if we add uncertainty with dropout layer
        if settings.dropout:
            original_shape = speaker_tensor.shape
            speaker_tensor = speaker_tensor.view(original_shape[0], original_shape[1] * original_shape[2])
            # Slicing the tensor to select dimensions corresponding to the distractor, e.g. 512:1024
            dropout_slice = speaker_tensor[:, original_shape[2]:(original_shape[2] * original_shape[1])]
            # Applying dropout
            dropout = torch.nn.Dropout(p=p_notseedist)
            dropout_tensor = dropout(dropout_slice)
            # Concatenating the dropout slice with the remaining dimensions
            speaker_tensor = torch.cat([speaker_tensor[:, :original_shape[2]], dropout_tensor, speaker_tensor[:, (original_shape[2] * original_shape[1]):]], dim=1) 
            speaker_tensor = speaker_tensor.view(original_shape[0], original_shape[1], original_shape[2])
        
        # if we implement probability of seeing the distractor
        elif settings.see_probabilities:
            # Create a mask of the same shape as the tensor
            mask = torch.ones_like(speaker_tensor, dtype=bool)
            # Generate random indices to apply the mask
            num_tensors = speaker_tensor.shape[0]
            indices = np.random.choice(num_tensors, int(num_tensors * p_notseedist), replace=False)
            # Apply the mask to the selected indices
            mask[indices, 1, :] = False
            # Apply the mask to the tensor
            speaker_tensor = torch.where(mask, speaker_tensor, torch.tensor(0.0, device=settings.device))
     
    return speaker_tensor, listener_tensor, label_tensor, embeddings


def get_unique_labels(dataset):
    unique_topnames = get_unique_by_field(dataset, 'topname')
    unique_responses = set()
    for responses in dataset['responses']:
        for k in responses.keys():
            unique_responses.add(k)
    return unique_topnames, unique_responses


def get_unique_by_field(dataset, fieldname):
    uniques = set()
    for elt in dataset[fieldname]:
        uniques.add(elt)
    return uniques


def get_entry_for_labels(dataset, labels, fieldname='topname', num_repeats=1):
    rows = []
    for _ in range(num_repeats):
        for label in labels:
            matching_rows = dataset.loc[dataset[fieldname] == label].index.tolist()
            rand_idx = int(np.random.random() * len(matching_rows))
            rows.append(matching_rows[rand_idx])
    big_data = dataset[dataset.index.isin(rows)]
    #big_data = dataset.iloc[rows]
    return big_data


def get_rand_entries(dataset, num_entries):
    big_data = dataset.sample(n=num_entries)
    return big_data


def get_embedding_batch(all_data, embed_data, batch_size, fieldname, vae=None):
    if settings.see_distractors_pragmatics:
        all_features = all_data['t_features']
    else:
        all_features = all_data['features']
    features = []
    embeddings = []
    while len(features) < batch_size:
        targ_idx = int(np.random.random() * len(all_features))
        # Get the embedding for the word
        if fieldname == 'responses':
            responses = all_data['responses'][targ_idx]
            words = []
            probs = []
            for k, v in responses.items():
                parsed_word = k.split(' ')
                if len(parsed_word) > 1:
                    # Skip "words" like "tennis player" etc. because they won't be in glove data
                    continue
                words.append(k)
                probs.append(v)
            if len(words) == 0:
                # Failed to find any legal words (e.g., all like "tennis player")
                continue
            total = np.sum(probs)
            probs = [p / total for p in probs]
            sampled_word = np.random.choice(words, p=probs)
        elif fieldname == 'vg_domain':
            sampled_word = all_data['vg_domain'][targ_idx]
        elif fieldname == 'topname':
            sampled_word = all_data['topname'][targ_idx]
        elif fieldname == 'vg_obj_name':
            sampled_word = all_data['vg_obj_name'][targ_idx]
        embedding = get_glove_embedding(embed_data, sampled_word)
        if embedding is None:
            continue
        # Get the features
        features.append(all_features[targ_idx])
        embeddings.append(embedding)
    feature_tensor = torch.Tensor(np.vstack(features)).to(settings.device)
    if vae is not None:
        with torch.no_grad():
            feature_tensor, _ = vae(feature_tensor)
    emb_tensor = torch.Tensor(np.vstack(embeddings)).to(settings.device)
    return feature_tensor, emb_tensor


def get_glove_embedding(dataset, word):
    if word == 'animals_plants':  # Gross, but important for vg_domain
        word = 'animals'
    try:
        cached_embed = settings.embedding_cache.get(word)
        if cached_embed is not None:
            return cached_embed
        embed = dataset.loc[word]
        settings.embedding_cache[word] = embed
        return embed
    except KeyError:
        # print("Couldn't find word", word)
        return None


def get_all_embeddings(glove_dataset, words):
    all_embeddings = []
    for word in words:
        emb = get_glove_embedding(glove_dataset, word)
        if emb is None:
            continue
        all_embeddings.append(emb.to_numpy())
    stacked_embeddings = np.vstack(all_embeddings)
    return stacked_embeddings


def intersection_over_union(boxA, boxB):
    # needs boxes of format [x1, x2, y1, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


