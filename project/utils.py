import numpy as np
import os
from keras.models import model_from_json, model_from_yaml
import tensorflow as tf
from tensorflow.python.ops import array_ops

def dataset_import(data_dir, data_type):
    sets = []
    print("Load from %s" % data_dir)
    for dir_name, subdir_list, file_list in os.walk(data_dir):
        for file_name in file_list:
            if(data_type in file_name):
                sets.append(dir_name + file_name)
    sets.sort()
    print(str(len(sets))+ " files loaded!")
    return sets

# Create directories for different runs of training
def runs_management(path):
    if not os.path.isdir(path + "run_0"):
        new_dir = path + "run_0"
        os.mkdir(new_dir)
    else:
        dir_list = []
        for d in os.listdir(path):
            if d != '':
                index = int(d.split("_")[-1])
                dir_list.append(index)

        dir_list.sort()
        new_dir = path + "run_" + str(dir_list[-1] + 1)

        os.mkdir(new_dir)

    os.mkdir(new_dir + "/model")
    os.mkdir(new_dir + "/logging")
    return new_dir

def padding_v2(score, context):
    # Pad zeros to star and end
    extended_score = np.array(score)
    
    padding_dimensions = (context, ) + extended_score.shape[1:]

    padding_start = np.zeros(padding_dimensions)
    padding_end = np.zeros(padding_dimensions)

    extended_score = np.concatenate((padding_start,
                                     extended_score,
                                     padding_end),
                                     axis=0)
    return extended_score

def get_metas(score, beat_resolution, len_context):
    # Beat locations, start symbol, end symbol

    # Number of bits
    n_symbols = 2 # start and end
    n_bits_beats = 1
    n_bits_measures = 1
    metas = np.zeros((score.shape[0], n_symbols + n_bits_beats + n_bits_measures))

    # Adding information for current beat location
    for time in range(0, len(score)):
        # Start symbol
        if time < len_context:
            metas[time, 0] = 1
        # End symbol
        elif time >= (len(score) - len_context):
            metas[time, 1] = 1
        else:
            # Within beat location,  0~beat_resolution-1 
            metas[time, 2] = (time % beat_resolution)/beat_resolution
            # Within measures location,  0~4*(beat_resolution-1) 
            metas[time, 3] = (time % (beat_resolution*4))/(beat_resolution*4)
            
    return metas

def get_representation(score, pitch_range, beat_resolution, len_context):
    pitch_table = np.array(score[0])
    pitch_counter = np.zeros(pitch_range, dtype="int64")
    score_progress = (np.concatenate([score[1:], np.zeros((1,  pitch_range))]) - score).astype('int64')
    # Intensity as length
    # 1 as onset, gradually decrease to zero 
    new_score = np.zeros((score.shape[0], score.shape[1], 3))
    for t ,tt in enumerate(score):
        for p in np.nonzero(pitch_table)[0]:
            if pitch_table[p] and pitch_counter[p] == 0:
                new_score[t, p, 0] = 1
                pitch_counter[p] = 1   
            
            elif pitch_table[p] == -1:
                new_score[t, p, 1] = 1
                pitch_counter[p] -= 1
                new_score[t - pitch_counter[p]:t, p, 2] = new_score[t - pitch_counter[p]:t, p, 2][::-1]
                pitch_counter[p] = 0
            
        for p in np.nonzero(pitch_counter)[0]:
            new_score[t, p, 2] = pitch_counter[p]
            pitch_counter[p] += 1
            
        pitch_table = score_progress[t]
        
    t +=1
    for p in np.nonzero(pitch_table)[0]:
        if pitch_table[p] == -1:
            pitch_counter[p] -= 1
            new_score[t - pitch_counter[p]:t, p, 2] = new_score[t - pitch_counter[p]:t, p, 2][::-1]

    score_t  = padding_v2(new_score , len_context)
    meta = get_metas(score_t , beat_resolution, len_context)
    
    return score_t, meta

    
def focal_loss(y_true, y_pred):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    alpha=0.6
    gamma=2
    threash_hold = 1e-4
    
    pos_p_sub = tf.where(tf.greater(y_true, threash_hold), y_pred, tf.ones_like(y_pred))
    pos_p_sub1 = tf.where(tf.greater(y_true, threash_hold), y_true, tf.ones_like(y_pred))
    
    neg_p_sub = tf.where(tf.less_equal(y_true, threash_hold), y_pred, tf.zeros_like(y_pred))
    per_entry_cross_ent = - alpha * ((1 - pos_p_sub) ** gamma) * tf.math.log(tf.clip_by_value(pos_p_sub, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.math.log(tf.clip_by_value(1.0 - neg_p_sub, 1e-8, 1.0))
#     per_entry_cross_ent = - alpha * (tf.abs(pos_p_sub1 - pos_p_sub) ** gamma) * tf.math.log(tf.clip_by_value(1.0 - tf.abs(pos_p_sub1 - pos_p_sub), 1e-8, 1.0)) - (1 - alpha) * (neg_p_sub ** gamma) * tf.math.log(tf.clip_by_value(1.0 - neg_p_sub, 1e-8, 1.0))

    return tf.reduce_mean(per_entry_cross_ent)


def partial_focal_loss(y_true, y_pred):
    # Based on binary cross entropy
    # Loss function for transformer, update only for the masked timesteps
    mask = tf.where(tf.equal(y_true, -1), tf.zeros_like(y_pred), tf.ones_like(y_pred))
    alpha=0.6
    gamma=2
    threash_hold = 1e-4
    
    pos_p_sub = tf.where(tf.greater(y_true, threash_hold), y_pred, tf.ones_like(y_pred))
    neg_p_sub = tf.where(tf.less_equal(y_true, threash_hold), y_pred, tf.zeros_like(y_pred))
    per_entry_cross_ent = - alpha * ((1 - pos_p_sub) ** gamma) * tf.math.log(tf.clip_by_value(pos_p_sub, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.math.log(tf.clip_by_value(1.0 - neg_p_sub, 1e-8, 1.0))
    
    per_entry_cross_ent *= mask
    
    return tf.reduce_mean(per_entry_cross_ent)

def partial_loss(y_true, y_pred):
    # Based on binary cross entropy
    # Loss function for transformer, update only for the masked timesteps
    mask = tf.where(tf.equal(y_true, -1), tf.zeros_like(y_pred), tf.ones_like(y_pred))
    
    per_entry_cross_ent = - y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-8, 1.0)) - (1 - y_true) * tf.math.log(tf.clip_by_value(1.0 - y_pred, 1e-8, 1.0))
    per_entry_cross_ent *= mask
    
    return tf.reduce_mean(per_entry_cross_ent)

def binary_crossentropy_mixup(y_true, y_pred):
    # Based on binary cross entropy
    per_entry_cross_ent = - y_true * tf.math.log(tf.clip_by_value(1.0 - tf.abs(y_true - y_pred), 1e-8, 1.0)) - (1 - y_true) * tf.math.log(tf.clip_by_value(1.0 - y_pred, 1e-8, 1.0))

    return tf.reduce_mean(per_entry_cross_ent)

def partial_loss_mixup(y_true, y_pred):
    # Based on binary cross entropy
    # Loss function for transformer, update only for the masked timesteps
    mask = tf.where(tf.equal(y_true, -1), tf.zeros_like(y_pred), tf.ones_like(y_pred))
    
    per_entry_cross_ent = - y_true * tf.math.log(tf.clip_by_value(1.0 - tf.abs(y_true - y_pred), 1e-8, 1.0)) - (1 - y_true) * tf.math.log(tf.clip_by_value(1.0 - y_pred, 1e-8, 1.0))
    per_entry_cross_ent *= mask
    
    return tf.reduce_mean(per_entry_cross_ent)

def partial_binary_accuracy(y_true, y_pred, threshold=0.5):
    # Use with partial loss
    if threshold != 0.5:
        threshold = tf.cast(threshold, y_pred.dtype)
        y_pred = tf.cast(y_pred > threshold, y_pred.dtype)
    
    mask0 = tf.where(tf.equal(y_true, -1), tf.zeros_like(y_pred), tf.ones_like(y_pred))
    y_true = tf.where(tf.equal(y_true, -1), -1e9*tf.ones_like(y_pred), y_true)

    return tf.reduce_sum(tf.cast(tf.equal(y_true, tf.round(y_pred)), dtype=y_pred.dtype))/tf.reduce_sum(mask0)

def current_l_binary_accuracy(y_true, y_pred, threshold=0.5):
    # Use with partial loss, give only the accuracy for the current timestep
    if threshold != 0.5:
        threshold = tf.cast(threshold, y_pred.dtype)
        y_pred = tf.cast(y_pred > threshold, y_pred.dtype)
    
    mask0 = tf.where(tf.equal(y_true, -1), tf.zeros_like(y_pred), tf.ones_like(y_pred))
    y_true = tf.where(tf.equal(y_true, -1), -1e9*tf.ones_like(y_pred), y_true)
    
    return tf.reduce_sum(tf.cast(tf.equal(y_true, tf.round(y_pred))[:, -1, :], dtype=y_pred.dtype))/tf.reduce_sum(mask0[:, -1, :])
    
def current_r_binary_accuracy(y_true, y_pred, threshold=0.5):
    # Use with partial loss, give only the accuracy for the current timestep
    if threshold != 0.5:
        threshold = tf.cast(threshold, y_pred.dtype)
        y_pred = tf.cast(y_pred > threshold, y_pred.dtype)
    
    mask0 = tf.where(tf.equal(y_true, -1), tf.zeros_like(y_pred), tf.ones_like(y_pred))
    y_true = tf.where(tf.equal(y_true, -1), -1e9*tf.ones_like(y_pred), y_true)
    
    return tf.reduce_sum(tf.cast(tf.equal(y_true, tf.round(y_pred))[:, 0, :], dtype=y_pred.dtype))/tf.reduce_sum(mask0[:, 0, :])
