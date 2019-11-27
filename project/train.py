import os
import numpy as np
import pickle
import pypianoroll as pr
from Music_Style_Transfer_master.project.utils import add_beat, padding, dataset_import
import tensorflow.keras.callbacks as cbs
from keras.utils import to_categorical

# Perform mixup on two dataset
class generator_random_mask_2mixup_onehot(object):
    def __init__(self, batch_size, pitch_range, len_context, 
                 path_dataset, path_dataset_mixup=None,
                 phase='train', percentage_train=0.8, mask_ratio=0):
        
        if path_dataset_mixup != None:
            self.mixup = True
        else:
            self.mixup = False
            path_dataset_mixup = path_dataset
        
        # Load major dataset
        d_scores = dataset_import(path_dataset, ".npy")
        d_metas = []
        for i in d_scores:
            if "meta" in i:
                d_metas.append(i)
                d_scores.remove(i)
         
        d_metas.sort(key=lambda x: (int(os.path.basename(x)[:-9])))
    
        # Load mixup dataset
        d_scores_mixup = dataset_import(path_dataset_mixup, ".npy")
        d_metas_mixup = []
        for i in d_scores_mixup:
            if "meta" in i:
                d_metas_mixup.append(i)
                d_scores_mixup.remove(i)
         
        d_metas_mixup.sort(key=lambda x: (int(os.path.basename(x)[:-9])))
    
        # Train valid test split:  percentage_train,  0.5*(1-percentage_train), 0.5*(1-percentage_train)
        if phase == 'train':
            self.score_indices = np.arange(int(len(d_scores) * percentage_train))
            self.score_indices_mixup = np.arange(int(len(d_scores_mixup) * percentage_train))
        if phase == "valid":
            self.score_indices = np.arange(int(len(d_scores) * percentage_train), int((1 + percentage_train)/2 * len(d_scores)))
            self.score_indices_mixup = np.arange(int(len(d_scores_mixup) * percentage_train), int((1 + percentage_train)/2 * len(d_scores_mixup)))
        elif phase == 'test':
            self.score_indices = np.arange(int((1 + percentage_train)/2 * len(d_scores)), len(d_scores))
            self.score_indices_mixup = np.arange(int((1 + percentage_train)/2 * len(d_scores_mixup)), len(d_scores_mixup))
        elif phase == 'all':
            self.score_indices = np.arange(len(d_scores))
            self.score_indices_mixup = np.arange(len(d_scores_mixup))
            
        self.d_scores = d_scores
        self.d_metas = d_metas
        self.d_scores_mixup = d_scores_mixup
        self.d_metas_mixup = d_metas_mixup
            
        self.batch_size = batch_size
        self.pitch_range = pitch_range
        self.len_context = len_context
        self.mask_ratio = mask_ratio
        
        self.central_features = []
        self.central_metas = []
        self.right_features = []
        self.right_metas = []
        self.left_features = []
        self.left_metas = []
        self.labels = []
        self.labels_aux_left = []
        self.labels_aux_right = []
    
        self.batch = 0
        self.change_score = 0
        self.one_hot_scaler = np.arange(1, pitch_range + 1)
        
        self.one_hot_eye = np.eye(pitch_range + 2)
        
    def get_sample(self, score, score_onset, score_offset, score_original, score_one_hot,
                   meta, score_length
              ):
    
        time_index = np.random.randint(self.len_context, score_length - self.len_context)
        midi_index = np.random.randint(self.pitch_range)
                
        central_feature = np.array(score[time_index, :self.pitch_range])
     
        label_aux_left = np.array(np.concatenate([score[time_index - (self.len_context):time_index + 1, :self.pitch_range], 
                                                  score_onset[time_index - (self.len_context):time_index + 1, :self.pitch_range],  
                                                  score_offset[time_index - (self.len_context):time_index + 1, :self.pitch_range]], axis=1))
        
        label_aux_right = np.array(np.concatenate([score[(time_index):(time_index + 1) + self.len_context, :self.pitch_range], 
                                                   score_onset[(time_index):(time_index + 1) + self.len_context, :self.pitch_range],  
                                                   score_offset[(time_index):(time_index + 1) + self.len_context, :self.pitch_range]], axis=1))
        
        central_feature[midi_index: ] = 2
        central_feature = np.reshape(central_feature, (1, -1))
        
        central_meta = meta[time_index, :]
        central_meta = np.reshape(central_meta, (1, -1))
        
        left_feature = np.array(score_one_hot[time_index - (self.len_context):time_index, :])
        left_meta = meta[time_index - (self.len_context):time_index, :] 
        
        right_feature = np.array(score_one_hot[(time_index + 1):(time_index + 1) + self.len_context, :]) 
        right_meta = meta[(time_index + 1):(time_index + 1) + self.len_context, :]

        # Label is refererenced from original dataset
        label = int(score_original[time_index, midi_index] > 0)
            
        return central_feature, central_meta, left_feature, left_meta, right_feature, right_meta, label, label_aux_left, label_aux_right
        
    def generator(self):
        while True:
            if self.change_score == 0:
                score_index = np.random.choice(self.score_indices)
                score = np.load(self.d_scores[score_index])
                score_length = score.shape[0]
            
                # Transpose
                edge = np.nonzero(np.sum(score, axis=0))[0]
                edge_up = min(13, score.shape[1] - edge[-1]) 
                edge_low = max(-12, -edge[0])
                shift = np.random.choice(np.arange(edge_low, edge_up))
                score = np.roll(score, shift=shift, axis=1)
            
                # Load meta
                meta = np.load(self.d_metas[int(os.path.basename(self.d_scores[score_index]).split("_")[0])])
            
                # Mixup, now we allign the start and end padding of two scores
                if self.mixup:
                    score_index_mixup = np.random.choice(self.score_indices_mixup)
                    score_mixup = np.load(self.d_scores_mixup[score_index_mixup])
                    score_length_mixup = score_mixup.shape[0]
                    # Transpose
                    edge = np.nonzero(np.sum(score_mixup, axis=0))[0]
                    edge_up = min(13, score_mixup.shape[1] - edge[-1]) 
                    edge_low = max(-12, -edge[0])
                    shift = np.random.choice(np.arange(edge_low, edge_up))
                    score_mixup = np.roll(score_mixup, shift=shift, axis=1)
                    # Load meta
                    meta_mixup= np.load(self.d_metas_mixup[int(os.path.basename(self.d_scores_mixup[score_index_mixup]).split("_")[0])])
                
                    score_onset_mixup = np.array(score_mixup[:, :, 0])
                    score_offset_mixup = np.array(score_mixup[:, :, 1])
                    score_sparse_mixup = np.array(score_mixup[:, :, 2]*score_mixup[:, :, 0])
                    score_one_hot_mixup = score_sparse_mixup*self.one_hot_scaler
                    score_mixup = (np.array(score_mixup[:, :, 2]) > 0).astype(int)  
                    
                    score_original_mixup = np.array(score_mixup)
                
                score_onset = np.array(score[:, :, 0])
                score_offset = np.array(score[:, :, 1])
                score_sparse = np.array(score[:, :, 2]*score[:, :, 0])
                score = (np.array(score[:, :, 2]) > 0).astype(int)
                # Preserve ground truth
                score_original = np.array(score)  
                score_one_hot = score*self.one_hot_scaler
                
            central_feature, central_meta, \
            left_feature, left_meta, \
            right_feature, right_meta, \
            label, label_aux_left, label_aux_right = self.get_sample(score, score_onset, score_offset, score_original, score_one_hot,
                                                                     meta, score_length)
        
            if self.mixup:
                central_feature_mixup, central_meta_mixup, \
                left_feature_mixup, left_meta_mixup, \
                right_feature_mixup, right_meta_mixup, \
                label_mixup, label_aux_left_mixup, label_aux_right_mixup = self.get_sample(score_mixup, score_onset_mixup, score_offset_mixup,
                                                                                           score_original_mixup, score_one_hot, meta_mixup, 
                                                                                           score_length_mixup)
      
                lam = np.random.beta(1, 4)
                
                left_feature = lam*left_feature + (1 - lam)*left_feature_mixup
                left_meta = lam*left_meta + (1 - lam)*left_meta_mixup
                central_feature = lam*central_feature + (1 - lam)*central_feature_mixup
                central_meta = lam*central_meta + (1 - lam)*central_meta_mixup 
                right_feature = lam*right_feature + (1 - lam)*right_feature_mixup
                right_meta = lam*right_meta + (1 - lam)*right_meta_mixup 
                label = lam*label + (1 - lam)*label_mixup 
                label_aux_left = lam*label_aux_left + (1 - lam)*label_aux_left_mixup
                label_aux_right = lam*label_aux_right + (1 - lam)*label_aux_right_mixup 
            
            # Random masking                                 
            mask = np.random.choice(np.arange(self.len_context), replace=False, size=self.len_context)
            split_point = int(self.len_context*self.mask_ratio)
            left_feature[mask[:split_point], :] = (self.pitch_range + 1)
            label_aux_left[mask[split_point:], :] = -1
        
            mask = np.random.choice(np.arange(self.len_context), replace=False, size=self.len_context)
            split_point = int(self.len_context*self.mask_ratio)
            right_feature[mask[:split_point], :] = (self.pitch_range + 1)
            label_aux_right[mask[split_point:] + 1, :] = -1
            
            self.central_features.append(central_feature)
            self.central_metas.append(central_meta)
            self.left_features.append(left_feature)
            self.left_metas.append(left_meta)
            self.right_features.append(right_feature)
            self.right_metas.append(right_meta)
            self.labels.append(label)
            self.labels_aux_left.append(label_aux_left)
            self.labels_aux_right.append(label_aux_right)

            self.batch += 1
            self.change_score = (self.change_score + 1) % (self.batch_size//4)
        
            # if there is a full batch
            if self.batch == self.batch_size:
                left_features = np.array(self.left_features, dtype=np.float32)
                left_metas = np.array(self.left_metas, dtype=np.float32)
                central_features = np.array(self.central_features, dtype=np.float32)
                central_metas = np.array(self.central_metas, dtype=np.float32)
                right_features = np.array(self.right_features, dtype=np.float32)
                right_metas = np.array(self.right_metas, dtype=np.float32)
                labels = np.array(self.labels, dtype=np.float32)
                labels_aux_left = np.array(self.labels_aux_left, dtype=np.float32)
                labels_aux_right = np.array(self.labels_aux_right, dtype=np.float32)
        
                next_element = (left_features,
                                left_metas,
                                central_features,
                                central_metas, 
                                right_features, 
                                right_metas, 
   
                                labels, 
                                labels_aux_left,
                                labels_aux_right
                )
            
                yield next_element
                self.batch = 0
                self.central_features = []
                self.central_metas = []
                self.right_features = []
                self.right_metas = []
                self.left_features = []
                self.left_metas = []
                self.labels = []
                self.labels_aux_left = []
                self.labels_aux_right = []
                                  


            
def train(model, path_model, path_dataset, path_logging, beat_resolution, len_context, pitch_range, epoch=80,
          steps_per_epoch=8000, batch_size=88*5, mask_ratio=0.15, path_dataset_mixup=None
         ):
    


    gen = generator_random_mask_2mixup_onehot(batch_size, pitch_range=pitch_range, len_context=len_context, phase='train',
                                              path_dataset=path_dataset, path_dataset_mixup=path_dataset_mixup,
                                              mask_ratio=mask_ratio
                                              )
    gen_valid = generator_random_mask_2mixup_onehot(batch_size, pitch_range=pitch_range, len_context=len_context, phase='valid',
                                                    path_dataset=path_dataset, path_dataset_mixup=path_dataset_mixup,
                                                    mask_ratio=mask_ratio
                                                   )
    gen_test = generator_random_mask_2mixup_onehot(batch_size, pitch_range=pitch_range, len_context=len_context, phase='test',
                                                   path_dataset=path_dataset, path_dataset_mixup=path_dataset_mixup,
                                                   mask_ratio=mask_ratio
                                                   )
        
        
    generator_train = (({'left_features': left_features,
                         'left_metas': left_metas,
                         'central_features': central_features,
                         'central_metas': central_metas,
                         'right_features': right_features,
                         'right_metas': right_metas
                         }, 
                        {'prediction': labels,
                          'central_l': labels_aux_left,
                          'central_r': labels_aux_right
                        })
                          for(left_features,
                              left_metas,
                              central_features,
                              central_metas,
                              right_features,
                              right_metas,
                              labels,
                              labels_aux_left,
                              labels_aux_right
                             ) in gen.generator())

    generator_val = (({'left_features': left_features,
                       'left_metas': left_metas,
                       'central_features': central_features,
                       'central_metas': central_metas,
                       'right_features': right_features,
                        'right_metas': right_metas
                       }, 
                      {'prediction': labels,
                       'central_l': labels_aux_left,
                       'central_r': labels_aux_right
                       })
                          for(left_features,
                              left_metas,
                              central_features,
                              central_metas,
                              right_features,
                              right_metas,
                              labels,
                              labels_aux_left,
                              labels_aux_right
                             ) in gen_valid.generator())
    
    generator_test = (({'left_features': left_features,
                        'left_metas': left_metas,
                         'central_features': central_features,
                         'central_metas': central_metas,
                         'right_features': right_features,
                         'right_metas': right_metas
                         }, 
                       {'prediction': labels,
                         'central_l': labels_aux_left,
                         'central_r': labels_aux_right
                        })
                          for(left_features,
                               left_metas,
                               central_features,
                               central_metas,
                               right_features,
                               right_metas,
                               labels,
                               labels_aux_left,
                               labels_aux_right
                             ) in gen_test.generator())
    
    
    cb_tensorboard = cbs.TensorBoard(log_dir=path_logging + "train/")
    cb_checkpoint = cbs.ModelCheckpoint(path_model + ".hdf5", 
                                        save_weights_only=True, save_best_only=True, monitor='val_loss')
    
    csv_logger = cbs.CSVLogger(path_logging + 'training_log.csv', append=True, separator=';')
    
    model.fit_generator(generator_train, steps_per_epoch=steps_per_epoch,
                        epochs=epoch, verbose=1, validation_data=generator_val,
                        validation_steps=len_valid, use_multiprocessing=False,
                        max_queue_size=batch_size, workers=1,
                        callbacks=[cb_tensorboard, cb_checkpoint, csv_logger]
                       )
    # Testing
    test_result= model.evaluate_generator(generator_test, steps=len_test, max_queue_size=100, workers=1,
                                          use_multiprocessing=False, verbose=1)

    return model, test_result

