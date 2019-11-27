import tensorflow as tf
import tensorflow.keras.layers as tkl
import numpy as np

#Transformer functions
def scaled_dot_product_attention(query, key, value, direction, srel=None):      
    """Calculate the attention weights. """
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    
    # relative embedding
    if srel is not None:
        matmul_qk += srel
    
    # scale matmul_qk
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)
    
    # Mask current timestep
    mask_0 = tf.zeros(shape=(tf.shape(matmul_qk)[0], tf.shape(matmul_qk)[1], tf.shape(matmul_qk)[2], tf.shape(matmul_qk)[3] - 1))
    mask_1 = -1e9*tf.ones(shape=(tf.shape(matmul_qk)[0], tf.shape(matmul_qk)[1], tf.shape(matmul_qk)[2], 1))
    if direction == "left":
        mask = tf.concat([mask_0, mask_1], axis=-1)
    elif direction == "right":
        mask = tf.concat([mask_1, mask_0], axis=-1)
    
    # Mask current timestep, left and right transformers have different direction
    # len_q == len_k
#     mask = tf.ones(shape=(tf.shape(matmul_qk)[3], tf.shape(matmul_qk)[3]))
#     if direction == "left":
#         mask = mask - tf.linalg.band_part(mask, -1, 0) 
#     elif direction == "right":
#         mask = mask - tf.linalg.band_part(mask, 0, -1) 
        
#     mask = -1e9*tf.tensordot(tf.ones(shape=(tf.shape(matmul_qk)[0], tf.shape(matmul_qk)[1], 1)), mask[tf.newaxis,...], axes=[[-1], [0]])
        
    logits #+= mask

    # softmax is normalized on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(logits, axis=-1)

    output = tf.matmul(attention_weights, value)

    return output

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, direction, len_context, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model
        self.direction = direction
        
        # plus 1 because we have to include central step
        self.len_context = len_context + 1
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = tkl.Dense(units=d_model)
        self.key_dense = tkl.Dense(units=d_model)
        self.value_dense = tkl.Dense(units=d_model)

        self.dense = tkl.Dense(units=d_model)

    def build(self, input_shape):
        self.rel = self.add_weight("rel_emb", shape=[self.len_context, self.depth])
        
    def _get_rel_embedding(self, len_q):
        starting_point = tf.maximum(0, self.len_context - len_q)
        e = self.rel[starting_point:,:]
        return e

    @staticmethod
    def _qe_masking(qe):
        mask = tf.sequence_mask(
            tf.range(tf.shape(qe)[-1] - 1, tf.shape(qe)[-1] - tf.shape(qe)[-2] - 1, -1), tf.shape(qe)[-1])

        mask = tf.logical_not(mask)
        mask = tf.cast(mask, tf.float32)

        return mask * qe

    def _skewing(self, tensor: tf.Tensor, len_q, len_k):
        padded = tf.pad(tensor, [[0, 0], [0,0], [0, 0], [1, 0]])
        reshaped = tf.reshape(padded, shape=(-1, tf.shape(padded)[1], tf.shape(padded)[-1], tf.shape(padded)[-2]))
        Srel = reshaped[:, :, 1:, :]
        # print('Sre: {}'.format(Srel))
        
        f1 = lambda: tf.pad(Srel, [[0,0], [0,0], [0,0], [0, len_k - len_q]])
        f2 = lambda: Srel[:, :, :, :len_k]
        Srel = tf.case({tf.greater(len_k, len_q): f1,
                        tf.less(len_k, len_q): f2,
                       }, default=lambda: Srel, exclusive=True)

#         if len_k > len_q:
#             Srel = tf.pad(Srel, [[0,0], [0,0], [0,0], [0, len_k - len_q]])
#         elif len_k < len_q:
#             Srel = Srel[:,  :,:, :len_k]

        return Srel 
        
    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value = inputs['query'], inputs['key'], inputs['value']
        batch_size = tf.shape(query)[0]
        
        # Notes: linear than multi-head V.S. multi-head than linear
        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        # relative embedding
        e = self._get_rel_embedding(tf.shape(query)[2])
        qe = tf.einsum('bhld,md->bhlm', query, e)
        qe = self._qe_masking(qe)
        # print(QE.shape)
        srel = self._skewing(qe, tf.shape(query)[2], tf.shape(key)[2])

        # scaled dot-product attention
        scaled_attention = scaled_dot_product_attention(query, key, value, self.direction, srel)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatenation of heads
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # final linear layer
        outputs = self.dense(concat_attention)

        return outputs
    
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model, sum_or_concat):
        super(PositionalEncoding, self).__init__()
        self.sum_or_concat  = sum_or_concat
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        
        angle_rads = self.get_angles(position=tf.range(position, dtype=tf.float32)[:, tf.newaxis], 
                                                 i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
                                                d_model=d_model)
        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        if self.sum_or_concat == "s":
            pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        if self.sum_or_concat == "s":
            return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
        elif self.sum_or_concat == "c":
            self.pos_encoding = self.pos_encoding[:tf.shape(inputs)[1], :, None] * tf.ones([1, tf.shape(inputs)[0]]) 
            self.pos_encoding = tf.transpose(self.pos_encoding, perm=[2, 0, 1])
            return tf.concat([inputs, self.pos_encoding], axis=-1)
        
def point_wise_feed_forward_network(d_model, units):
    return tf.keras.Sequential([
      tkl.Dense(units, activation='relu'),  # (batch_size, seq_len, dff)
      tkl.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, units, d_model, num_heads, rate_dropout, direction, len_context, name="encoder_layer"):
        super(EncoderLayer, self).__init__(name)
        self.attention_layer = MultiHeadAttention(d_model, num_heads, direction, len_context, name="attention")
        self.ffn = point_wise_feed_forward_network(d_model, units)#recurrent_feed_forward_network(d_model, units)

        self.layernorm1 = tkl.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tkl.LayerNormalization(epsilon=1e-6)
    
        self.dropout1 = tkl.Dropout(rate_dropout)
        self.dropout2 = tkl.Dropout(rate_dropout)
    
    def call(self, x):
        # PRE_LN transformer fix
        attention = self.layernorm1(x)
        attention = self.attention_layer({'query': attention, 'key': attention, 'value': attention})  # (batch_size, input_seq_len, d_model)
        attention = self.dropout1(attention)
        attention = x + attention  # (batch_size, input_seq_len, d_model)
    
        outputs = self.layernorm2(attention)
        outputs = self.ffn(outputs)  # (batch_size, input_seq_len, d_model)
        outputs = self.dropout2(outputs)
        outputs = attention + outputs  # (batch_size, input_seq_len, d_model)
    
        return outputs
    
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class Pitch_position_embedding(tf.keras.layers.Layer):
    def __init__(self, pitch_range, name="pitch_pos"):
        super(Pitch_position_embedding, self).__init__(name)
        # Pitch position array
        pitch_pos = []
        for i in range(pitch_range):
            pitch_pos.append(np.array(list(f'{i:07b}')).astype("int64"))
        pitch_pos = np.array(pitch_pos)
        
        self.pitch_pos = tf.Variable(tf.convert_to_tensor(pitch_pos, dtype=tf.float32), trainable=False)[tf.newaxis, ]

    def call(self, x):
        self.pitch_pos = tf.tensordot(tf.ones([tf.shape(x)[0], 1]), self.pitch_pos, axes=[[-1], [0]])
        x = tkl.concatenate([x, self.pitch_pos])
    
        return x
    

def _2way_transformer_wavenet_absolute_pitch(len_context = 8*4*24,
                                             n_octave = 7,
                                             size_embedding = 84,
                                             n_transf_layers = 6,
                                             n_transf_downsample = 2,
                                             n_conv_layers = 7,
                                             context_layers = 3):
    
    left_features = tkl.Input(shape=(len_context, n_octave*12), name='left_features')
    left_metas = tkl.Input(shape=(len_context, 4), name='left_metas')

    right_features = tkl.Input(shape=(len_context, n_octave*12), name='right_features')
    right_metas = tkl.Input(shape=(len_context, 4), name='right_metas')

    central_features = tkl.Input(shape=(1, n_octave*12), name='central_features')
    central_metas = tkl.Input(shape=(1, 4), name='central_metas')
        
    # Embedding for each timestep
    embedding = tf.keras.Sequential()
    embedding.add(tkl.Dense(size_embedding))

    left = tkl.Reshape([len_context, n_octave*12])(left_features)
    left = tkl.Dropout(0.3)(left)
    left = tkl.TimeDistributed(embedding)(left)
    left = tkl.Reshape([len_context, size_embedding])(left)
    left = tkl.concatenate([left, left_metas])

    right = tkl.Reshape([len_context, n_octave*12])(right_features)
    right = tkl.Dropout(0.3)(right)
    right = tkl.TimeDistributed(embedding)(right)
    right = tkl.Reshape([len_context, size_embedding])(right)
    right = tkl.concatenate([right, right_metas])
    
    # Mask central infromation from transformer
    central = -1*tf.ones(shape=(tf.shape(central_features)))
    central = tkl.TimeDistributed(embedding)(central)
    central = tkl.concatenate([central, central_metas])
        
    context = tkl.concatenate([left, central, right], axis = 1)
    context *= tf.math.sqrt(tf.cast(size_embedding + 4, tf.float32))
    context = PositionalEncoding(position=len_context*2 + 1, d_model= size_embedding+4, sum_or_concat="c")(context)
    
    # Central frames is contained in both left and right
    left = context[:, :len_context + 1, :]
    right = context[:, len_context:, :] 
    
    left = tkl.Reshape([len_context + 1 , 2*(size_embedding + 4)])(left)
    right = tkl.Reshape([len_context + 1 , 2*(size_embedding + 4)])(right)
    
    # Transformer context handling
    # Left and right transforemer
    centrals_l = []
    centrals_r = []
    for i in range(n_transf_layers):
        left = EncoderLayer(units=256, d_model=2*(size_embedding+4), num_heads=4, 
                                 rate_dropout=0.3, direction="left", len_context=len_context, name="transforemer_encoder_l_" + str(i))(left)
        
        right = EncoderLayer(units=256, d_model=2*(size_embedding+4), num_heads=4, 
                                 rate_dropout=0.3, direction="right", len_context=len_context, name="transforemer_encoder_r_" + str(i))(right)
        if i == n_transf_layers - 1:
            left = tkl.LayerNormalization(epsilon=1e-6)(left)
            right = tkl.LayerNormalization(epsilon=1e-6)(right)
            
            left = tkl.Bidirectional(tkl.GRU(2*(size_embedding + 4), return_sequences=True), merge_mode='concat')(left)
            right = tkl.Bidirectional(tkl.GRU(2*(size_embedding + 4), return_sequences=True), merge_mode='concat')(right)
            left = tkl.LayerNormalization(epsilon=1e-6)(left)
            right = tkl.LayerNormalization(epsilon=1e-6)(right)
            
            central_l = tkl.Reshape([1, 4*(size_embedding + 4)])(left[:, -1, :])
            central_r = tkl.Reshape([1, 4*(size_embedding + 4)])(right[:, 0, :])


    # Inverse embedding    
    embedding_i = tf.keras.Sequential()
    embedding_i.add(tkl.Dense(n_octave*12*3))
    embedding_i.add(tkl.Activation("sigmoid"))
    
    aux_out_left = tkl.TimeDistributed(embedding_i, name="central_l")(left)
    aux_out_right = tkl.TimeDistributed(embedding_i, name="central_r")(right)
    
    context = tkl.concatenate([central_l, central_r])
    context =  tkl.Dropout(0.3)(context)
    
    # TODO: Use Dense not tile here
    context = tkl.Dense(n_octave*12*10)(context)
    context = tkl.Reshape([n_octave*12, 10])(context)
    # Wavenet 
    skips = []
    # Input for conv1d has shape=(batches, steps, channels)
    # tf.one_hot output shape = (batches, features, depth) 
    central_features_seq = tf.one_hot(tf.cast(central_features, dtype=tf.int32), depth=3, axis=-1)
    central_features_seq = tkl.Reshape([n_octave*12, 3])(central_features_seq)
    # Adding position informaiton to pich axis
    central_features_seq = Pitch_position_embedding(pitch_range=n_octave*12)(central_features_seq)
    
    ch_0 = 64
    ch_1 = 16
    skip = tkl.Conv1D(ch_1, 1, padding='same')(central_features_seq)
    
    for i in range(n_conv_layers):
        conv_central_t = tkl.Conv1D(ch_0, 2, dilation_rate=2 ** (i), padding='causal')(skip)
        conv_central_s = tkl.Conv1D(ch_0, 2, dilation_rate=2 ** (i), padding='causal')(skip)
        if (i < context_layers):
            conv_context_t = tkl.Conv1D(ch_0, 2, dilation_rate=2 ** (i), padding='causal')(context)
            conv_context_s = tkl.Conv1D(ch_0, 2, dilation_rate=2 ** (i), padding='causal')(context)
            conv_t = tf.keras.activations.tanh(tkl.add([conv_central_t, conv_context_t]))
            conv_s = tf.keras.activations.sigmoid(tkl.add([conv_central_s, conv_context_s]))
        else:
            conv_t = tf.keras.activations.tanh(conv_central_t)
            conv_s = tf.keras.activations.sigmoid(conv_central_s)
            
        conv_prev = tkl.Multiply()([conv_t, conv_s])
        conv = tkl.Conv1D(ch_1, 1, padding='same')(conv_prev)
        skip_new = tkl.Conv1D(ch_1, 1, padding='same')(conv_prev)
        skip = tkl.add([conv, skip])
            
        skip = tkl.Dropout(0.3)(skip)
        skips.append(skip_new)

    out = tf.nn.relu(tkl.add(skips))
    out = tf.nn.relu(tkl.Conv1D(4, 1)(out))
    out = tkl.Flatten()(out)
    out = tkl.Dense(1, activation="sigmoid", name='prediction')(out)
               
    
    model = tf.keras.Model(inputs=[left_features, left_metas,
                                   central_features, central_metas,
                                   right_features, right_metas],
                                   outputs=[out, aux_out_left, aux_out_right])
    return model

def _2way_transformer_wavenet_absolute_pitch_new_embedding(len_context = 8*4*24,
                                                           n_octave = 7,
                                                           size_embedding = 84,
                                                           n_transf_layers = 6,
                                                           n_transf_downsample = 2,
                                                           n_conv_layers = 7,
                                                           context_layers = 3):
    
    left_features = tkl.Input(shape=(len_context, n_octave*12), name='left_features')
    left_metas = tkl.Input(shape=(len_context, 4), name='left_metas')

    right_features = tkl.Input(shape=(len_context, n_octave*12), name='right_features')
    right_metas = tkl.Input(shape=(len_context, 4), name='right_metas')

    central_features = tkl.Input(shape=(1, n_octave*12), name='central_features')
    central_metas = tkl.Input(shape=(1, 4), name='central_metas')
        
    # Embedding for each pitch event
    embedding_p = tf.keras.Sequential()
    embedding_p.add(tkl.Dense(size_embedding))
    
    # Embedding for each timestep
    embedding_t = tf.keras.Sequential()
    embedding_t.add(tkl.Dense(size_embedding))
#    embedding_t.add(tkl.Activation('relu'))

    left = tkl.Reshape([len_context, n_octave*12])(left_features)
    left = tf.one_hot(tf.cast(left, dtype=tf.int32), depth=size_embedding, axis=-1)
    left = embedding_p(left)
    left = tf.reduce_sum(left, axis = 2)
    left = tkl.Dropout(0.3)(left)
    left = tkl.TimeDistributed(embedding_t)(left)
    left = tkl.Reshape([len_context, size_embedding])(left)
    left = tkl.concatenate([left, left_metas])

    right = tkl.Reshape([len_context, n_octave*12])(right_features)
    right = tf.one_hot(tf.cast(right, dtype=tf.int32), depth=size_embedding, axis=-1)
    right = embedding_p(right)
    right = tf.reduce_sum(right, axis = 2)
    right = tkl.Dropout(0.3)(right)
    right = tkl.TimeDistributed(embedding_t)(right)
    right = tkl.Reshape([len_context, size_embedding])(right)
    right = tkl.concatenate([right, right_metas])
    
    # Mask central infromation from transformer
    central = (size_embedding - 1)*tf.ones(shape=(tf.shape(central_features)))
    central = tf.one_hot(tf.cast(central, dtype=tf.int32), depth=size_embedding, axis=-1)
    central = embedding_p(central)
    central = tf.reduce_sum(central, axis = 2)
    central = tkl.TimeDistributed(embedding_t)(central)
    central = tkl.concatenate([central, central_metas])
       
    context = tkl.concatenate([left, central, right], axis = 1)
    context *= tf.math.sqrt(tf.cast(size_embedding + 4, tf.float32))
    context = PositionalEncoding(position=len_context*2 + 1, d_model= size_embedding+4, sum_or_concat="c")(context)
    
    # Central frames is contained in both left and right
    left = context[:, :len_context + 1, :]
    right = context[:, len_context:, :] 
    
    left = tkl.Reshape([len_context + 1 , 2*(size_embedding+4)])(left)
    right = tkl.Reshape([len_context + 1 , 2*(size_embedding+4)])(right)
    
    # Transformer context handling
    # Left and right transforemer
    centrals_l = []
    centrals_r = []
    for i in range(n_transf_layers):
        left = EncoderLayer(units=256, d_model=2*(size_embedding+4), num_heads=4, 
                                 rate_dropout=0.3, direction="left", len_context=len_context, name="transforemer_encoder_l_" + str(i))(left)
        
        right = EncoderLayer(units=256, d_model=2*(size_embedding+4), num_heads=4, 
                                 rate_dropout=0.3, direction="right", len_context=len_context, name="transforemer_encoder_r_" + str(i))(right)
        if i == n_transf_layers - 1:
            left = tkl.LayerNormalization(epsilon=1e-6)(left)
            right = tkl.LayerNormalization(epsilon=1e-6)(right)
            
            left = tkl.Bidirectional(tkl.GRU(2*(size_embedding + 4), return_sequences=True), merge_mode='concat')(left)
            right = tkl.Bidirectional(tkl.GRU(2*(size_embedding + 4), return_sequences=True), merge_mode='concat')(right)
            left = tkl.LayerNormalization(epsilon=1e-6)(left)
            right = tkl.LayerNormalization(epsilon=1e-6)(right)
            
           
            central_l = tkl.Reshape([1, 4*(size_embedding + 4)])(left[:, -1, :])
            central_r = tkl.Reshape([1, 4*(size_embedding + 4)])(right[:, 0, :])


    # Inverse embedding    
    embedding_i = tf.keras.Sequential()
    embedding_i.add(tkl.Dense(n_octave*12*3))
    embedding_i.add(tkl.Activation("sigmoid"))
    
    aux_out_left = tkl.TimeDistributed(embedding_i, name="central_l")(left)
    aux_out_right = tkl.TimeDistributed(embedding_i, name="central_r")(right)
    
    context = tkl.concatenate([central_l, central_r])
    context =  tkl.Dropout(0.3)(context)
    
    # TODO: Use Dense not tile here
    context = tkl.Dense(n_octave*12*10)(context)
    context = tkl.Reshape([n_octave*12, 10])(context)

    # Wavenet 
    skips = []
    # Input for conv1d has shape=(batches, steps, channels)
    # tf.one_hot output shape = (batches, features, depth) 
    central_features_seq = tf.one_hot(tf.cast(central_features, dtype=tf.int32), depth=3, axis=-1)
    central_features_seq = tkl.Reshape([n_octave*12, 3])(central_features_seq)
#    central_features_seq = tf.transpose(central_features, perm=[0, 2, 1])
    # Adding position informaiton to pich axis
    central_features_seq = Pitch_position_embedding(pitch_range=n_octave*12)(central_features_seq)
    
    ch_0 = 64
    ch_1 = 16
    skip = tkl.Conv1D(ch_1, 1, padding='same')(central_features_seq)
    
    for i in range(n_conv_layers):
        conv_central_t = tkl.Conv1D(ch_0, 2, dilation_rate=2 ** (i), padding='causal')(skip)
        conv_central_s = tkl.Conv1D(ch_0, 2, dilation_rate=2 ** (i), padding='causal')(skip)
        if (i < context_layers):
            conv_context_t = tkl.Conv1D(ch_0, 2, dilation_rate=2 ** (i), padding='causal')(context)
            conv_context_s = tkl.Conv1D(ch_0, 2, dilation_rate=2 ** (i), padding='causal')(context)
            conv_t = tf.keras.activations.tanh(tkl.add([conv_central_t, conv_context_t]))
            conv_s = tf.keras.activations.sigmoid(tkl.add([conv_central_s, conv_context_s]))
        else:
            conv_t = tf.keras.activations.tanh(conv_central_t)
            conv_s = tf.keras.activations.sigmoid(conv_central_s)
            
        conv_prev = tkl.Multiply()([conv_t, conv_s])
        conv = tkl.Conv1D(ch_1, 1, padding='same')(conv_prev)
        skip_new = tkl.Conv1D(ch_1, 1, padding='same')(conv_prev)
        skip = tkl.add([conv, skip])
            
        skip = tkl.Dropout(0.3)(skip)
        skips.append(skip_new)

    out = tf.nn.relu(tkl.add(skips))
    out = tf.nn.relu(tkl.Conv1D(4, 1)(out))
    out = tkl.Flatten()(out)
    out = tkl.Dense(1, activation="sigmoid", name='prediction')(out)
              
    
    model = tf.keras.Model(inputs=[left_features, left_metas,
                                   central_features, central_metas,
                                   right_features, right_metas],
                                   outputs=[out, aux_out_left, aux_out_right])
    return model



