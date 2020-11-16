import tensorflow as tf
from tensorflow.python.ops import rnn
#import my_rnn
import pdb

eps = 1e-6
def batch_norm(x, is_training=False):
    return tf.layers.batch_normalization(x, momentum=0.8, training=is_training)

def sense_Global_Local_att(sense = None, context_repres = None, context_mask = None, window_size = 8):

    ''' 
    Args:
        sense: A 3-D tensor with shape (batch, mp, dim), corresonding to the each sense.
        context_repres: A 3-D tensor with shape (batch, L, dim), the lstm encoding context representation.
        context_mask: A 3-D tensor with shape (batch, L, 1), the index mask of the target word to extract the local context from context_repres.
        window_size: integer, size for the local context.
    Returns:
        Two 3-D tensor with shape (batch, mp, dim), one is sense based on global context, the other is sense based on local context.
    '''

    def singel_instance(x):
        s = x[0] # mp, dim, 4, 3
        c = x[1] # L, dim, 5, 3
        m = x[2] # L, 1 -> L
        #print c
        #local context generation
        c_shape = tf.shape(c)
        idx = tf.argmax(m, 0, output_type="int32")
        left_idx = tf.math.maximum(0, idx-window_size)
        right_idx = tf.math.minimum(idx+window_size, c_shape[0])
        indice = tf.range(left_idx[0], right_idx[0])
        local_c = tf.gather(c, indice, axis = 0)# L'', dim 

        _s_c = tf.nn.softmax(tf.matmul(s, c, transpose_b=True), axis = 1) # mp, L
        _s_local_c = tf.nn.softmax(tf.matmul(s, local_c, transpose_b=True), axis = 1) # mp, L''
        #print "Global",_s_c, c # 4,5 5,3
        #print "Local",_s_local_c, local_c # 4,? ?,3
       
        g_c = tf.matmul(_s_c, c) #mp, dim= mp, L * L, dim
        l_c = tf.matmul(_s_local_c, local_c) #mp, dim = mp, L'' * L'', dim
        #print "matmul",g_c, l_c 
        global_s = s + g_c
        local_s = s + l_c
        return (global_s, local_s)

    elems = (sense, context_repres, context_mask)
    output = tf.map_fn(singel_instance, elems, dtype=(tf.float32,tf.float32))
    return output

# For version compatibility
def reduce_sum(input_tensor, axis=None, keepdims=False):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims)


# For version compatibility
def softmax(logits, axis=None):
    try:
        return tf.nn.softmax(logits, axis=axis)
    except:
        return tf.nn.softmax(logits, dim=axis)
def get_shape(inputs, name=None):
    name = "shape" if name is None else name
    with tf.name_scope(name):
        static_shape = inputs.get_shape().as_list()
        dynamic_shape = tf.shape(inputs)
        shape = []
        for i, dim in enumerate(static_shape):
            dim = dim if dim is not None else dynamic_shape[i]
            shape.append(dim)
        return(shape)
def routing(input, b_IJ, num_outputs=10, num_dims=16, iter_routing = 3):
    ''' The routing algorithm.
    Args:
        input: A Tensor with [batch_size, num_caps_l=1152, 1, length(u_i)=8, 1]
               shape, num_caps_l meaning the number of capsule in the layer l.
        num_outputs: the number of output capsules.
        num_dims: the number of dimensions for output capsule.
    Returns:
        A Tensor of shape [batch_size, num_caps_l_plus_1, length(v_j)=16, 1]
        representing the vector output `v_j` in the layer l+1
    Notes:
        u_i represents the vector output of capsule i in the layer l, and
        v_j the vector output of capsule j in the layer l+1.
     '''

    # W: [1, num_caps_i, num_caps_j * len_v_j, len_u_j, 1] (1, 10, 1000, 100, 1)
    input_shape = get_shape(input)#batch_size, Mp, dim, 1
    W = tf.get_variable('Weight', shape=[1, input_shape[1], num_dims * num_outputs] + input_shape[-2:],
                        dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))
    #print (W.shape)
    biases = tf.get_variable('bias', shape=(1, 1, num_outputs, num_dims, 1))

    # Eq.2, calc u_hat
    # Since tf.matmul is a time-consuming op,
    # A better solution is using element-wise multiply, reduce_sum and reshape
    # ops instead. Matmul [a, b] x [b, c] is equal to a series ops as
    # element-wise multiply [a*c, b] * [a*c, b], reduce_sum at axis=1 and
    # reshape to [a, c]
    input = tf.tile(input, [1, 1, num_dims * num_outputs, 1, 1])
    # assert input.get_shape() == [batch_size, 1152, 160, 8, 1]

    u_hat = reduce_sum(W * input, axis=3, keepdims=True)
    u_hat = tf.reshape(u_hat, shape=[-1, input_shape[1], num_outputs, num_dims, 1])
    # assert u_hat.get_shape() == [batch_size, 1152, 10, 16, 1]

    # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    # line 3,for r iterations do
    for r_iter in range(iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            # line 4:
            # => [batch_size, 1152, 10, 1, 1]
            c_IJ = softmax(b_IJ, axis=2)

            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == iter_routing - 1:
                # line 5:
                # weighting u_hat with c_IJ, element-wise in the last two dims
                # => [batch_size, 1152, 10, 16, 1]
                s_J = tf.multiply(c_IJ, u_hat)
                # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                s_J = reduce_sum(s_J, axis=1, keepdims=True) + biases
                # assert s_J.get_shape() == [batch_size, 1, num_outputs, num_dims, 1]

                # line 6:
                # squash using Eq.1,
                v_J = squash(s_J)
                # assert v_J.get_shape() == [batch_size, 1, 10, 16, 1]
            elif r_iter < iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = reduce_sum(s_J, axis=1, keepdims=True) + biases
                v_J = squash(s_J)

                # line 7:
                # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 1152, 10, 16, 1]
                # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                # batch_size dim, resulting in [1, 1152, 10, 1, 1]
                v_J_tiled = tf.tile(v_J, [1, input_shape[1], 1, 1, 1])
                u_produce_v = reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keepdims=True)
                # assert u_produce_v.get_shape() == [batch_size, 1152, 10, 1, 1]

                # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                b_IJ += u_produce_v

    return(v_J)


def squash(vector):
    '''Squashing function corresponding to Eq. 1
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    vec_squared_norm = reduce_sum(tf.square(vector), -2, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + eps)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)

'''
def sense_selection(gloss, context, W_parameter):

    batch_size = gloss.get_shape().as_list()[0]
    memory_size = gloss.get_shape().as_list()[-1]
    Cin = Ain = gloss# * sense_mask  # [batch_size, max_n_sense, 2*n_units]
    #Bin = tf.reshape(context, [batch_size, memory_size, 1])  # [batch_size, 2*n_units, 1]
    Bin = tf.expand_dims(context, axis =2) 
    Aout = tf.matmul(Ain, Bin)  # [batch_size, max_n_sense, 1]
    #Aout_exp = tf.exp(Aout) #* sense_mask[:, :, :1]
    #p = Aout_exp / tf.reduce_sum(Aout_exp, axis=1, keepdims=True)  # [batch_size, max_n_sense, 1]
    p = tf.nn.softmax(Aout)
    #memory_p.append(tf.squeeze(p))
    Mout = tf.squeeze(tf.matmul(Cin, p, transpose_a=True), axis = 2)  # [batch_size, 2*n_units]
    state = tf.nn.relu(tf.add(Mout, tf.matmul(context, W_parameter)))  # [batch_size, 2*n_units]

    return Mout, state, tf.squeeze(Aout), tf.squeeze(p)
'''

def sense_selection(senses, context, W_parameter):
    senses = batch_norm(senses)
    context = batch_norm(context)
    batch_size = senses.get_shape().as_list()[0]
    sense_size = senses.get_shape().as_list()[-1]
    #print "senses",senses
    #print "context",context
    #pdb.set_trace()
    Cin = Ain = senses  # [batch_size, mp_dim, shared_dim]

    Bin = tf.expand_dims(context, axis =2)  # [batch_size, shared_dim, 1]
    #print "Bin",Bin
    #pdb.set_trace()
    Aout = tf.matmul(Ain, Bin)  # [batch_size, mp_dim, 1]
    #Aout_exp = tf.exp(Aout)
    #p = Aout_exp / tf.reduce_sum(Aout_exp, axis=1, keepdims=True)  # [batch_size, mp_dim, 1]
    p = tf.nn.softmax(Aout, axis = 1)
    #print "p",p
    #pdb.set_trace()
    #memory_p.append(tf.squeeze(p)) #[batch_size, mp_dim] 
    #print "Cin",Cin
    #pdb.set_trace() 
    Sout = tf.squeeze(tf.matmul(Cin, p, transpose_a=True), axis = 2) # [batch_size, shared_dim]
    #print "Sout",Sout
    #print "matmul",tf.matmul(context, W_parameter)
    #pdb.set_trace() 
    state = tf.nn.relu(batch_norm(tf.add(Sout, tf.matmul(context, W_parameter)))) # [batch_size, shared_dim]
    #print "state",state
    #pdb.set_trace()

    #if memory_update_type == 'concat':
    #    state = tf.concat((Mout, context), 1)  # [batch_size, 4*n_units]
    #    state = tf.nn.relu(batch_norm(tf.matmul(state, W_memory)))
    #else:  # linear
    #    state = batch_norm(tf.add(Mout, tf.matmul(context, U_memory)))  # [batch_size, 2*n_units]

    return Sout, state, tf.squeeze(Aout), p

def soft_gate_for_f_b_context(f_context_representation, #diff part
                                      b_context_representation,#sentence part 
                                      output_size, scope=None):
    #pre_context_representation from single BiLSTM with shape (batch, dim); match_representation from BiLSTM in BiMPM with shape (batch, dim);
    #the module aims to make a selection between sentence part by BiMPM matching and diff part by BiLSTM;
    #two representations help to generate a "gate" in order to make a selection between the sentence part and diff part;
    #The formular of calculating Gate is "sentence*gate+diff*(1-gate)"; If gate tends to be 1, then common part is conclusive, or diff part is conclusive
    
    with tf.variable_scope(scope or "gate_selection_layer"):
        highway_1 = tf.get_variable("highway_1", [output_size, output_size], dtype=tf.float32)
        highway_2 = tf.get_variable("highway_2", [output_size, output_size], dtype=tf.float32)
        highway_b = tf.get_variable("highway_b", [output_size], dtype=tf.float32)
        full_w = tf.get_variable("full_w", [output_size, output_size], dtype=tf.float32)
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)

        gate = tf.nn.sigmoid(tf.nn.xw_plus_b(f_context_representation, highway_1, highway_b)+tf.nn.xw_plus_b(b_context_representation, highway_2, highway_b))

        representation_f = tf.nn.tanh(tf.nn.xw_plus_b(f_context_representation, full_w, full_b))#common
        representation_b = tf.nn.tanh(tf.nn.xw_plus_b(b_context_representation, full_w, full_b))
        outputs = tf.add(tf.multiply(representation_b, gate),tf.multiply(representation_f,tf.subtract(1.0, gate)),"representation") 

    return outputs
def semantic_under_condition(word_embedding_multiperspective, word_Representatioin_multiperspective, w_list):
     
    w_1, b_1, w_2, b_2, w_attention = w_list

    #w_1 = tf.get_variable("mapping_w_1", [1,MP_dim], dtype=tf.float32) #1*P
    #b_1 = tf.get_variable("mapping_b_1", [shared_dim], dtype=tf.float32)
    #w_2 = tf.get_variable("mapping_w_2", [shared_dim, shared_dim], dtype=tf.float32) # D*D
    #b_2 = tf.get_variable("mapping_b_2", [shared_dim], dtype=tf.float32)
    #w_attention = tf.get_variable("attention_w", [context_lstm_dim*2, 1], dtype=tf.float32) # D*1

    #pdb.set_trace()
    def singel_instance(x):
        return tf.matmul(w_1, x)+b_1
    semantic_embedding = tf.map_fn(singel_instance, word_embedding_multiperspective, dtype=tf.float32) #batch * 1 * D
    #batch * P * D
    def singel_instance_condition(x):
        return tf.matmul(x,w_2)+b_2
    #semantic_Representation = tf.matmul(word_embedding_multiperspective, w_2)+b_2
    semantic_Representation = tf.map_fn(singel_instance_condition, word_Representatioin_multiperspective, dtype=tf.float32)

    def singel_instance_attention(x):
        return tf.matmul(x,w_attention)
    weights_attention = tf.nn.softmax(tf.map_fn(singel_instance_attention, semantic_Representation, dtype=tf.float32), axis = 1) #batch, P, 1

    #weights_attention = tf.nn.softmax(tf.matmul(semantic_Representation, w_attention), axis = 1) #batch, P, 1
    attention_R = tf.reduce_sum(tf.multiply(semantic_Representation, weights_attention), axis = 1) # batch, P, D-->batch, 1, D
          
    embedding_based_Representation = tf.reduce_sum(semantic_embedding,axis = 1, keepdims = False)+attention_R # batch, 1, D
    #print semantic_embedding # batch*1*D
    #print semantic_Representation # batch*P*D
    #print weights_attention
    #print attention_R
    #print embedding_based_Representation
    #pdb.set_trace()
    return embedding_based_Representation

def cosine_distance(y1,y2):
    # y1 [....,a, 1, d]
    # y2 [....,1, b, d]
#     cosine_numerator = T.sum(y1*y2, axis=-1)
    cosine_numerator = tf.reduce_sum(tf.multiply(y1, y2), axis=-1)
#     y1_norm = T.sqrt(T.maximum(T.sum(T.sqr(y1), axis=-1), eps)) #be careful while using T.sqrt(), like in the cases of Euclidean distance, cosine similarity, for the gradient of T.sqrt() at 0 is undefined, we should add an Eps or use T.maximum(original, eps) in the sqrt.
    y1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1), axis=-1), eps)) 
    y2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y2), axis=-1), eps)) 
    return cosine_numerator / y1_norm / y2_norm

def cal_relevancy_matrix(in_question_repres, in_passage_repres):
    in_question_repres_tmp = tf.expand_dims(in_question_repres, 1) # [batch_size, 1, question_len, dim]
    in_passage_repres_tmp = tf.expand_dims(in_passage_repres, 2) # [batch_size, passage_len, 1, dim]
    relevancy_matrix = cosine_distance(in_question_repres_tmp,in_passage_repres_tmp) # [batch_size, passage_len, question_len]
    return relevancy_matrix

def mask_relevancy_matrix(relevancy_matrix, question_mask, passage_mask):
    # relevancy_matrix: [batch_size, passage_len, question_len]
    # question_mask: [batch_size, question_len]
    # passage_mask: [batch_size, passsage_len]
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(question_mask, 1))
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(passage_mask, 2))
    return relevancy_matrix

def cal_cosine_weighted_question_representation(question_representation, cosine_matrix, normalize=False):
    # question_representation: [batch_size, question_len, dim]
    # cosine_matrix: [batch_size, passage_len, question_len]
    if normalize: cosine_matrix = tf.nn.softmax(cosine_matrix)
    expanded_cosine_matrix = tf.expand_dims(cosine_matrix, axis=-1) # [batch_size, passage_len, question_len, 'x']
    weighted_question_words = tf.expand_dims(question_representation, axis=1) # [batch_size, 'x', question_len, dim]
    weighted_question_words = tf.reduce_sum(tf.multiply(weighted_question_words, expanded_cosine_matrix), axis=2)# [batch_size, passage_len, dim]
    if not normalize:
        weighted_question_words = tf.div(weighted_question_words, tf.expand_dims(tf.add(tf.reduce_sum(cosine_matrix, axis=-1),eps),axis=-1))
    return weighted_question_words # [batch_size, passage_len, dim]

def multi_perspective_expand_for_3D(in_tensor, decompose_params):
    in_tensor = tf.expand_dims(in_tensor, axis=2) #[batch_size, passage_len, 'x', dim]
    decompose_params = tf.expand_dims(tf.expand_dims(decompose_params, axis=0), axis=0) # [1, 1, decompse_dim, dim]
    return tf.multiply(in_tensor, decompose_params)#[batch_size, passage_len, decompse_dim, dim]

def multi_perspective_expand_for_2D(in_tensor, decompose_params):
    in_tensor = tf.expand_dims(in_tensor, axis=1) #[batch_size, 'x', dim]
    decompose_params = tf.expand_dims(decompose_params, axis=0) # [1, decompse_dim, dim]
    return tf.multiply(in_tensor, decompose_params) # [batch_size, decompse_dim, dim]

def multi_perspective_expand_for_1D(in_tensor, decompose_params):
    in_tensor = tf.expand_dims(in_tensor, axis=0) #['x', dim]
    return tf.multiply(in_tensor, decompose_params) # [decompse_dim, dim]


def cal_full_matching_bak(passage_representation, full_question_representation, decompose_params):
    # passage_representation: [batch_size, passage_len, dim]
    # full_question_representation: [batch_size, dim]
    # decompose_params: [decompose_dim, dim]
    mp_passage_rep = multi_perspective_expand_for_3D(passage_representation, decompose_params) # [batch_size, passage_len, decompse_dim, dim]
    mp_full_question_rep = multi_perspective_expand_for_2D(full_question_representation, decompose_params) # [batch_size, decompse_dim, dim]
    return cosine_distance(mp_passage_rep, tf.expand_dims(mp_full_question_rep, axis=1)) #[batch_size, passage_len, decompse_dim]

def cal_full_matching(passage_representation, full_question_representation, decompose_params):
    # passage_representation: [batch_size, passage_len, dim]
    # full_question_representation: [batch_size, dim]
    # decompose_params: [decompose_dim, dim]
    def singel_instance(x):
        p = x[0]
        q = x[1]
        # p: [pasasge_len, dim], q: [dim]
        p = multi_perspective_expand_for_2D(p, decompose_params) # [pasasge_len, decompose_dim, dim]
        q = multi_perspective_expand_for_1D(q, decompose_params) # [decompose_dim, dim]
        q = tf.expand_dims(q, 0) # [1, decompose_dim, dim]
        return cosine_distance(p, q) # [passage_len, decompose]
    elems = (passage_representation, full_question_representation)
    return tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, passage_len, decompse_dim]
    
def cal_maxpooling_matching_bak(passage_rep, question_rep, decompose_params):
    # passage_representation: [batch_size, passage_len, dim]
    # qusetion_representation: [batch_size, question_len, dim]
    # decompose_params: [decompose_dim, dim]
    passage_rep = multi_perspective_expand_for_3D(passage_rep, decompose_params) # [batch_size, passage_len, decompse_dim, dim]
    question_rep = multi_perspective_expand_for_3D(question_rep, decompose_params) # [batch_size, question_len, decompse_dim, dim]

    passage_rep = tf.expand_dims(passage_rep, 2) # [batch_size, passage_len, 1, decompse_dim, dim]
    question_rep = tf.expand_dims(question_rep, 1) # [batch_size, 1, question_len, decompse_dim, dim]
    matching_matrix = cosine_distance(passage_rep,question_rep) # [batch_size, passage_len, question_len, decompse_dim]
    return tf.concat( axis = 2, values = [tf.reduce_max(matching_matrix, axis=2), tf.reduce_mean(matching_matrix, axis=2)])# [batch_size, passage_len, 2*decompse_dim]

def cal_maxpooling_matching(passage_rep, question_rep, decompose_params):
    # passage_representation: [batch_size, passage_len, dim]
    # qusetion_representation: [batch_size, question_len, dim]
    # decompose_params: [decompose_dim, dim]
    
    def singel_instance(x):
        p = x[0]
        q = x[1]
        # p: [pasasge_len, dim], q: [question_len, dim]
        p = multi_perspective_expand_for_2D(p, decompose_params) # [pasasge_len, decompose_dim, dim]
        q = multi_perspective_expand_for_2D(q, decompose_params) # [question_len, decompose_dim, dim]
        p = tf.expand_dims(p, 1) # [pasasge_len, 1, decompose_dim, dim]
        q = tf.expand_dims(q, 0) # [1, question_len, decompose_dim, dim]
        return cosine_distance(p, q) # [passage_len, question_len, decompose]
    elems = (passage_rep, question_rep)
    matching_matrix = tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, passage_len, question_len, decompse_dim]
    return tf.concat( axis = 2, values = [tf.reduce_max(matching_matrix, axis=2), tf.reduce_mean(matching_matrix, axis=2)])# [batch_size, passage_len, 2*decompse_dim]

def cal_maxpooling_matching_for_word(passage_rep, question_rep, decompose_params):
    # passage_representation: [batch_size, passage_len, dim]
    # qusetion_representation: [batch_size, question_len, dim]
    # decompose_params: [decompose_dim, dim]
    
    def singel_instance(x):
        p = x[0]
        q = x[1]
        q = multi_perspective_expand_for_2D(q, decompose_params) # [question_len, decompose_dim, dim]
        # p: [pasasge_len, dim], q: [question_len, dim]
        def single_instance_2(y):
            # y: [dim]
            y = multi_perspective_expand_for_1D(y, decompose_params) #[decompose_dim, dim]
            y = tf.expand_dims(y, 0) # [1, decompose_dim, dim]
            matching_matrix = cosine_distance(y, q)#[question_len, decompose_dim]
            return tf.concat( axis = 0, values = [tf.reduce_max(matching_matrix, axis=0), tf.reduce_mean(matching_matrix, axis=0)]) #[2*decompose_dim]
        return tf.map_fn(single_instance_2, p, dtype=tf.float32) # [passage_len, 2*decompse_dim]
    elems = (passage_rep, question_rep)
    return tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, passage_len, 2*decompse_dim]


def cal_attentive_matching(passage_rep, att_question_rep, decompose_params):
    # passage_rep: [batch_size, passage_len, dim]
    # att_question_rep: [batch_size, passage_len, dim]
    def singel_instance(x):
        p = x[0]
        q = x[1]
        # p: [pasasge_len, dim], q: [pasasge_len, dim]
        p = multi_perspective_expand_for_2D(p, decompose_params) # [pasasge_len, decompose_dim, dim]
        q = multi_perspective_expand_for_2D(q, decompose_params) # [pasasge_len, decompose_dim, dim]
        return cosine_distance(p, q) # [pasasge_len, decompose_dim]

    elems = (passage_rep, att_question_rep)
    return tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, passage_len, decompse_dim]

def cross_entropy(logits, truth, mask):
    # logits: [batch_size, passage_len]
    # truth: [batch_size, passage_len]
    # mask: [batch_size, passage_len]

#     xdev = x - x.max()
#     return xdev - T.log(T.sum(T.exp(xdev)))
    logits = tf.multiply(logits, mask)
    xdev = tf.sub(logits, tf.expand_dims(tf.reduce_max(logits, 1), -1))
    log_predictions = tf.sub(xdev, tf.expand_dims(tf.log(tf.reduce_sum(tf.exp(xdev),-1)),-1))
#     return -T.sum(targets * log_predictions)
    result = tf.multiply(tf.multiply(truth, log_predictions), mask) # [batch_size, passage_len]
    return tf.multiply(-1.0,tf.reduce_sum(result, -1)) # [batch_size]
    
def highway_layer(in_val, output_size, scope=None):
    # in_val: [batch_size, passage_len, dim]
    input_shape = tf.shape(in_val)
    batch_size = input_shape[0]
    passage_len = input_shape[1]
#     feat_dim = input_shape[2]
    in_val = tf.reshape(in_val, [batch_size * passage_len, output_size])
    with tf.variable_scope(scope or "highway_layer"):
        highway_w = tf.get_variable("highway_w", [output_size, output_size], dtype=tf.float32)
        highway_b = tf.get_variable("highway_b", [output_size], dtype=tf.float32)
        full_w = tf.get_variable("full_w", [output_size, output_size], dtype=tf.float32)
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        trans = tf.nn.tanh(tf.nn.xw_plus_b(in_val, full_w, full_b))
        gate = tf.nn.sigmoid(tf.nn.xw_plus_b(in_val, highway_w, highway_b))
        outputs = tf.add(tf.multiply(trans, gate), tf.multiply(in_val, tf.sub(1.0, gate)), "y")
    outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])
    return outputs

def multi_highway_layer(in_val, output_size, num_layers, scope=None):
    scope_name = 'highway_layer'
    if scope is not None: scope_name = scope
    for i in xrange(num_layers):
        cur_scope_name = scope_name + "-{}".format(i)
        in_val = highway_layer(in_val, output_size, scope=cur_scope_name)
    return in_val

def cal_max_question_representation(question_representation, cosine_matrix):
    # question_representation: [batch_size, question_len, dim]
    # cosine_matrix: [batch_size, passage_len, question_len]
    question_index = tf.argmax(cosine_matrix, 2) # [batch_size, passage_len]
    def singel_instance(x):
        q = x[0]
        c = x[1]
        return tf.gather(q, c)
    elems = (question_representation, question_index)
    return tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, passage_len, dim]

def cal_linear_decomposition_representation(passage_representation, passage_lengths, cosine_matrix,is_training, 
                                            lex_decompsition_dim, dropout_rate):
    # passage_representation: [batch_size, passage_len, dim]
    # cosine_matrix: [batch_size, passage_len, question_len]
    passage_similarity = tf.reduce_max(cosine_matrix, 2)# [batch_size, passage_len]
    similar_weights = tf.expand_dims(passage_similarity, -1) # [batch_size, passage_len, 1]
    dissimilar_weights = tf.subtract(1.0, similar_weights)
    similar_component = tf.multiply(passage_representation, similar_weights)
    dissimilar_component = tf.multiply(passage_representation, dissimilar_weights)
    all_component = tf.concat( axis =2, values = [similar_component, dissimilar_component])
    if lex_decompsition_dim==-1:
        return all_component
    with tf.variable_scope('lex_decomposition'):
        lex_lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(lex_decompsition_dim)
        lex_lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(lex_decompsition_dim)
        if is_training:
            lex_lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(lex_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
            lex_lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(lex_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
        lex_lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([lex_lstm_cell_fw])
        lex_lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([lex_lstm_cell_bw])

        (lex_features_fw, lex_features_bw), _ = rnn.bidirectional_dynamic_rnn(
                    lex_lstm_cell_fw, lex_lstm_cell_bw, all_component, dtype=tf.float32, sequence_length=passage_lengths)

        lex_features = tf.concat( axis =2, values = [lex_features_fw, lex_features_bw])
    return lex_features


def match_passage_with_question(passage_context_representation_fw, passage_context_representation_bw, mask,
                                question_context_representation_fw, question_context_representation_bw,question_mask,
                                MP_dim, context_lstm_dim, scope=None,
                                with_full_match=True, with_maxpool_match=True, with_attentive_match=True, with_max_attentive_match=True):

    all_question_aware_representatins = []
    dim = 0
    with tf.variable_scope(scope or "match_passage_with_question"):
        fw_question_full_rep = question_context_representation_fw[:,-1,:]
        bw_question_full_rep = question_context_representation_bw[:,0,:]

        question_context_representation_fw = tf.multiply(question_context_representation_fw, tf.expand_dims(question_mask,-1))
        question_context_representation_bw = tf.multiply(question_context_representation_bw, tf.expand_dims(question_mask,-1))
        passage_context_representation_fw = tf.multiply(passage_context_representation_fw, tf.expand_dims(mask,-1))
        passage_context_representation_bw = tf.multiply(passage_context_representation_bw, tf.expand_dims(mask,-1))

        forward_relevancy_matrix = cal_relevancy_matrix(question_context_representation_fw, passage_context_representation_fw)
        forward_relevancy_matrix = mask_relevancy_matrix(forward_relevancy_matrix, question_mask, mask)

        backward_relevancy_matrix = cal_relevancy_matrix(question_context_representation_bw, passage_context_representation_bw)
        backward_relevancy_matrix = mask_relevancy_matrix(backward_relevancy_matrix, question_mask, mask)
        if MP_dim > 0:
            if with_full_match:
                # forward Full-Matching: passage_context_representation_fw vs question_context_representation_fw[-1]
                fw_full_decomp_params = tf.get_variable("forward_full_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                fw_full_match_rep = cal_full_matching(passage_context_representation_fw, fw_question_full_rep, fw_full_decomp_params)
                all_question_aware_representatins.append(fw_full_match_rep)
                dim += MP_dim

                # backward Full-Matching: passage_context_representation_bw vs question_context_representation_bw[0]
                bw_full_decomp_params = tf.get_variable("backward_full_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                bw_full_match_rep = cal_full_matching(passage_context_representation_bw, bw_question_full_rep, bw_full_decomp_params)
                all_question_aware_representatins.append(bw_full_match_rep)
                dim += MP_dim

            if with_maxpool_match:
                # forward Maxpooling-Matching
                fw_maxpooling_decomp_params = tf.get_variable("forward_maxpooling_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                fw_maxpooling_rep = cal_maxpooling_matching(passage_context_representation_fw, question_context_representation_fw, fw_maxpooling_decomp_params)
                all_question_aware_representatins.append(fw_maxpooling_rep)
                dim += 2*MP_dim
                # backward Maxpooling-Matching
                bw_maxpooling_decomp_params = tf.get_variable("backward_maxpooling_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                bw_maxpooling_rep = cal_maxpooling_matching(passage_context_representation_bw, question_context_representation_bw, bw_maxpooling_decomp_params)
                all_question_aware_representatins.append(bw_maxpooling_rep)
                dim += 2*MP_dim
            
            if with_attentive_match:
                # forward attentive-matching
                # forward weighted question representation: [batch_size, question_len, passage_len] [batch_size, question_len, context_lstm_dim]
                att_question_fw_contexts = cal_cosine_weighted_question_representation(question_context_representation_fw, forward_relevancy_matrix)
                fw_attentive_decomp_params = tf.get_variable("forward_attentive_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                fw_attentive_rep = cal_attentive_matching(passage_context_representation_fw, att_question_fw_contexts, fw_attentive_decomp_params)
                all_question_aware_representatins.append(fw_attentive_rep)
                dim += MP_dim

                # backward attentive-matching
                # backward weighted question representation
                att_question_bw_contexts = cal_cosine_weighted_question_representation(question_context_representation_bw, backward_relevancy_matrix)
                bw_attentive_decomp_params = tf.get_variable("backward_attentive_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                bw_attentive_rep = cal_attentive_matching(passage_context_representation_bw, att_question_bw_contexts, bw_attentive_decomp_params)
                all_question_aware_representatins.append(bw_attentive_rep)
                dim += MP_dim
            
            if with_max_attentive_match:
                # forward max attentive-matching
                max_att_fw = cal_max_question_representation(question_context_representation_fw, forward_relevancy_matrix)
                fw_max_att_decomp_params = tf.get_variable("fw_max_att_decomp_params", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                fw_max_attentive_rep = cal_attentive_matching(passage_context_representation_fw, max_att_fw, fw_max_att_decomp_params)
                all_question_aware_representatins.append(fw_max_attentive_rep)
                dim += MP_dim

                # backward max attentive-matching
                max_att_bw = cal_max_question_representation(question_context_representation_bw, backward_relevancy_matrix)
                bw_max_att_decomp_params = tf.get_variable("bw_max_att_decomp_params", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                bw_max_attentive_rep = cal_attentive_matching(passage_context_representation_bw, max_att_bw, bw_max_att_decomp_params)
                all_question_aware_representatins.append(bw_max_attentive_rep)
                dim += MP_dim

        all_question_aware_representatins.append(tf.reduce_max(forward_relevancy_matrix, axis=2,keepdims=True))
        all_question_aware_representatins.append(tf.reduce_mean(forward_relevancy_matrix, axis=2,keepdims=True))
        all_question_aware_representatins.append(tf.reduce_max(backward_relevancy_matrix, axis=2,keepdims=True))
        all_question_aware_representatins.append(tf.reduce_mean(backward_relevancy_matrix, axis=2,keepdims=True))
        dim += 4
    return (all_question_aware_representatins, dim)
        
def unidirectional_matching(in_question_repres, in_passage_repres,question_lengths, passage_lengths,
                            question_mask, mask, MP_dim, input_dim, with_filter_layer, context_layer_num,
                            context_lstm_dim,is_training,dropout_rate,with_match_highway,aggregation_layer_num,
                            aggregation_lstm_dim,highway_layer_num,with_aggregation_highway,with_lex_decomposition, lex_decompsition_dim,
                            with_full_match=True, with_maxpool_match=True, with_attentive_match=True, with_max_attentive_match=True):
    # ======Filter layer======
    cosine_matrix = cal_relevancy_matrix(in_question_repres, in_passage_repres)
    cosine_matrix = mask_relevancy_matrix(cosine_matrix, question_mask, mask)
    raw_in_passage_repres = in_passage_repres
    if with_filter_layer:
        relevancy_matrix = cosine_matrix # [batch_size, passage_len, question_len]
        relevancy_degrees = tf.reduce_max(relevancy_matrix, axis=2) # [batch_size, passage_len]
        relevancy_degrees = tf.expand_dims(relevancy_degrees,axis=-1) # [batch_size, passage_len, 'x']
        in_passage_repres = tf.multiply(in_passage_repres, relevancy_degrees)
        
    # =======Context Representation Layer & Multi-Perspective matching layer=====
    all_question_aware_representatins = []
    # max and mean pooling at word level
    all_question_aware_representatins.append(tf.reduce_max(cosine_matrix, axis=2,keepdims=True))
    all_question_aware_representatins.append(tf.reduce_mean(cosine_matrix, axis=2,keepdims=True))
    question_aware_dim = 2
    
    if MP_dim>0:
        if with_max_attentive_match:
            # max_att word level
            max_att = cal_max_question_representation(in_question_repres, cosine_matrix)
            max_att_decomp_params = tf.get_variable("max_att_decomp_params", shape=[MP_dim, input_dim], dtype=tf.float32)
            max_attentive_rep = cal_attentive_matching(raw_in_passage_repres, max_att, max_att_decomp_params)
            all_question_aware_representatins.append(max_attentive_rep)
            question_aware_dim += MP_dim
    
    # lex decomposition
    if with_lex_decomposition:
        lex_decomposition = cal_linear_decomposition_representation(raw_in_passage_repres, passage_lengths, cosine_matrix,is_training, 
                                            lex_decompsition_dim, dropout_rate)
        all_question_aware_representatins.append(lex_decomposition)
        if lex_decompsition_dim== -1: question_aware_dim += 2 * input_dim
        else: question_aware_dim += 2* lex_decompsition_dim
        
    with tf.variable_scope('context_MP_matching'):
        for i in xrange(context_layer_num):
            with tf.variable_scope('layer-{}'.format(i)):
                with tf.variable_scope('context_represent'):
                    # parameters
                    context_lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(context_lstm_dim)
                    context_lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(context_lstm_dim)
                    if is_training:
                        context_lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
                        context_lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
                    context_lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_fw])
                    context_lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_bw])

                    # question representation
                    (question_context_representation_fw, question_context_representation_bw), _ = my_rnn.bidirectional_dynamic_rnn(
                                        context_lstm_cell_fw, context_lstm_cell_bw, in_question_repres, dtype=tf.float32, 
                                        sequence_length=question_lengths) # [batch_size, question_len, context_lstm_dim]
                    in_question_repres = tf.concat( axis =2, values = [question_context_representation_fw, question_context_representation_bw])

                    # passage representation
                    tf.get_variable_scope().reuse_variables()
                    (passage_context_representation_fw, passage_context_representation_bw), _ = my_rnn.bidirectional_dynamic_rnn(
                                        context_lstm_cell_fw, context_lstm_cell_bw, in_passage_repres, dtype=tf.float32, 
                                        sequence_length=passage_lengths) # [batch_size, passage_len, context_lstm_dim]
                    in_passage_repres = tf.concat( axis =2, values = [passage_context_representation_fw, passage_context_representation_bw])
                    
                # Multi-perspective matching
                with tf.variable_scope('MP_matching'):
                    (matching_vectors, matching_dim) = match_passage_with_question(passage_context_representation_fw, 
                                passage_context_representation_bw, mask,
                                question_context_representation_fw, question_context_representation_bw,question_mask,
                                MP_dim, context_lstm_dim, scope=None,
                                with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                                with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match)
                    all_question_aware_representatins.extend(matching_vectors)
                    question_aware_dim += matching_dim
        
    all_question_aware_representatins = tf.concat( axis =2, values = all_question_aware_representatins) # [batch_size, passage_len, dim]

    if is_training:
        all_question_aware_representatins = tf.nn.dropout(all_question_aware_representatins, (1 - dropout_rate))
    else:
        all_question_aware_representatins = tf.multiply(all_question_aware_representatins, (1 - dropout_rate))
        
    # ======Highway layer======
    if with_match_highway:
        with tf.variable_scope("matching_highway"):
            all_question_aware_representatins = multi_highway_layer(all_question_aware_representatins, question_aware_dim,highway_layer_num)
        
    #========Aggregation Layer======
    aggregation_representation = []
    aggregation_dim = 0
    aggregation_input = all_question_aware_representatins
    with tf.variable_scope('aggregation_layer'):
        for i in xrange(aggregation_layer_num):
            with tf.variable_scope('layer-{}'.format(i)):
                aggregation_lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(aggregation_lstm_dim)
                aggregation_lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(aggregation_lstm_dim)
                if is_training:
                    aggregation_lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(aggregation_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
                    aggregation_lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(aggregation_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
                aggregation_lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([aggregation_lstm_cell_fw])
                aggregation_lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([aggregation_lstm_cell_bw])

                cur_aggregation_representation, _ = my_rnn.bidirectional_dynamic_rnn(
                        aggregation_lstm_cell_fw, aggregation_lstm_cell_bw, aggregation_input, 
                        dtype=tf.float32, sequence_length=passage_lengths)

                fw_rep = cur_aggregation_representation[0][:,-1,:]
                bw_rep = cur_aggregation_representation[1][:,0,:]
                aggregation_representation.append(fw_rep)
                aggregation_representation.append(bw_rep)
                aggregation_dim += 2* aggregation_lstm_dim
                aggregation_input = tf.concat( axis =2, values = cur_aggregation_representation)# [batch_size, passage_len, 2*aggregation_lstm_dim]
        
    #
    aggregation_representation = tf.concat( axis =1, values = aggregation_representation) # [batch_size, aggregation_dim]

    # ======Highway layer======
    if with_aggregation_highway:
        with tf.variable_scope("aggregation_highway"):
            agg_shape = tf.shape(aggregation_representation)
            batch_size = agg_shape[0]
            aggregation_representation = tf.reshape(aggregation_representation, [1, batch_size, aggregation_dim])
            aggregation_representation = multi_highway_layer(aggregation_representation, aggregation_dim, highway_layer_num)
            aggregation_representation = tf.reshape(aggregation_representation, [batch_size, aggregation_dim])

    return (aggregation_representation, aggregation_dim)
        
def bilateral_match_func1(in_question_repres, in_passage_repres,
                        question_lengths, passage_lengths, question_mask, mask, MP_dim, input_dim, 
                        with_filter_layer, context_layer_num, context_lstm_dim,is_training,dropout_rate,
                        with_match_highway,aggregation_layer_num, aggregation_lstm_dim,highway_layer_num,
                        with_aggregation_highway,with_lex_decomposition,lex_decompsition_dim,
                        with_full_match=True, with_maxpool_match=True, with_attentive_match=True, with_max_attentive_match=True,
                        with_left_match=True, with_right_match=True):
    init_scale = 0.01
    initializer = tf.random_uniform_initializer(-init_scale, init_scale)
    match_representation = []
    match_dim = 0
        
    reuse_match_params = None
    if with_left_match:
        reuse_match_params = True
        with tf.name_scope("match_passsage"):
            with tf.variable_scope("MP-Match", reuse=None, initializer=initializer):
                (passage_match_representation, passage_match_dim) = unidirectional_matching(in_question_repres, in_passage_repres,
                            question_lengths, passage_lengths, question_mask, mask, MP_dim, input_dim, 
                            with_filter_layer, context_layer_num, context_lstm_dim,is_training,dropout_rate,
                            with_match_highway,aggregation_layer_num, aggregation_lstm_dim,highway_layer_num,
                            with_aggregation_highway,with_lex_decomposition,lex_decompsition_dim,
                            with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                            with_attentive_match=with_attentive_match,
                            with_max_attentive_match=with_max_attentive_match)
                match_representation.append(passage_match_representation)
                match_dim += passage_match_dim
    if with_right_match:
        with tf.name_scope("match_question"):
            with tf.variable_scope("MP-Match", reuse=reuse_match_params, initializer=initializer):
                (question_match_representation, question_match_dim) = unidirectional_matching(in_passage_repres, in_question_repres, 
                            passage_lengths, question_lengths, mask, question_mask, MP_dim, input_dim, 
                            with_filter_layer, context_layer_num, context_lstm_dim,is_training,dropout_rate,
                            with_match_highway,aggregation_layer_num, aggregation_lstm_dim,highway_layer_num,
                            with_aggregation_highway, with_lex_decomposition,lex_decompsition_dim,
                            with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                            with_attentive_match=with_attentive_match,
                            with_max_attentive_match=with_max_attentive_match)
                match_representation.append(question_match_representation)
                match_dim += question_match_dim
    match_representation = tf.concat( axis =1, values = match_representation)
    return (match_representation, match_dim)



def bilateral_match_func2(in_question_repres, in_passage_repres,
                        question_lengths, passage_lengths, question_mask, mask, MP_dim, input_dim, 
                        with_filter_layer, context_layer_num, context_lstm_dim,is_training,dropout_rate,
                        with_match_highway,aggregation_layer_num, aggregation_lstm_dim,highway_layer_num,
                        with_aggregation_highway,with_lex_decomposition,lex_decompsition_dim,
                        with_full_match=True, with_maxpool_match=True, with_attentive_match=True, with_max_attentive_match=True,
                        with_left_match=True, with_right_match=True, with_mean_aggregation=True):

    cosine_matrix = cal_relevancy_matrix(in_question_repres, in_passage_repres) # [batch_size, passage_len, question_len]
    cosine_matrix = mask_relevancy_matrix(cosine_matrix, question_mask, mask)
    cosine_matrix_transpose = tf.transpose(cosine_matrix, perm=[0,2,1])# [batch_size, question_len, passage_len]

    # ====word level matching======
    question_aware_representatins = []
    question_aware_dim = 0
    passage_aware_representatins = []
    passage_aware_dim = 0

    # max and mean pooling at word level
    question_aware_representatins.append(tf.reduce_max(cosine_matrix, axis=2,keepdims=True)) # [batch_size, passage_length, 1]
    question_aware_representatins.append(tf.reduce_mean(cosine_matrix, axis=2,keepdims=True))# [batch_size, passage_length, 1]
    question_aware_dim += 2
    passage_aware_representatins.append(tf.reduce_max(cosine_matrix_transpose, axis=2,keepdims=True))# [batch_size, question_len, 1]
    passage_aware_representatins.append(tf.reduce_mean(cosine_matrix_transpose, axis=2,keepdims=True))# [batch_size, question_len, 1]
    passage_aware_dim += 2
    

    if MP_dim>0:
        if with_max_attentive_match:
            # max_att word level
            qa_max_att = cal_max_question_representation(in_question_repres, cosine_matrix)# [batch_size, passage_len, dim]
            qa_max_att_decomp_params = tf.get_variable("qa_word_max_att_decomp_params", shape=[MP_dim, input_dim], dtype=tf.float32)
            qa_max_attentive_rep = cal_attentive_matching(in_passage_repres, qa_max_att, qa_max_att_decomp_params)# [batch_size, passage_len, decompse_dim]
            question_aware_representatins.append(qa_max_attentive_rep)
            question_aware_dim += MP_dim

            pa_max_att = cal_max_question_representation(in_passage_repres, cosine_matrix_transpose)# [batch_size, question_len, dim]
            pa_max_att_decomp_params = tf.get_variable("pa_word_max_att_decomp_params", shape=[MP_dim, input_dim], dtype=tf.float32)
            pa_max_attentive_rep = cal_attentive_matching(in_question_repres, pa_max_att, pa_max_att_decomp_params)# [batch_size, question_len, decompse_dim]
            passage_aware_representatins.append(pa_max_attentive_rep)
            passage_aware_dim += MP_dim

    with tf.variable_scope('context_MP_matching'):
        for i in xrange(context_layer_num): # support multiple context layer
            with tf.variable_scope('layer-{}'.format(i)):
                with tf.variable_scope('context_represent'):
                    # parameters
                    context_lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(context_lstm_dim)
                    context_lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(context_lstm_dim)
                    if is_training:
                        context_lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
                        context_lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
                    context_lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_fw])
                    context_lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_bw])

                    # question representation
                    (question_context_representation_fw, question_context_representation_bw), _ = my_rnn.bidirectional_dynamic_rnn(
                                        context_lstm_cell_fw, context_lstm_cell_bw, in_question_repres, dtype=tf.float32, 
                                        sequence_length=question_lengths) # [batch_size, question_len, context_lstm_dim]
                    in_question_repres = tf.concat( axis =2, values = [question_context_representation_fw, question_context_representation_bw])

                    # passage representation
                    tf.get_variable_scope().reuse_variables()
                    (passage_context_representation_fw, passage_context_representation_bw), _ = my_rnn.bidirectional_dynamic_rnn(
                                        context_lstm_cell_fw, context_lstm_cell_bw, in_passage_repres, dtype=tf.float32, 
                                        sequence_length=passage_lengths) # [batch_size, passage_len, context_lstm_dim]
                    in_passage_repres = tf.concat( axis =2, values = [passage_context_representation_fw, passage_context_representation_bw])
                    
                # Multi-perspective matching
                with tf.variable_scope('left_MP_matching'):
                    (matching_vectors, matching_dim) = match_passage_with_question(passage_context_representation_fw, 
                                passage_context_representation_bw, mask,
                                question_context_representation_fw, question_context_representation_bw,question_mask,
                                MP_dim, context_lstm_dim, scope=None,
                                with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                                with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match)
                    question_aware_representatins.extend(matching_vectors)
                    question_aware_dim += matching_dim
                
                with tf.variable_scope('right_MP_matching'):
                    (matching_vectors, matching_dim) = match_passage_with_question(question_context_representation_fw, 
                                question_context_representation_bw, question_mask,
                                passage_context_representation_fw, passage_context_representation_bw,mask,
                                MP_dim, context_lstm_dim, scope=None,
                                with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                                with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match)
                    passage_aware_representatins.extend(matching_vectors)
                    passage_aware_dim += matching_dim
        

        
    question_aware_representatins = tf.concat( axis =2, values = question_aware_representatins) # [batch_size, passage_len, question_aware_dim]
    passage_aware_representatins = tf.concat( axis =2, values = passage_aware_representatins) # [batch_size, question_len, question_aware_dim]

    if is_training:
        question_aware_representatins = tf.nn.dropout(question_aware_representatins, (1 - dropout_rate))
        passage_aware_representatins = tf.nn.dropout(passage_aware_representatins, (1 - dropout_rate))
    else:
        question_aware_representatins = tf.multiply(question_aware_representatins, (1 - dropout_rate))
        passage_aware_representatins = tf.multiply(passage_aware_representatins, (1 - dropout_rate))
        
    # ======Highway layer======
    if with_match_highway:
        with tf.variable_scope("left_matching_highway"):
            question_aware_representatins = multi_highway_layer(question_aware_representatins, question_aware_dim,highway_layer_num)
        with tf.variable_scope("right_matching_highway"):
            passage_aware_representatins = multi_highway_layer(passage_aware_representatins, passage_aware_dim,highway_layer_num)
    aggregation_representation = tf.concat([tf.reduce_max(question_aware_representatins,1),tf.reduce_max(passage_aware_representatins,1)],1)
    aggregation_dim = question_aware_dim+passage_aware_dim
    
    # ======Highway layer======
    if with_aggregation_highway:
        with tf.variable_scope("aggregation_highway"):
            agg_shape = tf.shape(aggregation_representation)
            batch_size = agg_shape[0]
            aggregation_representation = tf.reshape(aggregation_representation, [1, batch_size, aggregation_dim])
            aggregation_representation = multi_highway_layer(aggregation_representation, aggregation_dim, highway_layer_num)
            aggregation_representation = tf.reshape(aggregation_representation, [batch_size, aggregation_dim])
    
    return (aggregation_representation, aggregation_dim)

