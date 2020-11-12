
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
            c_IJ = softmax(b_IJ, axis=2)

            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == iter_routing - 1:
                s_J = tf.multiply(c_IJ, u_hat)
                s_J = reduce_sum(s_J, axis=1, keepdims=True) + biases

                v_J = squash(s_J)
            elif r_iter < iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = reduce_sum(s_J, axis=1, keepdims=True) + biases
                v_J = squash(s_J)


                v_J_tiled = tf.tile(v_J, [1, input_shape[1], 1, 1, 1])
                u_produce_v = reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keepdims=True)

                b_IJ += u_produce_v

    return(v_J)

def sense_Global_Local_att(sense = None, context_repres = None, context_mask = None, window_size = 8):
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


class Model(object):
  def __init__(self,
               config,
               is_training,
               input_ids,
               index_a,
               index_b,
               ids_a,
               ids_b,
               idx4lmms,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=True,
               scope=None,
               MP_dim = 10,
               shared_dim =50):
    config = copy.deepcopy(config)
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0
    
    input_shape = get_shape_list(input_ids, expected_rank=2)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    if input_mask is None:
      input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

    if token_type_ids is None:
      token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    with tf.variable_scope(scope, default_name="bert"):
      with tf.variable_scope("embeddings"):
        # Perform embedding lookup on the word ids.
        (self.embedding_output, self.embedding_table) = embedding_lookup(
            input_ids=input_ids,
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range,
            word_embedding_name="word_embeddings",
            use_one_hot_embeddings=use_one_hot_embeddings)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        self.embedding_output = embedding_postprocessor(
            input_tensor=self.embedding_output,
            use_token_type=True,
            token_type_ids=token_type_ids,
            token_type_vocab_size=config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)

        (self.embedding_output_ids_a, _) = embedding_lookup(
            input_ids=ids_a,
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range,
            word_embedding_name="word_a_embeddings",
            use_one_hot_embeddings=use_one_hot_embeddings)  

        (self.embedding_output_ids_b, _) = embedding_lookup(
            input_ids=ids_b,
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range,
            word_embedding_name="word_b_embeddings",
            use_one_hot_embeddings=use_one_hot_embeddings)
        
        lmms_emb_table = tf.get_variable("lmms_embedding", shape=None, dtype=tf.float32, initializer=config.lmms_emb, regularizer=None, trainable=True, collections=None)
        self.lmms_representation = tf.nn.embedding_lookup(lmms_emb_table, idx4lmms)# batch, 1024/2048


      with tf.variable_scope("encoder"):
        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.
        attention_mask = create_attention_mask_from_input_mask(
            input_ids, input_mask)

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        self.all_encoder_layers = transformer_model(
            input_tensor=self.embedding_output,
            attention_mask=attention_mask,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_act_fn=get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            do_return_all_layers=True)

      self.sequence_output = self.all_encoder_layers[-1]+self.all_encoder_layers[-2]+self.all_encoder_layers[-3]+self.all_encoder_layers[-4]# sum of top-4 layers


      question_mask = tf.expand_dims(tf.one_hot(index_a, seq_length, dtype=tf.float32), axis = 2)
      passage_mask = tf.expand_dims(tf.one_hot(index_b, seq_length, dtype=tf.float32), axis = 2)
      q_embedd_index = tf.reduce_sum(tf.multiply(question_mask,self.sequence_output), axis = 1, keepdims = False) # ...., index = [batch, dim], ....
      p_embedd_index = tf.reduce_sum(tf.multiply(passage_mask, self.sequence_output), axis = 1, keepdims = False) # ...., index = [batch, dim], ....
      
      with tf.variable_scope("pooler"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        _first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
        first_token_tensor = q_embedd_index+p_embedd_index#tf.concat([q_embedd_index, p_embedd_index],axis= 1)
        self.pooled_output = tf.layers.dense(
            first_token_tensor,
            config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=create_initializer(config.initializer_range))

      with tf.variable_scope("semantic_decomposing"):
        self.embedding_output_ids_a = tf.squeeze(self.embedding_output_ids_a, axis = 1)
        self.embedding_output_ids_b = tf.squeeze(self.embedding_output_ids_b, axis = 1)
        dense_q_embedd_index = tf.layers.dense(self.embedding_output_ids_a,shared_dim,activation=tf.nn.relu, kernel_initializer=create_initializer(config.initializer_range), name="dense_q_embedd_index") 
        dense_p_embedd_index = tf.layers.dense(self.embedding_output_ids_b,shared_dim,activation=tf.nn.relu, kernel_initializer=create_initializer(config.initializer_range), name="dense_p_embedd_index") 
        orthogonal_sememe = tf.get_variable("orthogonal_sememe", shape=[MP_dim, shared_dim], dtype=tf.float32, initializer=tf.orthogonal_initializer(1.0))
        q_sense = match_utils.multi_perspective_expand_for_2D(dense_q_embedd_index,orthogonal_sememe) #batch_size, Mp, dim
        p_sense = match_utils.multi_perspective_expand_for_2D(dense_p_embedd_index,orthogonal_sememe)

        q_sense = tf.expand_dims(tf.expand_dims(q_sense, axis = -1), axis = 2) # batch_size, Mp, dim, 1
        p_sense = tf.expand_dims(tf.expand_dims(p_sense, axis = -1), axis = 2) 

        capsules_q = q_sense# [32, 10, 1, 100, 1]
        capsules_p = p_sense
        with tf.variable_scope('capsule_routing'):
            for i in range(3): # support multiple capsule layers
                with tf.variable_scope('layer-{}'.format(i)):
                  b_IJ_q = tf.constant(np.zeros(shape = [1, MP_dim, MP_dim, 1, 1], dtype='float32'))
                  b_IJ_p = tf.constant(np.zeros(shape = [1, MP_dim, MP_dim, 1, 1], dtype='float32'))
                  capsules_q = match_utils.routing(capsules_q, b_IJ_q, num_outputs=MP_dim, num_dims=shared_dim, iter_routing=3)#(32, 1, 10, 100, 1)
                  tf.get_variable_scope().reuse_variables()
                  capsules_p = match_utils.routing(capsules_p, b_IJ_p, num_outputs=MP_dim, num_dims=shared_dim, iter_routing=3)#(32, 1, 10, 100, 1)
                  capsules_q = tf.expand_dims(tf.squeeze(capsules_q, axis=1), axis = 2)
                  capsules_p = tf.expand_dims(tf.squeeze(capsules_p, axis=1), axis = 2)
        #print (capsules_q.shape,capsules_p.shape)#(32, 1, 10, 100, 1) (32, 1, 10, 100, 1)

        capsules_q = tf.squeeze(tf.squeeze(capsules_q, axis=2), axis = -1) #batch_size, Mp, 1, dim, 1 => batch_size, Mp, dim 
        capsules_p = tf.squeeze(tf.squeeze(capsules_p, axis=2), axis = -1) #batch_size, Mp, 1, dim, 1 => batch_size, Mp, dim

        #sense = None, context_repres = None, context_mask = None, window_size = 5
        seq_output = tf.layers.dense(self.sequence_output,shared_dim,activation=tf.nn.relu, kernel_initializer=create_initializer(config.initializer_range), name="dense_sequence_output")
        q_global_s, q_local_s = match_utils.sense_Global_Local_att(capsules_q, seq_output, question_mask)
        p_global_s, p_local_s = match_utils.sense_Global_Local_att(capsules_p, seq_output, passage_mask)

        #Global and Local context
        q_m_sense = capsules_q + q_global_s + q_local_s #batch, p, dim
        p_m_sense = capsules_p + p_global_s + p_local_s


        q_m_norm_weight = tf.nn.softmax(tf.norm(tensor = q_m_sense, ord ='euclidean', axis = -1, keepdims = True), axis = 1) #batch, p, 1
        p_m_norm_weight = tf.nn.softmax(tf.norm(tensor = p_m_sense, ord ='euclidean', axis = -1, keepdims = True), axis = 1) 
        q_repre = tf.reduce_sum(tf.multiply(q_m_norm_weight, q_m_sense), axis = 1)
        p_repre = tf.reduce_sum(tf.multiply(p_m_norm_weight, p_m_sense), axis = 1)
        self.q_m_norm_weight = q_repre
        self.p_m_norm_weight = p_repre
        merge_Representation = tf.concat([q_repre, p_repre, q_embedd_index, p_embedd_index,self.lmms_representation],axis= 1) 
        self.pooled_output = tf.layers.dense(
            merge_Representation,
            config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=create_initializer(config.initializer_range), name="pooled_output")