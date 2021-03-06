import tensorflow as tf
# import util
# import data
# import numpy as np

"""
The network is designed around having a recurrent layer's memory be set dynamically, based on other information in the 
text, hence the name dynamic memory network (DMN).

The person gets a chance, first of all, to read the context and create memories of the facts inside. With those facts 
in mind, they then read the question, and re-examine the context specifically searching for the answer to that question, 
comparing the question to each of the facts.

Sometimes, one fact guides us to another. In the bAbI data set, the network might want to find the location of a football. 
It might search for sentences about the football to find that John was the last person to touch the football, 
then search for sentences about John to find that John had been in both the bedroom and the hallway. Once it realizes 
that John had been last in the hallway, it can then answer the question and confidently say that the football is in the hallway.
"""

class QA_DMNN_Model:
    def __init__(self, params):
        self.params = params
        # pass
        input_gru = tf.contrib.rnn.GRUCell(self.params.cell_size)
        self.gru_drop = tf.contrib.rnn.DropoutWrapper(input_gru, self.params.dropout_input, self.params.dropout_output)
        # attends: A list of all tensors that represent what the network attends to.
    
    def model(self, input_sentence_endings, context, query, input_query_lengths):
        context_placeholder = context  # context would be used later to represent something else
        
        # ------------------ Input Module
        # Create the input_module_output
        context_state = self.input_module(input_sentence_endings, context)
        # s = input_module_output = context_state

        # ------------------ Question Module
        # Create the question_rnn_state
        question_rnn_state = self.question_module(query, input_query_lengths)

        # ------------------ Episodic Memory
        memory_update = self.episodic_memory(question_rnn_state, input_sentence_endings)

        # ------------------ Answer Module
        # logits: # a mask over which words exist using logit
        logit, logits = self.answer_module(question_rnn_state, memory_update, context, input_sentence_endings)
        
        return context_state, logit, logits
        
    def input_module(self, input_sentence_endings, context):
        """
        The input module is the first of the four modules that a dynamic memory network uses to come up with its answer,
        and consists of a simple pass over the input with a gated recurrent unit, or GRU, to gather pieces of evidence.

        Each piece of evidence, or fact, corresponds to a single sentence in the context, and is represented by the
        output (input_module_outputs) at that timestep.
        """

        # dynamic_rnn also returns the final internal state. For this task, we don't need that.
        # It would be needed e.g. for attention network
        input_module_outputs, _ = tf.nn.dynamic_rnn(self.gru_drop, context, dtype=tf.float32, scope="input_module")

        # cs: the facts gathered from the context rnn final state
        self.facts = tf.gather_nd(input_module_outputs, input_sentence_endings)

        # to use every word as a fact, useful for tasks with one-sentence context
        return input_module_outputs

    def question_module(self, query, input_query_lengths):
        """
        The question module is the second module. It consists of another GRU pass, this time over the text of the
        question. Instead of pieces of evidence, we can simply pass forward the end state, as the question is guaranteed
        by the data set to be one sentence long.
        """
        self.question_module_outputs, _ = tf.nn.dynamic_rnn(self.gru_drop, query, dtype=tf.float32,
                                                       scope=tf.VariableScope(True, "input_module"))
        # q: the question states. a [batch_size, recurrent_cell_size] tensor.
        question_rnn_state = tf.gather_nd(self.question_module_outputs, input_query_lengths)

        return question_rnn_state

    def episodic_memory(self, question_rnn_state, input_sentence_endings):
        """
        Our third module, the episodic memory module, uses attention to do multiple passes, each pass consisting of GRUs
        iterating over the input. Each iteration inside each pass has a weighted update on current memory, based on how
        much attention is being paid to the corresponding fact at that time.

        We calculate attention in this model by constructing similarity measures between each fact, our current memory,
        and the original question. (Note that this is different from normal attention, which only constructs similarity
        measures between facts and current memory.)

        We pass the results through a two-layer feed-forward network to get an attention constant for each fact. We then
        modify the memory by doing a weighted pass with a GRU over the input facts (weighted by the corresponding
        attention constant). In order to avoid adding incorrect information into memory when the context is shorter than
        the full length of the matrix, we create a mask for which facts exist and don't attend at all (i.e., retain the
        same memory) when the fact does not exist.

        Note: attention mask is nearly always wrapped around a representation used by a layer. For images, that wrapping
        is most likely to happen around a convolutional layer (most likely one with a direct mapping to locations in the
        image), and for natural language, around a recurrent layer.
        """

        # tf.tile: This operation creates a new tensor by replicating input multiples times.
        # For example, tiling [a b c d] by [2] produces [a b c d a b c d]

        # make sure the current memory (i.e. the question vector) is broadcasted along the facts dim
        facts_size = tf.stack([tf.constant(1), tf.shape(self.facts)[1], tf.constant(1)])    # [1, range, 1]
        # Note: this would be used when the context vector is compared with the question. tile: expand the question by facts_size times
        re_question_rnn = tf.tile(tf.reshape(question_rnn_state, [-1, 1, self.params.cell_size]), facts_size)

        # Final output for attention needs to be 1 in order to create a mask
        output_size = 1

        # Weights and biases
        attend_init = tf.random_normal_initializer(stddev=0.1)
        w_1 = tf.get_variable("attend_w1", [1, self.params.cell_size*7, self.params.cell_size], tf.float32,
                                   initializer=attend_init)     # 7: i guess multi-head attn
        w_2 = tf.get_variable("attend_w2", [1, self.params.cell_size, output_size], tf.float32, initializer=attend_init)
        b_1 = tf.get_variable("attend_b1", [1, self.params.cell_size], tf.float32, initializer=attend_init)
        b_2 = tf.get_variable("attend_b2", [1, output_size], tf.float32, initializer=attend_init)
        self.w_1 = w_1      # make accessible outside

        # Regulate all weights and biases
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(w_1))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(b_1))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(w_2))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(b_2))

        # facts_0s: a [batch_size, max_facts_length, 1] tensor whose values are 1 if the corresponding fact exists
        # and 0 if not.
        self.facts_0s = tf.cast(tf.count_nonzero(input_sentence_endings[:, :, -1:], -1, keep_dims=True), tf.float32)

        with tf.variable_scope("Episodes") as scope:
            attention_gru = tf.contrib.rnn.GRUCell(self.params.cell_size)
            # memory: A list of all tensors that are the (current or past) memory state of the attention mechanism.
            memory = [question_rnn_state]
            self.attends = []

            for a in range(self.params.passes):
                # tile memory to facts size so attention can be performed on it
                current_mem = tf.tile(tf.reshape(memory[-1], [-1, 1, self.params.cell_size]), facts_size)
                # attention mask
                attend_to = self.attention(w_1, b_1, w_2, b_2, self.facts, current_mem, self.facts_0s, re_question_rnn)

                # Inverse attention mask, for what's retained in the state
                retain = 1 - attend_to

                # GRU pass over the facts, according to the attention mask
                while_valid_index = (lambda state, index: index < tf.shape(self.facts)[1])
                update_state = (lambda state, index: (attend_to[:, index, :] *
                                                      attention_gru(self.facts[:, index, :], state)[0]
                                                      + retain[:, index, :] * state))

                # start loop with most recent memory and at the first index
                memory.append(tuple(tf.while_loop(while_valid_index, (lambda state, index: (update_state(state, index),
                                                                                            index + 1)),
                                                  loop_vars=[memory[-1], 0]))[0])
                self.attends.append(attend_to)

                # Reuse variables so the GRU pass uses the same variables every pass
                scope.reuse_variables()

            return memory

    def attention(self, w_1, b_1, w_2, b_2, context_facts, current_mem, existing_facts, re_question_rnn):
        """
        Custom attention mechanism (constructing similarity measures between each fact, our
        current memory (i.e. the question vector), and the original question)
        :param context_facts: a [batch_size, maximum_sentence_count, recurrent_cell_size] tensor that contains all
                    the facts from the contexts.
        :param current_mem: a [batch_size, maximum_sentence_count, recurrent_cell_size] tensor that contains
                    the current memory. It should be the same memory for all facts for accurate results.
        :param existing_facts: a [batch_size, maximum_sentence_count, 1] tensor that acts as a binary
                    mask for which facts exist and which do not.
        :return:
        """

        with tf.variable_scope("attending") as scope:
            # attending: the metrics by which we decide what to attend to
            attending = tf.concat([context_facts, current_mem, re_question_rnn,
                                   context_facts * re_question_rnn,     # compare each fact to the question
                                   context_facts * current_mem,         # compare each fact with memory
                                   (context_facts - re_question_rnn)**2,
                                   (context_facts - current_mem)**2], 2)

            # m1:  First layer of multiplied weights for the feed-forward network.
            # We tile the weights in order to manually broadcast, since tf.matmul does not automatically broadcast batch
            # matrix multiplication (as of TensorFlow 1.2).
            m1 = tf.matmul((attending * existing_facts), tf.tile(w_1, tf.stack([tf.shape(attending)[0], 1, 1]))) \
                 * existing_facts

            # bias_1: A masked version of the first feed-forward layer's bias over only existing facts.
            bias_1 = b_1 * existing_facts

            # tnhan: First nonlinearity. In the original paper, this is a tanh nonlinearity; choosing relu was a design
            # choice intended to avoid issues with low gradient magnitude when the tanh returned values close to 1 or -1
            tnhan = tf.nn.relu(m1 + bias_1)

            # m2: Second layer of multiplied weights for the feed-forward network.
            m2 = tf.matmul(tnhan, tf.tile(w_2, tf.stack([tf.shape(attending)[0], 1, 1])))

            # bias_2: A masked version of the second feed-forward layer's bias.
            bias_2 = b_2 * existing_facts

            # norm_m2: A normalized version of the second layer of weights, which is used to help make sure the softmax
            # nonlinearity doesn't saturate.
            norm_m2 = tf.nn.l2_normalize(m2 + bias_2, -1)

            # softmaxable: A hack in order to use sparse_softmax on an otherwise dense tensor.
            # We make norm_m2 a sparse tensor, then make it dense again after the operation.
            softmax_id = tf.where(tf.not_equal(norm_m2, 0))[:,:-1]
            softmax_gather = tf.gather_nd(norm_m2[...,0], softmax_id)
            softmax_shape = tf.shape(norm_m2, out_type=tf.int64)[:-1]
            softmaxable = tf.SparseTensor(softmax_id, softmax_gather, softmax_shape)

            res = tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_softmax(softmaxable)), -1)
            return res

    def answer_module(self, question_rnn_state, memory, context, input_sentence_endings):
        """
        The final module is the answer module, which regresses from the question and episodic memory modules' outputs
        using a fully connected layer to a "final result" word vector, and the word in the context that is closest in
        distance to that result is our final output (to guarantee the result is an actual word).
        We calculate the closest word by creating a "score" for each word, which indicates the final result's distance
        from the word.
        """

        # a0: Final memory state. (Input to answer module)
        a0 = tf.concat([memory[-1], question_rnn_state], -1)

        # fc_init: Initializer for the final fully connected layer's weights.
        fc_init = tf.random_normal_initializer(stddev=0.1)

        with tf.variable_scope("answer"):
            # w_answer: The final fully connected layer's weights.
            w_answer = tf.get_variable("weight", [self.params.cell_size * 2, self.params.glove_dim], tf.float32,
                                       initializer=fc_init)

            # regulate the FC layer's weights
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(w_answer))

            # The regressed word. Not yet an actual word; we still have to find the closest match.
            logit = tf.expand_dims(tf.matmul(a0, w_answer), 1)

            # Make a mask over which words exist
            with tf.variable_scope("ending"):
                all_ends = tf.reshape(input_sentence_endings, [-1,2])
                range_ends = tf.range(tf.shape(all_ends)[0])
                ends_indices = tf.stack([all_ends[:,0], range_ends], axis=1)
                ind = tf.reduce_max(tf.scatter_nd(ends_indices,
                                                  all_ends[:,1],
                                                  [tf.shape(question_rnn_state)[0], tf.shape(all_ends)[0]]),
                                    axis=-1)
                range_ind = tf.range(tf.shape(ind)[0])
                mask_ends = tf.cast(tf.scatter_nd(tf.stack([ind, range_ind], axis=1),
                                                  tf.ones_like(range_ind),
                                                  [tf.reduce_max(ind) + 1, tf.shape(ind)[0]]),
                                    bool)

                # A bit of a trick. With the locations of the ends of the mask (the last periods in each of the
                # contexts) as 1 and the rest as 0, we can scan with exclusive or (starting from all 1).
                # For each context in the batch, this will result in 1s up until the marker (the location of that last
                # period) and 0s afterwards.
                mask = tf.scan(tf.logical_xor, mask_ends, tf.ones_like(range_ind, dtype=bool))

            # We score each possible word inversely with their Euclidean distance to the regressed word.
            # The highest score (lowest distance) will correspond to the selected word.
            logits = -tf.reduce_sum(
                tf.square(context * tf.transpose(tf.expand_dims(tf.cast(mask, tf.float32), -1), [1,0,2]) - logit),
                axis=-1)

            return logit, logits

    def optimization_module(self, context, gold_standard, logits):
        # Training
        # gold_standard: The real answers.

        """
        Adam estimates the first two moments of the gradient by calculating an exponentially decaying average of past
        iterations' gradients and squared gradients, which correspond to the estimated mean and the estimated variance
        of these gradients. The calculations use two additional hyperparameters to indicate how quickly the averages
        decay with the addition of new information. The averages are initialized as zero, which leads to bias toward
        zero, especially when those hyperparameters near one.

        In order to counteract that bias, Adam computes bias-corrected moment estimates that are greater in magnitude
        than the originals. The corrected estimates are then used to update the weights throughout the network. The
        combination of these estimates make Adam one of the best choices overall for optimization, especially for
        complex networks. This applies doubly to data that is very sparse, such as is common in nlp tasks.
        """

        with tf.variable_scope('accuracy'):
            eq = tf.equal(context, gold_standard)
            self.corrbool = tf.reduce_all(eq, -1)
            logloc = tf.reduce_max(logits, -1, keep_dims=True)

            # locs: A boolean tensor that indicates where the score matches the minimum score. This happens on multiple
            # dimensions, so in the off chance there's one or two indexes that match we make sure it matches in all
            # indexes
            locs = tf.equal(logits, logloc)

            # correctsbool: A boolean tensor that indicates for which words in the context the score always matches the
            # minimum score.
            correctsbool = tf.reduce_any(tf.logical_and(locs, self.corrbool), -1)
            # corrects: A tensor that is simply correctsbool cast to floats.
            corrects = tf.where(correctsbool, tf.ones_like(correctsbool, dtype=tf.float32),
                                tf.zeros_like(correctsbool, dtype=tf.float32))

            # corr: corrects, but for the right answer instead of our selected answer.
            corr = tf.where(self.corrbool, tf.ones_like(self.corrbool, dtype=tf.float32),
                            tf.zeros_like(self.corrbool, dtype=tf.float32))

        with tf.variable_scope("loss"):
            # Use sigmoid cross entropy as the base loss, with our distances as the relative probabilities.
            # There are multiple correct labels, for each location of the answer word within the context.
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.nn.l2_normalize(logits, -1), labels=corr)

            # Add regularization losses, weighted by weight_decay.
            total_loss = tf.reduce_mean(loss) + self.params.weight_decay * \
                         tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # TensorFlow's default implementation of the Adam optimizer works. We can adjust more than just the learning
        # rate, but it's not necessary to find a very good optimum.
        optimizer = tf.train.AdamOptimizer(self.params.learning_rate)

        # Once we have an optimizer, we ask it to minimize the loss in order to work towards the proper training.
        opt_op = optimizer.minimize(total_loss)

        return locs, corrects, corr, total_loss, opt_op
