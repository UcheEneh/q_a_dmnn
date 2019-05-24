import tensorflow as tf
import util
import data

"""
The network is designed around having a recurrent layer's memory be set dynamically, 
based on other information in the text, hence the name dynamic memory network (DMN).

The person gets a chance, first of all, to read the context and create memories of the facts inside. 
With those facts in mind, they then read the question, and re-examine the context specifically 
searching for the answer to that question, comparing the question to each of the facts.

Sometimes, one fact guides us to another. In the bAbI data set, the network might want to find the 
location of a football. It might search for sentences about the football to find that John was the 
last person to touch the football, then search for sentences about John to find that John had been 
in both the bedroom and the hallway. Once it realizes that John had been last in the hallway, 
it can then answer the question and confidently say that the football is in the hallway.
"""

class QA_DMNN_Model:
    def __init__(self):
        pass

    def inference(self):
        # recurrent_cell_size
        input_gru = tf.contrib.rnn.GRUCell(util.CELL_SIZE)
        self.gru_drop = tf.contrib.rnn.DropoutWrapper(input_gru, util.DROPUT_INPUT, util.DROPUT_OUTPUT)

        # input_sentence_endings: a [batch_size, maximum_sentence_count, 2] tensor that
        # contains the locations of the ends of sentences.
        self.input_sentence_endings = tf.placeholder(tf.int32, [None, None, 2], "sentence")

    def input_module(self):
        """
        The input module is the first of the four modules that a dynamic memory network uses to come up
        with its answer, and consists of a simple pass over the input with a gated recurrent unit, or GRU,
        to gather pieces of evidence.

        Each piece of evidence, or fact, corresponds to a single sentence in the context, and is
        represented by the output (input_module_outputs) at that timestep.
        """
        context = tf.placeholder(tf.float32, [None, None, util.GLOVE_DIM], "context")
        context_placeholder = context

        # dynamic_rnn also returns the final internal state. For this task, we don't need that.
        # It would be needed e.g. for attention network
        input_module_outputs, _ = tf.nn.dynamic_rnn(self.gru_drop, context, dtype=tf.float32,
                                                    scope="input_module")

        # cs: the facts gathered from the context
        self.eos_facts_from_ctxt = tf.gather_nd(input_module_outputs, self.input_sentence_endings)

        # to use every word as a fact, useful for tasks with one-sentence context
        s = input_module_outputs

    def question_module(self):
        """
        The question module is the second module. It consists of another GRU pass, this time over
        the text of the question. Instead of pieces of evidence, we can simply pass forward the end state,
        as the question is guaranteed by the data set to be one sentence long.
        """

        # query: a [batch_size, maximum_question_length, word_vector_dim] tensor that contains
        # all of the questions.
        query = tf.placeholder(tf.float32, [None, None, util.GLOVE_DIM], "query")

        # input_query_lengths: a [batch_size, 2] tensor that contains question length information.
        # input_query_lengths[:,1] has the actual lengths; input_query_lengths[:,0] is a simple range()
        # so that it plays nice with gather_nd.
        input_query_lengths = tf.placeholder(tf.int32, [None, 2], "query_lengths")
        question_module_outputs, _ = tf.nn.dynamic_rnn(self.gru_drop, query, dtype=tf.float32,
                                                       scope=tf.VariableScope(True, "input_module"))
        # q: the question states. a [batch_size, recurrent_cell_size] tensor.
        self.question_rnn_state = tf.gather_nd(question_module_outputs, input_query_lengths)

    def episodic_memory(self):
        """
        Our third module, the episodic memory module, uses attention to do multiple passes, each pass
        consisting of GRUs iterating over the input. Each iteration inside each pass has a weighted
        update on current memory, based on how much attention is being paid to the corresponding fact
        at that time.

        We calculate attention in this model by constructing similarity measures between each fact, our
        current memory, and the original question. (Note that this is different from normal attention,
        which only constructs similarity measures between facts and current memory.)

        We pass the results through a two-layer feed-forward network to get an attention constant for
        each fact. We then modify the memory by doing a weighted pass with a GRU over the input facts
        (weighted by the corresponding attention constant). In order to avoid adding incorrect
        information into memory when the context is shorter than the full length of the matrix, we
        create a mask for which facts exist and don't attend at all (i.e., retain the same memory) when
        the fact does not exist.

        Note: attention mask is nearly always wrapped around a representation used by a layer. For images,
        that wrapping is most likely to happen around a convolutional layer (most likely one with a
        direct mapping to locations in the image), and for natural language, around a recurrent layer.
        """

        # make sure the current memory (i.e. the question vector) is broadcasted along the facts dim
        facts_size = tf.stack([tf.constant(1), tf.shape(self.eos_facts_from_ctxt)])

        """
        tf.tile:
            - This operation creates a new tensor by replicating input multiples times.
              For example, tiling [a b c d] by [2] produces [a b c d a b c d]
        """
        # Note: this would be used when the context vector is compared with the question
        self.compare_question_facts = tf.tile(tf.reshape(self.question_rnn_state, [-1, 1, util.CELL_SIZE]),
                                     facts_size)

        # Final output for attention needs to be 1 in order to create a mask
        output_size = 1

        # Weights and biases
        attend_init = tf.random_normal_initializer(stddev=0.1)
        self.w_1 = tf.get_variable("attend_w1", [1, util.CELL_SIZE*7, util.CELL_SIZE], tf.float32,
                              initializer=attend_init)
        self.w_2 = tf.get_variable("attend_w2", [1, util.CELL_SIZE, output_size], tf.float32,
                              initializer=attend_init)
        self.b_1 = tf.get_variable("attend_b1", [1, util.CELL_SIZE], tf.float32,
                              initializer=attend_init)
        self.b_2 = tf.get_variable("attend_b2", [1, output_size], tf.float32,
                              initializer=attend_init)

        # Regulate all weights and biases
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(self.w_1))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(self.b_1))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(self.w_2))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(self.b_2))

        def attention(self, context_facts, current_mem, existing_facts):
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
                attending = tf.concat([context_facts, current_mem, self.compare_question_facts,
                                       context_facts * self.compare_question_facts,
                                       context_facts * current_mem,
                                       (context_facts - self.compare_question_facts)**2,
                                       (context_facts - current_mem)**2], 2)

                # m1:  First layer of multiplied weights for the feed-forward network.
                # We tile the weights in order to manually broadcast, since tf.matmul does not
                # automatically broadcast batch matrix multiplication (as of TensorFlow 1.2).
                m1 = tf.matmul(attending * existing_facts,
                               tf.tile(self.w_1, tf.stack([tf.shape(attending)[0], 1, 1]))) \
                     * existing_facts

                # bias_1: A masked version of the first feed-forward layer's bias over only existing facts.
                bias_1 = self.b_1 * existing_facts

                # tnhan: First nonlinearity. In the original paper, this is a tanh nonlinearity;
                #  choosing relu was a design choice intended to avoid issues with low gradient magnitude
                #  when the tanh returned values close to 1 or -1.
                tnhan = tf.nn.relu(m1 + bias_1)

                # m2: Second layer of multiplied weights for the feed-forward network.
                m2 = tf.matmul(tnhan, tf.tile(self.w_2, tf.stack([tf.shape(attending)[0], 1, 1])))

                # bias_2: A masked version of the second feed-forward layer's bias.
                bias_2 = self.b_2 * existing_facts

                # norm_m2: A normalized version of the second layer of weights, which is used to help
                # make sure the softmax nonlinearity doesn't saturate.
                norm_m2 = tf.nn.l2_normalize(m2 + bias_2, -1)

                # softmaxable: A hack in order to use sparse_softmax on an otherwise dense tensor.
                # We make norm_m2 a sparse tensor, then make it dense again after the operation.
                softmax_id = tf.where(tf.not_equal(norm_m2, 0))[:,:-1]
                softmax_gather = tf.gather_nd(norm_m2[...,0], softmax_id)
                softmax_shape = tf.shape(norm_m2, out_type=tf.int64)[:-1]
                softmaxable = tf.SparseTensor(softmax_id, softmax_gather, softmax_shape)

                return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_softmax(softmaxable)), -1)

        # facts_0s: a [batch_size, max_facts_length, 1] tensor whose values are 1 if the corresponding
        # fact exists and 0 if not.
        facts_0s = tf.cast(tf.count_nonzero(self.input_sentence_endings[:,:,-1:], -1, keep_dims=True),
                           tf.float32)

        with tf.variable_scope("Episodes") as scope:
            attention_gru = tf.contrib.rnn.GRUCell(util.CELL_SIZE)

            # memory: A list of all tensors that are the (current or past) memory state of the
            # attention mechanism.
            self.memory = [self.question_rnn_state]

            # attends: A list of all tensors that represent what the network attends to.
            attends = []
            for a in range(util.PASSES):
                # attention mask
                attend_to = attention(self.eos_facts_from_ctxt,
                                      tf.tile(tf.reshape(self.memory[-1], [-1, 1, util.CELL_SIZE]),
                                              facts_size),
                                      facts_0s)

                # Inverse attention mask, for what's retained in the state
                retain = 1- attend_to

                # GRU pass over the facts, according to the attention mask
                while_valid_index = (lambda state, index: index < tf.shape(self.eos_facts_from_ctxt)[1])
                update_state = (lambda state, index: (attend_to[:,index,:] *
                                                      attention_gru(self.eos_facts_from_ctxt[:,index,:],
                                                                    state)[0]
                                                      + retain[:,index,:] * state))

                # start loop with most recent memory and at the first index
                self.memory.append(tuple(tf.while_loop(while_valid_index,
                                                  (lambda state, index: (update_state(state,index),
                                                                         index + 1)),
                                                  loop_vars=[self.memory[-1], 0])
                                    )[0])
                attends.append(attend_to)

                # Reuse variables so the GRU pass uses the same variables every pass
                scope.reuse_variables()

    def answer_module(self):
        """
        The final module is the answer module, which regresses from the question and episodic memory
        modules' outputs using a fully connected layer to a "final result" word vector, and the word in
        the context that is closest in distance to that result is our final output (to guarantee the
        result is an actual word).
        We calculate the closest word by creating a "score" for each word, which indicates the final
        result's distance from the word.
        """

        # a0: Final memory state. (Input to answer module)
        a0 = tf.concat([self.memory[-1], q], -1)

        # fc_init: Initializer for the final fully connected layer's weights.
        fc_init = tf.random_normal_initializer(stddev=0.1)

        with tf.variable_scope("answer"):
            # w_answer: The final fully connected layer's weights.
            w_answer = tf.get_variable("weight", [util.CELL_SIZE * 2, util.GLOVE_DIM],
                                       tf.float32, initializer=fc_init)

            # regulate the FC layer's weights
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(w_answer))

            # The regressed word. Not yet an actual word; we still have to find the closest match.
            logit = tf.expand_dims(tf.matmul(a0, w_answer), 1)

            # Make a mask over which words exist
            with tf.variable_scope("ending"):
                all_ends = tf.reshape(self.input_sentence_endings, [-1,2])
                range_ends = tf.range(tf.shape(all_ends)[0])
                ends_indices = tf.stack([all_ends[:,0], range_ends], axis=1)
                ind = tf.reduce_max(tf.scatter_nd(ends_indices, all_ends[:,1],
                                                  [tf.shape(self.question_rnn_state)[0],
                                                   tf.shape(all_ends)[0]]), axis=-1)
                range_ind = tf.range(tf.shape(ind)[0])
                mask_ends = tf.cast(tf.scatter_nd(tf.stack([ind, range_ind], axis=1),
                                                  tf.ones_like(range_ind), [tf.reduce_max(ind) + 1,
                                                                            tf.shape(ind)[0]]), bool)

                # A bit of a trick. With the locations of the ends of the mask (the last periods in each of
                # the contexts) as 1 and the rest as 0, we can scan with exclusive or (starting from all 1).
                # For each context in the batch, this will result in 1s up until the marker (the location of
                # that last period) and 0s afterwards.
                mask = tf.scan(tf.logical_xor, mask_ends, tf.ones_like(range_ind, dtype=bool))

                # We score each possible word inversely with their Euclidean distance to the regressed word.
                #  The highest score (lowest distance) will correspond to the selected word.
                logits = -tf.reduce_sum(
                            tf.square(data.context *
                                      tf.transpose(tf.expand_dims(tf.cast(mask, tf.float32), -1),
                                                   [1,0,2]) - logit),
                            axis=-1)

    def optimization(self):
        """
        Adam estimates the first two moments of the gradient by calculating an exponentially decaying
        average of past iterations' gradients and squared gradients, which correspond to the estimated
        mean and the estimated variance of these gradients. The calculations use two additional
        hyperparameters to indicate how quickly the averages decay with the addition of new information.
        The averages are initialized as zero, which leads to bias toward zero, especially when those
        hyperparameters near one.

        In order to counteract that bias, Adam computes bias-corrected moment estimates that are greater
        in magnitude than the originals. The corrected estimates are then used to update the weights
        throughout the network. The combination of these estimates make Adam one of the best choices
        overall for optimization, especially for complex networks. This applies doubly to data that is
        very sparse, such as is common in natural language processing tasks.
        """


