import data
import model
import util
import tensorflow as tf
import numpy as np
import os
import argparse
import configparser
from matplotlib.pyplot import plot as plt


def train(iterations):
    """
    :param iterations: The number of data pieces to train on in total
    :param batch_size: The number of data pieces per batch
    :return:
    """
    # Prepare dataset
    dataset = data.BabiCorpus(params)
    final_train_data, final_test_data = dataset.prepare_data()

    #  Start by clearing out the TF default graph so we can always run the network again if we want to change something.
    tf.reset_default_graph()

    """
    input_sentence_endings:     a [batch_size, maxi_sent_count, 2] tensor that contains the locs of the ends of sents.
    context:                    a [batch_size, max_ctxt_len, word_vectztn_dims] tensor that contains all the ctxt info.
    query:                      a [batch_size, max_quest_len, word_vector_dim] tensor that contains all of the questions.
    input_query_lengths:        a [batch_size, 2] tensor that contains question length information.
                input_query_lengths[:,1] has the actual lengths;
                input_query_lengths[:,0] is a simple range() so that it plays nice with tf.gather_nd.
    gold_standard:              The real answers i.e. labels
    """

    input_sentence_endings = tf.placeholder(tf.int32, [None, None, 2], "sentence")
    context = tf.placeholder(tf.float32, [None, None, params.glove_dim], "context")
    query = tf.placeholder(tf.float32, [None, None, params.glove_dim], "query")
    input_query_lengths = tf.placeholder(tf.int32, [None, 2], "query_lengths")
    gold_standard = tf.placeholder(tf.float32, [None, 1, params.glove_dim], "answer")

    # logits: a mask over which words exist using logit
    # context_state = input_module_output
    context_state, logit, logits = model.model(input_sentence_endings, context, query, input_query_lengths)

    # ------------------ Optimization / Training
    locs, corrects, corr, total_loss, opt_op = model.optimization_module(context, gold_standard, logits)

    # Initialize variables
    init = tf.global_variables_initializer()

    # Launch the TensorFlow session
    sess = tf.Session()
    sess.run(init)

    # Prepare validation batch
    batch = np.random.randint(final_test_data.shape[0], size=params.batch_size * 10)
    batch_data = final_test_data[batch]
    validation_set, val_context_words, val_cqas = dataset.prep_batch(batch_data, context, input_sentence_endings,
                                                                     query, input_query_lengths, gold_standard,
                                                                     more_data=True)

    training_iterations = range(0, iterations, params.batch_size)
    # Use TQDM if installed
    try:
        from tqdm import tqdm
        # Add a progress bar if TQDM is installed
        training_iterations = tqdm(training_iterations)
    except:
        pass

    # Start training: Prepare training batch
    wordz = []
    for j in training_iterations:
        batch = np.random.randint(final_train_data.shape[0], size=params.batch_size)
        batch_data = final_train_data[batch]

        sess.run([opt_op], feed_dict=dataset.prep_batch(batch_data, context, input_sentence_endings,
                                                        query, input_query_lengths, gold_standard))
        # Validation step
        if (j / params.batch_size) % params.validation_step == 0:
            # Calculate batch accuracy, context(input) state, loss, ...
            acc, ccs, tmp_loss, log, con, cor, loc = sess.run([corrects, context_state, total_loss, logit, context,
                                                               corr, locs],
                                                              feed_dict=validation_set)

            # Display results
            print("Iter " + str(j / params.batch_size) + ", Minibatch Loss= ", tmp_loss, "Accuracy= ", np.mean(acc))

        # Visualize attention
        if params.view_attention:
            if j == 30000:
                visualize_attention(sess, locs, total_loss, logits, query, context_state, validation_set,
                                    val_context_words, val_cqas)

    # Final testing accuracy
    print("Final accuracy: ", np.mean(sess.run([corrects], feed_dict=dataset.prep_batch(final_test_data, context,
                                                                                        input_sentence_endings, query,
                                                                                        input_query_lengths,
                                                                                        gold_standard))[0]))
    sess.close()


def visualize_attention(sess, locs, total_loss, logits, query, context_state, validation_set, val_context_words, val_cqas):
    """
        After a little bit of training, see what kinds of answers we're getting from the network.

        In the diagrams plotted, we visualize attention over each of the episodes (rows) for all the sentences (columns) in
        our context; darker colors represent more attention paid to that sentence on that episode.

        You should see attention change between at least two episodes for each question, but sometimes attention will be
        able to find the answer within one, and sometimes it will take all four episodes.

        If the attention appears to be blank, it may be saturating and paying attention to everything at once.
        In that case, you can try training with a higher weight_decay in order to discourage that from happening.
        Later on in training, saturation becomes extremely common.
    """

    ancr = sess.run([model.corrbool, locs, total_loss, logits, model.facts_0s, model.w_1] +
                    model.attends +
                    [query, context_state, model.question_module_outputs],
                    feed_dict=validation_set)
    a = ancr[0]
    n = ancr[1]
    cr = ancr[2]
    attenders = np.array(ancr[6:-3])
    faq = np.sum(ancr[4], axis=(-1, -2))  # Number of facts in each context

    if not os.path.join("save", "visualize_attention"):
        os.mkdir(os.path.join("save", "visualize_attention"))

    limit = 5
    for question in range(min(limit, params.batch_size)):
        plt.yticks(range(params.passes, 0, -1))
        plt.ylabel("Episode")
        plt.xlabel("Question " + str(question + 1))
        pltdata = attenders[:, question, :int(faq[question]), 0]
        # Display only information about facts that actually exist, all others are 0
        pltdata = (pltdata - pltdata.mean()) / (pltdata.max()-pltdata.min()+0.001) * 256
        plt.pcolor(pltdata, cmap=plt.cm.BuGn, alpha=0.7)
        # plt.show()
        plt.savefig(os.path.join("save", "visualize_attention", "Question_" + str(question + 1) + ".png"))

    # In order to see what the answers for the above questions were, we can use the location of our distance score
    # in the context as an index and see what word is at that index.
    # Locations of responses within contexts
    indices = np.argmax(n, axis=1)

    # Locations of actual answers within contexts
    indicesc = np.argmax(a, axis=1)

    for idx_model, idx_label, ctxt_word, ctxt_q_a in list(zip(indices, indicesc, val_context_words, val_cqas))[:limit]:
        ccc = " ".join(ctxt_word)
        print("TEXT: ", ccc)
        print("QUESTION: ", " ".join(ctxt_q_a[3]))
        print("RESPONSE: ", ctxt_word[idx_model], ["Correct", "Incorrect"][idx_model != idx_label])
        print("EXPECTED: ", ctxt_word[idx_label])
        print()


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('default_params.ini')
    defaults = config['defaults']
    parser = argparse.ArgumentParser()
    parser.add_argument('--babi_dataset_link', type=str, default=defaults['babi_dataset_link'])
    # parser.add_argument('--babi_dataset_path', type=str, default=defaults['babi_dataset_path'])
    parser.add_argument('--babi_dataset_zip', type=str, default=defaults['babi_dataset_zip'])
    parser.add_argument('--glove_vector_file', type=str, default=defaults['glove_vector_file'])
    parser.add_argument('--interior_relative_path', type=str, default=defaults['interior_relative_path'])
    parser.add_argument('--train_set', type=str, default=defaults['train_set'])
    parser.add_argument('--test_set', type=str, default=defaults['test_set'])
    parser.add_argument('--train_set_post', type=str, default=defaults['train_set_post'])
    parser.add_argument('--test_set_post', type=str, default=defaults['test_set_post'])
    parser.add_argument('--batch_size', type=int, default=defaults['batch_size'])
    parser.add_argument('--cell_size', type=int, default=defaults['cell_size'])
    parser.add_argument('--dropout_input', type=float, default=defaults['dropout_input'])
    parser.add_argument('--dropout_output', type=float, default=defaults['dropout_output'])
    parser.add_argument('--ff_hidden_size', type=int, default=defaults['ff_hidden_size'])
    parser.add_argument('--glove_dim', type=int, default=defaults['glove_dim'])
    parser.add_argument('--iterations', type=int, default=defaults['iterations'])
    parser.add_argument('--learning_rate', type=float, default=defaults['learning_rate'])
    parser.add_argument('--passes', type=int, default=defaults['passes'])
    parser.add_argument('--validation_step', type=int, default=defaults['validation_step'])
    parser.add_argument('--weight_decay', type=float, default=defaults['weight_decay'])
    parser.add_argument('--view_attention', action='store_true')
    params = parser.parse_args()
    print(params)

    model = model.QA_DMNN_Model(params)
    train(iterations=params.iterations)