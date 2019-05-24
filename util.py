BABI_DATASET_LINK = "https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz"
BABI_DATASET_PATH = "data_set_zip"
BABI_DATASET_ZIP = "tasks_1-20_v1-2.tar.gz"     # 15 MB

# Select "task 5"
TRAIN_SET = "qa5_three-arg-relations_train.txt"
TEST_SET = "qa5_three-arg-relations_test.txt"

TRAIN_SET_POST = "tasks_1-20_v1-2/en/" + TRAIN_SET
TEST_SET_POST = "tasks_1-20_v1-2/en/" + TEST_SET

GLOVE_VECTOR_FILE = "../sent_analysis_lstm/glove.6B/glove.6B.50d.txt"


# MODEL HYPERPARAMETERS
CELL_SIZE = 128     # no of dims between recurrent layer

GLOVE_DIM = 50 # The number of dims in GloVe wordvec
LEARNING_RATE = 0.005

DROPUT_INPUT, DROPUT_OUTPUT = 0.5, 0.5  # Dropout probability
BATCH_SIZE = 128
PASSES = 4  # No of passes in episodic memory
FF_HIDDEN_SIZE = 256
WEIGHT_DECAY = 0.00000001   # The strength of our regularization. Increase to encourage sparsity
            # in episodic memory, but makes training slower. Don't make this larger than leraning_rate

ITERATION = 400000  # How many questions the network trains on each time it is trained.
                    # Some questions are counted multiple times.

VALIDATION_STEP = 100   # How many iterations of training occur before each validation check.
