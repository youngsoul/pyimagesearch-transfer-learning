# Transfer Learning Configuration File
import os


# intialize the path to the *original* input directory of images
ORIG_INPUT_DATASET = os.path.sep.join(['/Volumes', 'MacBackup', 'Food-5k', 'Food-5k'])

# initialize the base path to the *new* or processed directory contain our images
# after computeing the training and test split
BASE_PATH = os.path.sep.join(['/Volumes', 'MacBackup', 'Food-5k'])

MODEL_DATASET_PATH = os.path.sep.join([BASE_PATH, 'model_dataset'])

# define the names of the training, testing, and validation directories
TRAIN = "training"
TEST = "evaluation"
VAL = "validation"

# initialize the list of class labels
# the reason non_food is in index 0 is because of the way the files are provided and name encoded.
# all non-food images are of the form:  0_<num>.jpg all food images are of the form: 1_<num>.jpg
CLASSES=['non_food', 'food']

BATCH_SIZE = 32

# initialize the label encoder file path and the output directory to where the
# extracted features ( in CSV file format ) will be stored.
output_dir = "output"
LE_PATH = os.path.sep.join([BASE_PATH,'{}', output_dir])
LE_FILE = os.path.sep.join([LE_PATH, "le.cpickle"])

BASE_CSV_PATH = os.path.sep.join([BASE_PATH, '{}', output_dir])

# set the path to the serialized  model after training
MODEL_PATH = os.path.sep.join([BASE_PATH, '{}', output_dir, "model.cpickle"])

