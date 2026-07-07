import os

# paths to dataset folders - just put the files in these folders
# the pipeline will run whichever ones it finds
CICIDS_FOLDER = "./cicids2017/"
UNSW_FOLDER = "./unsw-nb15/"

# train test split
TEST_SIZE = 0.2
RANDOM_STATE = 42
SCALER = 'standard'

# feature selection params
MI_NEIGHBOURS = 5
RFE_ESTIMATORS = 50
RFE_STEP = 3
N_FEATURES = 15
CORRELATION_THRESHOLD = 0.95

# model params
FFNN_LAYERS = (128, 64, 32)
FFNN_MAX_ITER = 300

CNN_ESTIMATORS = 100
CNN_MAX_DEPTH = 4        # kept lower to avoid overfitting
CNN_LEARNING_RATE = 0.05

LSTM_ESTIMATORS = 150
LSTM_MAX_DEPTH = 8

OUTPUT_DIR = "./outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
