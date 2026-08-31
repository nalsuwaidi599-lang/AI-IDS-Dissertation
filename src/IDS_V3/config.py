# ---------------- EXPERIMENT SETTINGS ----------------

# Change this to:
# 'cicids' for CICIDS2017
# 'unsw' for UNSW-NB15
DATASET = 'unsw'

CICIDS_FOLDER = '/content/drive/MyDrive/CICIDS2017/'
UNSW_FOLDER = '/content/drive/MyDrive/UNSW-NB15/'
RESULTS_FOLDER = '/content/drive/MyDrive/IDS_Corrected_Results/'

QUICK_MODE = False

MAX_RECORDS = 100_000

# Five repeated runs for statistical reporting
SEEDS = [42, 43, 44, 45, 46]

# Maximum number kept after SHAP refinement
TOP_FEATURES = 15

EPOCHS = 20

BATCH_SIZE = 512
