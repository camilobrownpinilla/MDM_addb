import os

#FILE PARAMS
DATA_PATH = "../data/addb_dataset/"
GEOMETRY_PATH = "../data/Geometry/"
DATASET = 'addb'
SAVE_DIR = "../save/"
OVERWRITE = False

#DIFFUSION PARAMS
DIFFUSION_STEPS = 1000
NOISE_SCHEDULE = 'cosine'
SIGMA_SMALL = False

#LOSS PARAMS
LAMBDA_RCXYZ = 0.0
LAMBA_VEL = 0.0
LAMBA_FC = 0.0

#TRAIN PARAMS
EVAL_SPLIT = 'test'
EVAL_DURING_TRAINING = False
EVAL_BATCH_SIZE = 32
BATCH_SIZE = 32
LR = 0.0002
LR_ANNEAL_STEPS = 0
WEIGHT_DECAY = 0.0
NUM_STEPS = 600_000

#LOGGING/SAVING
LOG_INTERVAL = 1000
SAVE_INTERVAL = 50_000
RESUME_CHECKPOINT = ""
