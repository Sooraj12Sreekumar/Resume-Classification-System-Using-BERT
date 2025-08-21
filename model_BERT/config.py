import os
from pathlib import Path

DATA_PATH = "data/processed/"
TRAIN_FILE = DATA_PATH + "train.csv"
TEST_FILE = DATA_PATH + "test.csv"
VAL_FILE = DATA_PATH + "val.csv"
LOG_DIR = Path("logs/")


MODEL_SAVE_PATH = "model_BERT/checkpoints/"
TOKENIZER_SAVE_PATH = "model_BERT/tokenizers/"

MAX_LEN = 128
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
EPOCHS = 6
LEARNING_RATE = 5e-5
PATIENCE = 2
