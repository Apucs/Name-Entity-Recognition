import os

DATA_BASE_PATH = "dataset"

FILE_NAME1 = os.path.sep.join([DATA_BASE_PATH, "conllpp_dev.txt"])
FILE_NAME2 = os.path.sep.join([DATA_BASE_PATH, "conllpp_test.txt"])
FILE_NAME3 = os.path.sep.join([DATA_BASE_PATH, "conllpp_train.txt"])


NEW_FILE_NAME1 = os.path.sep.join([DATA_BASE_PATH, "conllpp_dev.csv"])
NEW_FILE_NAME2 = os.path.sep.join([DATA_BASE_PATH, "conllpp_test.csv"])
NEW_FILE_NAME3 = os.path.sep.join([DATA_BASE_PATH, "conllpp_train.csv"])


UP_FILE_NAME1 = os.path.sep.join([DATA_BASE_PATH, "conllpp_up_dev.csv"])
UP_FILE_NAME2 = os.path.sep.join([DATA_BASE_PATH, "conllpp_up_test.csv"])
UP_FILE_NAME3 = os.path.sep.join([DATA_BASE_PATH, "conllpp_up_train.csv"])


CHECKPOINT_PATH = "checkpoint"

CHECKPOINT1 = os.path.sep.join([CHECKPOINT_PATH, "checkpoint_saved.pth"])
CHECKPOINT2 = os.path.sep.join([CHECKPOINT_PATH, "checkpoint.pth"])
CHECKPOINT3 = os.path.sep.join([CHECKPOINT_PATH, "model_scripted.pt"])

CHECKPOINT4 = os.path.sep.join([CHECKPOINT_PATH, "model_last.pt"])

