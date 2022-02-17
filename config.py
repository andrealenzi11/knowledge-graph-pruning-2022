import os


# ==================== fields names ==================== #
HEAD = "HEAD"
RELATION = "RELATION"
TAIL = "TAIL"
# ====================================================== #


# ==================== folders ==================== #
RESOURCES_DIR = os.path.join(os.environ['HOME'], "resources", "graph_pruning")

DATASETS_DIR = os.path.join(RESOURCES_DIR, "datasets")
if not os.path.isdir(DATASETS_DIR):
    os.makedirs(DATASETS_DIR)

# repo with data
# https://github.com/villmow/datasets_knowledge_embedding

# COUNTRIES DATASET
COUNTRIES_ORIGINAL_TRAIN_PATH = os.path.join(DATASETS_DIR, "Countries", "original", "training.tsv")
COUNTRIES_ORIGINAL_VALID_PATH = os.path.join(DATASETS_DIR, "Countries", "original", "validation.tsv")
COUNTRIES_ORIGINAL_TEST_PATH = os.path.join(DATASETS_DIR, "Countries", "original", "testing.tsv")

# FB15k237 DATASET
FB15k237_ORIGINAL_TRAIN_PATH = os.path.join(DATASETS_DIR, "FB15k237", "original", "training.tsv")
FB15k237_ORIGINAL_VALID_PATH = os.path.join(DATASETS_DIR, "FB15k237", "original", "validation.tsv")
FB15k237_ORIGINAL_TEST_PATH = os.path.join(DATASETS_DIR, "FB15k237", "original", "testing.tsv")

# WN18RR DATASET
WN18RR_ORIGINAL_TRAIN_PATH = os.path.join(DATASETS_DIR, "WN18RR", "original", "training.tsv")
WN18RR_ORIGINAL_VALID_PATH = os.path.join(DATASETS_DIR, "WN18RR", "original", "validation.tsv")
WN18RR_ORIGINAL_TEST_PATH = os.path.join(DATASETS_DIR, "WN18RR", "original", "testing.tsv")

# WN18RR DATASET
YAGO310_ORIGINAL_TRAIN_PATH = os.path.join(DATASETS_DIR, "YAGO3-10", "original", "training.tsv")
YAGO310_ORIGINAL_VALID_PATH = os.path.join(DATASETS_DIR, "YAGO3-10", "original", "validation.tsv")
YAGO310_ORIGINAL_TEST_PATH = os.path.join(DATASETS_DIR, "YAGO3-10", "original", "testing.tsv")


MODELS_DIR = os.path.join(RESOURCES_DIR, "models")
if not os.path.isdir(MODELS_DIR):
    os.makedirs(MODELS_DIR)

CHECKPOINTS_DIR = os.path.join(RESOURCES_DIR, "checkpoints")
if not os.path.isdir(CHECKPOINTS_DIR):
    os.makedirs(CHECKPOINTS_DIR)
# ================================================= #
