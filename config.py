import os


# ==================== fields names ==================== #
SUBJ = "subject"
RELATION = "relation"
OBJ = "object"
# ====================================================== #


# ==================== folders ==================== #
RESOURCES_DIR = "/home/andrea/resources/graph_pruning/"

DATASETS_DIR = os.path.join(RESOURCES_DIR, "datasets")
if not os.path.isdir(DATASETS_DIR):
    os.makedirs(DATASETS_DIR)

MODELS_DIR = os.path.join(RESOURCES_DIR, "models")
if not os.path.isdir(MODELS_DIR):
    os.makedirs(MODELS_DIR)

CHECKPOINTS_DIR = os.path.join(RESOURCES_DIR, "checkpoints")
if not os.path.isdir(CHECKPOINTS_DIR):
    os.makedirs(CHECKPOINTS_DIR)
# ================================================= #
