import os


# ==================== fields names ==================== #
HEAD = "HEAD"
RELATION = "RELATION"
TAIL = "TAIL"
FAKE_FLAG = "FAKE_FLAG"
# ====================================================== #


# ==================== folders ==================== #
RESOURCES_DIR = os.path.join(os.environ['HOME'], "resources", "graph_pruning")

DATASETS_DIR = os.path.join(RESOURCES_DIR, "datasets")
if not os.path.isdir(DATASETS_DIR):
    os.makedirs(DATASETS_DIR)

# repo with data
# https://github.com/villmow/datasets_knowledge_embedding

MODELS_DIR = os.path.join(RESOURCES_DIR, "models")
if not os.path.isdir(MODELS_DIR):
    os.makedirs(MODELS_DIR)

CHECKPOINTS_DIR = os.path.join(RESOURCES_DIR, "checkpoints")
if not os.path.isdir(CHECKPOINTS_DIR):
    os.makedirs(CHECKPOINTS_DIR)
# ================================================= #
