import os
import subprocess
from typing import Sequence


def create_non_existent_folder(folder_path: str):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)


def create_non_existent_folders(root_folder_path: str, sub_folders_paths: Sequence[str]):
    for sf_p in sub_folders_paths:
        folder_path = os.path.join(root_folder_path, sf_p)
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)


# ==================== Random Seeds ==================== #
RANDOM_SEED_HEAD_SAMPLING = 11
RANDOM_SEED_RELATION_SAMPLING = 500
RANDOM_SEED_TAIL_SAMPLING = 1200
# ====================================================== #

# ==================== Fields Names ==================== #
HEAD = "HEAD"
RELATION = "RELATION"
TAIL = "TAIL"

FAKE_FLAG = "Y_FAKE"
# ====================================================== #

# ==================== Datasets Names ==================== #
FB15K237 = "FB15K237"
WN18RR = "WN18RR"
YAGO310 = "YAGO310"
COUNTRIES = "COUNTRIES"
CODEXSMALL = "CODEXSMALL"
# ======================================================== #

# ==================== Noise Levels ==================== #
ORIGINAL = "original"
NOISE_1 = "noise_1"
NOISE_5 = "noise_5"
NOISE_10 = "noise_10"
NOISE_15 = "noise_15"
# ====================================================== #

# ==================== partitions names ==================== #
TRAINING = "training"
VALIDATION = "validation"
TESTING = "testing"
# ========================================================== #

# ==================== First Level Resources Folders ==================== #
RESOURCES_DIR = os.path.join(os.environ['HOME'], "resources", "graph_pruning")

DATASETS_DIR = os.path.join(RESOURCES_DIR, "datasets")
create_non_existent_folder(folder_path=DATASETS_DIR)

MODELS_DIR = os.path.join(RESOURCES_DIR, "models")
create_non_existent_folder(folder_path=MODELS_DIR)

CHECKPOINTS_DIR = os.path.join(RESOURCES_DIR, "checkpoints")
create_non_existent_folder(folder_path=CHECKPOINTS_DIR)

RESULTS_DIR = os.path.join(RESOURCES_DIR, "results")
create_non_existent_folder(folder_path=RESULTS_DIR)
# ======================================================================= #


# ==================== Datasets ==================== #
# Datasets Files Names
TRAINING_TSV = "training.tsv"
TRAINING_Y_FAKE_TSV = "training_y_fake.tsv"
VALIDATION_TSV = "validation.tsv"
VALIDATION_Y_FAKE_TSV = "validation_y_fake.tsv"
TESTING_TSV = "testing.tsv"
TESTING_Y_FAKE_TSV = "testing_y_fake.tsv"

# fb15k237 sub-folder
FB15K237_DATASETS_FOLDER_PATH = os.path.join(DATASETS_DIR, FB15K237)
create_non_existent_folder(folder_path=FB15K237_DATASETS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=FB15K237_DATASETS_FOLDER_PATH,
                            sub_folders_paths=[ORIGINAL, NOISE_1, NOISE_5, NOISE_10, NOISE_15])

# wn18rr sub-folder
WN18RR_DATASETS_FOLDER_PATH = os.path.join(DATASETS_DIR, WN18RR)
create_non_existent_folder(folder_path=WN18RR_DATASETS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=WN18RR_DATASETS_FOLDER_PATH,
                            sub_folders_paths=[ORIGINAL, NOISE_1, NOISE_5, NOISE_10, NOISE_15])

# yago310 sub-folder
YAGO310_DATASETS_FOLDER_PATH = os.path.join(DATASETS_DIR, YAGO310)
create_non_existent_folder(folder_path=YAGO310_DATASETS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=YAGO310_DATASETS_FOLDER_PATH,
                            sub_folders_paths=[ORIGINAL, NOISE_1, NOISE_5, NOISE_10, NOISE_15])

# countries sub-folder
COUNTRIES_DATASETS_FOLDER_PATH = os.path.join(DATASETS_DIR, COUNTRIES)
create_non_existent_folder(folder_path=COUNTRIES_DATASETS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=COUNTRIES_DATASETS_FOLDER_PATH,
                            sub_folders_paths=[ORIGINAL, NOISE_1, NOISE_5, NOISE_10, NOISE_15])

# CoDExSmall sub-folder
CODEXSMALL_DATASETS_FOLDER_PATH = os.path.join(DATASETS_DIR, CODEXSMALL)
create_non_existent_folder(folder_path=CODEXSMALL_DATASETS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=CODEXSMALL_DATASETS_FOLDER_PATH,
                            sub_folders_paths=[ORIGINAL, NOISE_1, NOISE_5, NOISE_10, NOISE_15])
# ================================================== #


# ==================== Models ==================== #
# fb15k237 sub-folder
FB15K237_MODELS_FOLDER_PATH = os.path.join(MODELS_DIR, FB15K237)
create_non_existent_folder(folder_path=FB15K237_MODELS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=FB15K237_MODELS_FOLDER_PATH,
                            sub_folders_paths=[ORIGINAL, NOISE_1, NOISE_5, NOISE_10, NOISE_15])

# wn18rr sub-folder
WN18RR_MODELS_FOLDER_PATH = os.path.join(MODELS_DIR, WN18RR)
create_non_existent_folder(folder_path=WN18RR_MODELS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=WN18RR_MODELS_FOLDER_PATH,
                            sub_folders_paths=[ORIGINAL, NOISE_1, NOISE_5, NOISE_10, NOISE_15])

# yago310 sub-folder
YAGO310_MODELS_FOLDER_PATH = os.path.join(MODELS_DIR, YAGO310)
create_non_existent_folder(folder_path=YAGO310_MODELS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=YAGO310_MODELS_FOLDER_PATH,
                            sub_folders_paths=[ORIGINAL, NOISE_1, NOISE_5, NOISE_10, NOISE_15])


# countries sub-folder
COUNTRIES_MODELS_FOLDER_PATH = os.path.join(MODELS_DIR, COUNTRIES)
create_non_existent_folder(folder_path=COUNTRIES_MODELS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=COUNTRIES_MODELS_FOLDER_PATH,
                            sub_folders_paths=[ORIGINAL, NOISE_1, NOISE_5, NOISE_10, NOISE_15])

# CoDExSmall sub-folder
CODEXSMALL_MODELS_FOLDER_PATH = os.path.join(MODELS_DIR, CODEXSMALL)
create_non_existent_folder(folder_path=CODEXSMALL_MODELS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=CODEXSMALL_MODELS_FOLDER_PATH,
                            sub_folders_paths=[ORIGINAL, NOISE_1, NOISE_5, NOISE_10, NOISE_15])
# ================================================== #

# ==================== Checkpoints ==================== #
# fb15k237 sub-folder
FB15K237_CHECKPOINTS_FOLDER_PATH = os.path.join(CHECKPOINTS_DIR, FB15K237)
create_non_existent_folder(folder_path=FB15K237_CHECKPOINTS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=FB15K237_CHECKPOINTS_FOLDER_PATH,
                            sub_folders_paths=[ORIGINAL, NOISE_1, NOISE_5, NOISE_10, NOISE_15])

# wn18rr sub-folder
WN18RR_CHECKPOINTS_FOLDER_PATH = os.path.join(CHECKPOINTS_DIR, WN18RR)
create_non_existent_folder(folder_path=WN18RR_CHECKPOINTS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=WN18RR_CHECKPOINTS_FOLDER_PATH,
                            sub_folders_paths=[ORIGINAL, NOISE_1, NOISE_5, NOISE_10, NOISE_15])

# yago310 sub-folder
YAGO310_CHECKPOINTS_FOLDER_PATH = os.path.join(CHECKPOINTS_DIR, YAGO310)
create_non_existent_folder(folder_path=YAGO310_CHECKPOINTS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=YAGO310_CHECKPOINTS_FOLDER_PATH,
                            sub_folders_paths=[ORIGINAL, NOISE_1, NOISE_5, NOISE_10, NOISE_15])

# countries sub-folder
COUNTRIES_CHECKPOINTS_FOLDER_PATH = os.path.join(CHECKPOINTS_DIR, COUNTRIES)
create_non_existent_folder(folder_path=COUNTRIES_CHECKPOINTS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=COUNTRIES_CHECKPOINTS_FOLDER_PATH,
                            sub_folders_paths=[ORIGINAL, NOISE_1, NOISE_5, NOISE_10, NOISE_15])

# CoDExSmall sub-folder
CODEXSMALL_CHECKPOINTS_FOLDER_PATH = os.path.join(CHECKPOINTS_DIR, CODEXSMALL)
create_non_existent_folder(folder_path=CODEXSMALL_CHECKPOINTS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=CODEXSMALL_CHECKPOINTS_FOLDER_PATH,
                            sub_folders_paths=[ORIGINAL, NOISE_1, NOISE_5, NOISE_10, NOISE_15])
# ===================================================== #


# ===== Download FB15K237 Entities Mapping Json file
#       from Repo https://github.com/villmow/datasets_knowledge_embedding ===== #
FB15K237_MAPPING_FILE = os.path.join(FB15K237_DATASETS_FOLDER_PATH, "entity2wikidata.json")
FB15K237_MAPPING_URL = \
    "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237/entity2wikidata.json"

if not os.path.isfile(FB15K237_MAPPING_FILE):
    try:
        subprocess.run(['wget', '--no-check-certificate', FB15K237_MAPPING_URL, '-O', FB15K237_MAPPING_FILE])
    except Exception:
        raise ValueError(f"Error in download FB15K237 mapping file! \n"
                         f"\t\t (1) Download it manually from the following URL: {FB15K237_MAPPING_URL} \n"
                         f"\t\t (2) Put this Json file inside the folder: {FB15K237_DATASETS_FOLDER_PATH} \n")
# ============================================================================== #


# ===== Download CODEXSMALL Entities/Relations Mapping Json files from Repo https://github.com/tsafavi/codex ===== #

# entities map
CODEXSMALL_ENTITIES_MAPPING_FILE = os.path.join(CODEXSMALL_DATASETS_FOLDER_PATH, "entities.json")
CODEXSMALL_ENTITIES_MAPPING_URL = \
    "https://raw.githubusercontent.com/tsafavi/codex/master/data/entities/en/entities.json"

if not os.path.isfile(CODEXSMALL_ENTITIES_MAPPING_FILE):
    try:
        subprocess.run([
            'wget', '--no-check-certificate', CODEXSMALL_ENTITIES_MAPPING_URL, '-O', CODEXSMALL_ENTITIES_MAPPING_FILE
        ])
    except Exception:
        raise ValueError(f"Error in download CODEXSMALL mapping file for Entities! \n"
                         f"\t\t (1) Download it manually from the following URL: {CODEXSMALL_ENTITIES_MAPPING_URL} \n"
                         f"\t\t (2) Put this Json file inside the folder: {CODEXSMALL_DATASETS_FOLDER_PATH} \n")

# relations map
CODEXSMALL_RELATIONS_MAPPING_FILE = os.path.join(CODEXSMALL_DATASETS_FOLDER_PATH, "relations.json")
CODEXSMALL_RELATIONS_MAPPING_URL = \
    "https://raw.githubusercontent.com/tsafavi/codex/master/data/relations/en/relations.json"

if not os.path.isfile(CODEXSMALL_RELATIONS_MAPPING_FILE):
    try:
        subprocess.run([
            'wget', '--no-check-certificate', CODEXSMALL_RELATIONS_MAPPING_URL, '-O', CODEXSMALL_RELATIONS_MAPPING_FILE
        ])
    except Exception:
        raise ValueError(f"Error in download CODEXSMALL mapping file for Relations! \n"
                         f"\t\t (1) Download it manually from the following URL: {CODEXSMALL_RELATIONS_MAPPING_URL} \n"
                         f"\t\t (2) Put this Json file inside the folder: {CODEXSMALL_DATASETS_FOLDER_PATH} \n")
# =====================================================================================================================#
