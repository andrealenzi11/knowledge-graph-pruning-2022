import os
from typing import Sequence

import requests


def create_non_existent_folder(folder_path: str):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)


def create_non_existent_folders(root_folder_path: str, sub_folders_paths: Sequence[str]):
    for sf_p in sub_folders_paths:
        folder_path = os.path.join(root_folder_path, sf_p)
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)


# ==================== Random Seeds ==================== #
RANDOM_SEED_TRAINING = (28, 2, 1995)
RANDOM_SEED_VALIDATION = (29, 12, 1958)
RANDOM_SEED_TESTING = (1, 5, 1000)
# ====================================================== #

# ==================== Fields Names ==================== #
HEAD = "HEAD"
RELATION = "RELATION"
TAIL = "TAIL"

FAKE_FLAG = "Y_FAKE"
# ====================================================== #

# ==================== Models Names ==================== #
RESCAL = "RESCAL"
TRANSE = "TransE"
DISTMULT = "DistMult"
TRANSH = "TransH"
# TRANSR = "TransR"
# TRANSD = "TransD"
COMPLEX = "ComplEx"
HOLE = "HolE"
CONVE = "ConvE"
# CONVKB = "ConvKB"
# RGCN = "RGCN"
ROTATE = "RotatE"
PAIRRE = "PairRE"
AUTOSF = "AutoSF"
BOXE = "BoxE"
# ======================================================== #

# ==================== Datasets Names ==================== #
FB15K237 = "FB15K237"
WN18RR = "WN18RR"
YAGO310 = "YAGO310"
COUNTRIES = "COUNTRIES"
CODEXSMALL = "CODEXSMALL"
NATIONS = "NATIONS"
# ======================================================== #

# ==================== Noise Levels ==================== #
ORIGINAL = "original"
TOTAL_RANDOM = "random"
NOISE_1 = "noise_1"
NOISE_5 = "noise_5"
NOISE_10 = "noise_10"
NOISE_15 = "noise_15"
NOISE_20 = "noise_20"
NOISE_25 = "noise_25"
NOISE_30 = "noise_30"
NOISE_100 = "noise_100"

ALL_NOISE_LEVELS = [ORIGINAL, TOTAL_RANDOM, NOISE_1, NOISE_5, NOISE_10, NOISE_15, NOISE_20, NOISE_25, NOISE_30, NOISE_100]
# ====================================================== #

# ==================== partitions names ==================== #
TRAINING = "training"
VALIDATION = "validation"
TESTING = "testing"
# ========================================================== #


# ==================== metrics names and evaluation strategy ==================== #
# Link Prediction Metrics
MR = "mr"  # arithmetic_mean_rank
MRR = "mrr"  # inverse_harmonic_mean_rank
HITS_AT_1 = "hits_at_1"  # hits_at_K=1
HITS_AT_3 = "hits_at_3"  # hits_at_K=3
HITS_AT_5 = "hits_at_5"  # hits_at_K=5
HITS_AT_10 = "hits_at_10"  # hits_at_K=10

# Triple Classification Metrics
F1_MACRO = "f1_macro"
F1_NEG = "f1_neg"
F1_POS = "f1_pos"
NORM_DIST = "norm_dist"
Z_STAT = "z_stat"

BOTH_STRATEGY = "both"
HEAD_STRATEGY = "head"
TAIL_STRATEGY = "tail"

REALISTIC_STRATEGY = "realistic"
OPTIMISTIC_STRATEGY = "optimistic"
PESSIMISTIC_STRATEGY = "pessimistic"
# ============================================================================== #


# ==================== First Level Resources Folders ==================== #
RESOURCES_DIR = os.path.join(os.environ['HOME'], "resources", "graph_pruning")

create_non_existent_folder(folder_path=RESOURCES_DIR)

DATASETS_DIR = os.path.join(RESOURCES_DIR, "datasets")
create_non_existent_folder(folder_path=DATASETS_DIR)

MODELS_DIR = os.path.join(RESOURCES_DIR, "models")
create_non_existent_folder(folder_path=MODELS_DIR)

CHECKPOINTS_DIR = os.path.join(RESOURCES_DIR, "checkpoints")
create_non_existent_folder(folder_path=CHECKPOINTS_DIR)

TUNING_DIR = os.path.join(RESOURCES_DIR, "tuning")
create_non_existent_folder(folder_path=TUNING_DIR)

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
                            sub_folders_paths=ALL_NOISE_LEVELS)

# wn18rr sub-folder
WN18RR_DATASETS_FOLDER_PATH = os.path.join(DATASETS_DIR, WN18RR)
create_non_existent_folder(folder_path=WN18RR_DATASETS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=WN18RR_DATASETS_FOLDER_PATH,
                            sub_folders_paths=ALL_NOISE_LEVELS)

# yago310 sub-folder
YAGO310_DATASETS_FOLDER_PATH = os.path.join(DATASETS_DIR, YAGO310)
create_non_existent_folder(folder_path=YAGO310_DATASETS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=YAGO310_DATASETS_FOLDER_PATH,
                            sub_folders_paths=ALL_NOISE_LEVELS)

# countries sub-folder
COUNTRIES_DATASETS_FOLDER_PATH = os.path.join(DATASETS_DIR, COUNTRIES)
create_non_existent_folder(folder_path=COUNTRIES_DATASETS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=COUNTRIES_DATASETS_FOLDER_PATH,
                            sub_folders_paths=ALL_NOISE_LEVELS)

# CoDExSmall sub-folder
CODEXSMALL_DATASETS_FOLDER_PATH = os.path.join(DATASETS_DIR, CODEXSMALL)
create_non_existent_folder(folder_path=CODEXSMALL_DATASETS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=CODEXSMALL_DATASETS_FOLDER_PATH,
                            sub_folders_paths=ALL_NOISE_LEVELS)

# Nations sub-folder
NATIONS_DATASETS_FOLDER_PATH = os.path.join(DATASETS_DIR, NATIONS)
create_non_existent_folder(folder_path=NATIONS_DATASETS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=NATIONS_DATASETS_FOLDER_PATH,
                            sub_folders_paths=ALL_NOISE_LEVELS)
# ================================================== #


# ==================== Models ==================== #
# fb15k237 sub-folder
FB15K237_MODELS_FOLDER_PATH = os.path.join(MODELS_DIR, FB15K237)
create_non_existent_folder(folder_path=FB15K237_MODELS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=FB15K237_MODELS_FOLDER_PATH,
                            sub_folders_paths=ALL_NOISE_LEVELS)

# wn18rr sub-folder
WN18RR_MODELS_FOLDER_PATH = os.path.join(MODELS_DIR, WN18RR)
create_non_existent_folder(folder_path=WN18RR_MODELS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=WN18RR_MODELS_FOLDER_PATH,
                            sub_folders_paths=ALL_NOISE_LEVELS)

# yago310 sub-folder
YAGO310_MODELS_FOLDER_PATH = os.path.join(MODELS_DIR, YAGO310)
create_non_existent_folder(folder_path=YAGO310_MODELS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=YAGO310_MODELS_FOLDER_PATH,
                            sub_folders_paths=ALL_NOISE_LEVELS)

# countries sub-folder
COUNTRIES_MODELS_FOLDER_PATH = os.path.join(MODELS_DIR, COUNTRIES)
create_non_existent_folder(folder_path=COUNTRIES_MODELS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=COUNTRIES_MODELS_FOLDER_PATH,
                            sub_folders_paths=ALL_NOISE_LEVELS)

# CoDExSmall sub-folder
CODEXSMALL_MODELS_FOLDER_PATH = os.path.join(MODELS_DIR, CODEXSMALL)
create_non_existent_folder(folder_path=CODEXSMALL_MODELS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=CODEXSMALL_MODELS_FOLDER_PATH,
                            sub_folders_paths=ALL_NOISE_LEVELS)

# Nations sub-folder
NATIONS_MODELS_FOLDER_PATH = os.path.join(MODELS_DIR, NATIONS)
create_non_existent_folder(folder_path=NATIONS_MODELS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=NATIONS_MODELS_FOLDER_PATH,
                            sub_folders_paths=ALL_NOISE_LEVELS)
# ================================================== #


# ==================== Checkpoints ==================== #
# fb15k237 sub-folder
FB15K237_CHECKPOINTS_FOLDER_PATH = os.path.join(CHECKPOINTS_DIR, FB15K237)
create_non_existent_folder(folder_path=FB15K237_CHECKPOINTS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=FB15K237_CHECKPOINTS_FOLDER_PATH,
                            sub_folders_paths=ALL_NOISE_LEVELS)

# wn18rr sub-folder
WN18RR_CHECKPOINTS_FOLDER_PATH = os.path.join(CHECKPOINTS_DIR, WN18RR)
create_non_existent_folder(folder_path=WN18RR_CHECKPOINTS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=WN18RR_CHECKPOINTS_FOLDER_PATH,
                            sub_folders_paths=ALL_NOISE_LEVELS)

# yago310 sub-folder
YAGO310_CHECKPOINTS_FOLDER_PATH = os.path.join(CHECKPOINTS_DIR, YAGO310)
create_non_existent_folder(folder_path=YAGO310_CHECKPOINTS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=YAGO310_CHECKPOINTS_FOLDER_PATH,
                            sub_folders_paths=ALL_NOISE_LEVELS)

# countries sub-folder
COUNTRIES_CHECKPOINTS_FOLDER_PATH = os.path.join(CHECKPOINTS_DIR, COUNTRIES)
create_non_existent_folder(folder_path=COUNTRIES_CHECKPOINTS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=COUNTRIES_CHECKPOINTS_FOLDER_PATH,
                            sub_folders_paths=ALL_NOISE_LEVELS)

# CoDExSmall sub-folder
CODEXSMALL_CHECKPOINTS_FOLDER_PATH = os.path.join(CHECKPOINTS_DIR, CODEXSMALL)
create_non_existent_folder(folder_path=CODEXSMALL_CHECKPOINTS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=CODEXSMALL_CHECKPOINTS_FOLDER_PATH,
                            sub_folders_paths=ALL_NOISE_LEVELS)

# Nations sub-folder
NATIONS_CHECKPOINTS_FOLDER_PATH = os.path.join(CHECKPOINTS_DIR, NATIONS)
create_non_existent_folder(folder_path=NATIONS_CHECKPOINTS_FOLDER_PATH)
create_non_existent_folders(root_folder_path=NATIONS_CHECKPOINTS_FOLDER_PATH,
                            sub_folders_paths=ALL_NOISE_LEVELS)
# ===================================================== #


# ==================== Tuning ==================== #
# fb15k237 sub-folder
FB15K237_TUNING_FOLDER_PATH = os.path.join(TUNING_DIR, FB15K237)
create_non_existent_folder(folder_path=FB15K237_TUNING_FOLDER_PATH)

# wn18rr sub-folder
WN18RR_TUNING_FOLDER_PATH = os.path.join(TUNING_DIR, WN18RR)
create_non_existent_folder(folder_path=WN18RR_TUNING_FOLDER_PATH)

# yago310 sub-folder
YAGO310_TUNING_FOLDER_PATH = os.path.join(TUNING_DIR, YAGO310)
create_non_existent_folder(folder_path=YAGO310_TUNING_FOLDER_PATH)

# countries sub-folder
COUNTRIES_TUNING_FOLDER_PATH = os.path.join(TUNING_DIR, COUNTRIES)
create_non_existent_folder(folder_path=COUNTRIES_TUNING_FOLDER_PATH)

# CoDExSmall sub-folder
CODEXSMALL_TUNING_FOLDER_PATH = os.path.join(TUNING_DIR, CODEXSMALL)
create_non_existent_folder(folder_path=CODEXSMALL_TUNING_FOLDER_PATH)

# Nations sub-folder
NATIONS_TUNING_FOLDER_PATH = os.path.join(TUNING_DIR, NATIONS)
create_non_existent_folder(folder_path=NATIONS_TUNING_FOLDER_PATH)
# ===================================================== #


# ==================== Results ==================== #
# fb15k237 sub-folder
FB15K237_RESULTS_FOLDER_PATH = os.path.join(RESULTS_DIR, FB15K237)
create_non_existent_folder(folder_path=FB15K237_RESULTS_FOLDER_PATH)

# wn18rr sub-folder
WN18RR_RESULTS_FOLDER_PATH = os.path.join(RESULTS_DIR, WN18RR)
create_non_existent_folder(folder_path=WN18RR_RESULTS_FOLDER_PATH)

# yago310 sub-folder
YAGO310_RESULTS_FOLDER_PATH = os.path.join(RESULTS_DIR, YAGO310)
create_non_existent_folder(folder_path=YAGO310_RESULTS_FOLDER_PATH)

# countries sub-folder
COUNTRIES_RESULTS_FOLDER_PATH = os.path.join(RESULTS_DIR, COUNTRIES)
create_non_existent_folder(folder_path=COUNTRIES_RESULTS_FOLDER_PATH)

# CoDExSmall sub-folder
CODEXSMALL_RESULTS_FOLDER_PATH = os.path.join(RESULTS_DIR, CODEXSMALL)
create_non_existent_folder(folder_path=CODEXSMALL_RESULTS_FOLDER_PATH)

# Nations sub-folder
NATIONS_RESULTS_FOLDER_PATH = os.path.join(RESULTS_DIR, NATIONS)
create_non_existent_folder(folder_path=NATIONS_RESULTS_FOLDER_PATH)
# ===================================================== #


# ===== Download FB15K237 Entities Mapping Json file
#       from Repo https://github.com/villmow/datasets_knowledge_embedding ===== #
FB15K237_MAPPING_FILE = os.path.join(FB15K237_DATASETS_FOLDER_PATH, "entity2wikidata.json")
FB15K237_MAPPING_URL = \
    "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237/entity2wikidata.json"

if not os.path.isfile(FB15K237_MAPPING_FILE):
    try:
        resp1 = requests.get(url=FB15K237_MAPPING_URL, verify=False)
        with open(FB15K237_MAPPING_FILE, 'wb') as fw1:
            fw1.write(resp1.content)
        # subprocess.run(['wget', '--no-check-certificate', FB15K237_MAPPING_URL, '-O', FB15K237_MAPPING_FILE])
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
        resp2 = requests.get(url=CODEXSMALL_ENTITIES_MAPPING_URL, verify=False)
        with open(CODEXSMALL_ENTITIES_MAPPING_FILE, 'wb') as fw2:
            fw2.write(resp2.content)
        # subprocess.run([
        #    'wget', '--no-check-certificate', CODEXSMALL_ENTITIES_MAPPING_URL, '-O', CODEXSMALL_ENTITIES_MAPPING_FILE
        # ])
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
        resp3 = requests.get(url=CODEXSMALL_RELATIONS_MAPPING_URL, verify=False)
        with open(CODEXSMALL_RELATIONS_MAPPING_FILE, 'wb') as fw3:
            fw3.write(resp3.content)
        # subprocess.run([
        #    'wget', '--no-check-certificate', CODEXSMALL_RELATIONS_MAPPING_URL, '-O', CODEXSMALL_RELATIONS_MAPPING_FILE
        # ])
    except Exception:
        raise ValueError(f"Error in download CODEXSMALL mapping file for Relations! \n"
                         f"\t\t (1) Download it manually from the following URL: {CODEXSMALL_RELATIONS_MAPPING_URL} \n"
                         f"\t\t (2) Put this Json file inside the folder: {CODEXSMALL_DATASETS_FOLDER_PATH} \n")
# =====================================================================================================================#
