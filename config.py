import os
import subprocess


def create_non_existent_folder(folder_path: str):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)


# ==================== fields names ==================== #
HEAD = "HEAD"
RELATION = "RELATION"
TAIL = "TAIL"

FAKE_FLAG = "Y_FAKE"
# ====================================================== #


# ==================== First Level Resources Folders ==================== #
RESOURCES_DIR = os.path.join(os.environ['HOME'], "resources", "graph_pruning")

DATASETS_DIR = os.path.join(RESOURCES_DIR, "datasets")
create_non_existent_folder(folder_path=DATASETS_DIR)

MODELS_DIR = os.path.join(RESOURCES_DIR, "models")
create_non_existent_folder(folder_path=MODELS_DIR)

CHECKPOINTS_DIR = os.path.join(RESOURCES_DIR, "checkpoints")
create_non_existent_folder(folder_path=CHECKPOINTS_DIR)
# ======================================================================= #


# ==================== Datasets ==================== #
# datasets names
FB15K237 = "FB15K237"
WN18RR = "WN18RR"
YAGO310 = "YAGO310"
COUNTRIES = "COUNTRIES"

# noise levels
ORIGINAL = "original"
NOISE_1 = "noise_1"
NOISE_5 = "noise_5"
NOISE_10 = "noise_10"

# partitions names
TRAINING = "training"
VALIDATION = "validation"
TESTING = "testing"

# files names
TRAINING_TSV = "training.tsv"
TRAINING_Y_FAKE_TSV = "training_y_fake.tsv"
VALIDATION_TSV = "validation.tsv"
VALIDATION_Y_FAKE_TSV = "validation_y_fake.tsv"
TESTING_TSV = "testing.tsv"
TESTING_Y_FAKE_TSV = "testing_y_fake.tsv"

# fb15k237 sub-folder
FB15K237_FOLDER_PATH = os.path.join(DATASETS_DIR, FB15K237)
create_non_existent_folder(folder_path=FB15K237_FOLDER_PATH)
create_non_existent_folder(folder_path=os.path.join(FB15K237_FOLDER_PATH, ORIGINAL))
create_non_existent_folder(folder_path=os.path.join(FB15K237_FOLDER_PATH, NOISE_1))
create_non_existent_folder(folder_path=os.path.join(FB15K237_FOLDER_PATH, NOISE_5))
create_non_existent_folder(folder_path=os.path.join(FB15K237_FOLDER_PATH, NOISE_10))

# wn18rr sub-folder
WN18RR_FOLDER_PATH = os.path.join(DATASETS_DIR, WN18RR)
create_non_existent_folder(folder_path=WN18RR_FOLDER_PATH)
create_non_existent_folder(folder_path=os.path.join(WN18RR_FOLDER_PATH, ORIGINAL))
create_non_existent_folder(folder_path=os.path.join(WN18RR_FOLDER_PATH, NOISE_1))
create_non_existent_folder(folder_path=os.path.join(WN18RR_FOLDER_PATH, NOISE_5))
create_non_existent_folder(folder_path=os.path.join(WN18RR_FOLDER_PATH, NOISE_10))

# yago310 sub-folder
YAGO310_FOLDER_PATH = os.path.join(DATASETS_DIR, YAGO310)
create_non_existent_folder(folder_path=YAGO310_FOLDER_PATH)
create_non_existent_folder(folder_path=os.path.join(YAGO310_FOLDER_PATH, ORIGINAL))
create_non_existent_folder(folder_path=os.path.join(YAGO310_FOLDER_PATH, NOISE_1))
create_non_existent_folder(folder_path=os.path.join(YAGO310_FOLDER_PATH, NOISE_5))
create_non_existent_folder(folder_path=os.path.join(YAGO310_FOLDER_PATH, NOISE_10))

# countries sub-folder
COUNTRIES_FOLDER_PATH = os.path.join(DATASETS_DIR, COUNTRIES)
create_non_existent_folder(folder_path=COUNTRIES_FOLDER_PATH)
create_non_existent_folder(folder_path=os.path.join(COUNTRIES_FOLDER_PATH, ORIGINAL))
create_non_existent_folder(folder_path=os.path.join(COUNTRIES_FOLDER_PATH, NOISE_1))
create_non_existent_folder(folder_path=os.path.join(COUNTRIES_FOLDER_PATH, NOISE_5))
create_non_existent_folder(folder_path=os.path.join(COUNTRIES_FOLDER_PATH, NOISE_10))
# ================================================== #


# ===== Download FB15K237 Mapping Json file ===== #
# repo with mapping: https://github.com/villmow/datasets_knowledge_embedding
FB15K237_MAPPING_FILE = os.path.join(FB15K237_FOLDER_PATH, "entity2wikidata.json")
FB15K237_MAPPING_URL = \
    "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237/entity2wikidata.json"

if not os.path.isfile(FB15K237_MAPPING_FILE):
    try:
        subprocess.run(['wget', '--no-check-certificate', FB15K237_MAPPING_URL, '-O', FB15K237_MAPPING_FILE])
    except Exception:
        raise ValueError(f"Error in download FB15K237 mapping file! \n"
                         f"\t\t (1) Download it manually from the following URL: {FB15K237_MAPPING_URL} \n"
                         f"\t\t (2) Put this Json file inside the folder: {FB15K237_FOLDER_PATH} \n")
# ================================================== #
