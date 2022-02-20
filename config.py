import os
import subprocess


def create_non_existent_folder(folder_path: str):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)


# ==================== fields names ==================== #
HEAD = "HEAD"
RELATION = "RELATION"
TAIL = "TAIL"
FAKE_FLAG = "FAKE_FLAG"
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
FB15K237 = "FB15K237"
WN18RR = "WN18RR"
YAGO310 = "YAGO310"
COUNTRIES = "COUNTRIES"

FB15K237_FOLDER_PATH = os.path.join(DATASETS_DIR, FB15K237)
create_non_existent_folder(folder_path=FB15K237_FOLDER_PATH)
create_non_existent_folder(folder_path=os.path.join(FB15K237_FOLDER_PATH, "original"))
create_non_existent_folder(folder_path=os.path.join(FB15K237_FOLDER_PATH, "noise_1"))
create_non_existent_folder(folder_path=os.path.join(FB15K237_FOLDER_PATH, "noise_5"))
create_non_existent_folder(folder_path=os.path.join(FB15K237_FOLDER_PATH, "noise_10"))

WN18RR_FOLDER_PATH = os.path.join(DATASETS_DIR, WN18RR)
create_non_existent_folder(folder_path=WN18RR_FOLDER_PATH)
create_non_existent_folder(folder_path=os.path.join(WN18RR_FOLDER_PATH, "original"))
create_non_existent_folder(folder_path=os.path.join(WN18RR_FOLDER_PATH, "noise_1"))
create_non_existent_folder(folder_path=os.path.join(WN18RR_FOLDER_PATH, "noise_5"))
create_non_existent_folder(folder_path=os.path.join(WN18RR_FOLDER_PATH, "noise_10"))

YAGO310_FOLDER_PATH = os.path.join(DATASETS_DIR, YAGO310)
create_non_existent_folder(folder_path=YAGO310_FOLDER_PATH)
create_non_existent_folder(folder_path=os.path.join(YAGO310_FOLDER_PATH, "original"))
create_non_existent_folder(folder_path=os.path.join(YAGO310_FOLDER_PATH, "noise_1"))
create_non_existent_folder(folder_path=os.path.join(YAGO310_FOLDER_PATH, "noise_5"))
create_non_existent_folder(folder_path=os.path.join(YAGO310_FOLDER_PATH, "noise_10"))

COUNTRIES_FOLDER_PATH = os.path.join(DATASETS_DIR, COUNTRIES)
create_non_existent_folder(folder_path=COUNTRIES_FOLDER_PATH)
create_non_existent_folder(folder_path=os.path.join(COUNTRIES_FOLDER_PATH, "original"))
create_non_existent_folder(folder_path=os.path.join(COUNTRIES_FOLDER_PATH, "noise_1"))
create_non_existent_folder(folder_path=os.path.join(COUNTRIES_FOLDER_PATH, "noise_5"))
create_non_existent_folder(folder_path=os.path.join(COUNTRIES_FOLDER_PATH, "noise_10"))
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
