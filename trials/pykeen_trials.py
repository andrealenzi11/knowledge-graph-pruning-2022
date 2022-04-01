import os
from typing import Optional, Tuple

import pandas as pd
import torch
from pykeen.pipeline import pipeline, PipelineResult
from pykeen.triples import TriplesFactory

from config.config import DATASETS_DIR, CHECKPOINTS_DIR, MODELS_DIR

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


dbpedia50_dir_path = os.path.join(DATASETS_DIR, "dbpedia50")
if not os.path.isdir(dbpedia50_dir_path):
    os.makedirs(dbpedia50_dir_path)


def get_training_validation_test(training_set_path: str,
                                 test_set_path: str,
                                 validation_set_path: Optional[str] = None,
                                 create_inverse_triples: bool = False) -> Tuple[TriplesFactory,
                                                                                Optional[TriplesFactory],
                                                                                TriplesFactory]:
    training = TriplesFactory.from_path(path=training_set_path,
                                        create_inverse_triples=create_inverse_triples)
    testing = TriplesFactory.from_path(path=test_set_path,
                                       entity_to_id=training.entity_to_id,
                                       relation_to_id=training.relation_to_id,
                                       create_inverse_triples=create_inverse_triples)
    if validation_set_path:
        validation = TriplesFactory.from_path(path=validation_set_path,
                                              entity_to_id=training.entity_to_id,
                                              relation_to_id=training.relation_to_id,
                                              create_inverse_triples=create_inverse_triples)
    else:
        validation = None
    return training, validation, testing


def train(training: TriplesFactory,
          validation: TriplesFactory,
          testing: TriplesFactory) -> PipelineResult:
    return pipeline(
        training=training,
        validation=validation,
        testing=testing,
        model='TransE',
        model_kwargs=dict(
            embedding_dim=50
        ),
        training_kwargs=dict(
            num_epochs=20,
            checkpoint_directory=CHECKPOINTS_DIR,
            checkpoint_name='my_checkpoint.pt',
            checkpoint_frequency=10,
            checkpoint_on_failure=True,
        ),
        optimizer='Adam',
        training_loop='sLCWA',
        negative_sampler='basic',
        evaluator='RankBasedEvaluator',
        stopper='early',
        random_seed=1,
        device='gpu',  # 'cpu'
    )


def store(result_model: PipelineResult, out_dir_path: str):
    result_model.save_to_directory(os.path.join(MODELS_DIR, out_dir_path))


def load(in_dir_path: str) -> torch.nn:
    return torch.load(os.path.join(MODELS_DIR, in_dir_path, 'trained_model.pkl'))


def get_entities_embeddings(model: "pykeen trained model") -> torch.FloatTensor:
    # Entity representations and relation representations
    entity_representation_modules = model.entity_representations
    # Most models  only have one representation for entities and one for relations
    entity_embeddings = entity_representation_modules[0]
    # Invoke the forward() (__call__) and get the values
    entity_embedding_tensor: torch.FloatTensor = entity_embeddings(indices=None)  # .detach().numpy()
    print(f"\n >>> entity_embedding_tensor (shape={entity_embedding_tensor.shape}): \n{entity_embedding_tensor}")
    return entity_embedding_tensor


def get_relations_embeddings(model: "pykeen trained model") -> torch.FloatTensor:
    print(type(model))
    # Entity representations and relation representations
    relations_representation_modules = model.relation_representations
    # Most models  only have one representation for entities and one for relations
    relations_embeddings = relations_representation_modules[0]
    # Invoke the forward() (__call__) and get the values
    relations_embedding_tensor: torch.FloatTensor = relations_embeddings(indices=None)  # .detach().numpy()
    print(f"\n>>> relation_embedding_tensor (shape={relations_embedding_tensor.shape}): \n{relations_embedding_tensor}")
    return relations_embedding_tensor


def relation_prediction(model: "pykeen trained model",
                        training: TriplesFactory,
                        head: str,
                        tail: str) -> pd.DataFrame:
    predicted_relations_df = model.get_relation_prediction_df(head_label=head,
                                                              tail_label=tail,
                                                              triples_factory=training,
                                                              add_novelties=True,
                                                              remove_known=False)
    print(f"\n >>> Relation Prediction: \n{predicted_relations_df}")
    return predicted_relations_df


def head_prediction(model: "pykeen trained model",
                    training: TriplesFactory,
                    relation: str,
                    tail: str) -> pd.DataFrame:
    predicted_head_df = model.get_head_prediction_df(relation_label=relation,
                                                     tail_label=tail,
                                                     triples_factory=training,
                                                     add_novelties=True,
                                                     remove_known=False)
    print(f"\n >>> Head Prediction: \n{predicted_head_df}")
    return predicted_head_df


def tail_prediction(model: "pykeen trained model",
                    training: TriplesFactory,
                    head: str,
                    relation: str) -> pd.DataFrame:
    predicted_tail_df = model.get_tail_prediction_df(head_label=head,
                                                     relation_label=relation,
                                                     triples_factory=training,
                                                     add_novelties=True,
                                                     remove_known=False)
    print(f"\n >>> Tail Prediction: \n{predicted_tail_df}")
    return predicted_tail_df


def all_prediction(model: "pykeen trained model",
                   training: TriplesFactory,
                   k: Optional[int] = None) -> pd.DataFrame:
    """
        Very slow
    """
    top_k_predictions_df = model.get_all_prediction_df(k=k,
                                                       triples_factory=training,
                                                       add_novelties=True,
                                                       remove_known=False)
    print(f"\n >>> Top K all Prediction: \n{top_k_predictions_df}")
    return top_k_predictions_df


if __name__ == '__main__':

    # Download tsv datasets from the following link and put them inside the 'dbpedia50_dir' folder:
    # https://github.com/ZhenfengLei/KGDatasets/tree/master/DBpedia50
    dbpedia50_training_path = os.path.join(dbpedia50_dir_path, "train.tsv")
    dbpedia50_validation_path = os.path.join(dbpedia50_dir_path, "validation.tsv")
    dbpedia50_test_path = os.path.join(dbpedia50_dir_path, "test.tsv")

    training_, validation_, testing_ = get_training_validation_test(training_set_path=dbpedia50_training_path,
                                                                    test_set_path=dbpedia50_test_path,
                                                                    validation_set_path=dbpedia50_validation_path,
                                                                    create_inverse_triples=False,)
    print("\n >>> Training Info:")
    print(training_)
    print("\n >>> Validation Info:")
    print(validation_)
    print(f"Top 3 most frequent relations: {validation_.get_most_frequent_relations(n=3)}")
    print("\n >>> Test Info:")
    print(testing_)
    print(f"Top 3 most frequent relations: {testing_.get_most_frequent_relations(n=3)}")

    res = train(training=training_, validation=validation_, testing=testing_)

    store(result_model=res, out_dir_path="prova1")

    trained_kge_model = load(in_dir_path="prova1")

    get_entities_embeddings(model=trained_kge_model)

    get_relations_embeddings(model=trained_kge_model)

    relation_prediction(model=trained_kge_model,
                        head='Albert_Einstein',
                        tail='award',
                        training=training_)

    tail_prediction(model=trained_kge_model,
                    head='Albert_Einstein',
                    relation='Nobel_Prize_in_Physics',
                    training=training_)

    head_prediction(model=trained_kge_model,
                    relation='Nobel_Prize_in_Physics',
                    tail="award",
                    training=training_)
