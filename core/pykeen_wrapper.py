import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from pykeen.pipeline import pipeline, PipelineResult
from pykeen.triples import TriplesFactory


def get_train_test_validation(training_set_path: str,
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
    return training, testing, validation


def get_train_test_validation_2(knowledge_graph_path: str,
                                create_inverse_triples: bool = False,
                                train_fraction: float = 0.8,
                                test_fraction: float = 0.1,
                                validation_fraction: float = 0.1,
                                random_state: Optional[int] = None) -> Tuple[TriplesFactory,
                                                                             TriplesFactory,
                                                                             TriplesFactory]:
    tf = TriplesFactory.from_path(path=knowledge_graph_path,
                                  create_inverse_triples=create_inverse_triples)
    training, testing, validation = tf.split(ratios=[train_fraction,
                                                     test_fraction,
                                                     validation_fraction],
                                             random_state=random_state)
    return training, testing, validation


def train(training: TriplesFactory,
          testing: TriplesFactory,
          validation: Optional[TriplesFactory],
          model_name: str,
          model_kwargs: Optional[dict] = None,
          training_kwargs: Optional[dict] = None,
          loss_kwargs: Optional[dict] = None,
          regularizer_kwargs: Optional[dict] = None,
          optimizer_kwargs: Optional[dict] = None,
          negative_sampler_kwargs: Optional[dict] = None,
          stopper: Optional[str] = None) -> PipelineResult:
    # manage training kwargs
    if training_kwargs is None:
        training_kwargs = {
            "num_epochs": 100,
            "batch_size": 128,
            "use_tqdm_batch": False
        }
        batch_size = 128
    else:
        training_kwargs["use_tqdm_batch"] = False
        batch_size = training_kwargs["batch_size"]
    # training and evaluation
    return pipeline(
        training=training,
        validation=validation,
        testing=testing,
        dataset_kwargs={
            "create_inverse_triples": False,
        },
        model=model_name,
        model_kwargs=model_kwargs,
        training_kwargs=training_kwargs,
        optimizer='Adam',
        optimizer_kwargs=optimizer_kwargs,
        clear_optimizer=True,
        loss_kwargs=loss_kwargs,
        regularizer_kwargs=regularizer_kwargs,
        training_loop='slcwa',
        negative_sampler='basic',
        negative_sampler_kwargs=negative_sampler_kwargs,
        stopper=stopper,
        evaluator="RankBasedEvaluator",
        evaluator_kwargs={
            "batch_size": batch_size,
        },
        evaluation_kwargs={
            "use_tqdm": True
        },
        use_testing_data=True,
        device='gpu',  # 'cpu'
        use_tqdm=True,
        evaluation_fallback=True,
        filter_validation_when_testing=False,
    )


def store(result_model: PipelineResult, out_dir_path: str):
    result_model.save_to_directory(directory=out_dir_path)


def load(in_dir_path: str) -> torch.nn:
    return torch.load(os.path.join(in_dir_path, 'trained_model.pkl'))


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
    print(predicted_head_df["head_label"].values[:5])
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
    print(predicted_tail_df["tail_label"].values[:5])
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


def triples_prediction(model: "pykeen trained model",
                       triples: "indices of (h, r, t) triples.") -> np.ndarray:
    return model.predict_scores(triples=triples)
