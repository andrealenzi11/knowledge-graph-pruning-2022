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
          validation: TriplesFactory,
          kge_model_obj: "Pykeen KGE model name") -> PipelineResult:
    return pipeline(
        training=training,
        validation=validation,
        testing=testing,
        model=kge_model_obj,
        # model_kwargs=dict(
        #     embedding_dim=50
        # ),
        # training_kwargs=dict(
        #     num_epochs=3,
        #     checkpoint_directory=checkpoint_folder_path,
        #     checkpoint_name='my_checkpoint.pt',
        #     checkpoint_frequency=10,
        #     checkpoint_on_failure=True,
        # ),
        optimizer='Adam',
        training_loop='sLCWA',
        negative_sampler='basic',
        evaluator='RankBasedEvaluator',
        stopper='early',
        random_seed=1,
        device='gpu',  # 'cpu'
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
