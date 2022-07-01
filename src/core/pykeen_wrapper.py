import gzip
import os
from typing import Optional, Tuple, Sequence, Dict

import numpy as np
import pandas as pd
import torch
from pykeen.models.base import Model
from pykeen.models.predict import predict_triples_df
from pykeen.pipeline import pipeline, PipelineResult
from pykeen.triples.triples_factory import CoreTriplesFactory
from pykeen.triples.triples_factory import TriplesFactory


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


def print_partitions_info(training_triples: TriplesFactory,
                          training_triples_path: str,
                          validation_triples: TriplesFactory,
                          validation_triples_path: str,
                          testing_triples: TriplesFactory,
                          testing_triples_path: str):
    assert "training" in training_triples_path
    assert "validation" in validation_triples_path
    assert "testing" in testing_triples_path
    # training
    print("\n\t (*) training_triples:")
    print(f"\t\t\t path={training_triples_path}")
    print(f"\t\t\t #triples={training_triples.num_triples}  | "
          f" #entities={training_triples.num_entities}  | "
          f" #relations={training_triples.num_relations} \n")
    # validation
    print("\t (*) validation_triples:")
    print(f"\t\t\t path={validation_triples_path}")
    print(f"\t\t\t #triples={validation_triples.num_triples}  | "
          f" #entities={validation_triples.num_entities}  | "
          f" #relations={validation_triples.num_relations} \n")
    # testing
    print("\t (*) testing_triples:")
    print(f"\t\t\t path={testing_triples_path}")
    print(f"\t\t\t #triples={testing_triples.num_triples}  | "
          f" #entities={testing_triples.num_entities}  | "
          f" #relations={testing_triples.num_relations} \n")


def train(training: TriplesFactory,
          testing: TriplesFactory,
          validation: Optional[TriplesFactory],
          model_name: str,
          model_kwargs: Optional[dict] = None,
          loss_kwargs: Optional[dict] = None,
          regularizer_kwargs: Optional[dict] = None,
          optimizer_kwargs: Optional[dict] = None,
          negative_sampler_kwargs: Optional[dict] = None,
          training_kwargs: Optional[dict] = None) -> PipelineResult:
    # === Manage training kwargs === #
    if training_kwargs is None:
        # create a new predefined dict with default values
        training_kwargs = {
            "num_epochs": 100,
            "batch_size": 128,
            "use_tqdm_batch": False,
        }
    else:
        # add a new entry to an already existent dict
        assert isinstance(training_kwargs, dict)
        training_kwargs["use_tqdm_batch"] = False

    # === Manage negative_sampler_kwargs === #
    if negative_sampler_kwargs is None:
        # create a new predefined dict with default values
        negative_sampler_kwargs = {
            "filtered": True,
            "filterer": "python-set",  # "bloom"
        }
    else:
        # add new entries to an already existent dict
        assert isinstance(negative_sampler_kwargs, dict)
        negative_sampler_kwargs["filtered"] = True
        negative_sampler_kwargs["filterer"] = "python-set"  # "bloom"

    # === Training and Evaluation === #
    return pipeline(
        # dataset args
        training=training,
        validation=validation,
        testing=testing,
        # model args
        model=model_name,
        model_kwargs=model_kwargs,
        # loss args
        loss_kwargs=loss_kwargs,
        # regularize args
        regularizer_kwargs=regularizer_kwargs,
        # optimizer args
        optimizer='Adam',
        optimizer_kwargs=optimizer_kwargs,
        clear_optimizer=True,
        # training Loop args
        training_loop='slcwa',
        negative_sampler='basic',
        negative_sampler_kwargs=negative_sampler_kwargs,
        # training args
        training_kwargs=training_kwargs,
        stopper=None,
        # evaluation args
        evaluator="RankBasedEvaluator",
        evaluator_kwargs={
            "filtered": True,
            # "batch_size": batch_size,
        },
        evaluation_kwargs={
            "use_tqdm": True,
            "additional_filter_triples": [
                training.mapped_triples,
                validation.mapped_triples,
            ],
        },
        # misc args
        device='cuda:0',  # 'cpu'
        # random_seed=11,
        use_testing_data=True,
        evaluation_fallback=True,
        filter_validation_when_testing=True,
        use_tqdm=True,
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


def get_triples_scores(trained_kge_model: Model,
                       triples: Sequence[Tuple[str, str, str]],
                       triples_factory: CoreTriplesFactory) -> np.ndarray:
    pred_df = predict_triples_df(
        model=trained_kge_model,
        triples=triples,
        triples_factory=triples_factory,
        batch_size=None,
        mode=None,  # "testing",
    )
    assert pred_df.shape[0] == len(triples)
    return pred_df["score"].values


def get_label_id_map(gzip_training_triples_path: str) -> Dict[str, int]:
    assert os.path.isfile(gzip_training_triples_path)
    assert gzip_training_triples_path.endswith(".gz")
    with gzip.open(gzip_training_triples_path, 'rb') as fr_entity:
        df_ids_labels = pd.read_csv(fr_entity, sep="\t", header="infer", encoding="utf-8")
    assert "id" in df_ids_labels.columns
    assert "label" in df_ids_labels.columns
    df_ids_labels["id"] = df_ids_labels["id"].astype(int)
    df_ids_labels["label"] = df_ids_labels["label"].astype(str)
    return dict(zip(df_ids_labels["label"], df_ids_labels["id"]))


def get_triples_scores2(trained_kge_model: Model,
                        triples: Sequence[Tuple[str, str, str]],
                        entities_label_id_map: Dict[str, int],
                        relation_label_id_map: Dict[str, int],
                        debug_info: bool = False) -> np.ndarray:
    mapped_triples = []
    num_error_triples = 0
    num_valid_triples = 0
    for h, r, t in triples:
        try:
            h_id = entities_label_id_map[h]
            r_id = relation_label_id_map[r]
            t_id = entities_label_id_map[t]
        except KeyError:
            num_error_triples += 1
            continue
        num_valid_triples += 1
        mapped_triples.append([h_id, r_id, t_id])
    mapped_triples_tensor = torch.tensor(mapped_triples,
                                         dtype=torch.long,
                                         device="cpu",
                                         requires_grad=False)
    if debug_info:
        print(f"#error_triples: {num_error_triples}  |  #valid_triples: {num_valid_triples}")
    pred_df = predict_triples_df(
        model=trained_kge_model,
        triples=mapped_triples_tensor,
        triples_factory=None,
        batch_size=None,
        mode=None,  # "testing",
    )
    assert pred_df.shape[0] == num_valid_triples
    return pred_df["score"].values


def get_triples_scores3(trained_kge_model: Model,
                         triples: Sequence[Tuple[str, str, str]],
                         entities_label_id_map: Dict[str, int],
                         relation_label_id_map: Dict[str, int]) -> np.ndarray:
    mapped_triples = []
    for h, r, t in triples:
        h_id = entities_label_id_map[h]
        r_id = relation_label_id_map[r]
        t_id = entities_label_id_map[t]
        mapped_triples.append([h_id, r_id, t_id])
    mapped_triples_tensor = torch.tensor(mapped_triples,
                                         dtype=torch.long,
                                         device="cpu",
                                         requires_grad=False)
    pred_df = predict_triples_df(
        model=trained_kge_model,
        triples=mapped_triples_tensor,
        triples_factory=None,
        batch_size=None,
        mode=None,  # "testing",
    )
    return pred_df["score"].values
