import json
import os

from optuna.pruners import PercentilePruner
from optuna.samplers import TPESampler
from pykeen.hpo import hpo_pipeline

from config.config import ORIGINAL, COUNTRIES, WN18RR, FB15K237, YAGO310, CODEXSMALL
from config.config import RESCAL, TRANSE, DISTMULT, TRANSH, COMPLEX, HOLE, CONVE, ROTATE, PAIRRE, AUTOSF, BOXE
from core.fabrication import DatasetPathFactory
from core.pykeen_wrapper import get_train_test_validation
from dao.dataset_loading import TsvDatasetLoader

all_datasets_names = {
    COUNTRIES,
    WN18RR,
    FB15K237,
    YAGO310,
    CODEXSMALL,
}

valid_kge_models = {
    RESCAL,
    TRANSE,
    DISTMULT,
    TRANSH,
    COMPLEX,
    HOLE,
    CONVE,
    ROTATE,
    PAIRRE,
    AUTOSF,
    BOXE,
}


if __name__ == '__main__':

    # === Set your Configuration === #
    configuration = dict(
        dataset_name=COUNTRIES,
        model_name=BOXE,
        num_trials=30,
        num_startup_trials_sampler=20,  # default: 10
        num_expected_improvement_candidates_sampler=32,  # default: 24
        num_startup_trials_pruner=5,  # default: 5
        percentile_pruner=70.0,
        max_batch_size=256,
        max_num_epoch=200,
    )

    # === Check Configuration === #
    if configuration["dataset_name"] not in all_datasets_names:
        raise ValueError(f"Invalid dataset name '{configuration['dataset_name']}'!")
    if configuration["model_name"] not in valid_kge_models:
        raise ValueError(f"Invalid model name '{configuration['model_name']}'!")

    # ==== Print Current Configuration=== #
    print("\n>>>>>>>> CONFIGURATION <<<<<<<<")
    for k, v in configuration.items():
        print(f"\t\t {k} = {v}")
    print(">>>>>>>>>>>>>>><<<<<<<<<<<<<<< \n")

    # === Get Input Dataset: Training, Validation, Testing === #
    datasets_loader = TsvDatasetLoader(dataset_name=configuration["dataset_name"],
                                       noise_level=ORIGINAL)
    training_path, validation_path, testing_path = \
        datasets_loader.get_training_validation_testing_dfs_paths(noisy_test_flag=False)
    training, testing, validation = get_train_test_validation(training_set_path=training_path,
                                                              test_set_path=testing_path,
                                                              validation_set_path=validation_path,
                                                              create_inverse_triples=False)
    # m_obj = TransE(triples_factory=training).half()  # convert to float16 format

    # === Start HPO Study === #
    hpo_pipeline_result = hpo_pipeline(
        training=training,
        validation=validation,
        testing=testing,
        model=configuration["model_name"],
        n_trials=configuration["num_trials"],
        sampler=TPESampler(
            consider_prior=True,
            prior_weight=1.0,
            consider_magic_clip=True,
            consider_endpoints=False,
            n_startup_trials=configuration["num_startup_trials_sampler"],
            n_ei_candidates=configuration["num_expected_improvement_candidates_sampler"],
        ),
        pruner=PercentilePruner(
            percentile=configuration["percentile_pruner"],
            n_startup_trials=configuration["num_startup_trials_pruner"],
        ),
        training_kwargs={
            "use_tqdm_batch": False,
        },
        training_kwargs_ranges=dict(
            num_epochs=dict(type=int, low=30, high=configuration["max_num_epoch"], q=5),
            batch_size=dict(type=int, low=64, high=configuration["max_batch_size"], q=64),
        ),
        optimizer="Adam",
        optimizer_kwargs_ranges=dict(
            lr=dict(type=float, low=0.0001, high=0.01, scale="log"),
        ),
        training_loop="slcwa",
        negative_sampler="basic",
        stopper=None,
        evaluator="RankBasedEvaluator",
        evaluation_kwargs={
            "use_tqdm": True,
        },
        metric="both.realistic.inverse_harmonic_mean_rank",  # MRR
        direction="maximize",
        device="gpu",  # 'cpu' | 'gpu'
    )

    # === See HPO Results === #
    print("\n\n >>>>> Study Best Result:")
    print(hpo_pipeline_result.study.best_value)
    print(hpo_pipeline_result.study.best_params)
    print(hpo_pipeline_result.study.best_trial)
    print(hpo_pipeline_result.study.best_trial.number)
    print(hpo_pipeline_result.study.best_trial.datetime_start.strftime("%Y/%m/%d %H:%M:%S"))
    print(hpo_pipeline_result.study.best_trial.datetime_complete.strftime("%Y/%m/%d %H:%M:%S"))
    print(hpo_pipeline_result.study.best_trial.user_attrs)

    print("\n\n >>>>>> Best Trials (Pareto front in the study):")
    for trial in hpo_pipeline_result.study.best_trials:
        print(trial.number, trial.values, trial.params)

    print("\n\n >>>>>> All Trials:")
    for trial in hpo_pipeline_result.study.get_trials():
        print(trial)

    # === Save Configuration and HPO results === #
    dataset_tuning_folder_path = DatasetPathFactory(dataset_name=configuration["dataset_name"]).get_tuning_folder_path()
    out_file_path = os.path.join(dataset_tuning_folder_path, f"{configuration['model_name']}_study.json")
    output_diz = dict(
        starting_configuration=dict(configuration),
        number=int(hpo_pipeline_result.study.best_trial.number),
        best_value=float(hpo_pipeline_result.study.best_value),
        best_params=dict(hpo_pipeline_result.study.best_params),
        start_time=str(hpo_pipeline_result.study.best_trial.datetime_start.strftime("%Y/%m/%d %H:%M:%S")),
        end_time=str(hpo_pipeline_result.study.best_trial.datetime_complete.strftime("%Y/%m/%d %H:%M:%S")),
        metrics=dict(hpo_pipeline_result.study.best_trial.user_attrs),
    )
    with open(out_file_path, "w") as outfile:
        json.dump(obj=output_diz, fp=outfile,
                  ensure_ascii=True, check_circular=True, allow_nan=True, indent=4)
