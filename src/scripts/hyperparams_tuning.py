import argparse
import json
import os

from optuna.pruners import PercentilePruner
from optuna.samplers import TPESampler
from pykeen.hpo import hpo_pipeline
from pykeen.version import VERSION as PYKEEN_VERSION
from torch.version import __version__ as torch_version

from src.config.config import ORIGINAL, COUNTRIES, WN18RR, FB15K237, YAGO310, CODEXSMALL, NATIONS, \
    TRANSE, DISTMULT, TRANSH, COMPLEX, HOLE, CONVE, ROTATE, PAIRRE, AUTOSF, BOXE, TUNING_DIR
from src.core.pykeen_wrapper import get_train_test_validation, print_partitions_info
from src.dao.dataset_loading import DatasetPathFactory, TsvDatasetLoader
from src.utils.cuda_info import get_cuda_info
from src.utils.printing import print_and_write

all_datasets_names = [
    FB15K237,  # "FB15K237"
    WN18RR,  # "WN18RR"
    YAGO310,  # "YAGO310"
    COUNTRIES,  # "COUNTRIES"
    CODEXSMALL,  # "CODEXSMALL"
    NATIONS,  # "NATIONS"
]

valid_kge_models = [
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
]

if __name__ == '__main__':

    fp = os.path.join(TUNING_DIR, "hyper_parameters_tuning_log.txt")

    with open(fp, "w") as fw_log:

        # Instantiate the parser for the command line arguments
        args_parser = argparse.ArgumentParser()

        # Add command line arguments entries
        args_parser.add_argument('dataset',
                                 help='Dataset Name',
                                 type=str,
                                 choices=all_datasets_names)
        args_parser.add_argument('-n', '--num_trials',
                                 dest="num_trials",
                                 help='Number of trials for the Sampler',
                                 type=int,
                                 required=False,
                                 default=50)
        args_parser.add_argument('-s', '--num_startup_trials',
                                 dest="num_startup_trials",
                                 help='Number of startup trials for the Sampler',
                                 type=int,
                                 required=False,
                                 default=30)

        # Parse command line arguments
        cl_args = args_parser.parse_args()

        # Access to the command line arguments
        print_and_write(out_file=fw_log, text=f"Argument values: \n\t {cl_args} \n")
        dataset_name = str(cl_args.dataset).upper().strip()
        print_and_write(out_file=fw_log, text=f"dataset_name: {dataset_name}")
        num_trials_sampler = int(cl_args.num_trials)
        print_and_write(out_file=fw_log, text=f"num_trials_sampler: {num_trials_sampler}")
        num_startup_trials_sampler = int(cl_args.num_startup_trials)
        print_and_write(out_file=fw_log, text=f"num_startup_trials_sampler: {num_startup_trials_sampler}")
        assert num_startup_trials_sampler < num_trials_sampler
        if num_trials_sampler > 15:
            num_startup_trials_pruner = 15
            assert num_startup_trials_pruner < num_trials_sampler
        else:
            num_startup_trials_pruner = num_startup_trials_sampler
            assert num_startup_trials_pruner == num_startup_trials_sampler
            assert num_startup_trials_pruner < num_trials_sampler
        print_and_write(out_file=fw_log, text=f"num_startup_trials_pruner: {num_startup_trials_pruner}")
        print_and_write(out_file=fw_log, text=f"pykeen version: {PYKEEN_VERSION}")
        print_and_write(out_file=fw_log, text=f"torch version: {torch_version}")

        # === Get Input Dataset: Training, Validation, Testing === #
        # check on input datset
        if dataset_name not in all_datasets_names:
            raise ValueError(f"Invalid dataset name '{dataset_name}'! \n"
                             f"Specify one of the following values: {all_datasets_names} \n")

        # check on cuda
        print_and_write(out_file=fw_log, text=f"{get_cuda_info()}")

        # dataset loader
        datasets_loader = TsvDatasetLoader(dataset_name=dataset_name,
                                           noise_level=ORIGINAL)

        # paths
        training_path, validation_path, testing_path = \
            datasets_loader.get_training_validation_testing_dfs_paths(noisy_test_flag=False)
        assert "training" in training_path
        assert ORIGINAL in training_path
        assert dataset_name in training_path
        assert "validation" in validation_path
        assert ORIGINAL in validation_path
        assert dataset_name in validation_path
        assert "testing" in testing_path
        assert ORIGINAL in testing_path
        assert dataset_name in testing_path

        # partitions (triples factories)
        training, testing, validation = get_train_test_validation(training_set_path=training_path,
                                                                  test_set_path=testing_path,
                                                                  validation_set_path=validation_path,
                                                                  create_inverse_triples=False)
        print_partitions_info(training_triples=training,
                              training_triples_path=training_path,
                              validation_triples=validation,
                              validation_triples_path=validation_path,
                              testing_triples=testing,
                              testing_triples_path=testing_path)

        # ===== iterate over all models ===== #
        for kge_model_name in valid_kge_models:

            print_and_write(out_file=fw_log, text=f"\n\n\n#################### {kge_model_name} ####################")
            if kge_model_name not in valid_kge_models:
                raise ValueError(f"Invalid model name '{kge_model_name}'!")

            # Special case for BoxE (bug: BoxE hpo_pipeline does not work on GPU whit negative sampler filtering)
            if kge_model_name == BOXE:
                negative_sampler_kwargs = None
            # Otherwise
            else:
                negative_sampler_kwargs = {
                    "filtered": True,
                    "filterer": "python-set",  # "bloom"
                }

            # === Set your Configuration === #
            configuration = dict(
                dataset_name=dataset_name,
                training_path=training_path,
                validation_path=validation_path,
                testing_path=testing_path,
                model_name=kge_model_name,
                device="cuda:0",  # "cpu"
                negative_sampler_kwargs=negative_sampler_kwargs,
                num_trials=num_trials_sampler,
                num_startup_trials_sampler=num_startup_trials_sampler,  # default: 10
                num_expected_improvement_candidates_sampler=32,  # default: 24
                num_startup_trials_pruner=num_startup_trials_pruner,  # default: 5
                percentile_pruner=70.0,
                max_batch_size=256,
                max_num_epoch=200,
            )

            # ==== Print Current Configuration=== #
            print_and_write(out_file=fw_log, text="\n>>>>>>>> CONFIGURATION <<<<<<<<")
            for k, v in configuration.items():
                print_and_write(out_file=fw_log, text=f"\t\t {k} = {v}")
            print_and_write(out_file=fw_log, text=">>>>>>>>>>>>>>><<<<<<<<<<<<<<< \n")

            # === Start HPO Study === #
            hpo_pipeline_result = hpo_pipeline(
                # dataset args
                training=training,
                validation=validation,
                testing=testing,
                # model args
                model=configuration["model_name"],
                # optimizer args
                optimizer="Adam",
                optimizer_kwargs_ranges=dict(
                    lr=dict(type=float, low=0.0001, high=0.01, scale="log"),
                ),
                # # training loop args
                training_loop="slcwa",
                negative_sampler="basic",
                negative_sampler_kwargs=negative_sampler_kwargs,
                # training args
                training_kwargs={
                    "use_tqdm_batch": False,
                },
                training_kwargs_ranges=dict(
                    num_epochs=dict(type=int, low=30, high=configuration["max_num_epoch"], q=5),
                    batch_size=dict(type=int, low=64, high=configuration["max_batch_size"], q=64),
                ),
                stopper=None,
                # evaluation args
                evaluator="RankBasedEvaluator",
                evaluation_kwargs={
                    "use_tqdm": True,
                    "additional_filter_triples": [
                        training.mapped_triples,
                        validation.mapped_triples,
                    ],
                },
                evaluator_kwargs={
                    "filtered": True,
                    # "batch_size": 128,
                },
                metric="both.realistic.inverse_harmonic_mean_rank",  # MRR
                filter_validation_when_testing=True,
                # misc args
                device=configuration["device"],
                # Optuna study args
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
                direction="maximize",
                n_trials=configuration["num_trials"],
            )

            # === See HPO Results === #
            print_and_write(out_file=fw_log, text="\n\n >>>>> Study Best Result:")
            print_and_write(out_file=fw_log, text=hpo_pipeline_result.study.best_trial.number)
            print_and_write(out_file=fw_log, text=hpo_pipeline_result.study.best_value)
            print_and_write(out_file=fw_log, text=hpo_pipeline_result.study.best_params)
            print_and_write(out_file=fw_log,
                            text=hpo_pipeline_result.study.best_trial.datetime_start.strftime("%Y/%m/%d %H:%M:%S"))
            print_and_write(out_file=fw_log,
                            text=hpo_pipeline_result.study.best_trial.datetime_complete.strftime("%Y/%m/%d %H:%M:%S"))
            # print_and_write(out_file=fw_log, text=hpo_pipeline_result.study.best_trial)
            # print_and_write(out_file=fw_log, text=hpo_pipeline_result.study.best_trial.user_attrs)

            print_and_write(out_file=fw_log, text="\n\n >>>>>> Best Trials (Pareto front in the study):")
            for trial in hpo_pipeline_result.study.best_trials:
                print_and_write(out_file=fw_log, text=f"({trial.number}, {trial.values}, {trial.params})")

            print_and_write(out_file=fw_log, text="\n\n >>>>>> All Trials:")
            for trial in hpo_pipeline_result.study.get_trials():
                print_and_write(out_file=fw_log, text=f"({trial.number}, {trial.values}, {trial.params})")

            # === Save Configuration and HPO results === #
            dataset_tuning_folder_path = DatasetPathFactory(
                dataset_name=configuration["dataset_name"]).get_tuning_folder_path()
            assert dataset_name in dataset_tuning_folder_path

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

            # conclude models iteration
            del kge_model_name, negative_sampler_kwargs, configuration, \
                hpo_pipeline_result, dataset_tuning_folder_path, out_file_path, output_diz
            print_and_write(out_file=fw_log, text=f"{'#' * 80} \n")

        print_and_write(out_file=fw_log, text="EXIT 0")
