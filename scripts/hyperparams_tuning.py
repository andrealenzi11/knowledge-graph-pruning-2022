from optuna.pruners import PercentilePruner
from optuna.samplers import TPESampler
from pykeen.hpo import hpo_pipeline

valid_datasets = {
    "Countries",
    "Nations",
    "CoDExSmall",
    "DBpedia50",
    "WN18RR",
    "FB15k237",
}

valid_kge_models = {
    "RESCAL",
    "TransE",
    "DistMult",
    "TransH",
    "TransR",
    "TransD",
    "ComplEx",
    "HolE",
    "ConvE",
    # "ConvKB",
    # "RGCN",
    "RotatE",
    "PairRE",
    "AutoSF",
    "BoxE",
}

if __name__ == '__main__':

    # === Set your Configuration === #
    pykeen_dataset = "Countries"
    pykeen_model = "TransE"

    num_trials = 30
    num_startup_trials_sampler = 3  # default: 10
    num_expected_improvement_candidates_sampler = 32  # default: 24
    num_startup_trials_pruner = 5  # default: 5
    percentile_pruner = 70.0

    max_batch_size = 256
    max_num_epoch = 200
    # ================================ #

    # === Start HPO study === #
    hpo_pipeline_result = hpo_pipeline(
        dataset=pykeen_dataset,
        model=pykeen_model,  # 'ComplEx',
        n_trials=num_trials,
        sampler=TPESampler(
            consider_prior=True,
            prior_weight=1.0,
            consider_magic_clip=True,
            consider_endpoints=False,
            n_startup_trials=num_startup_trials_sampler,
            n_ei_candidates=num_expected_improvement_candidates_sampler,
        ),
        pruner=PercentilePruner(
            percentile=percentile_pruner,
            n_startup_trials=num_startup_trials_pruner,
        ),
        training_kwargs={
            "use_tqdm_batch": False,
        },
        training_kwargs_ranges=dict(
            num_epochs=dict(type=int, low=30, high=max_num_epoch, q=5),
            batch_size=dict(type=int, low=64, high=max_batch_size, q=64),
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
        metric="inverse_harmonic_mean_rank",  # MRR
        direction="maximize",
        device="gpu",  # 'cpu' | 'gpu'
    )

    # === See HPO Results === #
    print("\n\n >>>>> Study Best Result:")
    print(hpo_pipeline_result.study.best_value)
    print(hpo_pipeline_result.study.best_params)
    print(hpo_pipeline_result.study.best_trial)

    print("\n\n >>>>>> Best Trials (Pareto front in the study):")
    for trial in hpo_pipeline_result.study.best_trials:
        print(trial.number, trial.values, trial.params)

    print("\n\n >>>>>> All Trials:")
    for trial in hpo_pipeline_result.study.get_trials():
        print(trial)
    # === Save Configuration and HPO results === #

