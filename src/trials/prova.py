from pprint import pprint
from pykeen.models import TransE
from pykeen.datasets.countries import Countries

from pykeen.pipeline import pipeline
import matplotlib.pyplot as plt

dataset_countries = Countries()
print(dataset_countries.__dict__)

result = pipeline(
    model=TransE().half(),
    dataset='WN18RR',
    training_kwargs={
        "num_epochs": 5,
        "batch_size": 512,
        "use_tqdm_batch": False,
    },
    optimizer='Adam',
    training_loop='sLCWA',
    negative_sampler='basic',
    use_testing_data=True,
    stopper='early',
    stopper_kwargs={
        "frequency": 100,
        "patience": 2,
        "metric": 'hits_at_k',
        "relative_delta": 0.01,
        "larger_is_better": True,
    },
    evaluator="RankBasedEvaluator",
    evaluator_kwargs={
        "batch_size": 256,
    },
    evaluation_kwargs={
        "use_tqdm": True
    },
    random_seed=1,
    device='gpu',  # 'cpu'
    use_tqdm=True,
)

result.plot_losses()
plt.show()
plt.close()
try:
    result.plot_early_stopping()
    plt.show()
    plt.close()
except ValueError:
    pass

print("\n\n #################### result ####################")
print(result)
print()
d1 = result.__dict__
pprint(d1)
print("#" * 80)

print("\n\n #################### model ####################")
print(result.model)
print()
d2 = result.model.__dict__
del d2['_entity_ids']
pprint(d2)
print("#" * 80)

print("\n\n #################### configuration ####################")
pprint(result.configuration)
print("#" * 80)

print("\n\n #################### metrics ####################")
for k, v in result.metric_results.to_dict()["both"]["realistic"].items():
    if k in {
        "arithmetic_mean_rank",
        "inverse_harmonic_mean_rank",
        "hits_at_1",
        "hits_at_3",
        "hits_at_5",
        "hits_at_10",
    }:
        print(k, ":", v)
print("#" * 80)
