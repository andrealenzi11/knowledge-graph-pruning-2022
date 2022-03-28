from pykeen.datasets import Nations, Countries, WN18RR
from pykeen.evaluation import RankBasedEvaluator
from pykeen.models import TransE
from pykeen.training import SLCWATrainingLoop
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, StepLR

# dataset = Nations()
dataset = WN18RR()

training_triples_factory = dataset.training

# Pick a model
model = TransE(triples_factory=training_triples_factory)

# Pick an optimizer from Torch
optimizer = Adam(params=model.get_grad_params(),
                 lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0, amsgrad=False)

# Pick a learning rate scheduler
lr_scheduler1 = ExponentialLR(optimizer=optimizer, gamma=0.9, last_epoch=- 1, verbose=True)
lr_scheduler2 = StepLR(optimizer, step_size=20, gamma=0.99)

# Pick a training approach (sLCWA or LCWA)
training_loop = SLCWATrainingLoop(
    model=model,
    triples_factory=training_triples_factory,
    optimizer=optimizer,
    # lr_scheduler=lr_scheduler2,
)

# Train like Cristiano Ronaldo
_ = training_loop.train(
    triples_factory=training_triples_factory,
    num_epochs=100,
    batch_size=256,
)

# Pick an evaluator
evaluator = RankBasedEvaluator()

# Get triples to test
mapped_triples = dataset.testing.mapped_triples

# Evaluate
result = evaluator.evaluate(
    model=model,
    mapped_triples=mapped_triples,
    batch_size=1024,
    additional_filter_triples=[
        dataset.training.mapped_triples,
        dataset.validation.mapped_triples,
    ],
)


print("\n\n #################### metrics ####################")
for k, v in result.to_dict()["both"]["realistic"].items():
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
