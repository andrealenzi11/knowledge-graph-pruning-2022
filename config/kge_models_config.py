from config.config import COUNTRIES, CODEXSMALL, NATIONS

from config.config import RESCAL, TRANSE, DISTMULT, TRANSH, COMPLEX, HOLE, CONVE, ROTATE, PAIRRE, AUTOSF, BOXE

HYPERPARAMS_CONFIG = {

    # ========== Countries Dataset ========== #
    COUNTRIES: {

        RESCAL: {
            "model.embedding_dim": 48,
            "loss.margin": 2.141771137678795,
            "regularizer.weight": 0.7029573807791926,
            "optimizer.lr": 0.008329192175591759,
            "negative_sampler.num_negs_per_pos": 4,
            "training.num_epochs": 175,
            "training.batch_size": 64,
        },
        TRANSE: {
            "model.embedding_dim": 48,
            "model.scoring_fct_norm": 2,
            "loss.margin": 2.5243942926015612,
            "optimizer.lr": 0.0033918253411746365,
            "negative_sampler.num_negs_per_pos": 17,
            "training.num_epochs": 125,
            "training.batch_size": 64,
        },
        DISTMULT: {
            "model.embedding_dim": 192,
            "loss.margin": 2.1555781444982585,
            "regularizer.weight": 0.011956117505770046,
            "optimizer.lr": 0.0036636488109368683,
            "negative_sampler.num_negs_per_pos": 87,
            "training.num_epochs": 145,
            "training.batch_size": 64,
        },
        TRANSH: {
            "model.embedding_dim": 96,
            "model.scoring_fct_norm": 1,
            "loss.margin": 1.7244747006769479,
            "regularizer.weight": 0.07712961562459153,
            "optimizer.lr": 0.007150891868887584,
            "negative_sampler.num_negs_per_pos": 56,
            "training.num_epochs": 170,
            "training.batch_size": 64,
        },
        COMPLEX: {
            "model.embedding_dim": 16,
            "optimizer.lr": 0.008344467027748435,
            "negative_sampler.num_negs_per_pos": 24,
            "training.num_epochs": 135,
            "training.batch_size": 64,
        },
        HOLE: {
            "model.embedding_dim": 128,
            "loss.margin": 2.7278586144315335,
            "optimizer.lr": 0.007884092022341537,
            "negative_sampler.num_negs_per_pos": 1,
            "training.num_epochs": 30,
            "training.batch_size": 256,
        },
        CONVE: {
            "model.output_channels": 32,
            "model.input_dropout": 0.1,
            "model.output_dropout": 0.0,
            "model.feature_map_dropout": 0.30000000000000004,
            "optimizer.lr": 0.004643931212906079,
            "negative_sampler.num_negs_per_pos": 53,
            "training.num_epochs": 90,
            "training.batch_size": 192,
        },
        ROTATE: {
            "model.embedding_dim": 592,
            "loss.margin": 2.775210652373596,
            "optimizer.lr": 0.002153178133751181,
            "negative_sampler.num_negs_per_pos": 3,
            "training.num_epochs": 100,
            "training.batch_size": 256,
        },
        PAIRRE: {
            "model.embedding_dim": 128,
            "model.p": 1,
            "loss.margin": 1.2820779503900268,
            "optimizer.lr": 0.0002893294058100564,
            "negative_sampler.num_negs_per_pos": 76,
            "training.num_epochs": 195,
            "training.batch_size": 192,
        },
        AUTOSF: {
            "model.embedding_dim": 16,
            "loss.margin": 2.2188954424388805,
            "optimizer.lr": 0.009547046189867365,
            "negative_sampler.num_negs_per_pos": 67,
            "training.num_epochs": 190,
            "training.batch_size": 64,
        },
        BOXE: {
            "model.embedding_dim": 160,
            "model.p": 2,
            "loss.margin": 21,
            # "loss.adversarial_temperature": 0.6716731481834158,
            "optimizer.lr": 0.009298475043383676,
            "negative_sampler.num_negs_per_pos": 48,
            "training.num_epochs": 110,
            "training.batch_size": 256,
        },
    },

    # ========== CoDExSmall Dataset ========== #
    NATIONS: {

        RESCAL:  {
            "model.embedding_dim": 224,
            "loss.margin": 1.2872050384481426,
            "regularizer.weight": 0.7655403769033309,
            "optimizer.lr": 0.007265992846208457,
            "negative_sampler.num_negs_per_pos": 2,
            "training.num_epochs": 165,
            "training.batch_size": 64,
        },
        TRANSE: {
            "model.embedding_dim": 80,
            "model.scoring_fct_norm": 2,
            "loss.margin": 1.7300440097610212,
            "optimizer.lr": 0.00795011583801911,
            "negative_sampler.num_negs_per_pos": 27,
            "training.num_epochs": 150,
            "training.batch_size": 256,
        },
        DISTMULT: {

        },
        TRANSH: {

        },
        COMPLEX: {

        },
        HOLE: {

        },
        CONVE: {

        },
        ROTATE: {

        },
        PAIRRE: {

        },
        AUTOSF: {

        },
        BOXE: {

        },
    }

    # ========== Countries Dataset ========== #

}
