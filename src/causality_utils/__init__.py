from .cpme import (cpme_dr_weights, cpme_dr_weights_crossfit,
                   cpme_dr_weights_crossfit_nystrom,
                   cpme_dr_weights_crossfit_projected_nystrom,
                   cpme_dr_weights_nystrom, cpme_dr_weights_projected_nystrom,
                   cpme_plugin_weights, lambda_n_half, select_regularization)
from .herding import evaluate_embedding, herding_diagnostics, kernel_herding
from .crossfit import make_crossfit_fold_ids
from .overlap import effective_sample_size
from .policies import (GaussianMixturePolicy, GaussianPolicy, Policy,
                       context_shift_gaussian, mean_preserving_mixture,
                       importance_weights, log_importance_weights)
