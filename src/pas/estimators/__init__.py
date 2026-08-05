from pas.estimators.simple_estimators import get_mle_estimators, get_pred_mean_estimators
from pas.estimators.ppi_estimators import get_vanilla_ppi_estimators, get_pt_ppi_estimators
from pas.estimators.pas_estimators import get_shrinkage_only_estimators, get_pas_estimators, get_shrinkage_to_mean_estimators
from pas.estimators.uni_pas_estimators import get_uni_pt_estimators, get_uni_pas_estimators
from pas.estimators.eb_estimators import get_eb_ppi_estimators, get_eb_unipt_ppi_estimators

__all__ = [
    "get_mle_estimators",
    "get_pred_mean_estimators",
    "get_vanilla_ppi_estimators",
    "get_pt_ppi_estimators",
    "get_shrinkage_only_estimators",
    "get_pas_estimators",
    "get_shrinkage_to_mean_estimators",
    "get_uni_pt_estimators",
    "get_uni_pas_estimators",
    "get_eb_ppi_estimators",
    "get_eb_unipt_ppi_estimators",
    "CORE_ESTIMATORS",
    "PAPER_TABLE3_ESTIMATORS",
    "ALL_ESTIMATORS",
]


CORE_ESTIMATORS = {
    "mle": get_mle_estimators,
    "pred_mean": get_pred_mean_estimators,
    "ppi": get_vanilla_ppi_estimators,
    "pt": get_pt_ppi_estimators,
    "shrinkage_only": get_shrinkage_only_estimators,
    "shrinkage_mean": get_shrinkage_to_mean_estimators,
    "pas": get_pas_estimators,
}


# The nine rows of Table 3, in the paper's order. `CORE_ESTIMATORS` alone is the
# seven rows of Table 2 (the synthetic study); the real-data table adds UniPT and
# UniPAS, which the paper includes "only for the real-world experiments, as they
# are specifically designed for settings where the second moments are unknown".
# `mle` must stay first: `run_benchmark` measures "% Improved" against it.
PAPER_TABLE3_ESTIMATORS = {
    **CORE_ESTIMATORS,
    "uni_pt": get_uni_pt_estimators,
    "uni_pas": get_uni_pas_estimators,
}


ALL_ESTIMATORS = {
    **PAPER_TABLE3_ESTIMATORS,
    "eb_ppi": get_eb_ppi_estimators,
    "eb_unipt_ppi": get_eb_unipt_ppi_estimators,
}
