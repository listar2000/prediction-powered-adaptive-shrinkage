from hmac import compare_digest
from estimators.simple_estimators import get_mle_estimators, get_pred_mean_estimators
from estimators.ppi_estimators import get_vanilla_ppi_estimators, get_pt_ppi_estimators
from estimators.pas_estimators import get_shrinkage_only_estimators, get_pas_estimators, get_shrinkage_to_mean_estimators
from estimators.uni_pas_estimators import get_uni_pt_estimators, get_uni_pas_estimators

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


ALL_ESTIMATORS = {
    **CORE_ESTIMATORS,
    "uni_pt": get_uni_pt_estimators,
    "uni_pas": get_uni_pas_estimators,
}
