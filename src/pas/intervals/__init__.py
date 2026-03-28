from pas.intervals.simple_cis import get_mle_cis, get_pred_mean_cis, get_bootstrap_cis
from pas.intervals.ppi_cis import get_vanilla_ppi_cis, get_pt_ppi_cis
from pas.intervals.eb_cis import get_eb_ppi_cis, get_eb_unipt_ppi_cis

__all__ = [
    "get_mle_cis",
    "get_pred_mean_cis",
    "get_bootstrap_cis",
    "get_vanilla_ppi_cis",
    "get_pt_ppi_cis",
    "get_eb_ppi_cis",
    "get_eb_unipt_ppi_cis",
]

CORE_CI_METHODS = {
    "mle_ci": get_mle_cis,
    "pred_mean_ci": get_pred_mean_cis,
    "bootstrap_ci": get_bootstrap_cis,
    "ppi_ci": get_vanilla_ppi_cis,
    "pt_ci": get_pt_ppi_cis,
    "eb_ppi_ci": get_eb_ppi_cis,
    "eb_unipt_ppi_ci": get_eb_unipt_ppi_cis,
}
