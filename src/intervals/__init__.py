from intervals.simple_cis import get_mle_cis, get_bootstrap_cis
from intervals.ppi_cis import get_vanilla_ppi_cis, get_pt_ppi_cis

__all__ = [
    "get_mle_cis",
    "get_bootstrap_cis",
    "get_vanilla_ppi_cis",
    "get_pt_ppi_cis",
]

CORE_CI_METHODS = {
    "mle_ci": get_mle_cis,
    "bootstrap_ci": get_bootstrap_cis,
    "ppi_ci": get_vanilla_ppi_cis,
    "pt_ci": get_pt_ppi_cis,
}
