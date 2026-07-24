from pas.intervals.simple_cis import get_mle_cis, get_pred_mean_cis, get_bootstrap_cis
from pas.intervals.ppi_cis import get_vanilla_ppi_cis, get_pt_ppi_cis
from pas.intervals.eb_cis import (
    get_eb_ppi_cis,
    get_eb_unipt_ppi_cis,
    get_eb_power_tuned_cis,
)
from pas.intervals.double_shrinkage_cis import (
    get_double_shrinkage_cis,
    get_double_shrinkage_mm1_cis,
    get_double_shrinkage_mm2_cis,
    get_double_shrinkage_mle_cis,
    get_double_shrinkage_ure_cis,
)

# NPMLE-prior EB CIs require the optional `npmle` extra (npeb + cvxpy + mosek).
# Imported lazily so that users without the extra installed can still use the
# rest of the package.
try:
    from pas.intervals.eb_npmle_ci import (
        get_npmle_eb_ppi_cis,
        get_npmle_eb_unipt_ppi_cis,
        get_npmle_eb_power_tuned_cis,
    )
    _HAS_NPMLE = True
except ImportError:
    _HAS_NPMLE = False

__all__ = [
    "get_mle_cis",
    "get_pred_mean_cis",
    "get_bootstrap_cis",
    "get_vanilla_ppi_cis",
    "get_pt_ppi_cis",
    "get_eb_ppi_cis",
    "get_eb_unipt_ppi_cis",
    "get_eb_power_tuned_cis",
    "get_double_shrinkage_cis",
    "get_double_shrinkage_mm1_cis",
    "get_double_shrinkage_mm2_cis",
    "get_double_shrinkage_mle_cis",
    "get_double_shrinkage_ure_cis",
]

CORE_CI_METHODS = {
    "mle_ci": get_mle_cis,
    "pred_mean_ci": get_pred_mean_cis,
    "bootstrap_ci": get_bootstrap_cis,
    "ppi_ci": get_vanilla_ppi_cis,
    "pt_ci": get_pt_ppi_cis,
    "eb_ppi_ci": get_eb_ppi_cis,
    "eb_unipt_ppi_ci": get_eb_unipt_ppi_cis,
    "eb_pt_ci": get_eb_power_tuned_cis,
    # Rosenman--Dominici--Miratrix (2023) double-shrinkage robust EBCIs.
    "double_shrinkage_mm1_ci": get_double_shrinkage_mm1_cis,
    "double_shrinkage_mm2_ci": get_double_shrinkage_mm2_cis,
    "double_shrinkage_mle_ci": get_double_shrinkage_mle_cis,
    "double_shrinkage_ure_ci": get_double_shrinkage_ure_cis,
}

if _HAS_NPMLE:
    __all__ += [
        "get_npmle_eb_ppi_cis",
        "get_npmle_eb_unipt_ppi_cis",
        "get_npmle_eb_power_tuned_cis",
    ]
    CORE_CI_METHODS["eb_npmle_ppi_ci"] = get_npmle_eb_ppi_cis
    CORE_CI_METHODS["eb_npmle_unipt_ppi_ci"] = get_npmle_eb_unipt_ppi_cis
    CORE_CI_METHODS["eb_npmle_pt_ci"] = get_npmle_eb_power_tuned_cis
