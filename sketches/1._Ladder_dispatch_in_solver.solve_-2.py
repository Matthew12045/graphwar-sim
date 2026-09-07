# ccf.py — the contract, kept narrow

from dataclasses import dataclass
from enum import Enum
from typing import Union

class CCFRejection(Enum):
    """Maps into the M5.1 outcome taxonomy — do not invent a parallel enum."""
    CORRIDOR_TOO_TIGHT = "corridor_too_tight"
    DEGREE_EXHAUSTED   = "degree_exhausted"
    NO_CERTIFICATE     = "no_certificate"      # fit found, containment unproven
    NUMERICALLY_UNSTABLE = "numerically_unstable"

@dataclass(frozen=True)
class CertifiedFit:
    expr: "Expr"
    certificate: "Certificate"   # witness: grid resolution + bound used
    cost: int                    # for M5.3 budget accounting

CCFResult = Union[CertifiedFit, CCFRejection]