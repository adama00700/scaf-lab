"""
SCAF — Structural Conservation Adaptive Flow
=============================================
قانون فيزيائي واحد. سبعة منتجات. 322 اختبار موثق.

المبدأ الأساسي:
    C = sum(x) / mean(mask)
    Ec = |sum(x*) - C| → 0

الاستخدام السريع:
    from scaf import SCAFInverse, SCAFVideo, SCAFGuard

    # تحسين MRI
    result = SCAFInverse().reconstruct(kspace, mask)

    # إصلاح Echo
    fixed  = SCAFVideo().repair(frames, drop_mask)

    # كشف هلوسة AI
    score  = SCAFGuard().verify(ai_output)
"""

__version__ = "1.0.0"
__author__  = "SCAF Research Laboratory"
__license__ = "MIT"

from .products.inverse  import SCAFInverse
from .products.video    import SCAFVideo
from .products.guard    import SCAFGuard
from .products.cert     import SCAFCert
from .products.sorl     import SCAFSORL
from .core.conservation import conservation_constant, conservation_error
from .core.diffusion    import scaf_step

__all__ = [
    "SCAFInverse",
    "SCAFVideo",
    "SCAFGuard",
    "SCAFCert",
    "SCAFSORL",
    "conservation_constant",
    "conservation_error",
    "scaf_step",
]
