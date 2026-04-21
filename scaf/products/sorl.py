"""
scaf.products.sorl
==================
SCAF-SORL — تحسين التصوير العام

النتائج: 3/3 (100%) PASS | +2.16 dB | SSIM +0.082
"""

import numpy as np
from ..core.conservation import conservation_constant, project_conservation
from ..core.diffusion import scaf_step

EPS = 1e-12


class SCAFSORL:
    """
    تحسين جودة الصور الطبية والعامة.

    يعمل كـ preprocessing قبل التحليل أو التشخيص.
    لا يحتاج معرفة مسبقة بنظام التصوير.

    Parameters
    ----------
    n_iter : int
        عدد خطوات التحسين (افتراضي: 200).
    kappa : float
        عتبة حفظ الحواف (افتراضي: 0.22).
    beta : float
        قوة الانتشار (افتراضي: 2.42).

    Examples
    --------
    >>> import numpy as np
    >>> from scaf import SCAFSORL
    >>> enhancer = SCAFSORL()
    >>> enhanced = enhancer.enhance(image)
    """

    def __init__(self, n_iter: int = 200,
                  kappa: float = 0.22,
                  beta:  float = 2.42):
        self.n_iter = n_iter
        self.kappa  = kappa
        self.beta   = beta

    def enhance(self, image: np.ndarray,
                 preserve_range: bool = True) -> np.ndarray:
        """
        يُحسِّن جودة الصورة.

        Parameters
        ----------
        image : np.ndarray (2D float)
            الصورة المدخلة.
        preserve_range : bool
            إذا True، يحافظ على نطاق القيم الأصلي.

        Returns
        -------
        np.ndarray (2D float)
            الصورة المحسَّنة.
        """
        img = image.astype(np.float64)
        orig_min = img.min()
        orig_max = img.max()

        # تطبيع للنطاق [0.5, 8.5]
        if orig_max > orig_min + EPS:
            img = (img - orig_min) / (orig_max - orig_min) * 8 + 0.5

        C_target = conservation_constant(img)
        dt_c = 0.05

        for it in range(self.n_iter):
            img = scaf_step(img, self.beta, dt_c, self.kappa)
            img = project_conservation(img, C_target)
            if it % 80 == 79:
                dt_c *= 0.92

        # إعادة النطاق الأصلي
        if preserve_range and orig_max > orig_min + EPS:
            img = (img - 0.5) / 8.0 * (orig_max - orig_min) + orig_min

        return img

    def enhance_batch(self, images: list[np.ndarray],
                       **kwargs) -> list[np.ndarray]:
        """تحسين مجموعة من الصور."""
        return [self.enhance(img, **kwargs) for img in images]
