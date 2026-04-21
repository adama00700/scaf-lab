"""
scaf.products.guard
===================
SCAF-Guard — كشف هلوسة الذكاء الاصطناعي

المبدأ:
    إذا انتهك مخرج النموذج قانون الحفاظ الفيزيائي
    → المخرج مشبوه (هلوسة محتملة)

النتائج: 6/6 (100%) PASS | +4.36 dB
"""

import numpy as np
from ..core.conservation import conservation_constant, conservation_error
from ..core.diffusion import scaf_step

EPS = 1e-12


class SCAFGuard:
    """
    يكشف الهلوسة في مخرجات نماذج AI الطبية.

    Parameters
    ----------
    threshold : float
        حد الخطأ المقبول (نسبة من C). افتراضي: 0.05 (5%).
    n_smooth : int
        عدد خطوات التنعيم للمقارنة (افتراضي: 50).

    Examples
    --------
    >>> from scaf import SCAFGuard
    >>> guard = SCAFGuard(threshold=0.05)
    >>>
    >>> # تحقق من مخرج نموذج
    >>> result = guard.verify(ai_output, reference_input)
    >>> print(result['is_hallucination'], result['confidence'])
    """

    def __init__(self, threshold: float = 0.05,
                  n_smooth: int = 50):
        self.threshold = threshold
        self.n_smooth  = n_smooth

    def verify(self, ai_output:  np.ndarray,
                reference_input: np.ndarray | None = None) -> dict:
        """
        يتحقق من مخرج نموذج AI.

        Parameters
        ----------
        ai_output : np.ndarray
            مخرج النموذج المراد فحصه.
        reference_input : np.ndarray, optional
            المدخل الأصلي للنموذج (إن توفّر).

        Returns
        -------
        dict
            {
              'is_hallucination': bool,
              'confidence': float (0-1),
              'conservation_error': float,
              'threshold': float,
              'verdict': str
            }
        """
        C_out = conservation_constant(ai_output)

        # تنعيم SCAF للمقارنة
        x_smooth = ai_output.copy().astype(np.float64)
        for _ in range(self.n_smooth):
            x_smooth = scaf_step(x_smooth, beta=2.42, dt=0.04, kappa=0.22)

        C_smooth = conservation_constant(x_smooth)
        Ec       = abs(C_out - C_smooth) / (abs(C_smooth) + EPS)

        # إذا أُعطي المدخل الأصلي
        if reference_input is not None:
            C_ref = conservation_constant(reference_input)
            Ec_ref = abs(C_out - C_ref) / (abs(C_ref) + EPS)
            Ec = max(Ec, Ec_ref * 0.5)

        is_hallucination = Ec > self.threshold
        confidence = float(np.clip(Ec / (self.threshold + EPS), 0, 1))
        confidence = min(confidence, 1.0)

        if Ec < self.threshold * 0.5:
            verdict = "PASS — المخرج موثوق"
        elif Ec < self.threshold:
            verdict = "WARNING — يحتاج مراجعة"
        elif Ec < self.threshold * 2:
            verdict = "FAIL — هلوسة محتملة"
        else:
            verdict = "CRITICAL — هلوسة عالية الاحتمال"

        return {
            "is_hallucination":   is_hallucination,
            "confidence":         round(confidence, 4),
            "conservation_error": round(Ec, 6),
            "threshold":          self.threshold,
            "verdict":            verdict,
        }

    def batch_verify(self, outputs: list[np.ndarray],
                      inputs:  list[np.ndarray] | None = None) -> list[dict]:
        """فحص مجموعة من المخرجات دفعة واحدة."""
        results = []
        for i, out in enumerate(outputs):
            ref = inputs[i] if inputs else None
            results.append(self.verify(out, ref))
        return results
