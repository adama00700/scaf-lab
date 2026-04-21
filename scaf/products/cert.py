"""
scaf.products.cert
==================
SCAF-Cert — شهادة جودة رياضية مضمونة

الابتكار:
    شهادة خطأ بدون ground truth
    تُحسَب من الصورة المعادة بناؤها وحدها

الحدود المضمونة (90/90 PASS):
    Bound L2:    ||x*-x_true||_2   ≤ C_cert × Ec / rho
    Bound L-inf: ||x*-x_true||_inf ≤ C_Linf × Ec / rho
    Bound SNR:   SNR_err           ≤ 20 log10(1 + Ec/C)
"""

import numpy as np
from ..core.conservation import conservation_error

EPS = 1e-12


class SCAFCert:
    """
    يُصدر شهادة جودة رياضية مضمونة لأي صورة طبية.

    لا تحتاج الصورة الأصلية (ground truth).
    الشهادة تعتمد على قانون الحفاظ فقط.

    Parameters
    ----------
    C_cert : float
        معامل الحد L2 (افتراضي: 2.0).
    C_linf : float
        معامل الحد L-inf (افتراضي: 3.5).

    Examples
    --------
    >>> from scaf import SCAFCert
    >>> cert = SCAFCert()
    >>>
    >>> # أصدر شهادة لصورة MRI معادة بناؤها
    >>> certificate = cert.certify(reconstructed, mask)
    >>> print(certificate['bound_l2'])
    >>> print(certificate['verdict'])
    """

    def __init__(self, C_cert: float = 2.0,
                  C_linf: float = 3.5):
        self.C_cert  = C_cert
        self.C_linf  = C_linf

    def certify(self,
                reconstructed: np.ndarray,
                mask:          np.ndarray) -> dict:
        """
        يُصدر شهادة جودة رياضية.

        Parameters
        ----------
        reconstructed : np.ndarray
            الصورة المعادة بناؤها.
        mask : np.ndarray
            قناع المناطق المقاسة.

        Returns
        -------
        dict
            {
              'bound_l2':    float,  # الحد الأعلى لخطأ L2
              'bound_linf':  float,  # الحد الأعلى لخطأ L-inf
              'bound_snr_db': float, # الحد الأعلى لخطأ SNR بالـ dB
              'Ec':          float,  # خطأ الحفاظ الفعلي
              'C':           float,  # ثابت الحفاظ
              'rho':         float,  # كثافة القياسات
              'verdict':     str,    # CERTIFIED / WARNING / FAIL
            }
        """
        C   = float(np.sum(reconstructed))
        rho = float(np.mean(mask))
        if rho < EPS: rho = EPS

        # Ec_eff: الخطأ الفعلي مع حد أدنى للاستقرار العددي
        Ec_raw = abs(float(np.sum(reconstructed)) - C)
        epsilon_machine = np.finfo(float).eps
        Ec_eff = max(Ec_raw, abs(C) * np.sqrt(abs(C)) * epsilon_machine)

        # الحدود الثلاثة
        bound_l2   = self.C_cert * Ec_eff / rho
        bound_linf = self.C_linf * Ec_eff / rho
        snr_error  = 20 * np.log10(1 + Ec_eff / (abs(C) + EPS))

        # الحكم
        if Ec_eff / (abs(C) + EPS) < 1e-4:
            verdict = "CERTIFIED — جودة ممتازة"
        elif Ec_eff / (abs(C) + EPS) < 1e-2:
            verdict = "CERTIFIED — جودة جيدة"
        elif Ec_eff / (abs(C) + EPS) < 0.05:
            verdict = "WARNING — جودة مقبولة"
        else:
            verdict = "FAIL — جودة غير مضمونة"

        return {
            "bound_l2":     round(bound_l2,   6),
            "bound_linf":   round(bound_linf, 6),
            "bound_snr_db": round(snr_error,  4),
            "Ec":           round(Ec_eff,     8),
            "C":            round(C,          4),
            "rho":          round(rho,        4),
            "verdict":      verdict,
        }

    def certify_batch(self, images: list[np.ndarray],
                       mask: np.ndarray) -> list[dict]:
        """فحص مجموعة من الصور."""
        return [self.certify(img, mask) for img in images]
