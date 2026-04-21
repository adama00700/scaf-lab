"""
scaf.core.conservation
======================
قانون الحفاظ — النواة الرياضية لكل SCAF

C  = sum(x) / mean(mask)
Ec = |sum(x*) - C|
"""

import numpy as np

EPS = 1e-12


def conservation_constant(reference: np.ndarray,
                           mask: np.ndarray | None = None) -> float:
    """
    يحسب ثابت الحفاظ C من الصورة المرجعية.

    Parameters
    ----------
    reference : np.ndarray
        الصورة أو الإشارة المرجعية.
    mask : np.ndarray, optional
        قناع المناطق المتاحة (0-1). إذا لم يُعطَ يُفترض mask=1.

    Returns
    -------
    float
        ثابت الحفاظ C.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(64, 64)
    >>> C = conservation_constant(img)
    """
    if mask is None:
        return float(np.sum(reference))
    rho = float(np.mean(mask))
    if rho < EPS:
        return float(np.sum(reference))
    return float(np.sum(reference)) / rho


def conservation_error(reconstructed: np.ndarray,
                        C_target: float) -> float:
    """
    يحسب خطأ الحفاظ Ec = |sum(x*) - C|

    Parameters
    ----------
    reconstructed : np.ndarray
        الصورة المعادة بناؤها.
    C_target : float
        ثابت الحفاظ المستهدف.

    Returns
    -------
    float
        خطأ الحفاظ Ec.
    """
    return abs(float(np.sum(reconstructed)) - C_target)


def project_conservation(x: np.ndarray,
                          C_target: float,
                          x_min: float = 0.0,
                          x_max: float = 9.0) -> np.ndarray:
    """
    يُسقط x على قيد الحفاظ: sum(x*) = C_target

    Parameters
    ----------
    x : np.ndarray
        الصورة الحالية.
    C_target : float
        ثابت الحفاظ المستهدف.
    x_min, x_max : float
        حدود النطاق الفيزيائي.

    Returns
    -------
    np.ndarray
        الصورة بعد الإسقاط.
    """
    s = float(np.sum(x))
    if s > EPS:
        return np.clip(x * (C_target / s), x_min, x_max)
    return x
