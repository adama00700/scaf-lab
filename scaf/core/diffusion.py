"""
scaf.core.diffusion
===================
Anisotropic Diffusion مع Perona-Malik

المعادلة:
    dx/dt = beta × div(w(|∇x|) × ∇x)
    w(s)  = 1 / (1 + (s/kappa)²)
"""

import numpy as np

EPS = 1e-12

# المعاملات المقفلة — Universal Constants
BETA_DEFAULT  = 2.42
DT_DEFAULT    = 0.05
KAPPA_DEFAULT = 0.22


def scaf_step(x: np.ndarray,
               beta:  float = BETA_DEFAULT,
               dt:    float = DT_DEFAULT,
               kappa: float = KAPPA_DEFAULT) -> np.ndarray:
    """
    خطوة SCAF الأساسية — Anisotropic Diffusion مع Perona-Malik.

    Parameters
    ----------
    x : np.ndarray (2D)
        الصورة الحالية.
    beta : float
        قوة الانتشار (افتراضي: 2.42).
    dt : float
        الخطوة الزمنية (افتراضي: 0.05).
    kappa : float
        عتبة حفظ الحواف (افتراضي: 0.22).

    Returns
    -------
    np.ndarray
        الصورة بعد خطوة انتشار واحدة.
    """
    gx = np.zeros_like(x)
    gy = np.zeros_like(x)
    gx[:, :-1] = x[:, 1:] - x[:, :-1]
    gy[:-1, :] = x[1:, :] - x[:-1, :]

    mag = np.sqrt(gx**2 + gy**2 + EPS)
    w   = 1.0 / (1.0 + (mag / kappa)**2)

    fx, fy = w * gx, w * gy

    div = np.zeros_like(x)
    div[:, 0]  = fx[:, 0]
    div[:, 1:] = fx[:, 1:] - fx[:, :-1]
    div[0, :]  += fy[0, :]
    div[1:, :] += fy[1:, :] - fy[:-1, :]

    return x + (beta * dt) * div


def scaf_diffuse(x: np.ndarray,
                  n_iter: int = 100,
                  beta:   float = BETA_DEFAULT,
                  dt:     float = DT_DEFAULT,
                  kappa:  float = KAPPA_DEFAULT,
                  decay:  float = 0.92,
                  decay_every: int = 80) -> np.ndarray:
    """
    تطبيق SCAF لعدد من الخطوات مع تناقص تدريجي.

    Parameters
    ----------
    x : np.ndarray
        الصورة المدخلة.
    n_iter : int
        عدد خطوات الانتشار.
    decay : float
        معدل تناقص beta و dt كل decay_every خطوة.
    decay_every : int
        عدد الخطوات بين كل تناقص.

    Returns
    -------
    np.ndarray
        الصورة المُنعَّمة.
    """
    dt_c = dt
    for it in range(n_iter):
        x = scaf_step(x, beta, dt_c, kappa)
        if it % decay_every == decay_every - 1:
            dt_c *= decay
    return x
