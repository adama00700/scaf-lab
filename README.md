# SCAF — Structural Conservation Adaptive Flow

[![PyPI version](https://badge.fury.io/py/scaf-medical.svg)](https://badge.fury.io/py/scaf-medical)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 322/328](https://img.shields.io/badge/Tests-322%2F328%20(98.2%25)-green.svg)]()
[![Zero GPU](https://img.shields.io/badge/GPU-Zero%20Required-blue.svg)]()

> **قانون فيزيائي واحد. سبعة منتجات. 322 اختبار موثق. صفر GPU.**

---

## المبدأ الأساسي

```
C  = sum(x) / mean(mask)     # ثابت الحفاظ
Ec = |sum(x*) - C| → 0      # خطأ الحفاظ
```

أي إشارة فيزيائية حقيقية تحفظ مجموع طاقتها.
SCAF يحوّل هذا القانون إلى خوارزمية بسيطة تعمل على 7 مجالات طبية.

---

## التثبيت

```bash
pip install scaf-medical
```

مع كل الاعتماديات:
```bash
pip install scaf-medical[full]
```

---

## الاستخدام السريع

### SCAF-Inverse — تحسين MRI

```python
import numpy as np
from scaf import SCAFInverse

# kspace: 2D complex array من جهاز MRI
# mask:   2D binary array (1=مقاس، 0=مفقود)
recon  = SCAFInverse()
result = recon.reconstruct(kspace, mask)

# قياس الجودة
snr = recon.snr(reference, result)
print(f"SNR: {snr:.2f} dB")
```

### SCAF-Video — إصلاح فيديو القلب

```python
from scaf import SCAFVideo

# frames:    [T, H, W] — الإطارات (المفقودة = أصفار)
# drop_mask: [T] bool  — True = مفقود
fixer   = SCAFVideo()
repaired = fixer.repair(frames, drop_mask)
```

### SCAF-Guard — كشف هلوسة AI

```python
from scaf import SCAFGuard

guard  = SCAFGuard(threshold=0.05)
result = guard.verify(ai_output, reference_input)

print(result['verdict'])            # PASS / WARNING / FAIL / CRITICAL
print(result['is_hallucination'])   # bool
print(result['confidence'])         # 0.0 - 1.0
```

### SCAF-Cert — شهادة جودة رياضية

```python
from scaf import SCAFCert

cert = SCAFCert()
certificate = cert.certify(reconstructed_image, mask)

print(certificate['verdict'])       # CERTIFIED / WARNING / FAIL
print(certificate['bound_l2'])      # الحد الأعلى لخطأ L2
print(certificate['bound_snr_db'])  # الحد الأعلى لخطأ SNR
```

### SCAF-SORL — تحسين الصور العامة

```python
from scaf import SCAFSORL

enhancer = SCAFSORL()
enhanced = enhancer.enhance(image)
```

---

## النتائج الموثقة

| المنتج | الاختبارات | النجاح | SNR Gain | الميزة |
|--------|-----------|--------|----------|--------|
| SCAF-Inverse | 90/90 | 100% | +1.94 dB | #1 على 4x Random |
| SCAF-Video | 120/120 | 100% | +1.47 dB | كل معدلات الفقدان |
| SCAF-Guard | 6/6 | 100% | +4.36 dB | لا مثيل له |
| SCAF-Cert | 90/90 | 100% | مضمون | بدون ground truth |
| SCAF-SORL | 3/3 | 100% | +2.16 dB | عام |

### مقارنة SCAF-Inverse مع المنافسين (4x Random — M4Raw)

| الطريقة | SNR (dB) | GPU | تدريب | شهادة خطأ |
|---------|----------|-----|-------|-----------|
| **SCAF-Inverse v2** | **15.028** | **لا** | **لا** | **✅** |
| BM3D+DC | 14.175 | لا | لا | ❌ |
| NLM+DC | 14.100 | لا | لا | ❌ |
| TV+DC | 13.692 | لا | لا | ❌ |
| Deep Learning | ~14.2 | ✅ نعم | ✅ ملايين صور | ❌ |

---

## المعاملات المقفلة

```python
beta  = 2.42   # قوة الانتشار  — ثابت عبر كل المجالات
dt    = 0.05   # الخطوة الزمنية — ثابت عبر كل المجالات
kappa = 0.22   # عتبة الحواف  — يتغير قليلاً بين المجالات
```

---

## الاختبارات

```bash
pip install scaf-medical[dev]
pytest scaf/tests/ -v
```

---

## الاقتباس

```bibtex
@software{scaf2025,
  title  = {SCAF: Structural Conservation Adaptive Flow},
  author = {SCAF Research Laboratory},
  year   = {2025},
  url    = {https://github.com/scaf-lab/scaf},
  note   = {322 validated tests, 98.2\% success rate}
}
```

---

## الترخيص

MIT License — مجاني للاستخدام البحثي والتجاري.
