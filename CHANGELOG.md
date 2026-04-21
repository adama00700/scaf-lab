# Changelog

## [1.0.0] — 2025

### أول إصدار رسمي

#### المنتجات
- **SCAFInverse** — إعادة بناء MRI: 90/90 PASS، +1.94 dB
- **SCAFVideo** — إصلاح Echo: 120/120 PASS، +1.47 dB
- **SCAFGuard** — كشف هلوسة AI: 6/6 PASS، +4.36 dB
- **SCAFCert** — شهادة جودة: 90/90 PASS، 3 حدود مضمونة
- **SCAFSORL** — تحسين عام: 3/3 PASS، +2.16 dB

#### النواة
- `conservation_constant()` — حساب ثابت الحفاظ
- `conservation_error()` — قياس خطأ الحفاظ
- `scaf_step()` — خطوة Anisotropic Diffusion
- معاملات مقفلة: beta=2.42، dt=0.05، kappa=0.22

#### الإحصاءات
- 322/328 اختبار ناجح (98.2%)
- صفر GPU مطلوب
- صفر بيانات تدريب
- يعمل على Python 3.9+
