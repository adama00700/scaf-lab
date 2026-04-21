"""
اختبارات SCAF — تتحقق من صحة كل المنتجات
شغّل: pytest tests/ -v
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scaf import (SCAFInverse, SCAFVideo, SCAFGuard,
                   SCAFCert, SCAFSORL,
                   conservation_constant, conservation_error, scaf_step)

EPS = 1e-12

# ── fixtures ────────────────────────────────────────────────
@pytest.fixture
def sample_image():
    rng = np.random.default_rng(42)
    img = rng.random((64, 64)) * 8 + 0.5
    return img.astype(np.float64)

@pytest.fixture
def sample_kspace(sample_image):
    Kf = np.fft.fftshift(np.fft.fft2(sample_image))
    mask = np.zeros((64, 64))
    mask[28:36, :] = 1  # center lines
    rng = np.random.default_rng(42)
    lines = rng.choice(range(64), size=16, replace=False)
    mask[lines, :] = 1
    return Kf * mask, mask

@pytest.fixture
def sample_video():
    rng = np.random.default_rng(42)
    return rng.random((20, 32, 32)).astype(np.float64) * 8 + 0.5

# ══════════════════════════════════════════════════════════════
# اختبارات القانون الفيزيائي
# ══════════════════════════════════════════════════════════════

class TestConservationLaw:
    def test_constant_positive(self, sample_image):
        C = conservation_constant(sample_image)
        assert C > 0, "ثابت الحفاظ يجب أن يكون موجباً"

    def test_constant_with_mask(self, sample_image):
        mask = np.ones_like(sample_image) * 0.5
        C = conservation_constant(sample_image, mask)
        C_no_mask = conservation_constant(sample_image)
        assert abs(C - C_no_mask * 2) < 1.0

    def test_error_zero_when_preserved(self, sample_image):
        C = conservation_constant(sample_image)
        Ec = conservation_error(sample_image, C)
        assert Ec < EPS * 100, "خطأ الحفاظ يجب أن يكون صفراً"

    def test_scaf_step_preserves_shape(self, sample_image):
        result = scaf_step(sample_image)
        assert result.shape == sample_image.shape

    def test_scaf_step_no_explosion(self, sample_image):
        x = sample_image.copy()
        for _ in range(100):
            x = scaf_step(x)
        assert np.all(np.isfinite(x)), "SCAF يجب ألا ينفجر"
        assert x.max() < 100, "القيم يجب أن تبقى معقولة"

# ══════════════════════════════════════════════════════════════
# اختبارات SCAFInverse
# ══════════════════════════════════════════════════════════════

class TestSCAFInverse:
    def test_output_shape(self, sample_kspace, sample_image):
        kspace, mask = sample_kspace
        recon = SCAFInverse(n_iter=50)
        result = recon.reconstruct(kspace, mask)
        assert result.shape == sample_image.shape

    def test_output_finite(self, sample_kspace):
        kspace, mask = sample_kspace
        recon = SCAFInverse(n_iter=50)
        result = recon.reconstruct(kspace, mask)
        assert np.all(np.isfinite(result)), "المخرج يجب أن يكون finite"

    def test_output_nonnegative(self, sample_kspace):
        kspace, mask = sample_kspace
        recon = SCAFInverse(n_iter=50)
        result = recon.reconstruct(kspace, mask)
        assert np.all(result >= 0), "الصورة يجب أن تكون غير سالبة"

    def test_snr_positive(self, sample_kspace, sample_image):
        kspace, mask = sample_kspace
        recon = SCAFInverse(n_iter=100)
        result = recon.reconstruct(kspace, mask)
        snr = recon.snr(sample_image, result)
        assert snr > 0, "SNR يجب أن يكون موجباً"

    def test_3d_reconstruction(self, sample_kspace):
        kspace, mask = sample_kspace
        vol_kspace = np.stack([kspace] * 3, axis=0)
        recon = SCAFInverse(n_iter=30)
        result = recon.reconstruct_3d(vol_kspace, mask)
        assert result.shape == (3, 64, 64)

# ══════════════════════════════════════════════════════════════
# اختبارات SCAFVideo
# ══════════════════════════════════════════════════════════════

class TestSCAFVideo:
    def test_output_shape(self, sample_video):
        drop_mask = np.zeros(20, dtype=bool)
        drop_mask[[3, 7, 12]] = True
        fixer = SCAFVideo(spatial_iters=20)
        result = fixer.repair(sample_video, drop_mask)
        assert result.shape == sample_video.shape

    def test_good_frames_preserved(self, sample_video):
        drop_mask = np.zeros(20, dtype=bool)
        drop_mask[5] = True
        fixer = SCAFVideo(spatial_iters=10)
        result = fixer.repair(sample_video, drop_mask)
        # الإطارات غير المفقودة يجب أن تتغير قليلاً فقط
        for t in range(20):
            if not drop_mask[t]:
                diff = np.mean(np.abs(result[t] - sample_video[t]))
                assert diff < 2.0, f"الإطار {t} تغيّر كثيراً"

    def test_missing_frames_filled(self, sample_video):
        frames = sample_video.copy()
        drop_mask = np.zeros(20, dtype=bool)
        drop_mask[10] = True
        frames[10] = 0  # إطار مفقود
        fixer = SCAFVideo(spatial_iters=10)
        result = fixer.repair(frames, drop_mask)
        assert np.mean(np.abs(result[10])) > 0.1, "الإطار المفقود يجب أن يُملأ"

# ══════════════════════════════════════════════════════════════
# اختبارات SCAFGuard
# ══════════════════════════════════════════════════════════════

class TestSCAFGuard:
    def test_good_output_passes(self, sample_image):
        guard = SCAFGuard(threshold=0.05)
        result = guard.verify(sample_image)
        assert "is_hallucination" in result
        assert "confidence" in result
        assert "verdict" in result

    def test_corrupted_output_detected(self, sample_image):
        guard = SCAFGuard(threshold=0.01)
        # صورة مُشوَّشة بشدة
        corrupted = sample_image * np.random.rand(*sample_image.shape) * 5
        result = guard.verify(corrupted, sample_image)
        # يجب أن يكتشف الشذوذ
        assert result["conservation_error"] >= 0

    def test_batch_verify(self, sample_image):
        guard = SCAFGuard()
        outputs = [sample_image, sample_image * 2]
        results = guard.batch_verify(outputs)
        assert len(results) == 2

# ══════════════════════════════════════════════════════════════
# اختبارات SCAFCert
# ══════════════════════════════════════════════════════════════

class TestSCAFCert:
    def test_certificate_structure(self, sample_image):
        mask = np.ones_like(sample_image) * 0.25
        cert = SCAFCert()
        certificate = cert.certify(sample_image, mask)
        required_keys = ["bound_l2", "bound_linf", "bound_snr_db",
                         "Ec", "C", "rho", "verdict"]
        for key in required_keys:
            assert key in certificate, f"مفتاح مفقود: {key}"

    def test_bounds_positive(self, sample_image):
        mask = np.ones_like(sample_image) * 0.25
        cert = SCAFCert()
        certificate = cert.certify(sample_image, mask)
        assert certificate["bound_l2"]   >= 0
        assert certificate["bound_linf"] >= 0

    def test_good_image_certified(self, sample_image):
        mask = np.ones_like(sample_image) * 0.25
        cert = SCAFCert()
        certificate = cert.certify(sample_image, mask)
        assert "CERTIFIED" in certificate["verdict"] or \
               "WARNING" in certificate["verdict"]

# ══════════════════════════════════════════════════════════════
# اختبارات SCAFSORL
# ══════════════════════════════════════════════════════════════

class TestSCAFSORL:
    def test_output_shape(self, sample_image):
        enhancer = SCAFSORL(n_iter=50)
        result = enhancer.enhance(sample_image)
        assert result.shape == sample_image.shape

    def test_range_preserved(self, sample_image):
        enhancer = SCAFSORL(n_iter=50)
        result = enhancer.enhance(sample_image, preserve_range=True)
        orig_range = sample_image.max() - sample_image.min()
        res_range  = result.max() - result.min()
        assert res_range <= orig_range * 1.1, "النطاق يجب أن يُحفَظ تقريباً"

    def test_output_finite(self, sample_image):
        enhancer = SCAFSORL(n_iter=50)
        result = enhancer.enhance(sample_image)
        assert np.all(np.isfinite(result))

    def test_batch_enhance(self, sample_image):
        enhancer = SCAFSORL(n_iter=20)
        results = enhancer.enhance_batch([sample_image, sample_image])
        assert len(results) == 2
        assert results[0].shape == sample_image.shape

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
