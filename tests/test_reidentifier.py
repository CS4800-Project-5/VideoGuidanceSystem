"""Re-identification matching (reidentifier.py), with a fake appearance model whose features are
plain unit vectors, so each test controls exactly how alike two views are."""

import os

import numpy as np
import pytest

from reidentifier import ColorHistogramExtractor, ReIdentifier, make_extractor, mean_feature

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeExtractor:
    default_threshold = 0.7

    def similarity(self, a, b):
        return float(np.dot(a, b))


def unit(*values):
    v = np.array(values, np.float32)
    return v / np.linalg.norm(v)


FRONT, BACK, TOP, OTHER = unit(1, 0, 0, 0), unit(0, 1, 0, 0), unit(0, 0, 1, 0), unit(0, 0, 0, 1)


def reid(**kwargs):
    return ReIdentifier(FakeExtractor(), **kwargs)


def test_view_from_new_angle_matches_its_own_sample():
    r = reid()
    for i, view in enumerate([FRONT, BACK, TOP]):
        r.add(1, view, i)
    # Seen from above: matches the sample from above, not an average with front and back views.
    assert r.score(unit(0.05, 0, 1, 0), 1) > 0.99
    assert r.match({"q": unit(0.05, 0, 1, 0)})["q"][0] == 1


def test_full_bank_drops_the_most_redundant_sample_not_the_oldest():
    r = reid(bank_size=3)
    for i, view in enumerate([FRONT, BACK, TOP]):
        r.add(1, view, i)
    r.add(1, unit(1, 0.3, 0, 0), 3)          # another front-ish view
    kinds = sorted(int(np.argmax(f[:3])) for f in r.banks[1])
    assert kinds == [0, 1, 2]                # front, back and top all kept


def test_near_duplicate_samples_are_skipped():
    r = reid()
    r.add(1, FRONT, 0)
    r.add(1, unit(1, 0.01, 0, 0), 1)
    assert len(r.banks[1]) == 1


def test_people_in_view_are_never_candidates():
    r = reid()
    r.add(1, FRONT, 0)
    assert r.match({"q": FRONT}, exclude={1})["q"][0] is None


def test_each_remembered_person_goes_to_one_track_only():
    r = reid()
    r.add(1, FRONT, 0)
    result = r.match({"close": unit(1, 0.1, 0, 0), "less": unit(1, 0.4, 0, 0)})
    assert result["close"][0] == 1 and result["less"][0] is None


def test_ambiguous_match_gets_a_new_id():
    r = reid(margin=0.05)
    r.add(1, unit(1, 0.02, 0, 0), 0)
    r.add(2, unit(1, -0.02, 0, 0), 0)
    assert r.match({"q": FRONT})["q"][0] is None


def test_below_threshold_is_a_new_person():
    r = reid()
    r.add(1, FRONT, 0)
    person, score, closest = r.match({"q": OTHER})["q"]
    assert person is None and closest == 1 and score < 0.7


def test_person_too_far_away_is_ruled_out():
    r = reid(max_speed=2.0, position_slack=1.0)
    r.add(1, FRONT, 0)
    r.seen(1, 10.0, (0.0, 0.0, 0.0))
    near, far = {"q": (2.0, 0.0, 0.0)}, {"q": (20.0, 0.0, 0.0)}
    assert r.match({"q": FRONT}, positions=far, stamp=11.0)["q"][0] is None
    assert r.match({"q": FRONT}, positions=near, stamp=11.0)["q"][0] == 1
    assert r.match({"q": FRONT}, positions=far, stamp=30.0)["q"][0] == 1     # had time to walk
    assert r.match({"q": FRONT}, positions={"q": None}, stamp=11.0)["q"][0] == 1


def test_memory_limit_forgets_people():
    r = reid(memory_seconds=5.0)
    r.add(1, FRONT, 0.0)
    r.expire(4.0)
    assert len(r) == 1
    r.expire(6.0)
    assert len(r) == 0


def test_mean_feature_ignores_missing_samples():
    assert mean_feature([None, None]) is None
    assert np.allclose(mean_feature([FRONT, None, FRONT]), FRONT)


def test_color_histogram_tells_different_clothes_apart():
    extractor = ColorHistogramExtractor()
    red = np.zeros((100, 40, 3), np.uint8)
    red[:] = (0, 0, 200)
    blue = np.zeros((100, 40, 3), np.uint8)
    blue[:] = (200, 0, 0)
    same = extractor.similarity(extractor.extract(red), extractor.extract(red.copy()))
    different = extractor.similarity(extractor.extract(red), extractor.extract(blue))
    assert same > 0.99 and different < 0.3
    assert extractor.extract(red[:5, :5]) is None        # too small to judge


def test_without_osnet_weights_falls_back_to_clothing_colors():
    assert isinstance(make_extractor(["", "no_such_file.pth"]), ColorHistogramExtractor)


@pytest.mark.parametrize("name", ["osnet_x0_5_msmt17.pth", "osnet_ibn_x1_0_msmt17.pth"])
def test_installed_osnet_weights_load(name):
    path = os.path.join(ROOT, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not downloaded (setup_zed.sh)")
    pytest.importorskip("torch")
    extractor = make_extractor([path])
    assert extractor.variant == name.replace("_msmt17.pth", "")
    feature = extractor.extract(np.zeros((256, 128, 3), np.uint8))
    assert feature.shape == (512,) and abs(np.linalg.norm(feature) - 1) < 1e-3
