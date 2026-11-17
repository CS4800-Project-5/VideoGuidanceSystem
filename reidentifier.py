"""Re-identification: recognize a person who comes back into view, by how they look.

The tracker (ByteTrack / BoT-SORT) follows people from frame to frame by position, so someone who
leaves the view, or the whole camera leaving the room, ends their track; when they come back they
get a new tracker ID. The ReIdentifier remembers what every person looked like, with no time limit,
and gives a returning person their old ID if their appearance matches.

How appearance is measured is a Strategy (AppearanceExtractor), so it can be swapped without
touching the matching:
  - OSNetExtractor: a neural network trained for person re-identification (recognizing the same
    person across different cameras and viewpoints). Default when its weights are installed.
  - ColorHistogramExtractor: clothing colors. No download needed, but people in similar clothes
    look alike to it. Used when the OSNet weights are missing.

Based on Angel's reidentifier.py (angel-branch): the extractor Strategy and the color histogram come
from there. Changed: no time limit by default, several appearance samples per person, matching
against everyone not currently in view (so two people in view never share an ID), and OSNet.
"""

from abc import ABC, abstractmethod

import cv2
import numpy as np


class AppearanceExtractor(ABC):
    """Turns image crops of people into feature vectors."""

    # Similarity at or above which two crops count as the same person (see match()).
    default_threshold = 0.7

    @abstractmethod
    def extract(self, crop):
        """Return a 1-D float32 feature vector, or None if the crop is unusable."""

    def extract_batch(self, crops):
        """Features for several crops at once (one inference call where the extractor can batch)."""
        return [self.extract(crop) for crop in crops]

    @abstractmethod
    def similarity(self, a, b):
        """Return how alike two features are: higher is more alike, 1 means identical."""


class ColorHistogramExtractor(AppearanceExtractor):
    """Hue/saturation histogram of the crop (Angel's): cheap on a CPU, and good at telling
    apart people in different clothes. Brightness is ignored so shadows and lighting changes
    matter less."""

    default_threshold = 0.7

    def __init__(self, hue_bins=16, sat_bins=16, min_pixels=200):
        self.bins = [hue_bins, sat_bins]
        self.min_pixels = min_pixels

    def extract(self, crop):
        if crop is None or crop.size == 0:
            return None
        if crop.shape[0] * crop.shape[1] < self.min_pixels:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, self.bins, [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
        return hist.flatten().astype(np.float32)

    def similarity(self, a, b):
        # Bhattacharyya distance is 0 for identical histograms, 1 for disjoint ones.
        return 1.0 - cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)


class OSNetExtractor(AppearanceExtractor):
    """OSNet person re-identification network (osnet.py, from torchreid). Each crop becomes a
    512-number "appearance fingerprint" of the whole body (clothes, build, colors, carried items),
    trained so the same person seen from different angles and cameras gives similar numbers.
    Similarity is the cosine similarity of two fingerprints."""

    default_threshold = 0.65

    MEAN = np.array([0.485, 0.456, 0.406], np.float32)   # ImageNet normalization, as in training
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, weights_path, variant="osnet_x0_5", device="cpu", min_height=40):
        import torch
        import osnet

        self.torch = torch
        self.device = device
        self.min_height = min_height
        self.model = getattr(osnet, variant)(num_classes=1, pretrained=False, loss="softmax")
        try:
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        except Exception:
            # torchreid checkpoints can hold extra non-tensor entries (epoch, scores) that
            # weights_only refuses. The file comes from torchreid's official model zoo.
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        own = self.model.state_dict()
        loaded = {}
        for key, value in state.items():
            key = key[7:] if key.startswith("module.") else key
            if key in own and own[key].shape == value.shape:
                loaded[key] = value
        # Everything except the training classifier (whose size depends on the training set)
        # must load, or the features are meaningless.
        needed = [k for k in own if not k.startswith("classifier.")]
        missing = [k for k in needed if k not in loaded]
        if missing:
            raise ValueError(f"{weights_path} doesn't match {variant}: {len(missing)} layers missing, "
                             f"e.g. {missing[:3]}")
        own.update(loaded)
        self.model.load_state_dict(own)
        self.model.eval().to(device)

    def extract(self, crop):
        return self.extract_batch([crop])[0]

    def extract_batch(self, crops):
        usable, batch = [], []
        for i, crop in enumerate(crops):
            if crop is None or crop.size == 0 or crop.shape[0] < self.min_height:
                continue
            rgb = cv2.cvtColor(cv2.resize(crop, (128, 256)), cv2.COLOR_BGR2RGB)   # width x height
            batch.append(((rgb.astype(np.float32) / 255.0 - self.MEAN) / self.STD).transpose(2, 0, 1))
            usable.append(i)
        features = [None] * len(crops)
        if not batch:
            return features
        with self.torch.no_grad():
            out = self.model(self.torch.from_numpy(np.stack(batch)).to(self.device)).cpu().numpy()
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
        for i, feature in zip(usable, out):
            features[i] = feature.astype(np.float32)
        return features

    def similarity(self, a, b):
        return float(np.dot(a, b))


def make_extractor(weights_path, logger=None):
    """OSNet if its weights load, otherwise the color histogram (with a warning)."""
    try:
        extractor = OSNetExtractor(weights_path)
        if logger:
            logger.info(f"Re-ID: OSNet appearance model loaded from {weights_path}")
        return extractor
    except Exception as error:
        if logger:
            logger.warn(f"Re-ID: OSNet not available ({error}); using clothing colors instead. "
                        "Run setup_zed.sh (with internet) to download the OSNet weights.")
        return ColorHistogramExtractor()


class ReIdentifier:
    """Remembers what each person looks like and finds the best match for a new track.

    Every person ID keeps a small bank of appearance samples from different moments (angles,
    poses, lighting). A new track is compared with every remembered person who is not in view
    right now; the best match above the threshold, clearly better than the second best, gets
    that person's ID back. There is no time limit unless memory_seconds is set.
    """

    def __init__(self, extractor, threshold=None, memory_seconds=0.0, bank_size=12, margin=0.05):
        self.extractor = extractor
        self.threshold = extractor.default_threshold if threshold is None or threshold < 0 else threshold
        self.memory_seconds = memory_seconds     # 0 = remember for as long as the program runs
        self.bank_size = bank_size
        self.margin = margin
        self.reset()

    def reset(self):
        self.banks = {}        # person ID -> list of appearance features
        self.last_seen = {}    # person ID -> image time last seen (only used for memory_seconds)

    def __len__(self):
        return len(self.banks)

    def score(self, feature, person_id):
        """How well a feature matches a person: mean of the 3 best similarities to their samples,
        so one lucky sample doesn't decide, but a person seen from the same angle still matches."""
        sims = sorted((self.extractor.similarity(feature, f) for f in self.banks[person_id]), reverse=True)
        return float(np.mean(sims[:3]))

    def match(self, queries, exclude=()):
        """queries: {key: feature or None}. Returns {key: (person_id or None, best_score, best_id)}.

        One-to-one: each remembered person is given to at most one query, best scores first, and
        people in `exclude` (in view right now) are never candidates.
        """
        candidates = [p for p in self.banks if p not in exclude]
        scored = {}
        for key, feature in queries.items():
            if feature is None or not candidates:
                scored[key] = []
                continue
            scored[key] = sorted(((self.score(feature, p), p) for p in candidates), reverse=True)

        result = {key: (None, s[0][0] if s else None, s[0][1] if s else None) for key, s in scored.items()}
        pairs = sorted(((s, key, p) for key, lst in scored.items() for s, p in lst
                        if s >= self.threshold), reverse=True)
        taken_keys, taken_people = set(), set()
        for s, key, p in pairs:
            if key in taken_keys or p in taken_people:
                continue
            # Ambiguous if another remembered person scores almost as well for this query.
            runner_up = next((other for other, q in scored[key] if q != p), None)
            if runner_up is not None and s - runner_up < self.margin:
                continue
            result[key] = (p, s, p)
            taken_keys.add(key)
            taken_people.add(p)
        return result

    def add(self, person_id, feature, stamp):
        """Add an appearance sample for a person. Near-duplicates of an existing sample are
        skipped so the bank keeps a variety of views; the oldest sample goes when it's full."""
        self.last_seen[person_id] = stamp
        if feature is None:
            return
        bank = self.banks.setdefault(person_id, [])
        if any(self.extractor.similarity(feature, f) > 0.97 for f in bank):
            return
        bank.append(feature)
        if len(bank) > self.bank_size:
            bank.pop(0)

    def seen(self, person_id, stamp):
        self.last_seen[person_id] = stamp

    def expire(self, stamp):
        """Forget people not seen for memory_seconds (only if memory_seconds > 0)."""
        if self.memory_seconds <= 0:
            return
        for person_id, seen_at in list(self.last_seen.items()):
            if stamp - seen_at > self.memory_seconds:
                self.banks.pop(person_id, None)
                del self.last_seen[person_id]


def mean_feature(features):
    """Average several samples of one track into one query (normalized for cosine similarity)."""
    features = [f for f in features if f is not None]
    if not features:
        return None
    mean = np.mean(features, axis=0).astype(np.float32)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean
