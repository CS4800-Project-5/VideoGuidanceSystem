"""Re-identification: give an object its old ID when it comes back into view.

ByteTrack only remembers a lost object for `track_buffer` frames and matches by
position, so an object that leaves for longer gets a new tracker ID. The
ReIdentifier sits on top of ByteTrack. It maps tracker IDs to stable object IDs
and remembers what departed objects looked like for `memory_seconds`. A new
tracker ID that looks like a remembered object gets that object's ID back.

How appearance is measured is a Strategy (AppearanceExtractor), so the color
histogram can be swapped for a learned embedding without touching the matching.
"""

from abc import ABC, abstractmethod

import cv2
import numpy as np


class AppearanceExtractor(ABC):
    """Turns an image crop of one object into a feature vector."""

    @abstractmethod
    def extract(self, crop):
        """Return a 1-D float32 feature vector, or None if the crop is unusable."""

    @abstractmethod
    def similarity(self, a, b):
        """Return how alike two features are, from 0 (different) to 1 (same)."""


class ColorHistogramExtractor(AppearanceExtractor):
    """Hue/saturation histogram of the crop: cheap on a CPU, and good at telling
    apart people in different clothes. Brightness is ignored so shadows and
    lighting changes matter less."""

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


class ReIdentifier:
    """Maps short-lived tracker IDs to stable object IDs."""

    def __init__(self, extractor, memory_seconds=60.0, match_threshold=0.7,
                 feature_smoothing=0.3):
        self.extractor = extractor
        self.memory_seconds = memory_seconds
        self.match_threshold = match_threshold
        self.feature_smoothing = feature_smoothing
        self.reset()

    def reset(self):
        self.tracker_to_object = {}   # tracker ID -> object ID
        self.features = {}            # object ID -> appearance feature
        self.departed = {}            # object ID -> time it left the view
        self.next_object_id = 1

    def assign(self, tracker_id, crop, stamp):
        """Return (object_id, how) for a tracked detection.

        `how` is "tracked" for a known tracker ID, "returned" when ByteTrack
        itself kept the ID through an absence, "reidentified" when appearance
        matched a departed object, and "new" otherwise.
        """
        self.expire(stamp)
        feature = self.extractor.extract(crop)

        if tracker_id in self.tracker_to_object:
            object_id = self.tracker_to_object[tracker_id]
            how = "returned" if self.departed.pop(object_id, None) is not None else "tracked"
        else:
            object_id = self._match_departed(feature)
            if object_id is not None:
                del self.departed[object_id]
                how = "reidentified"
            else:
                object_id = self.next_object_id
                self.next_object_id += 1
                how = "new"
            self.tracker_to_object[tracker_id] = object_id

        self._update_feature(object_id, feature)
        return object_id, how

    def depart(self, object_id, stamp):
        """Remember an object that left the view, so it can be re-identified."""
        if object_id in self.features:
            self.departed[object_id] = stamp

    def expire(self, stamp):
        """Forget departed objects that have been gone longer than memory_seconds."""
        for object_id, left_at in list(self.departed.items()):
            if stamp - left_at > self.memory_seconds:
                del self.departed[object_id]
                self.features.pop(object_id, None)
                self.tracker_to_object = {
                    t: o for t, o in self.tracker_to_object.items() if o != object_id
                }

    def _match_departed(self, feature):
        if feature is None:
            return None
        best_id, best_score = None, self.match_threshold
        for object_id in self.departed:
            score = self.extractor.similarity(feature, self.features[object_id])
            if score >= best_score:
                best_id, best_score = object_id, score
        return best_id

    def _update_feature(self, object_id, feature):
        if feature is None:
            return
        old = self.features.get(object_id)
        if old is None:
            self.features[object_id] = feature
        else:
            # Running average, so one odd frame (partial occlusion, motion blur)
            # does not overwrite what the object usually looks like.
            mixed = (1 - self.feature_smoothing) * old + self.feature_smoothing * feature
            self.features[object_id] = mixed / mixed.sum()
