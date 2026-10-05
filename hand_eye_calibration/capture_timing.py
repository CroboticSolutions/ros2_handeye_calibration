"""Camera latency model and raw ChArUco observations, independent of ROS.

The Femto Bolt colour stream arrives 0.5-1 s after its header stamp. Freshness
limits written for a ~50 ms camera (0.2 s, 0.35 s) then reject every frame and
the automatic run stalls. Two rules replace them:

  * Freshness limits scale with the measured stamp->arrival delay (p95 plus a
    margin), never below the original fixed value.
  * A calibration frame must be *exposed* after the robot became stationary:
    its header stamp must be later than the settle time. This is independent
    of latency, because it compares stamps with stamps.

The measured delay is observable in status so a slow camera is visible
instead of looking like "board lost".
"""
import collections
import json
import threading

import numpy as np


class LatencyModel:
    def __init__(self, window=60, margin_s=0.25, cap_s=3.0, configured_s=-1.0):
        self.ages = collections.deque(maxlen=int(window))
        self.margin_s = float(margin_s)
        self.cap_s = float(cap_s)
        # >= 0 overrides the measurement (e.g. for simulation or a known camera).
        self.configured_s = float(configured_s)
        self.lock = threading.Lock()

    def add(self, age_s):
        if np.isfinite(age_s) and -1.0 < age_s < 30.0:
            with self.lock:
                self.ages.append(float(age_s))

    def stats(self):
        with self.lock:
            ages = np.array(self.ages, dtype=float)
        if self.configured_s >= 0:
            return {'source': 'configured', 'median_s': self.configured_s, 'p95_s': self.configured_s,
                    'count': int(ages.size)}
        if ages.size < 5:
            return {'source': 'unmeasured', 'median_s': None, 'p95_s': None, 'count': int(ages.size)}
        return {'source': 'measured', 'median_s': float(np.median(ages)),
                'p95_s': float(np.percentile(ages, 95)), 'count': int(ages.size)}

    def limit(self, base_s):
        """Maximum acceptable stamp age: the original limit, or p95 + margin."""
        p95 = self.stats()['p95_s']
        if p95 is None:
            return float(base_s)
        return float(min(self.cap_s, max(base_s, p95 + self.margin_s)))

    def settle_wait(self, base_s):
        """Extra time to wait after the robot stops so a post-stop frame exists."""
        median = self.stats()['median_s']
        return float(base_s if median is None else min(self.cap_s, max(base_s, median + self.margin_s)))


def parse_observation(text):
    """Detector JSON -> dict with integer stamp_ns, or None when malformed."""
    try:
        data = json.loads(text)
        data['stamp_ns'] = int(data['stamp']['sec']) * 1_000_000_000 + int(data['stamp']['nanosec'])
        ids, image, obj = data['ids'], data['image_points'], data['object_points']
        if not (len(ids) == len(image) == len(obj)) or not ids:
            return None
        return data
    except (ValueError, TypeError, KeyError):
        return None


class ObservationBuffer:
    """Recent detector observations keyed by image stamp."""

    def __init__(self, capacity=90):
        self.items = collections.OrderedDict()
        self.capacity = int(capacity)
        self.lock = threading.Lock()

    def add(self, observation):
        with self.lock:
            self.items[observation['stamp_ns']] = observation
            while len(self.items) > self.capacity:
                self.items.popitem(last=False)

    def get(self, stamp_ns, tolerance_ns=2_000_000):
        with self.lock:
            if stamp_ns in self.items:
                return self.items[stamp_ns]
            best = min(self.items, key=lambda s: abs(s - stamp_ns), default=None)
            if best is not None and abs(best - stamp_ns) <= tolerance_ns:
                return self.items[best]
        return None


def frame_record(observation, robot_pose, stamp_ns, joints=None):
    """Compact per-frame record stored with a sample and in the dataset."""
    return {
        'stamp_ns': int(stamp_ns),
        'robot': [float(v) for v in robot_pose],
        'ids': [int(v) for v in observation['ids']],
        'image_points': [[float(a), float(b)] for a, b in observation['image_points']],
        'object_points': [[float(a), float(b), float(c)] for a, b, c in observation['object_points']],
        'reprojection_px': observation.get('reprojection_px'),
        'joints': joints,
    }
