"""Local image Jacobian learned from observed robot increments, not camera TF."""
from collections import deque
import numpy as np


class ImageMotionModel:
    def __init__(self, joints):
        self.joints = joints
        self.samples = deque(maxlen=32)
        self.good_predictions = 0

    def add(self, before_q, after_q, before_pixels, after_pixels):
        dq = np.asarray(after_q)-before_q
        if np.linalg.norm(dq) < .003:
            return
        predicted = self.predict(before_q, dq, before_pixels)
        if predicted is not None:
            error = float(np.max(np.linalg.norm(predicted-after_pixels, axis=1)))
            self.good_predictions = self.good_predictions+1 if error < .012 else 0
            if error > .035:
                self.samples.clear()  # moved board or invalid local model
        self.samples.append((np.asarray(after_q).copy(), dq.copy(),
                             (after_pixels-before_pixels).reshape(-1)))

    def jacobian(self, q):
        samples = [s for s in self.samples if np.max(np.abs(s[0]-q)) < .25]
        if not samples:
            return None, None
        d = np.array([s[1] for s in samples])
        y = np.array([s[2] for s in samples])
        return np.linalg.lstsq(d, y, rcond=.05)[0], np.linalg.pinv(d, rcond=.05) @ d

    def predict(self, q, dq, pixels):
        jac, projection = self.jacobian(q)
        if jac is None or np.linalg.norm(dq - dq@projection) > .001:
            return None
        return pixels + (dq@jac).reshape(-1, 2)

    def confident(self, q, dq):
        direction = dq / max(np.linalg.norm(dq), 1e-9)
        evidence = sum(1 for at, motion, _ in self.samples
                       if np.max(np.abs(at-q)) < .25
                       and abs(motion@direction)/np.linalg.norm(motion) > .7)
        return self.good_predictions >= 3 and evidence >= 2

    def centered_delta(self, q, pixels):
        jac, projection = self.jacobian(q)
        if jac is None:
            return np.zeros(self.joints)
        center_jac = jac.reshape(self.joints, -1, 2).mean(axis=1).T
        # Damped least squares; centering is secondary to useful excitation.
        error = np.array([.5, .5]) - (pixels.min(axis=0)+pixels.max(axis=0))/2
        return projection @ center_jac.T @ np.linalg.solve(center_jac@center_jac.T + .05*np.eye(2), error)
