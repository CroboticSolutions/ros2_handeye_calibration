"""Bound movement without useful measurements, including rejected steps."""
class AcquisitionProgress:
    MAX_WITHOUT_SAMPLE = 16
    MAX_LOCAL_WITHOUT_SAMPLE = 8
    MAX_PLANNED_TRANSIT = 48

    def __init__(self, views):
        self.limit = self.MAX_WITHOUT_SAMPLE
        self.views = views
        self.without_sample = 0
        self.local_without_sample = 0

    def observe(self, views):
        if views > self.views:
            self.views = views
            self.without_sample = self.local_without_sample = 0

    def before_step(self, planned, advancing=False):
        self.limit = self.MAX_PLANNED_TRANSIT if planned and advancing else self.MAX_WITHOUT_SAMPLE
        if self.without_sample >= self.limit:
            raise RuntimeError(f'Acquisition stalled: {self.limit} attempts without a new distinct view. Robot stopped; no new calibration was saved. Reposition the board to allow different viewing angles.')
        if not planned and self.local_without_sample >= self.MAX_LOCAL_WITHOUT_SAMPLE:
            raise RuntimeError('No feasible informative target and local search exhausted: 8 local attempts without a new distinct view. Robot stopped; no new calibration was saved. Reposition the board to allow different viewing angles.')
        self.without_sample += 1
        if not planned:
            self.local_without_sample += 1

    def status(self):
        return dict(attempts_without_sample=self.without_sample,
                    local_attempts_without_sample=self.local_without_sample,
                    max_attempts_without_sample=self.limit,
                    max_local_attempts_without_sample=self.MAX_LOCAL_WITHOUT_SAMPLE)
