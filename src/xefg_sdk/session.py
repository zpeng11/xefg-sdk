"""Synchronous history pairing. No swapchain or presentation ownership."""
from .types import Config, InterpolationResult


class Xefg2xSession:
    def __init__(self, runtime, config=None):
        self.runtime = runtime
        self.config = config or Config()
        self._previous = None

    def submit(self, frame):
        # Snapshot before execution so application resources may be reused
        # as soon as this synchronous call returns.
        owned = frame.copy()
        if owned.reset_history:
            self.reset()
        if self._previous is None:
            self._previous = owned
            return InterpolationResult("history_ready", None, None, owned.frame_id)
        result = self.runtime.interpolate_pair(self._previous, owned, self.config)
        self._previous = owned
        return result

    def reset(self):
        self._previous = None
