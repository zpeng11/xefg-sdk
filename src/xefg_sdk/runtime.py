"""Stateless frame-pair execution using only a model package and app inputs."""
import numpy as np

from .bindings import bind_inputs
from .package import ModelPackage
from .planner import Planner
from .preprocess import prepare_inputs
from .types import Config, ImageView, InterpolationResult


class Xefg2xRuntime:
    def __init__(self, package, executor):
        self.package = package
        self.executor = executor
        self.planner = Planner(package)
        self._handles = {}
        self._closed = False

    @classmethod
    def load(cls, package_path, executor=None):
        if executor is None:
            from .executors import OrtExecutor
            executor = OrtExecutor()
        return cls(ModelPackage(package_path), executor)

    def prepare(self, previous, current, config=None):
        """Return a runtime plan and feeds, without loading an inference engine."""
        if self._closed:
            raise RuntimeError("runtime is closed")
        config = config or Config()
        mode, resources = prepare_inputs(previous, current, config)
        height, width = resources["current_color"].shape[:2]
        motion_height, motion_width = resources["motion_vectors"].shape[:2]
        plan = self.planner.resolve(mode, (width, height), (motion_width, motion_height),
                                    config.profile_id)
        variant = self.package.variants[plan.variant_id]
        return mode, plan, bind_inputs(variant, resources, plan)

    def interpolate_pair(self, previous, current, config=None):
        mode, plan, feeds = self.prepare(previous, current, config)
        variant = self.package.variants[plan.variant_id]
        if plan.variant_id not in self._handles:
            self._handles[plan.variant_id] = self.executor.load(self.package.path(variant["model"]))
        output_name = variant["output"]
        values = self.executor.run(self._handles[plan.variant_id], feeds, [output_name])
        image = np.asarray(values[output_name])
        width, height = plan.output_size
        if image.dtype != np.uint8 or image.shape != (height, width, 4):
            raise ValueError("executor returned an unexpected generated_frame shape or dtype")
        return InterpolationResult(
            "generated", ImageView(image, "BGRA8_UNORM"), previous.frame_id, current.frame_id,
            ui_mode=mode, profile_id=plan.profile_id)

    def close(self):
        if not self._closed:
            for handle in self._handles.values():
                self.executor.close(handle)
            self._handles.clear()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
