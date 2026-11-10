"""Resolve internal graph metadata without capture tensors or shader plans."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RuntimePlan:
    variant_id: str
    profile_id: str
    output_size: tuple[int, int]
    motion_size: tuple[int, int]
    feeds: dict[str, np.ndarray]


class Planner:
    def __init__(self, package):
        self.package = package
        self._cache = {}

    def resolve(self, mode, output_size, motion_size, profile_id=None):
        key = int(mode), tuple(output_size), tuple(motion_size), profile_id
        if key in self._cache:
            return self._cache[key]
        matches = []
        for variant_id, variant in self.package.variants.items():
            if variant["mode"] != int(mode):
                continue
            for profile in self.package.profiles[variant_id]:
                if profile_id is not None and profile["id"] != profile_id:
                    continue
                if (tuple(profile["output_size"]) == tuple(output_size)
                        and tuple(profile["motion_size"]) == tuple(motion_size)):
                    matches.append((variant_id, profile))
        if len(matches) != 1:
            raise ValueError("expected one profile for mode %d, output %s, motion %s; found %d"
                             % (int(mode), output_size, motion_size, len(matches)))
        variant_id, profile = matches[0]
        feeds = {}
        for name, entry in profile["feeds"].items():
            if entry["dtype"] not in ("int64", "float32"):
                raise ValueError("unsupported metadata dtype: " + entry["dtype"])
            value = np.asarray(entry["values"], dtype=entry["dtype"]).reshape(entry["shape"])
            value.setflags(write=False)
            feeds[name] = value
        plan = RuntimePlan(variant_id, profile["id"], tuple(output_size), tuple(motion_size), feeds)
        self._cache[key] = plan
        return plan
