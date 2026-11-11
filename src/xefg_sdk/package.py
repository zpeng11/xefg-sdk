"""Read the versioned, language-neutral model package."""
import hashlib
import json
from pathlib import Path


class ModelPackage:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.manifest = self.read_json("manifest.json")
        if self.manifest.get("schema_version") != 1:
            raise ValueError("unsupported model package schema_version")
        # These describe normalized model-boundary tensors, not the engine's
        # native coordinate system or camera settings. Source capture metadata
        # is deliberately not interpreted by the runtime.
        supported = {
            "color_format": "BGRA8_UNORM", "motion_format": "RG16_FLOAT",
            "motion_units": "mv_pixels", "motion_direction": "current_to_previous",
            "depth_inverted": False, "jittered_motion": False,
            "ui_premultiplied": True,
        }
        for key, value in self.manifest.get("conventions", {}).items():
            if key not in supported or value != supported[key]:
                raise ValueError("unsupported model convention: %s=%s" % (key, value))
        self.variants = self.manifest.get("variants", {})
        if not self.variants:
            raise ValueError("model package contains no variants")
        self.profiles = {}
        for variant_id, variant in self.variants.items():
            if variant.get("mode") not in (1, 3, 4):
                raise ValueError("unsupported model mode: %s" % variant.get("mode"))
            model_path = self.path(variant["model"])
            digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
            if digest != variant["model_sha256"]:
                raise ValueError("model hash does not match manifest: " + variant_id)
            profiles = []
            for relative in variant.get("profiles", []):
                profile = self.read_json(relative)
                if profile.get("schema_version") != 1 or profile.get("variant") != variant_id:
                    raise ValueError("profile schema or variant does not match: " + relative)
                profiles.append(profile)
            if not profiles:
                raise ValueError("variant has no runtime profiles: " + variant_id)
            self.profiles[variant_id] = profiles

    def path(self, relative):
        path = (self.directory / relative).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError("model package paths must stay inside the package")
        return path

    def read_json(self, relative):
        with self.path(relative).open(encoding="utf-8") as handle:
            return json.load(handle)
