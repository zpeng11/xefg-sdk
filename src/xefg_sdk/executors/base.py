"""Small execution protocol; no device or graphics API is required."""
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


class Executor(Protocol):
    def load(self, model_path: Path) -> Any:
        """Return an opaque handle owned by this executor."""
        ...

    def run(self, handle: Any, inputs: Mapping[str, np.ndarray],
            output_names: Sequence[str]) -> Mapping[str, np.ndarray]:
        ...

    def close(self, handle: Any) -> None:
        ...
