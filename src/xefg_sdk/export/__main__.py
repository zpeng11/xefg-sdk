"""Run the supported exporter with ``python -m xefg_sdk.export``."""
import sys

from .cli import main


if __name__ == "__main__":
    sys.exit(main())
