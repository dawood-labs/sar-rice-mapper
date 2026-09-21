"""Allows `python -m sar_pipeline ...`."""
import sys

from .cli import main

sys.exit(main())
