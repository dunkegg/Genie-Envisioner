"""Recovered Za-adaptor training entrypoint.

The source file was removed by a rebase, but CPython's last successfully
compiled trainer was still present in ``runner/__pycache__``.  A stable copy
of that artifact is kept beside this file and loaded here so the original
training implementation is preserved exactly.

Supported runtime versions are Python 3.10 (the training environment) and
Python 3.13 (the local development environment).  Once the original source is
recovered from Git, this compatibility entrypoint and the two recovered
artifacts can be replaced by that source file.
"""

from importlib.machinery import SourcelessFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
import sys


_RUNTIME_TAG = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
_RECOVERED_PATH = Path(__file__).with_name(
    f"_za_adaptor_trainer_recovered.{_RUNTIME_TAG}.pyc"
)

if not _RECOVERED_PATH.is_file():
    raise RuntimeError(
        "The recovered ZaAdaptorTrainer bytecode does not support "
        f"Python {sys.version_info.major}.{sys.version_info.minor}. "
        "Use the project's Python 3.10 training environment."
    )

_MODULE_NAME = "runner._za_adaptor_trainer_recovered"
_LOADER = SourcelessFileLoader(_MODULE_NAME, str(_RECOVERED_PATH))
_SPEC = spec_from_loader(_MODULE_NAME, _LOADER)
if _SPEC is None:
    raise ImportError(f"Cannot create a module spec for {_RECOVERED_PATH}")

_RECOVERED_MODULE = module_from_spec(_SPEC)
sys.modules[_MODULE_NAME] = _RECOVERED_MODULE
_LOADER.exec_module(_RECOVERED_MODULE)

ZaAdaptorTrainer = _RECOVERED_MODULE.ZaAdaptorTrainer

__all__ = ["ZaAdaptorTrainer"]
