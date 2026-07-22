"""Helpers for modules recovered from CPython bytecode after a lost rebase."""

from importlib.machinery import SourcelessFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
import runpy
import sys


def recovered_path(source_file, recovered_stem):
    tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    path = Path(source_file).with_name(f"{recovered_stem}.{tag}.pyc")
    if not path.is_file():
        raise RuntimeError(
            f"{path.name} is unavailable for Python {sys.version_info.major}.{sys.version_info.minor}. "
            "Use a runtime for which a recovered cache is present."
        )
    return path


def load_recovered(namespace, source_file, recovered_stem):
    path = recovered_path(source_file, recovered_stem)
    module_name = f"_recovered.{recovered_stem}"
    loader = SourcelessFileLoader(module_name, str(path))
    spec = spec_from_loader(module_name, loader)
    if spec is None:
        raise ImportError(f"Cannot create a module spec for {path}")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    loader.exec_module(module)
    exported = getattr(module, "__all__", None)
    if exported is None:
        exported = [name for name in vars(module) if not name.startswith("_")]
    namespace.update({name: getattr(module, name) for name in exported})
    namespace["__all__"] = list(exported)


def run_recovered(source_file, recovered_stem):
    return runpy.run_path(str(recovered_path(source_file, recovered_stem)), run_name="__main__")

