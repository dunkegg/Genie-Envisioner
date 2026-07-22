"""Load source modules recovered from CPython bytecode after a lost rebase."""

from importlib.machinery import SourcelessFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
import sys


def load_recovered(namespace, source_file, recovered_stem):
    runtime_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    recovered_path = Path(source_file).with_name(f"{recovered_stem}.{runtime_tag}.pyc")
    if not recovered_path.is_file():
        raise RuntimeError(
            f"{recovered_path.name} does not support Python "
            f"{sys.version_info.major}.{sys.version_info.minor}; use the Python 3.10 training environment."
        )

    module_name = f"models.{recovered_stem}"
    loader = SourcelessFileLoader(module_name, str(recovered_path))
    spec = spec_from_loader(module_name, loader)
    if spec is None:
        raise ImportError(f"Cannot create a module spec for {recovered_path}")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    loader.exec_module(module)

    exported = getattr(module, "__all__", None)
    if exported is None:
        exported = [name for name in vars(module) if not name.startswith("_")]
    namespace.update({name: getattr(module, name) for name in exported})
    namespace["__all__"] = list(exported)

