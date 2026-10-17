"""Run the established 18 gates in a fresh packet, preserving historical files."""
import importlib.util
from pathlib import Path
source = Path('stage5/artifacts/reduction-structure/final/regressions-full/harness.py').resolve()
spec = importlib.util.spec_from_file_location('promotion_gates', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.OUT = Path(__file__).resolve().parent
module.main(cache=module.OUT / 'logs')
