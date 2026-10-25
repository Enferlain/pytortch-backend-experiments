"""Suite-wide settings, inherited by the subprocesses the tests start."""

import os

# A first-use candidate failing its comparison raises instead of being excluded
# (tuning.STRICT_ENV): the corpus and the suite have no expected exclusions.
os.environ.setdefault("CUBECL_TORCH_STRICT_CANDIDATES", "1")
