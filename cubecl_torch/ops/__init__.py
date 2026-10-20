"""torch.library custom ops backed by kernel artifacts, independent of Inductor.

Importing a submodule registers its op, e.g. ``import cubecl_torch.ops.sin_plus_one``
makes ``torch.ops.cubecl_torch.sin_plus_one`` available.
"""
