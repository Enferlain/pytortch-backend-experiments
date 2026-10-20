"""CubeCL kernels for PyTorch: runtime, descriptor compiler and front ends.

Layers depend downwards only: ``inductor``/``ops`` -> ``compiler`` -> ``runtime``.
Importing this package imports no layer; import the one you need.
"""
