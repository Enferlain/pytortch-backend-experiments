"""Errors shared by the descriptor compiler and its front ends."""


class UnsupportedCubeCL(NotImplementedError):
    """The construct cannot be expressed by the CubeCL backend; the message says why."""


class MalformedDescriptor(ValueError):
    """A descriptor does not conform to its schema; nothing was compiled."""
