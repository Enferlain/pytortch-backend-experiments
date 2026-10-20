# Spec Delta

## Purpose

Lets CubeCL kernels, hand-authored or later built from CubeK components, be registered as PyTorch
custom ops. They're usable from eager code and from compiled graphs, next to or instead of
compiler-generated kernels, without going through Inductor.

## ADDED Requirements

### Requirement: Custom op registration

The package SHALL provide a way to register a kernel artifact as a `torch.library` custom op with a
declared schema and a fake (meta) implementation, so the op can be traced.

#### Scenario: Registered op in eager mode

- **WHEN** a custom op backed by a kernel artifact is registered and called on CUDA tensors in eager mode
- **THEN** it runs the CubeCL kernel on the current stream and returns results that match its reference implementation

### Requirement: Opaque extern call inside compiled graphs

A registered custom op SHALL be traced by `torch.compile` as a single opaque extern call. Its output
SHALL be identical to calling the op eagerly, including when it sits between ordinary operations the
backend compiles.

#### Scenario: Mixed graph

- **WHEN** a function computing ordinary supported operations, then `cubecl_torch::sin_plus_one`, then ordinary supported operations is compiled and run
- **THEN** the compiled graph contains exactly one extern call for the custom op with no graph break around it, the surrounding operations are compiled as generated kernels, and the output equals the eager result

### Requirement: No overlay or Inductor dependency

Registering and calling a custom op in eager mode SHALL NOT import Inductor or Dynamo, and SHALL NOT
install the PyTorch overlay.

#### Scenario: Eager-only process

- **WHEN** a process registers and calls a custom op without using `torch.compile`
- **THEN** no `torch._inductor` or `torch._dynamo` module is loaded, and the overlay is not installed

### Requirement: Free-threaded custom ops

Calling custom ops from multiple Python threads on free-threaded CPython SHALL keep the GIL disabled
and produce correct results.

#### Scenario: Concurrent eager calls

- **WHEN** several threads call the same custom op on independent tensors and streams
- **THEN** every result is correct and the GIL remains disabled
