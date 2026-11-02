## ADDED Requirements

### Requirement: PyTorch owns memory and streams

Bridged kernels SHALL read and write PyTorch-owned storage in place, SHALL be enqueued on the
caller's current stream of the tensors' device, and SHALL never cause CubeCL to free, pool, move or
reuse memory PyTorch allocated. Device memory the bridge needs for itself (launch metadata,
workspaces) SHALL be allocated through PyTorch's caching allocator on the launch's stream.

#### Scenario: In-place on PyTorch storage

- **WHEN** a bridged kernel runs on tensors PyTorch allocated
- **THEN** the results appear in those tensors' storage, and no CubeCL-owned copy of them exists

#### Scenario: Bridge allocations are PyTorch's

- **WHEN** a bridged kernel needs device memory of its own for a shape not seen before
- **THEN** that memory is reported by PyTorch's allocator statistics for the device, and none is held outside them

### Requirement: Launch ordering with PyTorch

A bridged launch SHALL be enqueued on its stream before the call that issues it returns, so that
work PyTorch issues on that stream before and after it is ordered around it without host
synchronization. Work on other streams SHALL follow PyTorch's cross-stream rules: the bridge adds
no ordering PyTorch would not, and drops none.

#### Scenario: Same-stream producer and consumer

- **WHEN** PyTorch writes a tensor, a bridged kernel reads it and writes another, and PyTorch reads that output, all on one current stream (default or not) with no synchronization between
- **THEN** the consumer observes the kernel's result for every iteration

#### Scenario: Cross-stream use

- **WHEN** a tensor produced on one stream is consumed by a bridged kernel on another after the caller makes the consuming stream wait on the producing one
- **THEN** the kernel reads the produced values

### Requirement: Storage lifetime

The bridge SHALL keep PyTorch storage used by a launch from being reused until the launch's stream
has passed it, by the same mechanism PyTorch uses for work on non-default streams, and SHALL hold no
reference to that storage after the issuing call returns.

#### Scenario: Freed while queued

- **WHEN** the caller drops its last reference to an input right after a bridged launch on a side stream, and new allocations follow under memory pressure
- **THEN** the kernel's output is correct and the input's memory is not handed out before the kernel has run

### Requirement: Graph capture

Bridged launches SHALL be capturable into a PyTorch CUDA graph and SHALL replay against the captured
storage, including launches whose shape the bridge first sees during the capture.

#### Scenario: Capture and replay

- **WHEN** bridged kernels are captured with PyTorch's graph capture, the inputs are overwritten in place, and the graph is replayed
- **THEN** the outputs match the kernels run eagerly on the new inputs

#### Scenario: Shape first seen while capturing

- **WHEN** a bridged kernel is first launched for a shape inside a capture
- **THEN** the capture succeeds, and any memory the bridge allocated for it belongs to the graph's memory pool

### Requirement: Launch host cost

The warm host time of a bridged launch from Python SHALL not exceed the host time of a raw Triton
kernel launch for the same kernel arguments beyond measurement noise, on the same machine and
Python.

#### Scenario: Warm launch microbenchmark

- **WHEN** a prepared bridged kernel and a Triton kernel with the same arguments are launched repeatedly from Python
- **THEN** the bridged median host time per launch does not exceed Triton's by more than the measured noise

### Requirement: Free-threaded use

Bridged launches SHALL be correct when issued concurrently from several Python threads in a
free-threaded interpreter, each on its own current stream.

#### Scenario: Concurrent threads

- **WHEN** several threads issue bridged launches concurrently, each on a different stream, without a global interpreter lock
- **THEN** every output is correct and no launch is lost or misordered within its stream

### Requirement: Numerics and provenance

Bridged kernels SHALL be compiled with the same numerics policy as descriptor-compiled kernels (FMA
contraction and division precision), only from the audited CubeCL and CubeK source state, and every
selection record naming a bridged kernel SHALL record that source state and the policy it was
compiled under.

#### Scenario: Policy reaches the compiler

- **WHEN** a bridged kernel performs a division under the default numerics policy
- **THEN** its compiled code uses the same division precision a descriptor-compiled kernel uses under that policy

#### Scenario: Provenance recorded

- **WHEN** a selection chooses or rejects a bridged candidate
- **THEN** its record names the CubeCL and CubeK pins and the patch state the kernel was built from

### Requirement: Bridged candidates in selection

Bridged kernels SHALL enter selection only as candidates beside existing plans, through the same
legality, numerical-band and first-use timing rules, and SHALL be offered only for kernels whose
shape and data they can run exactly as the descriptor specifies. A refused bridged candidate SHALL
appear among the rejected candidates with its reason. Existing plans and their generated code SHALL
be unchanged.

#### Scenario: Plain row sum

- **WHEN** `x.sum(-1)` on a contiguous floating-point tensor is compiled under the `first_use` policy
- **THEN** the candidate set includes the bridged reduce, or records the rule that rejected it

#### Scenario: Fused body

- **WHEN** a reduction with a fused per-element body (for example RMSNorm) is compiled
- **THEN** no bridged candidate is offered, the rejection reason names the fused body, and the selected plan is one the kernel could select before this change

#### Scenario: Numerical band

- **WHEN** a bridged candidate's summation is outside Triton's band for its class
- **THEN** it is rejected for that class with the band as its reason

### Requirement: Platform independence of the design

The bridge's contracts SHALL be stated in terms of a device's memory, streams and allocator, not
CUDA's, and every CubeCL change the bridge relies on SHALL have a defined counterpart for CubeCL's
HIP, Metal and wgpu servers, even where only CUDA is built and tested.

#### Scenario: Design review per platform

- **WHEN** the bridge's CubeCL changes are reviewed
- **THEN** each change records how the HIP, Metal and wgpu servers provide the same behavior, or why it does not apply to them
