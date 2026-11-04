use alloc::string::{String, ToString};
use alloc::vec::Vec;
use pliron::derive::format;

use crate::AddressType;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, serde::Serialize, serde::Deserialize)]
#[allow(missing_docs)]
#[format("`(` $x `, ` $y `, ` $z `)`")]
/// The number of units across all 3 axis totalling to the number of working units in a cube.
pub struct Dim3 {
    /// The number of units in the x axis.
    pub x: u32,
    /// The number of units in the y axis.
    pub y: u32,
    /// The number of units in the z axis.
    pub z: u32,
}

impl Dim3 {
    /// Create a new `Dim3` with x = y = z = 1.
    pub const fn new_single() -> Self {
        Self { x: 1, y: 1, z: 1 }
    }

    /// Create a new `Dim3` with the given x, and y = z = 1.
    pub const fn new_1d(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    /// Create a new dim3 with the given x and y, and z = 1.
    pub const fn new_2d(x: u32, y: u32) -> Self {
        Self { x, y, z: 1 }
    }

    /// Create a new `Dim3` with the given x, y and z.
    /// This is equivalent to the [new](Dim3::new) function.
    pub const fn new_3d(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Total numbers of units per cube
    pub const fn num_elems(&self) -> u32 {
        self.x * self.y * self.z
    }

    /// Whether this `Dim3` can fully contain `other`
    pub const fn can_contain(&self, other: Dim3) -> bool {
        self.x >= other.x && self.y >= other.y && self.z >= other.z
    }
}

impl From<(u32, u32, u32)> for Dim3 {
    fn from(value: (u32, u32, u32)) -> Self {
        Dim3::new_3d(value.0, value.1, value.2)
    }
}

impl From<Dim3> for (u32, u32, u32) {
    fn from(val: Dim3) -> Self {
        (val.x, val.y, val.z)
    }
}

/// The kind of execution to be performed.
#[derive(
    Default, Hash, PartialEq, Eq, Clone, Debug, Copy, serde::Serialize, serde::Deserialize,
)]
pub enum ExecutionMode {
    /// Checked kernels are safe.
    #[default]
    Checked,
    /// Validate OOB and alert if OOB access occurs
    Validate,
    /// Unchecked kernels are unsafe.
    Unchecked,
}

/// How much source-level debug data a kernel carries. Profilers and debuggers read it to map
/// kernel code back to the `#[cube]` source.
#[derive(
    Default,
    Hash,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Clone,
    Copy,
    Debug,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum DebugInfo {
    /// No debug data.
    #[default]
    #[serde(rename = "none")]
    None,
    /// Source lines, with each inlined `#[cube]` function as its own frame.
    #[serde(rename = "line-tables")]
    LineTables,
    /// Line tables and the embedded source text.
    #[serde(rename = "full")]
    Full,
}

/// Floating-point code generation a compiler applies to a whole kernel, beyond
/// what its IR says: whether `a * b + c` may contract to a fused multiply-add,
/// and whether a division is correctly rounded. `None` leaves the backend's
/// default (NVRTC's: contraction on, precise division).
///
/// A caller that must round like another compiler, under a policy that can
/// change between kernels, sets it around the launches it issues with
/// `with_fp_compilation` (in `cubecl-runtime`); each launch takes the setting
/// in force on its thread into its kernel's settings and id, so kernels compiled under different settings never share a
/// cache entry. Only macro-generated launches take the thread's setting; a
/// kernel that builds its own settings or id sets the field itself.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct FpCompilation {
    /// Whether multiplies and adds may contract to fused multiply-adds.
    pub contraction: Option<bool>,
    /// Whether divisions are correctly rounded (rather than approximate).
    pub precise_division: Option<bool>,
}

impl FpCompilation {
    /// Whether this is the backend default, which leaves kernel ids as they were.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// The sizes a kernel is compiled for, as Inductor writes static sizes as
/// literals: its launch records each index (`usize`) scalar and each tensor's
/// shape and strides as it registers them (the launch still carries them), and
/// the kernel reads them as constants, so a division or modulo by a size folds
/// in the target compiler and a loop over a fixed extent has a fixed trip
/// count. One kernel per set of sizes. Set around launches with
/// `with_static_sizes` (in `cubecl-runtime`); like `fp`, only macro-generated
/// launches take it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct StaticSizes {
    /// The index scalars, in registration order (the order of their ids).
    pub scalars: Vec<u64>,
    /// Each tensor's shape and strides, by extended metadata position.
    pub layouts: Vec<(Vec<usize>, Vec<usize>)>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelSettings {
    /// The cube dim of the kernel
    pub cube_dim: Dim3,
    /// The address type of the kernel
    pub address_type: AddressType,
    /// The name of the kernel
    pub kernel_name: String,
    /// The source-level debug data to include
    pub debug_info: DebugInfo,
    /// CUDA Cluster dim, if any
    pub cluster_dim: Option<Dim3>,
    /// Execution mode
    pub execution_mode: ExecutionMode,
    /// Floating-point code generation for the whole kernel
    pub fp: FpCompilation,
    /// The sizes the kernel is compiled for, if it is compiled for its sizes
    pub static_sizes: Option<StaticSizes>,
}

impl KernelSettings {
    pub fn new(cube_dim: Dim3, execution_mode: ExecutionMode, address_type: AddressType) -> Self {
        Self {
            cube_dim,
            address_type,
            kernel_name: String::new(),
            debug_info: DebugInfo::None,
            cluster_dim: None,
            execution_mode,
            fp: FpCompilation::default(),
            static_sizes: None,
        }
    }
}

impl KernelSettings {
    /// Set cube dimension.
    pub fn cube_dim(mut self, cube_dim: Dim3) -> Self {
        self.cube_dim = cube_dim;
        self
    }

    /// Set address type.
    pub fn address_type(mut self, ty: AddressType) -> Self {
        self.address_type = ty;
        self
    }

    /// Set kernel name.
    pub fn kernel_name<S: AsRef<str>>(mut self, name: S) -> Self {
        self.kernel_name = name.as_ref().to_string();
        self
    }

    /// Include full debug data, whatever the cargo profile asks for.
    #[must_use]
    pub fn debug_symbols(self) -> Self {
        self.debug_info(DebugInfo::Full)
    }

    /// Set the source-level debug data to include.
    #[must_use]
    pub fn debug_info(mut self, debug_info: DebugInfo) -> Self {
        self.debug_info = debug_info;
        self
    }

    /// Set cluster dim
    pub fn cluster_dim(mut self, cluster_dim: Dim3) -> Self {
        self.cluster_dim = Some(cluster_dim);
        self
    }
}
