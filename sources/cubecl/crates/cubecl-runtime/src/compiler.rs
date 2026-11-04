use crate::kernel::KernelDefinition;
use crate::poison::DevicePoison;
use alloc::string::{String, ToString};
use cubecl_environment::backtrace::BackTrace;
use thiserror::Error;

/// Told of every kernel a server compiles, as it compiles it: for a caller
/// that records what was built, such as a digest of the generated code.
///
/// Called on whichever thread finalizes the kernel, once it compiled: it must
/// not call into the server (which may be the caller of the compilation), only
/// record. A kernel the server takes from a persistent compilation store is not
/// compiled again, and not reported.
pub trait CompilationObserver: Send + Sync {
    /// `kernel` was compiled from `source` (the exact text the backend's
    /// compiler read, or `None` for a backend that compiles no text) into
    /// `binary` (PTX on CUDA).
    fn compiled(&self, kernel: &crate::id::KernelId, source: Option<&str>, binary: &[u8]);
}

/// A server's [`CompilationObserver`], shared with the compilers that report to it.
#[derive(Clone)]
pub struct ObserverSlot(
    alloc::sync::Arc<
        cubecl_environment::sync::RwLock<Option<alloc::sync::Arc<dyn CompilationObserver>>>,
    >,
);

impl Default for ObserverSlot {
    fn default() -> Self {
        Self(alloc::sync::Arc::new(
            cubecl_environment::sync::RwLock::new(None),
        ))
    }
}

impl core::fmt::Debug for ObserverSlot {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let set = self.0.read().is_some();
        f.debug_tuple("ObserverSlot")
            .field(&if set { "set" } else { "empty" })
            .finish()
    }
}

impl ObserverSlot {
    /// Reports to `observer` from now on, or to none.
    pub fn set(&self, observer: Option<alloc::sync::Arc<dyn CompilationObserver>>) {
        *self.0.write() = observer;
    }

    /// Tells the observer, if one is set, that `kernel` was compiled.
    pub fn report(&self, kernel: &crate::id::KernelId, source: Option<&str>, binary: &[u8]) {
        // Not under the lock: the observer may set another observer.
        let observer = self.0.read().clone();
        if let Some(observer) = observer {
            observer.compiled(kernel, source, binary);
        }
    }
}

/// JIT compilation error.
#[derive(Error, Clone)]
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
pub enum CompilationError {
    /// An instruction isn't supported.
    #[error(
        "An unsupported instruction caused the compilation to fail\nCaused by:\n  {reason}\nBacktrace:\n{backtrace}"
    )]
    UnsupportedInstruction {
        /// The caused of the error.
        reason: String,
        /// The backtrace for this error.
        #[cfg_attr(serializable, serde(skip))]
        backtrace: BackTrace,
    },

    /// A generic compilation error.
    #[error(
        "An error caused the compilation to fail\nCaused by:\n  {reason}\nBacktrace:\n{backtrace}"
    )]
    Generic {
        /// The error context.
        reason: String,
        /// The backtrace for this error.
        #[cfg_attr(serializable, serde(skip))]
        backtrace: BackTrace,
    },
    /// The device was poisoned before the compiled kernel could be loaded onto
    /// it. See [`ServerError::DevicePoisoned`](crate::server::ServerError::DevicePoisoned).
    #[error("The device is poisoned, the kernel could not be loaded\nCaused by:\n  {0}")]
    DevicePoisoned(#[from] DevicePoison),

    /// A generic compilation error.
    #[error(
        "A validation error caused the compilation to fail\nCaused by:\n  {reason}\nBacktrace:\n{backtrace}"
    )]
    Validation {
        /// The error context.
        reason: String,
        /// The backtrace for this error.
        #[cfg_attr(serializable, serde(skip))]
        backtrace: BackTrace,
    },
    /// The kernel's own code panicked while it expanded, an assertion in it included: a defect
    /// in the kernel, where every other variant is a compiler turning it down.
    #[error(
        "Expanding the kernel `{kernel}` panicked\nCaused by:\n  {message}\nBacktrace:\n{backtrace}"
    )]
    ExpansionPanicked {
        /// The kernel that panicked.
        kernel: String,
        /// The panic's message.
        message: String,
        /// The backtrace for this error.
        #[cfg_attr(serializable, serde(skip))]
        backtrace: BackTrace,
    },
}

impl CompilationError {
    /// Whether the device that emitted the error is poisoned.
    pub fn is_device_poisoned(&self) -> bool {
        matches!(self, Self::DevicePoisoned(_))
    }

    /// Whether this is the kernel being turned down, a candidate to drop or a case to skip,
    /// rather than a defect in it to report or something going wrong while building it.
    ///
    /// Every variant is, except two:
    /// - a kernel that panicked while it expanded: a kernel declines a configuration by
    ///   returning an error, so a panic is a bug that skipping would hide;
    /// - a device that died before the module could load.
    pub fn is_refusal(&self) -> bool {
        !matches!(self, Self::ExpansionPanicked { .. }) && !self.is_device_poisoned()
    }
}

impl core::fmt::Debug for CompilationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self}")
    }
}

impl From<pliron::result::Error> for CompilationError {
    fn from(value: pliron::result::Error) -> Self {
        CompilationError::Validation {
            reason: value.to_string(),
            backtrace: BackTrace::capture(),
        }
    }
}

/// Compiles the representation into its own representation that can be formatted into tokens.
pub trait Compiler: Sync + Send + 'static + Clone + core::fmt::Debug {
    /// The representation for the compiled code.
    type Representation: core::fmt::Display;
    /// The compilation options used to configure the compiler
    type CompilationOptions: Send + Default + core::fmt::Debug;

    /// Compiles the [kernel definition](KernelDefinition) into the compiler's representation.
    fn compile(
        &mut self,
        kernel: KernelDefinition,
        compilation_options: &Self::CompilationOptions,
    ) -> Result<Self::Representation, CompilationError>;

    /// What the compiled kernel does with each buffer binding, by buffer
    /// position — the visibility analysis's answer, when the representation
    /// kept it (see [`BufferIOAttr`](crate::kernel::BufferIOAttr)).
    ///
    /// `None` reads as every buffer both read and written, the conservative
    /// direction. A compiler overriding this must answer from the IR
    /// attributes the annotate pass stamped, never from what its shader
    /// language kept — wgpu's shader visibility, for one, is deliberately
    /// forced wider than the kernel's own behavior.
    fn buffer_io(
        _repr: &Self::Representation,
    ) -> Option<alloc::vec::Vec<crate::kernel::BufferIOAttr>> {
        None
    }

    /// The default extension for the runtime's kernel/shader code.
    /// Might change based on which compiler is used.
    fn extension(&self) -> &'static str;

    /// Short identifier of the language this compiler produces, such as
    /// `"wgsl"` or `"cuda"`.
    ///
    /// What a [`PrecompiledSource`](crate::kernel::PrecompiledSource) has to
    /// name to be accepted by this compiler.
    fn lang_tag(&self) -> &'static str;
}
