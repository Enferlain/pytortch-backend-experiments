use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use core::{
    any::{Any, TypeId},
    fmt::Display,
    hash::{Hash, Hasher},
};
use cubecl_common::{
    format::{DebugRaw, format_str},
    hash::{StableHash, StableHasher},
};
use cubecl_ir::{
    AddressType,
    settings::{DebugInfo, Dim3, ExecutionMode},
};
use derive_more::{Eq, PartialEq};

#[macro_export(local_inner_macros)]
/// Create a new storage ID type.
macro_rules! storage_id_type {
    ($name:ident) => {
        /// Storage ID.
        #[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, PartialOrd, Ord)]
        pub struct $name {
            value: usize,
        }

        impl $name {
            /// Create a new ID.
            pub fn new() -> Self {
                use core::sync::atomic::{AtomicUsize, Ordering};

                static COUNTER: AtomicUsize = AtomicUsize::new(0);

                let value = COUNTER.fetch_add(1, Ordering::Relaxed);
                if value == usize::MAX {
                    core::panic!("Memory ID overflowed");
                }
                Self { value }
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

/// Identifies a backend-owned captured graph.
///
/// [`end_capture`](crate::server::Server::end_capture) records a graph,
/// stores it in the backend's own registry, and returns this lightweight id;
/// [`replay`](crate::server::Server::replay) and
/// [`graph_destroy`](crate::server::Server::graph_destroy) take the id
/// back to look the graph up. Referencing the graph by id keeps the raw
/// executable inside the server — it never crosses the actor boundary in a box —
/// exactly as memory is referenced by [`Handle`](crate::server::Handle) rather
/// than by raw pointer.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, PartialOrd, Ord)]
pub struct GraphId {
    value: u64,
}

impl GraphId {
    /// Allocate a fresh, process-unique graph id.
    pub fn new() -> Self {
        use core::sync::atomic::{AtomicU64, Ordering};

        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let value = COUNTER.fetch_add(1, Ordering::Relaxed);
        if value == u64::MAX {
            core::panic!("Graph ID overflowed");
        }
        Self { value }
    }
}

impl Default for GraphId {
    fn default() -> Self {
        Self::new()
    }
}

pub use cubecl_ir::settings::FpCompilation;

#[cfg(feature = "std")]
std::thread_local! {
    static FP_COMPILATION: core::cell::Cell<FpCompilation> =
        const { core::cell::Cell::new(FpCompilation { contraction: None, precise_division: None }) };
}

/// Runs `f` with `fp` as the setting for every kernel launched on this thread
/// inside it (see [`FpCompilation`]), restoring the previous setting afterward,
/// including on unwind.
#[cfg(feature = "std")]
pub fn with_fp_compilation<R>(fp: FpCompilation, f: impl FnOnce() -> R) -> R {
    {
        struct Restore(FpCompilation);
        impl Drop for Restore {
            fn drop(&mut self) {
                FP_COMPILATION.with(|cell| cell.set(self.0));
            }
        }
        let _restore = Restore(FP_COMPILATION.with(|cell| cell.replace(fp)));
        f()
    }
}

/// The floating-point setting in force on this thread, which a launch puts in
/// its kernel's settings.
pub fn current_fp_compilation() -> FpCompilation {
    #[cfg(feature = "std")]
    return FP_COMPILATION.with(|cell| cell.get());
    #[cfg(not(feature = "std"))]
    FpCompilation::default()
}

pub use cubecl_ir::settings::StaticSizes;

#[cfg(feature = "std")]
std::thread_local! {
    static STATIC_SIZES: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

/// Runs `f` with every kernel launched on this thread inside it compiled for
/// its sizes (see [`StaticSizes`]), restoring the previous setting afterward,
/// including on unwind.
#[cfg(feature = "std")]
pub fn with_static_sizes<R>(f: impl FnOnce() -> R) -> R {
    {
        struct Restore(bool);
        impl Drop for Restore {
            fn drop(&mut self) {
                STATIC_SIZES.with(|cell| cell.set(self.0));
            }
        }
        let _restore = Restore(STATIC_SIZES.with(|cell| cell.replace(true)));
        f()
    }
}

/// Whether kernels launched on this thread are compiled for their sizes.
pub fn static_sizes_requested() -> bool {
    #[cfg(feature = "std")]
    return STATIC_SIZES.with(|cell| cell.get());
    #[cfg(not(feature = "std"))]
    false
}

/// Kernel unique identifier.
#[derive(Clone, PartialEq, Eq)]
pub struct KernelId {
    #[eq(skip)]
    type_name: &'static str,
    pub(crate) type_id: core::any::TypeId,
    /// The [`CubeDim`] for this kernel
    pub cube_dim: Dim3,
    /// The address type for this kernel
    pub address_type: AddressType,
    /// The execution mode for this kernel
    pub mode: ExecutionMode,
    /// The debug data of this kernel, after the configuration lowered it.
    pub debug_info: DebugInfo,
    /// The floating-point code generation in force when the id was built.
    pub fp: FpCompilation,
    /// The sizes the kernel is compiled for, if it is compiled for its sizes.
    pub static_sizes: Option<StaticSizes>,
    pub(crate) info: Option<Info>,
}

impl Hash for KernelId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.type_id.hash(state);
        self.address_type.hash(state);
        self.cube_dim.hash(state);
        self.mode.hash(state);
        self.info.hash(state);
        // A kernel without debug data keeps the hash it had before the level existed.
        if self.debug_info != DebugInfo::None {
            self.debug_info.hash(state);
        }
        // Likewise for the default floating-point code generation.
        if !self.fp.is_default() {
            self.fp.hash(state);
        }
        // Likewise for a kernel that reads its sizes from its launch.
        if let Some(sizes) = &self.static_sizes {
            sizes.hash(state);
        }
    }
}

impl core::fmt::Debug for KernelId {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        let mut debug_str = f.debug_struct("KernelId");
        debug_str
            .field("type", &DebugRaw(self.type_name))
            .field("address_type", &self.address_type);
        debug_str.field("cube_dim", &self.cube_dim);
        debug_str.field("mode", &self.mode);
        debug_str.field("debug_info", &self.debug_info);
        if !self.fp.is_default() {
            debug_str.field("fp", &self.fp);
        }
        if let Some(sizes) = &self.static_sizes {
            debug_str.field("static_sizes", sizes);
        }
        match &self.info {
            Some(info) => debug_str.field("info", info),
            None => debug_str.field("info", &self.info),
        };
        debug_str.finish()
    }
}

impl Display for KernelId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match &self.info {
            Some(info) => f.write_str(
                format_str(
                    format!("{info:?}").as_str(),
                    &[('(', ')'), ('[', ']'), ('{', '}')],
                    true,
                )
                .as_str(),
            ),
            None => f.write_str("No info"),
        }
    }
}

impl KernelId {
    /// The kernel's type name, trimmed to its last path segment — the short
    /// name a report can print without drowning the reader in type paths.
    pub fn short_name(&self) -> &'static str {
        let name = self.type_name.split('<').next().unwrap_or(self.type_name);
        name.rsplit("::").next().unwrap_or(name)
    }

    /// The kernel's type, in full: what names it before its comptime
    /// arguments tell instances apart.
    pub fn type_name(&self) -> &'static str {
        self.type_name
    }

    /// Create a new [kernel id](KernelId) for a type.
    pub fn new<T: 'static>() -> Self {
        Self {
            type_id: core::any::TypeId::of::<T>(),
            type_name: core::any::type_name::<T>(),
            info: None,
            cube_dim: Dim3::new_single(),
            mode: ExecutionMode::Checked,
            address_type: Default::default(),
            debug_info: DebugInfo::None,
            fp: FpCompilation::default(),
            static_sizes: None,
        }
    }

    /// Render the key in a standard format that can be used between runs.
    ///
    /// Can be used as a persistent kernel cache key.
    pub fn stable_format(&self) -> String {
        let format = format!(
            "{}-{}-{:?}-{:?}-{:?}",
            self.type_name, self.address_type, self.cube_dim, self.mode, self.info
        );
        let format = match self.debug_info {
            DebugInfo::None => format,
            level => format!("{format}-{level:?}"),
        };
        let format = match self.fp.is_default() {
            true => format,
            false => format!("{format}-{:?}", self.fp),
        };
        match &self.static_sizes {
            None => format,
            Some(sizes) => format!("{format}-{sizes:?}"),
        }
    }

    /// Hash the key in a stable way that can be used between runs.
    ///
    /// Can be used as a persistent kernel cache key.
    pub fn stable_hash(&self) -> StableHash {
        let mut hasher = StableHasher::new();
        self.type_name.hash(&mut hasher);
        self.address_type.hash(&mut hasher);
        self.cube_dim.hash(&mut hasher);
        self.mode.hash(&mut hasher);
        self.info.hash(&mut hasher);
        // A kernel without debug data keeps its entry point name and cache key.
        if self.debug_info != DebugInfo::None {
            self.debug_info.hash(&mut hasher);
        }
        if !self.fp.is_default() {
            self.fp.hash(&mut hasher);
        }
        if let Some(sizes) = &self.static_sizes {
            sizes.hash(&mut hasher);
        }

        hasher.finalize()
    }

    /// Return the entrypoint name disambiguated with a stable hash discriminator.
    pub fn entrypoint_name(&self, base: &str) -> String {
        format!("{base}_{:08x}", self.stable_hash() as u32)
    }

    /// Add information to the [kernel id](KernelId).
    ///
    /// The information is used to differentiate kernels of the same kind but with different
    /// configurations, which affect the generated code.
    pub fn info<I: 'static + PartialEq + Eq + Hash + core::fmt::Debug + Send + Sync>(
        mut self,
        info: I,
    ) -> Self {
        self.info = Some(Info::new(info));
        self
    }

    /// Set the [execution mode](ExecutionMode).
    pub fn mode(mut self, mode: ExecutionMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the debug data that the kernel asks for. The id keeps the level after the
    /// configuration lowers it, because that level changes the compiled kernel.
    #[must_use]
    pub fn debug_info(mut self, requested: DebugInfo) -> Self {
        self.debug_info = crate::config::compilation::effective_debug_info(requested);
        self
    }

    /// Set the sizes the kernel is compiled for (see [`StaticSizes`]).
    #[must_use]
    pub fn static_sizes(mut self, sizes: Option<StaticSizes>) -> Self {
        self.static_sizes = sizes;
        self
    }

    /// Set the floating-point code generation the kernel is compiled with.
    #[must_use]
    pub fn fp(mut self, fp: FpCompilation) -> Self {
        self.fp = fp;
        self
    }

    /// Set the [cube dim](CubeDim).
    pub fn cube_dim(mut self, cube_dim: Dim3) -> Self {
        self.cube_dim = cube_dim;
        self
    }

    /// Set the [`AddressType`].
    pub fn address_type(mut self, addr_ty: AddressType) -> Self {
        self.address_type = addr_ty;
        self
    }
}

impl core::fmt::Debug for Info {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.value.fmt(f)
    }
}

impl Info {
    fn new<T: 'static + PartialEq + Eq + Hash + core::fmt::Debug + Send + Sync>(id: T) -> Self {
        Self {
            value: Arc::new(id),
        }
    }
}

/// This trait allows various types to be used as keys within a single data structure.
///
/// The downside is that the hashing method is hardcoded and cannot be configured using the
/// [`core::hash::Hash`] function. The provided [Hasher] will be modified, but only based on the
/// result of the hash from the [`DefaultHasher`].
trait DynKey: core::fmt::Debug + Send + Sync {
    fn dyn_type_id(&self) -> TypeId;
    fn dyn_eq(&self, other: &dyn DynKey) -> bool;
    fn dyn_hash(&self, state: &mut dyn Hasher);
    fn dyn_hash_one(&self) -> StableHash;
    fn as_any(&self) -> &dyn Any;
}

impl PartialEq for Info {
    fn eq(&self, other: &Self) -> bool {
        self.value.dyn_eq(other.value.as_ref())
    }
}

/// Extra information
#[derive(Clone)]
pub(crate) struct Info {
    value: Arc<dyn DynKey>,
}
impl Eq for Info {}

impl Hash for Info {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.dyn_type_id().hash(state);
        self.value.dyn_hash(state)
    }
}

impl<T: 'static + PartialEq + Eq + Hash + core::fmt::Debug + Send + Sync> DynKey for T {
    fn dyn_eq(&self, other: &dyn DynKey) -> bool {
        if let Some(other) = other.as_any().downcast_ref::<T>() {
            self == other
        } else {
            false
        }
    }

    fn dyn_type_id(&self) -> TypeId {
        TypeId::of::<T>()
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        let hash = self.dyn_hash_one();
        state.write_u128(hash);
    }

    fn dyn_hash_one(&self) -> StableHash {
        let mut hasher = StableHasher::new();
        self.hash(&mut hasher);
        hasher.finalize()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use std::collections::HashSet;

    #[test_log::test]
    pub fn kernel_id_hash() {
        let value_1 = KernelId::new::<()>().info("1");
        let value_2 = KernelId::new::<()>().info("2");

        let mut set = HashSet::new();

        set.insert(value_1.clone());

        assert!(set.contains(&value_1));
        assert!(!set.contains(&value_2));
    }

    #[test]
    fn fp_compilation_keys_the_kernel() {
        let default = KernelId::new::<()>().info("1");
        let fast = FpCompilation {
            contraction: Some(true),
            precise_division: Some(false),
        };
        let precise = FpCompilation {
            contraction: Some(false),
            precise_division: Some(true),
        };
        let (a, b) = (
            KernelId::new::<()>().info("1").fp(fast),
            KernelId::new::<()>().info("1").fp(precise),
        );
        assert_eq!((a.fp, b.fp), (fast, precise));
        assert_eq!(with_fp_compilation(fast, current_fp_compilation), fast);
        // Restored after the scope.
        assert!(current_fp_compilation().is_default());
        for (x, y) in [(&default, &a), (&default, &b), (&a, &b)] {
            assert_ne!(x, y);
            assert_ne!(x.stable_hash(), y.stable_hash());
            assert_ne!(x.stable_format(), y.stable_format());
        }
        // The default keeps the hash an id had before the setting existed.
        let mut hasher = StableHasher::new();
        let id = &default;
        id.type_name.hash(&mut hasher);
        id.address_type.hash(&mut hasher);
        id.cube_dim.hash(&mut hasher);
        id.mode.hash(&mut hasher);
        id.info.hash(&mut hasher);
        assert_eq!(default.stable_hash(), hasher.finalize());
    }

    #[test]
    fn static_sizes_key_the_kernel() {
        let sizes = |scalars: &[u64]| StaticSizes {
            scalars: scalars.to_vec(),
            layouts: vec![(vec![300, 4096], vec![4096, 1])],
        };
        let dynamic = KernelId::new::<()>().info("1");
        let a = KernelId::new::<()>().info("1").static_sizes(Some(sizes(&[1])));
        let b = KernelId::new::<()>().info("1").static_sizes(Some(sizes(&[0])));
        for (x, y) in [(&dynamic, &a), (&dynamic, &b), (&a, &b)] {
            assert_ne!(x, y);
            assert_ne!(x.stable_hash(), y.stable_hash());
            assert_ne!(x.stable_format(), y.stable_format());
        }
        assert!(with_static_sizes(static_sizes_requested));
        // Restored after the scope, and only ever set on its own thread.
        assert!(!static_sizes_requested());
        with_static_sizes(|| {
            assert!(!std::thread::spawn(static_sizes_requested).join().unwrap());
        });
    }

    #[test]
    fn debug_info_changes_the_hash_and_the_cache_key() {
        let with = |debug_info| KernelId {
            debug_info,
            ..KernelId::new::<()>().info("1")
        };
        let (none, lines, full) = (
            with(DebugInfo::None),
            with(DebugInfo::LineTables),
            with(DebugInfo::Full),
        );
        assert_ne!(none.stable_hash(), lines.stable_hash());
        assert_ne!(lines.stable_hash(), full.stable_hash());
        assert_ne!(none.stable_format(), lines.stable_format());
        assert_ne!(lines.stable_format(), full.stable_format());
        assert_ne!(none, lines);
        let state = std::collections::hash_map::RandomState::new();
        assert_ne!(
            core::hash::BuildHasher::hash_one(&state, &none),
            core::hash::BuildHasher::hash_one(&state, &lines)
        );
        // Without debug data, the hash and the entry point name are the same as before.
        let plain = KernelId::new::<()>().info("1");
        assert_eq!(none.stable_hash(), plain.stable_hash());
        assert_eq!(none.stable_format(), plain.stable_format());
    }
}
