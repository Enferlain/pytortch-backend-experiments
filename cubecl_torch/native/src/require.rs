//! Checked indirect access precondition, represented as a side-effecting IR op.
//! Device traps are target lowering, never pointer accesses or neutral-value fallback.
use cubecl_core::ir::{HasSideEffects, NoMemoryEffect, cube_op, pliron, prelude::*};
#[cfg(feature = "hip-probe")]
use cubecl_cpp::target::Hip;
use cubecl_cpp::{
    shared::{CppValue, operation::OpToCPP},
    target::Cuda,
};
#[cube_op(name = "cubecl_bridge.require")]
#[result_ty(none)]
#[op_traits(HasSideEffects, NoMemoryEffect)]
pub struct RequireOp {
    pub condition: Value,
}
#[pliron::derive::op_interface_impl]
impl OpToCPP<Cuda> for RequireOp {
    fn to_cpp(&self, ctx: &Context) -> String {
        format!("if (!{}) {{ __trap(); }}\n", self.condition(ctx).name(ctx))
    }
}
#[cfg(feature = "hip-probe")]
#[pliron::derive::op_interface_impl]
impl OpToCPP<Hip> for RequireOp {
    fn to_cpp(&self, ctx: &Context) -> String {
        format!(
            "if (!{}) {{ __builtin_trap(); }}\n",
            self.condition(ctx).name(ctx)
        )
    }
}
