//! Unknown operands expose partial folds that all-constant tests would miss.
use cubecl_core::ir::{
    ConstantValue, ElemType, FloatKind,
    attributes::FloatAttr,
    interfaces::ConstantAttr,
    pliron::{
        attribute::attr_cast,
        builtin::ops::{ConstantOp, FuncOp},
        irfmt::parsers::spaced,
        linked_list::ContainsLinkedList,
        operation::verify_operation,
        parsable::parse_from_str,
        printable::Printable,
        result::ExpectOk,
    },
    prelude::*,
};
use cubecl_cpp::shared::CppConstantAttr;
use cubecl_opt::passes::{
    inst_combine::{InstCombine, InstCombinePass},
    sccp::SCCPPass,
};

fn run(text: &str, passes: &[&str]) -> (Context, Ptr<Operation>, String) {
    let mut ctx = Context::new();
    let op = parse_from_str(spaced(Operation::top_level_parser()), &mut ctx, text).expect_ok(&ctx);
    verify_operation(op, &ctx).expect_ok(&ctx);
    let mut analyses = AnalysisManager::default();
    for pass in passes {
        match *pass {
            "sccp" => SCCPPass.run(op, &mut ctx, &mut analyses),
            "simplify" => SimplifyOpsPass::default().run(op, &mut ctx, &mut analyses),
            "combine" => InstCombinePass::default().run(op, &mut ctx, &mut analyses),
            "contract" => InstCombinePass::new(InstCombine {
                allow_fp_contraction: true,
            })
            .run(op, &mut ctx, &mut analyses),
            _ => panic!("unknown test pass"),
        }
        .expect_ok(&ctx);
        verify_operation(op, &ctx).expect_ok(&ctx);
    }
    let printed = op.dyn_op(&ctx).disp(&ctx).to_string();
    (ctx, op, printed)
}

fn returned(ctx: &Context, func: Ptr<Operation>) -> Value {
    let func = func.as_op::<FuncOp>(ctx).unwrap();
    let region = func.get_operation().deref(ctx).get_region(0);
    let block = region.deref(ctx).iter(ctx).last().unwrap();
    let terminator = block.deref(ctx).iter(ctx).last().unwrap();
    terminator.deref(ctx).operands().next().unwrap()
}

fn function(ty: &str, arithmetic: &str, lhs: &str, rhs: &str, constant: &str) -> String {
    format!(
        "builtin.func @f: builtin.function <({ty}, {ty}) -> ({ty})> [] {{
        ^entry(x: {ty}, y: {ty}):
        c = builtin.constant <{constant}> : {ty};
        r = math.{arithmetic} ({lhs}, {rhs}) [] []: <({ty}, {ty}) -> ({ty})>;
        branch.return r
        }}"
    )
}

#[test]
fn unknown_float_operands_survive_both_pass_orders() {
    for ty in ["cube.f32", "cube.f16", "cube.bf16", "cube.f64"] {
        for (op, lhs, rhs, value) in [
            ("f_add", "c", "x", "0"),
            ("f_add", "x", "c", "0"),
            ("f_add", "c", "x", "-0"),
            ("f_add", "x", "c", "-0"),
            ("f_sub", "x", "x", "0"),
            ("f_sub", "x", "c", "-0"),
            ("f_mul", "c", "x", "0"),
            ("f_mul", "x", "c", "0"),
            ("f_mul", "c", "x", "-0"),
            ("f_mul", "x", "c", "-0"),
            ("f_div", "c", "y", "0"),
            ("f_div", "c", "y", "-0"),
            ("f_rem", "c", "y", "0"),
            ("f_rem", "c", "y", "-0"),
            ("f_rem", "x", "c", "1"),
            ("f_mod_floor", "c", "y", "0"),
            ("f_mod_floor", "c", "y", "-0"),
        ] {
            for passes in [
                vec!["simplify"],
                vec!["sccp"],
                vec!["sccp", "simplify"],
                vec!["simplify", "sccp"],
            ] {
                let text = function(ty, op, lhs, rhs, &format!("cube.float {ty}: {value}"));
                let (ctx, func, printed) = run(&text, &passes);
                let ret = returned(&ctx, func);
                assert_eq!(ret.get_type(&ctx).disp(&ctx).to_string().trim(), ty);
                let defining = ret.defining_op().unwrap();
                assert!(
                    defining
                        .dyn_op(&ctx)
                        .disp(&ctx)
                        .to_string()
                        .contains(&format!("math.{op}")),
                    "{ty} {op} {passes:?}: result replaced: {printed}"
                );
            }
        }
    }
}

#[test]
fn retained_safe_identities_still_return_the_input() {
    for (op, lhs, rhs, c) in [
        ("f_mul", "c", "x", "1"),
        ("f_mul", "x", "c", "1"),
        ("f_div", "x", "c", "1"),
        ("f_sub", "x", "c", "0"),
    ] {
        let text = function(
            "cube.f32",
            op,
            lhs,
            rhs,
            &format!("cube.float cube.f32: {c}"),
        );
        let (ctx, func, printed) = run(&text, &["simplify"]);
        assert!(returned(&ctx, func).defining_op().is_none(), "{printed}");
    }
}

#[test]
fn integer_remainder_by_one_is_never_the_input() {
    for (ty, op, kind) in [
        ("builtin.integer si64", "s_rem", "si64"),
        ("builtin.integer ui64", "u_rem", "ui64"),
    ] {
        let text = function(ty, op, "x", "c", &format!("builtin.integer <1: {kind}>"));
        let (ctx, func, _) = run(&text, &["simplify"]);
        assert!(returned(&ctx, func).defining_op().is_some());
        for passes in [vec!["sccp", "simplify"], vec!["simplify", "sccp"]] {
            let (ctx, func, printed) = run(&text, &passes);
            let value = returned(&ctx, func).defining_op().unwrap();
            assert!(
                value
                    .dyn_op(&ctx)
                    .disp(&ctx)
                    .to_string()
                    .contains(&format!("<0: {kind}>")),
                "{printed}"
            );
        }
    }
}

#[test]
fn contraction_requires_explicit_pass_permission() {
    let text =
        "builtin.func @f: builtin.function <(cube.f32, cube.f32, cube.f32) -> (cube.f32)> [] {
        ^entry(x: cube.f32, y: cube.f32, z: cube.f32):
        p = math.f_mul (x, y) [] []: <(cube.f32, cube.f32) -> (cube.f32)>;
        r = math.f_add (p, z) [] []: <(cube.f32, cube.f32) -> (cube.f32)>;
        branch.return r
    }";
    let (_, _, printed) = run(text, &["combine"]);
    assert!(printed.contains("math.f_mul") && printed.contains("math.f_add"));
    assert!(!printed.contains("math.fma"));
    let (_, _, printed) = run(text, &["contract"]);
    assert!(printed.contains("math.fma"));
}

#[test]
fn ordinary_constant_remainders_still_fold_to_their_value() {
    for (opcode, value, expected) in [
        ("f_rem", "1.5", 0.5),
        ("f_rem", "-1.5", -0.5),
        ("f_rem", "2", 0.0),
        ("f_mod_floor", "-1.5", 0.5),
    ] {
        let text = format!(
            "builtin.func @f: builtin.function <() -> (cube.f32)> [] {{
            ^entry():
            a = builtin.constant <cube.float cube.f32: {value}> : cube.f32;
            b = builtin.constant <cube.float cube.f32: 1> : cube.f32;
            r = math.{opcode} (a, b) [] []: <(cube.f32, cube.f32) -> (cube.f32)>;
            branch.return r
        }}"
        );
        let (ctx, func, _) = run(&text, &["sccp", "simplify"]);
        let op = returned(&ctx, func)
            .defining_op()
            .unwrap()
            .as_op::<ConstantOp>(&ctx)
            .unwrap();
        let value = op.get_attr_builtin_constant_value(&ctx).unwrap();
        let constant = attr_cast::<dyn ConstantAttr>(&**value).unwrap();
        assert_eq!(constant.float_as_f64(&ctx), Some(expected));
    }
}

#[test]
fn runtime_remainder_survives_a_control_flow_merge() {
    let text =
        "builtin.func @f: builtin.function <(cube.f32, cube.f32, cube.bool) -> (cube.f32)> [] {
        ^entry(x: cube.f32, y: cube.f32, cond: cube.bool):
        one = builtin.constant <cube.float cube.f32: 1> : cube.f32;
        cf.branch_conditional if cond ^left(x, one) else ^right(y, one)
        ^left(a: cube.f32, b: cube.f32):
        cf.branch ^merge(a, b)
        ^right(a2: cube.f32, b2: cube.f32):
        cf.branch ^merge(a2, b2)
        ^merge(v: cube.f32, divisor: cube.f32):
        r = math.f_rem (v, divisor) [] []: <(cube.f32, cube.f32) -> (cube.f32)>;
        branch.return r
    }";
    for passes in [vec!["sccp", "simplify"], vec!["simplify", "sccp"]] {
        let (ctx, func, printed) = run(text, &passes);
        let op = returned(&ctx, func).defining_op().unwrap();
        assert!(
            op.dyn_op(&ctx)
                .disp(&ctx)
                .to_string()
                .contains("math.f_rem"),
            "{printed}"
        );
    }
}

#[test]
fn floating_cpp_literals_preserve_zero_sign_and_large_integral_values() {
    let ctx = Context::new();
    for kind in [
        FloatKind::F16,
        FloatKind::BF16,
        FloatKind::F32,
        FloatKind::F64,
    ] {
        for value in [-0.0, 0.0, 1.0] {
            let attr = ConstantValue::Float(value).as_attribute(&ctx, ElemType::Float(kind));
            let literal = attr.downcast_ref::<FloatAttr>().unwrap().to_cpp(&ctx);
            assert!(literal.contains('.') || literal.contains('e') || literal.contains('E'));
            if value == 0.0 && value.is_sign_negative() {
                assert_eq!(literal, "-0.0");
            }
        }
    }
    let attr = ConstantValue::Float(1e20).as_attribute(&ctx, ElemType::Float(FloatKind::F32));
    let literal = attr.downcast_ref::<FloatAttr>().unwrap().to_cpp(&ctx);
    assert!(literal.contains('.') || literal.contains('e') || literal.contains('E'));
}
