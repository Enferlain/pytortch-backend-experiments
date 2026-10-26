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
            ("f_mul_rn", "c", "x", "0"),
            ("f_mul_rn", "x", "c", "-0"),
            ("f_div_rn", "c", "y", "0"),
            ("f_div_rn", "c", "y", "-0"),
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
        ("f_mul_rn", "c", "x", "1"),
        ("f_mul_rn", "x", "c", "1"),
        ("f_div_rn", "x", "c", "1"),
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
fn rounded_products_are_never_contracted() {
    for (multiply, contracted) in [("f_mul", true), ("f_mul_rn", false)] {
        let text = format!(
            "builtin.func @f: builtin.function <(cube.f32, cube.f32, cube.f32) -> (cube.f32)> [] {{
            ^entry(x: cube.f32, y: cube.f32, z: cube.f32):
            p = math.{multiply} (x, y) [] []: <(cube.f32, cube.f32) -> (cube.f32)>;
            r = math.f_add (p, z) [] []: <(cube.f32, cube.f32) -> (cube.f32)>;
            branch.return r
        }}"
        );
        let (_, _, printed) = run(&text, &["contract"]);
        assert_eq!(printed.contains("math.fma"), contracted, "{printed}");
    }
}

fn folded(ty: &str, opcode: &str, operands: &[&str]) -> f64 {
    let names = ["a", "b", "c"];
    let constants: String = operands
        .iter()
        .zip(names)
        .map(|(value, name)| {
            format!("{name} = builtin.constant <cube.float {ty}: {value}> : {ty};\n")
        })
        .collect();
    let args = names[..operands.len()].join(", ");
    let types = vec![ty; operands.len()].join(", ");
    let text = format!(
        "builtin.func @f: builtin.function <() -> ({ty})> [] {{
        ^entry():
        {constants}r = math.{opcode} ({args}) [] []: <({types}) -> ({ty})>;
        branch.return r
    }}"
    );
    let (ctx, func, printed) = run(&text, &["sccp", "simplify"]);
    let op = returned(&ctx, func)
        .defining_op()
        .and_then(|op| op.as_op::<ConstantOp>(&ctx))
        .unwrap_or_else(|| panic!("{opcode} not folded: {printed}"));
    let value = op.get_attr_builtin_constant_value(&ctx).unwrap();
    let constant = attr_cast::<dyn ConstantAttr>(&**value).unwrap();
    constant.float_as_f64(&ctx).unwrap()
}

#[test]
fn constant_fma_rounds_once() {
    // (1 + 2^-k)^2 - (1 + 2^-(k-1)) is 2^-2k exactly; rounding the product first loses it.
    for (ty, factor, addend, expected) in [
        (
            "cube.f32",
            "1.000244140625",
            "-1.00048828125",
            2f64.powi(-24),
        ),
        (
            "cube.f64",
            "1.0000000074505805969238281250",
            "-1.0000000149011611938476562500",
            2f64.powi(-54),
        ),
        ("cube.f16", "1.015625", "-1.03125", 2f64.powi(-12)),
    ] {
        let value = folded(ty, "fma", &[factor, factor, addend]);
        assert_eq!(value, expected, "{ty}");
    }
}

#[test]
fn constant_half_fma_that_would_round_twice_is_not_folded() {
    // (1 + 2^-4)^2 is a bf16 tie that the tiny addend breaks upwards; an f64 sum loses it.
    let text = "builtin.func @f: builtin.function <() -> (cube.bf16)> [] {
        ^entry():
        a = builtin.constant <cube.float cube.bf16: 1.0625> : cube.bf16;
        c = builtin.constant <cube.float cube.bf16: 8.673617379884035e-19> : cube.bf16;
        r = math.fma (a, a, c) [] []: <(cube.bf16, cube.bf16, cube.bf16) -> (cube.bf16)>;
        branch.return r
    }";
    let (ctx, func, printed) = run(text, &["sccp", "simplify"]);
    let op = returned(&ctx, func).defining_op().unwrap();
    assert!(
        op.dyn_op(&ctx).disp(&ctx).to_string().contains("math.fma"),
        "{printed}"
    );
}

#[test]
fn constant_rounded_arithmetic_folds_to_ieee_results() {
    let third = folded("cube.f32", "f_div_rn", &["1", "3"]);
    assert_eq!(third, f64::from(1f32 / 3f32));
    let product = folded("cube.f32", "f_mul_rn", &["3", "0.1"]);
    assert_eq!(product, f64::from(3f32 * 0.1f32));
    assert_eq!(folded("cube.f32", "f_div_rn", &["1", "0"]), f64::INFINITY);
    assert_eq!(
        folded("cube.f32", "f_div_rn", &["1", "-0"]),
        f64::NEG_INFINITY
    );
    assert!(folded("cube.f32", "f_div_rn", &["0", "0"]).is_nan());
    let negative_zero = folded("cube.f32", "f_mul_rn", &["-1", "0"]);
    assert!(negative_zero == 0.0 && negative_zero.is_sign_negative());
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
