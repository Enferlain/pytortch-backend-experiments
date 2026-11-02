"""The PTX equivalence check used to review CubeCL upgrades."""

from cubecl_torch.testing.ptx_equivalence import difference

# The shape seen at fed6c0c0: a sum and its compensation both start at zero, in the
# opposite order, and every later use is renamed.
OLD = """\
// header comment
.visible .entry cubecl5_aaaa(
\t.reg .f32 \t%f<9>;
\tmov.f32 \t%f6, 0f00000000;
\tmov.u64 \t%rd3, %rd1;
\tmov.f32 \t%f8, %f6;
$L__BB0_4:
\tadd.rn.f32 \t%f5, %f8, %f2;
\tadd.rn.f32 \t%f6, %f6, %f5;
\tmov.f32 \t%f8, %f5;
"""
NEW = """\
.visible .entry cubecl5_bbbb(
\t.reg .f32 \t%f<12>;
\tmov.f32 \t%f7, 0f00000000;
\tmov.u64 \t%rd3, %rd1;
\tmov.f32 \t%f6, %f7;
$L__BB0_4:
\tadd.rn.f32 \t%f5, %f7, %f2;
\tadd.rn.f32 \t%f6, %f6, %f5;
\tmov.f32 \t%f7, %f5;
"""


def test_renamed_registers_and_reordered_zeros_are_equivalent():
    assert difference(OLD, NEW) is None


def test_a_changed_instruction_is_found():
    changed = NEW.replace("add.rn.f32 \t%f6, %f6", "sub.rn.f32 \t%f6, %f6")
    assert "differs" in difference(OLD, changed)


def test_an_inconsistent_renaming_is_found():
    # The sum now reads the compensation register: not a renaming.
    swapped = NEW.replace("add.rn.f32 \t%f5, %f7, %f2", "add.rn.f32 \t%f5, %f6, %f2")
    assert difference(OLD, swapped) is not None


def test_a_register_that_is_not_zero_is_not_a_zero():
    nonzero = NEW.replace("mov.f32 \t%f7, 0f00000000", "mov.f32 \t%f7, 0f3F800000")
    assert difference(OLD, nonzero) is not None


def test_a_predicated_write_ends_a_zero():
    # %f6 may be overwritten under %p1 before it is copied: the copy is not a zero.
    old = OLD.replace(
        "\tmov.u64 \t%rd3, %rd1;\n", "\t@%p1 ld.global.f32 \t%f6, [%rd3];\n"
    )
    new = NEW.replace(
        "\tmov.u64 \t%rd3, %rd1;\n", "\t@%p1 ld.global.f32 \t%f7, [%rd3];\n"
    )
    assert difference(old, new) is not None


def test_a_read_between_zeros_keeps_their_order():
    # The first add reads a zero in one file and a register zeroed only later in the other.
    old = """\
\tmov.f32 \t%f6, 0f00000000;
\tadd.rn.f32 \t%f1, %f6, %f2;
\tmov.f32 \t%f7, 0f00000000;
\tadd.rn.f32 \t%f3, %f7, %f1;
"""
    new = """\
\tmov.f32 \t%f7, 0f00000000;
\tadd.rn.f32 \t%f1, %f6, %f2;
\tmov.f32 \t%f6, 0f00000000;
\tadd.rn.f32 \t%f3, %f7, %f1;
"""
    assert difference(old, new) is not None


def test_registers_keep_their_class():
    assert difference("\tmov.b32 \t%r1, %r2;\n", "\tmov.b32 \t%f1, %r2;\n") is not None
