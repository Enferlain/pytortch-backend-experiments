"""Fixed semantic benchmark families, authored before physical optimization."""

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class Case:
    family: str
    shape: tuple[int, ...]
    dtype: str = "float32"
    dynamic: bool = True

    @property
    def name(self):
        return f"{self.family}-{self.dtype}-{'x'.join(map(str, self.shape))}"

    def to_dict(self):
        return {"name": self.name, **asdict(self)}


def cases():
    result = []
    for dtype in ("float32", "float16", "bfloat16"):
        for family, shapes in (
            ("add", [(1,), (255,), (256,), (257,), (4099,), (1048583,), (16777219,)]),
            ("sin", [(257,), (1048583,)]),
            ("fused", [(257,), (1048583,)]),
            ("bias_row", [(3, 255), (64, 256), (64, 257), (256, 4099)]),
            ("bias_column", [(3, 255), (64, 256), (64, 257), (256, 4099)]),
            ("transpose", [(33, 257), (256, 513)]),
            ("slice", [(257,), (1048583,)]),
            ("stride_zero", [(64, 257)]),
            ("masked_cat", [(255,), (257,), (4099,)]),
            (
                "row_sum",
                [(64, n) for n in (31, 32, 33, 255, 256, 257, 1024)]
                + [(16, 4099), (1, 65539), (8, 65539)],
            ),
            ("column_sum", [(33, 257), (256, 513), (4099, 64)]),
            ("transpose_sum", [(3, 257), (3, 4099)]),
            ("mean_epilogue", [(64, 257), (64, 4099)]),
            ("rms_inference", [(64, 255), (64, 1024), (16, 4099)]),
            ("rms_backward", [(64, 255), (64, 1024), (16, 4099)]),
            ("gather", [(257,), (65539,)]),
            ("dropout_backward", [(257,), (65539,)]),
        ):
            result.extend(Case(family, tuple(shape), dtype) for shape in shapes)
    result.extend(
        Case("scatter_" + kind, (n,))
        for kind in ("distinct", "collision")
        for n in (257, 65539)
    )
    result.extend([Case("embedding_backward", (257, 7)), Case("training", (5, 9))])
    return result


def add(x):
    return x + 1


def sin(x):
    return torch.sin(x) + 1


def fused(x, y):
    return (x * y + x) / (y + 2)


def bias(x, y):
    return x + y


def masked_cat(x):
    return torch.cat([x, x[:1]]) + 1


def row_sum(x):
    return x.sum(-1)


def column_sum(x):
    return x.sum(0)


def mean_epilogue(x):
    return x.mean(-1) + 1


def rms(x):
    y = x.float()
    return (y * torch.rsqrt((y * y).mean(-1, keepdim=True) + 1e-5)).to(x.dtype)


def gather(x, indices):
    return torch.gather(x, 0, indices)


def scatter(x, indices, values):
    return torch.scatter_add(x, 0, indices, values)


def embedding(x, indices):
    return torch.nn.functional.embedding(indices, x)


def dropout(x):
    return torch.nn.functional.dropout(x, 0.25, True)


class Model(torch.nn.Module):
    """Exact authored Stage 4 model; SGD stays outside captured forward/backward."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(17, 7))
        self.linear = torch.nn.Linear(7, 9, bias=False)

    def forward(self, idx):
        x = self.linear(torch.nn.functional.embedding(idx, self.weight))
        x = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5)
        return torch.nn.functional.dropout(x, 0.25, True)


@dataclass
class Fixture:
    fn: object
    args: tuple
    upstream: object = None
    model: object = None
    optimizer: object = None
    stochastic: bool = False


def fixture(case):
    dtype = getattr(torch, case.dtype)
    x = torch.randn(case.shape, device="cuda", dtype=dtype)
    family = case.family
    if family == "fused":
        return Fixture(fused, (x, torch.rand_like(x)))
    if family.startswith("bias_"):
        b, n = case.shape
        y = torch.randn(
            (n,) if family == "bias_row" else (b, 1), device="cuda", dtype=dtype
        )
        return Fixture(bias, (x, y))
    if family == "transpose":
        return Fixture(sin, (x.T,))
    if family == "transpose_sum":
        return Fixture(
            row_sum,
            (torch.randn(case.shape[1], case.shape[0], device="cuda", dtype=dtype).T,),
        )
    if family == "slice":
        return Fixture(
            sin, (torch.randn(2 * case.shape[0] + 3, device="cuda", dtype=dtype)[3::2],)
        )
    if family == "stride_zero":
        return Fixture(
            sin,
            (
                torch.randn(case.shape[0], 1, device="cuda", dtype=dtype).expand(
                    case.shape
                ),
            ),
        )
    if family == "gather":
        idx = torch.arange(case.shape[0], device="cuda", dtype=torch.int64).flip(0)
        return Fixture(gather, (x, idx))
    if family.startswith("scatter_"):
        idx = torch.arange(case.shape[0], device="cuda", dtype=torch.int64)
        # Exactly representable updates isolate reset/ordering correctness from
        # legitimate atomic addition order; arbitrary gradients remain separate.
        x = ((idx % 17).to(torch.float32) - 8) / 8
        if family == "scatter_collision":
            idx = idx % 17
        return Fixture(scatter, (torch.zeros_like(x), idx, x))
    if family == "embedding_backward":
        x = torch.randn(17, case.shape[1], device="cuda", requires_grad=True)
        idx = torch.arange(case.shape[0], device="cuda", dtype=torch.int64) % 17
        return Fixture(embedding, (x, idx), torch.randn(*case.shape, device="cuda"))
    if family in ("rms_backward", "dropout_backward"):
        if family == "dropout_backward":
            x = torch.rand(case.shape, device="cuda", dtype=dtype) + 0.5
        x.requires_grad_()
        return Fixture(
            rms if family == "rms_backward" else dropout,
            (x,),
            torch.randn_like(x),
            stochastic=family == "dropout_backward",
        )
    if family == "training":
        model = Model().cuda()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9, foreach=False, fused=False
        )
        for parameter in model.parameters():
            optimizer.state[parameter]["momentum_buffer"] = torch.zeros_like(parameter)

        def loss(idx):
            return (model(idx).flatten() ** 2).mean()

        return Fixture(
            loss,
            (torch.tensor([0, 2, 2, 16, 1], device="cuda"),),
            model=model,
            optimizer=optimizer,
            stochastic=True,
        )
    return Fixture(
        {
            "add": add,
            "sin": sin,
            "masked_cat": masked_cat,
            "row_sum": row_sum,
            "column_sum": column_sum,
            "mean_epilogue": mean_epilogue,
            "rms_inference": rms,
        }[family],
        (x,),
    )
