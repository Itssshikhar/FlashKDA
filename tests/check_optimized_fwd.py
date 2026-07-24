import math
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import flash_kda


D = 128
LOWER_BOUND = -5.0


def make_inputs(total_t, h):
    torch.manual_seed(42)
    q = F.normalize(
        torch.randn((1, total_t, h, D), dtype=torch.float32, device="cuda"),
        p=2,
        dim=-1,
    ).to(torch.bfloat16)
    k = F.normalize(
        torch.randn((1, total_t, h, D), dtype=torch.float32, device="cuda"),
        p=2,
        dim=-1,
    ).to(torch.bfloat16)
    v = torch.randn((1, total_t, h, D), dtype=torch.bfloat16, device="cuda")
    g = torch.randn((1, total_t, h, D), dtype=torch.bfloat16, device="cuda")
    beta = torch.randn((1, total_t, h), dtype=torch.bfloat16, device="cuda")
    a_log = torch.rand(h, dtype=torch.float32, device="cuda")
    dt_bias = torch.rand(h, D, dtype=torch.float32, device="cuda")
    return q, k, v, g, beta, a_log, dt_bias


def run_case(name, seq_lens, state_dtype, has_in, has_out, h=2, kernel_only=False):
    total_t = sum(seq_lens)
    n = len(seq_lens)
    q, k, v, g, beta, a_log, dt_bias = make_inputs(total_t, h)
    scale = 1.0 / math.sqrt(D)

    cu_seqlens = None
    if n > 1:
        cu_seqlens = torch.tensor(
            [0] + list(torch.tensor(seq_lens).cumsum(0).tolist()),
            dtype=torch.long,
            device="cuda",
        )

    dtype = torch.bfloat16 if state_dtype == "bf16" else torch.float32
    state_shape = (n, h, D, D)
    state_values = torch.arange(
        math.prod(state_shape), dtype=torch.float32, device="cuda"
    ).reshape(state_shape).to(torch.bfloat16).to(dtype)

    initial_kernel = state_values.clone() if has_in else None
    initial_ref = state_values.clone() if has_in else None
    final_kernel = torch.zeros(state_shape, dtype=dtype, device="cuda") if has_out else None
    final_ref = torch.zeros(state_shape, dtype=dtype, device="cuda") if has_out else None
    out_kernel = torch.zeros_like(q)
    out_ref = torch.zeros_like(q)

    ref_kwargs = {"cu_seqlens": cu_seqlens} if cu_seqlens is not None else {}
    kernel_kwargs = dict(ref_kwargs)
    if n > 1 and all(seq_len == seq_lens[0] for seq_len in seq_lens):
        kernel_kwargs["uniform_seq_len"] = seq_lens[0]
    flash_kda.fwd(
        q,
        k,
        v,
        g,
        beta,
        scale,
        out_kernel,
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=LOWER_BOUND,
        initial_state=initial_kernel,
        final_state=final_kernel,
        **kernel_kwargs,
    )
    torch.cuda.synchronize()

    if kernel_only:
        print(f"PASS {name} (kernel launch)", flush=True)
        return

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tests"))
    from torch_ref import torch_ref

    torch_ref(
        q,
        k,
        v,
        g,
        beta,
        scale,
        out_ref,
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=LOWER_BOUND,
        initial_state=initial_ref,
        final_state=final_ref,
        **ref_kwargs,
    )

    if not torch.equal(out_kernel, out_ref):
        mismatch = (out_kernel != out_ref).nonzero()[0].tolist()
        raise AssertionError(
            f"{name}: output mismatch at {mismatch}: "
            f"kernel={out_kernel[tuple(mismatch)]} ref={out_ref[tuple(mismatch)]}"
        )
    if final_kernel is not None and not torch.equal(final_kernel, final_ref):
        mismatch = (final_kernel != final_ref).nonzero()[0].tolist()
        # Diagnostics: distinguish a transposed export from stale/garbage data.
        transposed = torch.equal(final_kernel, final_ref.transpose(-1, -2))
        near_t = torch.allclose(
            final_kernel.float(),
            final_ref.transpose(-1, -2).float(),
            rtol=1e-2,
            atol=1e-3,
        )
        initial_like = initial_ref is not None and torch.allclose(
            final_kernel.float(), initial_ref.float(), rtol=1e-2, atol=1e-3
        )
        frac_zero = float((final_kernel == 0).float().mean())
        raise AssertionError(
            f"{name}: final-state mismatch at {mismatch}: "
            f"kernel={final_kernel[tuple(mismatch)]} ref={final_ref[tuple(mismatch)]} "
            f"[transpose_exact={transposed} transpose_close={near_t} "
            f"initial_like={initial_like} frac_zero={frac_zero:.3f}]"
        )
    print(f"PASS {name}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-only", action="store_true")
    args = parser.parse_args()

    cases = [
        ("bf16-full", [16], "bf16", True, True),
        ("bf16-tail", [17], "bf16", True, True),
        ("fp32-tail", [37], "fp32", True, True),
        ("zero-state", [32], "bf16", False, False),
        ("output-state-only", [17], "bf16", False, True),
        ("varlen", [17, 33, 65], "bf16", True, True),
        ("uniform-packed", [33, 33, 33], "bf16", True, True),
        ("uniform-full-tile", [32, 32, 32], "bf16", True, True),
        ("uniform-stage-reuse", [48, 48, 48], "bf16", True, True),
    ]
    for case in cases:
        run_case(*case, kernel_only=args.kernel_only)
        if args.kernel_only:
            break


if __name__ == "__main__":
    main()
