import argparse
import math

import torch
import torch.nn.functional as F

import flash_kda


FIXED_CASES = {
    "fixed": [8192],
    "varlen": [1300, 547, 2048, 963, 271, 3063],
    "varlen_even": [1024] * 8,
}


def _make_inputs(seq_lens, h, d, state_mode):
    device = torch.device("cuda")
    torch.manual_seed(0)

    t_total = sum(seq_lens)
    n = len(seq_lens)
    scale = 1.0 / math.sqrt(d)

    q = F.normalize(
        torch.randn((1, t_total, h, d), dtype=torch.float32, device=device),
        p=2,
        dim=-1,
    ).to(torch.bfloat16)
    k = F.normalize(
        torch.randn((1, t_total, h, d), dtype=torch.float32, device=device),
        p=2,
        dim=-1,
    ).to(torch.bfloat16)
    v = torch.randn((1, t_total, h, d), dtype=torch.bfloat16, device=device)
    g = torch.randn((1, t_total, h, d), dtype=torch.bfloat16, device=device)
    beta = torch.randn((1, t_total, h), dtype=torch.bfloat16, device=device)
    out = torch.empty_like(q)

    a_log = torch.rand(h, dtype=torch.float32, device=device)
    dt_bias = torch.rand(h, d, dtype=torch.float32, device=device)

    kwargs = {}
    if len(seq_lens) > 1:
        kwargs["cu_seqlens"] = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()),
            dtype=torch.long,
            device=device,
        )

    if state_mode != "none":
        initial_state = torch.arange(
            n * h * d * d,
            dtype=torch.float32,
            device=device,
        ).reshape(n, h, d, d)
        if state_mode == "bf16":
            initial_state = initial_state.to(torch.bfloat16)
        kwargs["initial_state"] = initial_state
        kwargs["final_state"] = torch.empty_like(initial_state)

    return q, k, v, g, beta, scale, out, a_log, dt_bias, kwargs


def run_case(mode, h, d, state_mode, warmup, iters):
    seq_lens = FIXED_CASES[mode]
    q, k, v, g, beta, scale, out, a_log, dt_bias, kwargs = _make_inputs(
        seq_lens,
        h,
        d,
        state_mode,
    )

    print(
        f"case={mode} seq_lens={seq_lens} H={h} D={d} "
        f"state_mode={state_mode} warmup={warmup} iters={iters}",
        flush=True,
    )

    def dispatch():
        flash_kda.fwd(
            q,
            k,
            v,
            g,
            beta,
            scale,
            out,
            A_log=a_log,
            dt_bias=dt_bias,
            lower_bound=-5.0,
            **kwargs,
        )

    with torch.inference_mode():
        for _ in range(warmup):
            dispatch()
        torch.cuda.synchronize()

        for _ in range(iters):
            dispatch()
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(FIXED_CASES), required=True)
    parser.add_argument("--state-mode", choices=["bf16", "fp32", "none"], default="bf16")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--H", type=int, default=96)
    parser.add_argument("--D", type=int, default=128)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    print(f"torch={torch.__version__} torch_cuda={torch.version.cuda}", flush=True)
    print(f"device={torch.cuda.get_device_name(0)}", flush=True)
    run_case(args.mode, args.H, args.D, args.state_mode, args.warmup, args.iters)


if __name__ == "__main__":
    main()
