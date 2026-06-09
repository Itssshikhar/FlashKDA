from __future__ import annotations

import datetime as _datetime
import subprocess
from pathlib import Path

import modal


APP_DIR = "/workspace/FlashKDA"
REPORT_DIR = "/reports"
VOLUME_NAME = "flashkda-ncu"
GPU = "H100!"


app = modal.App("flashkda-ncu")
reports = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("build-essential", "ca-certificates", "git", "ninja-build")
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel packaging ninja",
        "apt-get update && "
        "(apt-get install -y cuda-nsight-compute-13-0 || "
        "apt-get install -y nsight-compute || "
        "apt-get install -y cuda-nsight-compute-12-9)",
    )
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("numpy")
    .env(
        {
            "NVIDIA_VISIBLE_DEVICES": "all",
            "NVIDIA_DRIVER_CAPABILITIES": "all",
        }
    )
    .workdir(APP_DIR)
    .add_local_file("setup.py", f"{APP_DIR}/setup.py")
    .add_local_dir("flash_kda", f"{APP_DIR}/flash_kda")
    .add_local_dir("csrc", f"{APP_DIR}/csrc")
    .add_local_file(
        "benchmarks/profile_fwd_ncu_target.py",
        f"{APP_DIR}/benchmarks/profile_fwd_ncu_target.py",
    )
    .add_local_dir("cutlass/include", f"{APP_DIR}/cutlass/include")
    .add_local_dir("cutlass/examples/common", f"{APP_DIR}/cutlass/examples/common")
    .add_local_dir("cutlass/tools/util/include", f"{APP_DIR}/cutlass/tools/util/include")
)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _ncu_collection_args(ncu_set: str) -> list[str]:
    if ncu_set.startswith("section:"):
        args: list[str] = []
        for section in _split_csv(ncu_set.removeprefix("section:")):
            args.extend(["--section", section])
        return args
    return ["--set", ncu_set]


@app.function(
    image=image,
    gpu=GPU,
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 3,
    volumes={REPORT_DIR: reports},
)
def run_ncu(
    modes: str = "fixed,varlen",
    state_mode: str = "bf16",
    ncu_set: str = "full",
    warmup: int = 0,
    iters: int = 1,
    h: int = 96,
    d: int = 128,
    diagnose_only: bool = False,
) -> dict[str, object]:
    import os
    import shutil
    import subprocess
    from pathlib import Path

    def find_ncu() -> str | None:
        preferred = [
            Path("/opt/nvidia/nsight-compute/2025.3.1/ncu"),
            Path("/opt/nvidia/nsight-compute/2025.3.1/target/linux-desktop-glibc_2_11_3-x64/ncu"),
            Path("/usr/local/cuda-13.0/bin/ncu"),
            Path("/opt/nvidia/nsight-compute/2025.2.1/ncu"),
            Path("/opt/nvidia/nsight-compute/2025.2.1/target/linux-desktop-glibc_2_11_3-x64/ncu"),
        ]
        for candidate in preferred:
            if candidate.exists():
                return str(candidate)
        return shutil.which("ncu")

    ncu = find_ncu()
    if ncu is None:
        raise RuntimeError("ncu was not found in the Modal image")

    os.chdir(APP_DIR)

    run_id = _datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(REPORT_DIR) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "MAX_JOBS": "8",
            "NVCC_THREADS": "8",
            "TORCH_CUDA_ARCH_LIST": "9.0a",
            "CC": "/usr/bin/gcc",
            "CXX": "/usr/bin/g++",
            "CUDAHOSTCXX": "/usr/bin/g++",
        }
    )

    def run_capture(cmd: list[str], path: Path | None = None) -> tuple[int, str]:
        proc = subprocess.run(
            cmd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        output = f"$ {' '.join(cmd)}\n{proc.stdout}"
        if path is not None:
            path.write_text(output)
        print(output, flush=True)
        return proc.returncode, output

    metadata_cmds = [
        ["nvidia-smi"],
        ["nvidia-smi", "-L"],
        ["nvidia-smi", "-q", "-d", "CLOCK,COMPUTE,PERFORMANCE"],
        ["bash", "-lc", "which ncu || true; readlink -f $(which ncu) || true"],
        ["bash", "-lc", "find /opt/nvidia/nsight-compute /usr/local/cuda* -name ncu 2>/dev/null | sort"],
        [ncu, "--version"],
        [ncu, "--list-sets"],
        [ncu, "--list-sections"],
        [
            "bash",
            "-lc",
            "env | sort | grep -E '^(CUDA|NVIDIA|LD_)' || true",
        ],
        [
            "bash",
            "-lc",
            "ldconfig -p | grep -E 'libcuda|libcupti|libnvidia' | head -200 || true",
        ],
        [
            "bash",
            "-lc",
            "find /usr /lib -name 'libcuda.so*' -o -name 'libcupti.so*' "
            "-o -name 'libnvidia-*.so*' 2>/dev/null | sort | head -300",
        ],
        [
            "python",
            "-c",
            "import torch; print('torch', torch.__version__); print('torch_cuda', torch.version.cuda)",
        ],
    ]
    metadata = []
    for cmd in metadata_cmds:
        _, output = run_capture(cmd)
        metadata.append(output)
    (out_dir / "metadata.txt").write_text("\n".join(metadata))

    preflight_cmds = {
        "torch_launchstats": [
            ncu,
            "--section",
            "LaunchStats",
            "--kernel-name-base",
            "function",
            "--clock-control",
            "none",
            "--target-processes",
            "all",
            "python",
            "-c",
            "import torch; x=torch.randn(1024, device='cuda'); y=x+x; torch.cuda.synchronize()",
        ],
        "torch_default": [
            ncu,
            "--set",
            "default",
            "--kernel-name-base",
            "function",
            "--clock-control",
            "none",
            "--target-processes",
            "all",
            "python",
            "-c",
            "import torch; x=torch.randn(1024, device='cuda'); y=x+x; torch.cuda.synchronize()",
        ],
    }
    preflight: dict[str, dict[str, object]] = {}
    for name, cmd in preflight_cmds.items():
        code, output = run_capture(cmd, out_dir / f"preflight_{name}.txt")
        preflight[name] = {"returncode": code, "tail": output[-8000:]}

    if diagnose_only:
        reports.commit()
        return {
            "ok": all(item["returncode"] == 0 for item in preflight.values()),
            "run_id": run_id,
            "volume": VOLUME_NAME,
            "remote_path": f"/{run_id}",
            "summaries": {name: str(item["tail"]) for name, item in preflight.items()},
        }

    build_cmd = ["python", "-m", "pip", "install", "-v", "--no-build-isolation", "."]
    print(f"$ {' '.join(build_cmd)}", flush=True)
    build = subprocess.run(
        build_cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (out_dir / "build.log").write_text(build.stdout)
    print(build.stdout, flush=True)
    ok = build.returncode == 0
    errors: list[str] = []
    if build.returncode != 0:
        errors.append(f"build exited {build.returncode}")

    summaries: dict[str, str] = {}
    for mode in _split_csv(modes) if ok else []:
        label = f"{mode}_{state_mode}"
        report_path = out_dir / f"{label}.ncu-rep"
        summary_path = out_dir / f"{label}.txt"
        cmd = [
            ncu,
            *_ncu_collection_args(ncu_set),
            "--kernel-name-base",
            "function",
            "-k",
            "regex:_flash_kda_fwd_(prepare|recurrence)",
            "--clock-control",
            "none",
            "--target-processes",
            "all",
            "--import-source",
            "yes",
            "--source-folders",
            APP_DIR,
            "--export",
            str(report_path),
            "python",
            "benchmarks/profile_fwd_ncu_target.py",
            "--mode",
            mode,
            "--state-mode",
            state_mode,
            "--warmup",
            str(warmup),
            "--iters",
            str(iters),
            "--H",
            str(h),
            "--D",
            str(d),
        ]
        print(f"$ {' '.join(cmd)}", flush=True)
        proc = subprocess.run(
            cmd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        summary_path.write_text(proc.stdout)
        print(proc.stdout, flush=True)
        summaries[label] = proc.stdout[-8000:]
        if proc.returncode != 0:
            ok = False
            errors.append(f"{label} ncu exited {proc.returncode}")
            break

    reports.commit()
    return {
        "ok": ok,
        "errors": errors,
        "run_id": run_id,
        "volume": VOLUME_NAME,
        "remote_path": f"/{run_id}",
        "summaries": summaries,
    }


@app.local_entrypoint()
def main(
    modes: str = "fixed,varlen",
    state_mode: str = "bf16",
    ncu_set: str = "full",
    warmup: int = 0,
    iters: int = 1,
    h: int = 96,
    d: int = 128,
    diagnose_only: bool = False,
    download: bool = True,
    local_dir: str = "artifacts/ncu",
):
    result = run_ncu.remote(
        modes=modes,
        state_mode=state_mode,
        ncu_set=ncu_set,
        warmup=warmup,
        iters=iters,
        h=h,
        d=d,
        diagnose_only=diagnose_only,
    )

    run_id = str(result["run_id"])
    remote_path = str(result["remote_path"])
    print(f"Modal Volume: {result['volume']}:{remote_path}")

    for label, summary in dict(result["summaries"]).items():
        print(f"\n===== {label} =====")
        print(summary)

    if download:
        destination = Path(local_dir) / run_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "modal",
                "volume",
                "get",
                "--force",
                str(result["volume"]),
                remote_path,
                str(destination),
            ],
            check=True,
        )
        print(f"Downloaded ncu artifacts to {destination}")

    if not bool(result.get("ok", False)):
        errors = ", ".join(str(error) for error in result.get("errors", []))
        raise SystemExit(f"Modal ncu run failed: {errors or 'see artifacts'}")
