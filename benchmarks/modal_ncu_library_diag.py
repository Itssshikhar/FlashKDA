from __future__ import annotations

import subprocess
from pathlib import Path

import modal


APP_DIR = "/workspace/FlashKDA"
GPU = "H100!"


app = modal.App("flashkda-ncu-library-diag")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("build-essential", "ca-certificates", "git", "strace")
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel packaging ninja",
        "apt-get update && "
        "(apt-get install -y cuda-nsight-compute-13-0 || "
        "apt-get install -y nsight-compute || "
        "apt-get install -y cuda-nsight-compute-12-9)",
    )
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu128")
    .env(
        {
            "NVIDIA_VISIBLE_DEVICES": "all",
            "NVIDIA_DRIVER_CAPABILITIES": "all",
        }
    )
    .workdir(APP_DIR)
)


@app.function(image=image, gpu=GPU, timeout=60 * 30)
def diagnose() -> dict[str, str]:
    import os
    import shutil

    def find_ncu() -> str:
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
        ncu = shutil.which("ncu")
        if ncu is None:
            raise RuntimeError("ncu not found")
        return ncu

    ncu = find_ncu()
    base_cmd = [
        ncu,
        "--verbose",
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
    ]

    def run(cmd: list[str], *, env: dict[str, str] | None = None) -> tuple[int, str]:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
        return proc.returncode, f"$ {' '.join(cmd)}\n{proc.stdout}"

    metadata_parts = []
    for cmd in [
        ["nvidia-smi"],
        [ncu, "--version"],
        ["bash", "-lc", "env | sort | grep -E '^(CUDA|NVIDIA|LD_)' || true"],
        ["bash", "-lc", "cat /proc/driver/nvidia/params 2>/dev/null | grep -i Restrict || true"],
        ["bash", "-lc", "ldconfig -p | grep -E 'libcuda|libcupti|libnvidia|PerfWorks|NvPerf' | sort || true"],
        [
            "bash",
            "-lc",
            "find /usr /lib /opt/nvidia -name 'libcuda.so*' -o -name 'libcupti.so*' "
            "-o -name 'libnvidia-*.so*' -o -name '*PerfWorks*.so*' -o -name '*NvPerf*.so*' "
            "-o -name '*nvperf*.so*' -o -name '*cuda-injection*.so*' "
            "2>/dev/null | sort",
        ],
    ]:
        _, output = run(cmd)
        metadata_parts.append(output)

    ld_env = os.environ.copy()
    ld_env["LD_DEBUG"] = "libs,files"
    ld_code, ld_output = run(base_cmd, env=ld_env)

    strace_path = Path("/tmp/ncu-file.strace")
    strace_cmd = [
        "strace",
        "-f",
        "-e",
        "trace=file",
        "-o",
        str(strace_path),
        *base_cmd,
    ]
    strace_code, strace_stdout = run(strace_cmd)
    strace_text = strace_path.read_text(errors="replace") if strace_path.exists() else ""

    interesting_strace = []
    for line in strace_text.splitlines():
        lower = line.lower()
        if ".so" not in lower:
            continue
        if not any(token in lower for token in ["nvidia", "cupti", "perf", "cuda", "ncu"]):
            continue
        if "enoent" in lower or "eacces" in lower or "eperm" in lower:
            interesting_strace.append(line)

    interesting_ld = []
    for line in ld_output.splitlines():
        lower = line.lower()
        if any(token in lower for token in ["not found", "error", "libcuda", "libcupti", "libnvidia", "perfworks", "nvperf"]):
            interesting_ld.append(line)

    return {
        "metadata": "\n".join(metadata_parts)[-20000:],
        "ld_code": str(ld_code),
        "ld_interesting": "\n".join(interesting_ld)[-30000:],
        "ld_tail": ld_output[-12000:],
        "strace_code": str(strace_code),
        "strace_stdout": strace_stdout[-12000:],
        "strace_interesting": "\n".join(interesting_strace)[-30000:],
    }


@app.local_entrypoint()
def main():
    result = diagnose.remote()
    for key, value in result.items():
        print(f"\n===== {key} =====")
        print(value)
