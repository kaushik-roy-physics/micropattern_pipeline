"""
Is this Python environment ready for the pipeline? Answers in a few seconds.

Run inside the venv by hpc/common.sh before every job (and by setup_env.sh). It
installs nothing; it only says what, if anything, has to be (re)installed:

exit 0  everything is in place: the job starts at once
exit 1  some packages from requirements.txt are missing or too old
exit 2  PyTorch is missing or is the wrong build (reinstall it from the cu128 index)

What "the wrong build" means: the gpu-b6000 nodes are Blackwell GPUs (sm_120)
run with CUDA 12.8. The wheel must be built for exactly that CUDA (the cu128
index): an older one (cu126, cu121) has no sm_120 kernels, imports fine, even
reports the GPU as available, then fails on the first real kernel; a newer one
(the plain PyPI wheel is now cu130) needs a newer GPU driver than the cluster's
CUDA 12.8 setup. So the check reads which CUDA the wheel was built for and which
GPU architectures it was compiled for, which works on a node without a GPU too.
torchvision must be built for the same CUDA as torch, or its operators fail.

Usage:  python check_env.py REQUIREMENTS.txt [--cuda 12.8] [--arch sm_120]
"""

from __future__ import annotations

import argparse
import re
import sys
import time

OK, PACKAGES, TORCH = 0, 1, 2


def _packaging():
    try:
        from packaging.requirements import Requirement
        from packaging.version import Version
    except ImportError:                                  # pip always vendors it
        from pip._vendor.packaging.requirements import Requirement
        from pip._vendor.packaging.version import Version
    return Requirement, Version


def check_requirements(path: str) -> list[str]:
    """Problems with the plain (non-torch) requirements, as human-readable lines."""
    from importlib.metadata import PackageNotFoundError, version
    Requirement, Version = _packaging()
    problems = []
    for raw in open(path, encoding="utf-8"):
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        req = Requirement(line)
        if req.marker is not None and not req.marker.evaluate():
            continue
        if req.name.lower() in ("torch", "torchvision"):
            continue                                     # checked separately below
        try:
            have = version(req.name)
        except PackageNotFoundError:
            problems.append(f"{req.name}: not installed (need {req.specifier or 'any'})")
            continue
        base = Version(have).base_version
        if req.specifier and not req.specifier.contains(base, prereleases=True):
            problems.append(f"{req.name}: {have} installed, need {req.specifier}")
    return problems


def _arch_flags(torch) -> tuple[list[str], list[str]]:
    """``(sm_XX binaries, compute_XX PTX)`` the wheel was compiled for (no GPU needed)."""
    try:
        flags = torch._C._cuda_getArchFlags() or ""
    except Exception:                                    # noqa: BLE001
        flags = ""
    if not flags:
        try:
            flags = " ".join(torch.cuda.get_arch_list())
        except Exception:                                # noqa: BLE001
            flags = ""
    sm = sorted(set(re.findall(r"sm_\d+", flags)), key=lambda a: int(a[3:]))
    ptx = sorted(set(re.findall(r"compute_\d+", flags)), key=lambda a: int(a[8:]))
    return sm, ptx


def _cc(arch: str) -> tuple[int, int]:
    """'sm_120' → (12, 0), 'sm_89' → (8, 9)."""
    digits = re.sub(r"\D", "", arch)
    return int(digits[:-1]), int(digits[-1])


def runs_on(sm: list[str], ptx: list[str], arch: str) -> bool:
    """Can a wheel with these binaries / PTX run kernels on *arch*?

    A binary runs on GPUs of the same major version with an equal or higher
    minor (sm_86 runs on sm_89; sm_90 does NOT run on sm_120). PTX for an older
    architecture is compiled for the GPU when first used.
    """
    major, minor = _cc(arch)
    if any(_cc(a)[0] == major and _cc(a)[1] <= minor for a in sm):
        return True
    return any(_cc(p) <= (major, minor) for p in ptx)


def _cuda_text(value) -> str | None:
    """'12.8' as is; torchvision's integer form 12080 → '12.8'."""
    if value is None:
        return None
    text = str(value)
    if text.isdigit() and len(text) >= 4:
        v = int(text)
        return f"{v // 1000}.{(v % 1000) // 10}"
    return text


def check_torch(cuda: str, arch: str) -> list[str]:
    _, Version = _packaging()
    problems = []
    try:
        import torch
    except Exception as exc:                             # noqa: BLE001
        return [f"torch: cannot be imported ({type(exc).__name__}: {exc})"]
    built = torch.version.cuda
    print(f"  torch {torch.__version__}, built for CUDA {built or 'none (CPU-only wheel)'}")
    if not built:
        problems.append("torch: CPU-only wheel; the GPU cannot be used")
    elif Version(built).release[:2] != Version(cuda).release[:2]:
        problems.append(f"torch: built for CUDA {built}, but this cluster's build is CUDA "
                        f"{cuda} (TORCH_INDEX)")
    sm, ptx = _arch_flags(torch)
    if built:
        print(f"  compiled for: {' '.join(sm + ptx) or '(unknown)'}")
        if (sm or ptx) and not runs_on(sm, ptx, arch):
            problems.append(f"torch: no kernels that run on {arch} in this wheel "
                            f"(has {', '.join(sm + ptx)})")
    try:
        import torchvision
        tv_cuda = _cuda_text(getattr(getattr(torchvision, "version", None), "cuda", None))
        print(f"  torchvision {torchvision.__version__}, built for CUDA {tv_cuda}")
        if built and tv_cuda and tv_cuda != built:
            problems.append(f"torchvision: built for CUDA {tv_cuda} but torch for {built}")
    except Exception as exc:                             # noqa: BLE001
        problems.append(f"torchvision: cannot be imported ({type(exc).__name__}: {exc})")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("requirements")
    ap.add_argument("--cuda", default="12.8", help="CUDA version the torch build must have")
    ap.add_argument("--arch", default="sm_120", help="GPU architecture that must be supported")
    args = ap.parse_args()

    start = time.perf_counter()
    print(f"Checking the Python environment ({sys.executable}, Python "
          f"{sys.version.split()[0]})")
    torch_problems = check_torch(args.cuda, args.arch)
    pkg_problems = check_requirements(args.requirements)
    for p in torch_problems + pkg_problems:
        print(f"  NEEDS FIXING: {p}")
    took = time.perf_counter() - start
    if torch_problems:
        print(f"PyTorch must be (re)installed. ({took:.1f} s)")
        return TORCH
    if pkg_problems:
        print(f"Some packages must be installed. ({took:.1f} s)")
        return PACKAGES
    print(f"Environment OK: nothing to install. ({took:.1f} s)")
    return OK


if __name__ == "__main__":
    sys.exit(main())
