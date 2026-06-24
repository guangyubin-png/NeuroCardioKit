"""NeuroCardioKit — EEG + ECG/HRV analysis toolbox, with OSEA auto-build."""
import os
import sys
import subprocess
from setuptools import setup

HERE = os.path.dirname(os.path.abspath(__file__))


def _osea_lib_name():
    if sys.platform == "win32":
        return "osea.dll"
    elif sys.platform == "darwin":
        return "osea.dylib"
    else:
        return "osea.so"


def _osea_compiler():
    if sys.platform == "win32":
        return "mingw"
    else:
        return "gcc"


def _build_osea():
    """Compile OSEA shared library from C++ source. Best-effort, warns on failure."""
    osea_dir = os.path.join(HERE, "osea")
    build_script = os.path.join(osea_dir, "build_osea.py")
    lib_path = os.path.join(osea_dir, _osea_lib_name())

    if os.path.exists(lib_path):
        print(f"[neurocardiokit] {_osea_lib_name()} already exists, skipping build")
        return

    if not os.path.exists(build_script):
        print("[neurocardiokit] build_osea.py not found, skipping OSEA build")
        return

    compiler = _osea_compiler()
    print(f"[neurocardiokit] Building {_osea_lib_name()} from C++ source "
          f"(platform={sys.platform}, compiler={compiler})...")

    try:
        subprocess.run(
            [sys.executable, build_script, "--compiler", compiler],
            cwd=osea_dir,
            check=True,
        )
    except Exception:
        try:
            print("[neurocardiokit] Retrying with auto-detection...")
            subprocess.run(
                [sys.executable, build_script],
                cwd=osea_dir,
                check=True,
            )
        except Exception:
            print(
                "[neurocardiokit] WARNING: OSEA build failed. "
                "OSEA functions will not be available. "
                "To build manually: python osea/build_osea.py"
            )


# Build OSEA at import time so it works for both `pip install .` and `pip install -e .`
_build_osea()

setup(
    name="neurocardiokit",
    version="0.1.0",
    description="NeuroCardioKit — EEG + ECG/HRV analysis toolbox for neurocardiac research",
    python_requires=">=3.8",
    package_dir={"neurocardiokit": "."},
    packages=[
        "neurocardiokit",
        "neurocardiokit.eeg",
        "neurocardiokit.hrv",
        "neurocardiokit.osea",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "hrv-analysis",
    ],
    include_package_data=True,
    package_data={
        "neurocardiokit.osea": ["*.dll", "*.so", "*.dylib", "*.cpp", "*.h", "build_osea.py"],
    },
)
