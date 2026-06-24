"""
Build OSEA shared library from C++ source.

Cross-platform: Windows (.dll), Linux (.so), macOS (.dylib).

Usage:
    python build_osea.py                    # Auto-detect platform + compiler
    python build_osea.py --compiler mingw   # Windows MinGW
    python build_osea.py --compiler msvc    # Windows MSVC
    python build_osea.py --compiler conda   # Windows conda MinGW
    python build_osea.py --compiler gcc     # Linux g++
    python build_osea.py --compiler clang   # macOS / Linux clang++
"""

import subprocess
import sys
import os
import shutil
import platform

HERE = os.path.dirname(os.path.abspath(__file__))
BRIDGE_SRC = os.path.join(HERE, "osea_bridge.cpp")

# Platform-specific shared library extension
if sys.platform == "win32":
    LIB_EXT = ".dll"
elif sys.platform == "darwin":
    LIB_EXT = ".dylib"
else:
    LIB_EXT = ".so"

OUT_LIB = os.path.join(HERE, "osea" + LIB_EXT)


# ---------------------------------------------------------------------------
# Compiler install guidance per platform
# ---------------------------------------------------------------------------

def _gcc_install_hint():
    """Return the command to install g++ for the current Linux distro."""
    if shutil.which("apt"):
        return "sudo apt install build-essential"
    if shutil.which("yum"):
        return "sudo yum install gcc-c++"
    if shutil.which("dnf"):
        return "sudo dnf install gcc-c++"
    if shutil.which("pacman"):
        return "sudo pacman -S gcc"
    if shutil.which("zypper"):
        return "sudo zypper install gcc-c++"
    return "install g++ via your system package manager"


# ---------------------------------------------------------------------------
# Windows compilers
# ---------------------------------------------------------------------------

def build_msvc():
    """Build using Microsoft Visual C++ compiler."""
    cl = shutil.which("cl")
    if cl is None:
        for vs_dir in [
            r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
            r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC",
            r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC",
        ]:
            if os.path.isdir(vs_dir):
                versions = sorted(os.listdir(vs_dir), reverse=True)
                for v in versions:
                    cl_path = os.path.join(vs_dir, v, "bin", "Hostx64", "x64", "cl.exe")
                    if os.path.isfile(cl_path):
                        return build_with_cl(cl_path)
        print("ERROR: cl.exe not found. Install Visual Studio Build Tools or use --compiler mingw.")
        return False
    return build_with_cl(cl)


def build_with_cl(cl_path):
    print(f"Using MSVC: {cl_path}")
    cmd = [cl_path, "/LD", "/O2", "/nologo", f"/Fe:{OUT_LIB}", BRIDGE_SRC]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=HERE)
    return result.returncode == 0


def _find_mingw_gxx():
    """Find MinGW g++ on Windows."""
    gxx = shutil.which("g++")
    if gxx:
        return gxx
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    for subpath in [
        os.path.join(conda_prefix, "Library", "mingw-w64", "bin", "g++.exe"),
        os.path.join(conda_prefix, "Library", "bin", "g++.exe"),
    ]:
        if os.path.isfile(subpath):
            return subpath
    gxx = shutil.which("x86_64-w64-mingw32-g++")
    if gxx:
        return gxx
    return None


def build_mingw():
    """Build using MinGW g++ on Windows."""
    gxx = _find_mingw_gxx()
    if gxx is None:
        print("ERROR: MinGW g++ not found.")
        print("  Install via conda:  conda install -c conda-forge m2w64-gcc")
        print("  Or via:             https://winlibs.com/")
        return False
    return _build_with_gxx(gxx, extra_flags=["-static"])


def build_conda():
    """Build using conda environment compiler (Windows)."""
    conda_prefix = os.environ.get("CONDA_PREFIX", r"D:\anaconda3\envs\ecg_ai")

    mingw_gxx = os.path.join(conda_prefix, "Library", "mingw-w64", "bin", "g++.exe")
    if os.path.isfile(mingw_gxx):
        return _build_with_gxx(mingw_gxx, extra_flags=["-static"])

    gxx = os.path.join(conda_prefix, "Library", "bin", "g++.exe")
    if os.path.isfile(gxx):
        return _build_with_gxx(gxx, extra_flags=["-static"])

    clang = os.path.join(conda_prefix, "Library", "bin", "clang++.exe")
    if os.path.isfile(clang):
        return _build_with_gxx(clang)

    print("No conda compiler found.")
    return False


# ---------------------------------------------------------------------------
# Linux / macOS / generic g++/clang
# ---------------------------------------------------------------------------

def _find_gxx():
    """Find a C++ compiler (g++ or clang++)."""
    for name in ["g++", "clang++", "c++"]:
        path = shutil.which(name)
        if path:
            return path
    return None


def _build_with_gxx(compiler_path, extra_flags=None):
    """Build shared library with g++/clang++."""
    print(f"Using compiler: {compiler_path}")
    flags = ["-shared", "-O2", "-fPIC"]
    if extra_flags:
        flags.extend(extra_flags)
    cmd = [compiler_path] + flags + ["-o", OUT_LIB, BRIDGE_SRC]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=HERE)
    return result.returncode == 0


def build_gcc():
    """Build using system g++ (Linux / macOS / generic)."""
    gxx = _find_gxx()
    if gxx is None:
        print("ERROR: No C++ compiler found (tried g++, clang++, c++).")
        if sys.platform == "linux":
            print(f"  Install:  {_gcc_install_hint()}")
        elif sys.platform == "darwin":
            print("  Install:  xcode-select --install")
        else:
            print("  Please install a C++ compiler and add it to PATH.")
        return False
    return _build_with_gxx(gxx)


build_clang = build_gcc  # alias


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    compiler = None
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--compiler" and i + 1 < len(sys.argv):
            compiler = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    # Map compiler name → build function
    builders = {
        "msvc": build_msvc,
        "mingw": build_mingw,
        "conda": build_conda,
        "gcc": build_gcc,
        "clang": build_clang,
    }

    if compiler and compiler in builders:
        success = builders[compiler]()
    elif compiler:
        print(f"Unknown compiler '{compiler}'. Choices: {', '.join(builders)}")
        sys.exit(1)
    else:
        # Auto-detect by platform
        success = _auto_detect()

    if success:
        if os.path.isfile(OUT_LIB):
            size_kb = os.path.getsize(OUT_LIB) / 1024
            print(f"\nBuild successful! {OUT_LIB} ({size_kb:.0f} KB)")
        else:
            print("\nBuild appeared to succeed but output file not found.")
    else:
        print("\nBuild failed. Try specifying compiler explicitly:")
        if sys.platform == "win32":
            print("  python build_osea.py --compiler mingw")
            print("  python build_osea.py --compiler msvc")
            print("  python build_osea.py --compiler conda")
        else:
            print("  python build_osea.py --compiler gcc")
            print("  python build_osea.py --compiler clang")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Auto-install compiler via conda
# ---------------------------------------------------------------------------

def _find_conda_exe():
    """Find the conda executable."""
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        for subpath in [
            os.path.join(conda_prefix, "Scripts", "conda.exe"),
            os.path.join(conda_prefix, "condabin", "conda.bat"),
            os.path.join(conda_prefix, "bin", "conda"),
        ]:
            if os.path.isfile(subpath):
                return subpath
    # Fallback: try PATH
    for name in ["conda", "conda.exe", "conda.bat"]:
        path = shutil.which(name)
        if path:
            return path
    return None


def _conda_install(package):
    """Install a package via conda. Returns True on success."""
    conda = _find_conda_exe()
    if conda is None:
        return False
    print(f"[conda] Installing {package} (this may take a minute)...")
    try:
        subprocess.run(
            [conda, "install", "-y", "-c", "conda-forge", package],
            check=True,
            timeout=300,
        )
        return True
    except Exception as e:
        print(f"[conda] Auto-install failed: {e}")
        return False


def _auto_install_windows_compiler():
    """Try to auto-install MinGW via conda on Windows."""
    if _find_conda_exe() is None:
        return False
    print("No C++ compiler found. Attempting auto-install via conda...")
    if _conda_install("m2w64-gcc"):
        # Re-check after install
        gxx = _find_mingw_gxx()
        if gxx:
            print(f"Installed successfully: {gxx}")
            return True
    return False


def _auto_detect():
    """Auto-detect compiler by platform and availability."""
    if sys.platform == "win32":
        # 1. Try existing MinGW
        gxx = _find_mingw_gxx()
        if gxx:
            print(f"Auto-detected MinGW: {gxx}")
            return build_mingw()

        # 2. Try conda env compiler
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if conda_prefix:
            print("Trying conda compiler...")
            if build_conda():
                return True

        # 3. Try MSVC
        cl = shutil.which("cl")
        if cl:
            print("Trying MSVC...")
            return build_msvc()

        # 4. Auto-install MinGW via conda
        if _auto_install_windows_compiler():
            return build_mingw()

        # 5. Give up with clear instructions
        print("ERROR: No C++ compiler found on Windows.")
        print("  Install MinGW via conda:  conda install -c conda-forge m2w64-gcc")
        print("  Or download from:         https://winlibs.com/")
        return False
    else:
        # Linux / macOS: try system g++ / clang++
        gxx = _find_gxx()
        if gxx:
            print(f"Auto-detected: {gxx}")
            return _build_with_gxx(gxx)

        # In conda env, try installing gcc
        if _find_conda_exe():
            print("No C++ compiler found. Attempting auto-install via conda...")
            if _conda_install("gcc"):
                gxx = _find_gxx()
                if gxx:
                    print(f"Installed successfully: {gxx}")
                    return _build_with_gxx(gxx)

        # Give up
        print("ERROR: No C++ compiler found.")
        if sys.platform == "linux":
            print(f"  Install:  {_gcc_install_hint()}")
            print("  Or via conda:  conda install -c conda-forge gcc")
        elif sys.platform == "darwin":
            print("  Install:  xcode-select --install")
            print("  Or via conda:  conda install -c conda-forge clang")
        return False


if __name__ == "__main__":
    main()
