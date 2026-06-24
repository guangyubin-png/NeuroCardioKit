"""
OSEA - Open Source ECG Analysis Python bridge.

Thin ctypes wrapper around the OSEA DLL. Provides sample-by-sample
QRS detection and beat classification for 200 Hz ECG data.

Usage:
    from neurocardiokit.osea import OseaEngine

    engine = OseaEngine()
    engine.reset()

    for sample in ecg_data:  # 200 Hz, baseline ~0, 200 units/mV
        delay = engine.detect(sample)
        if delay > 0:
            print(f"Beat: type={engine.beat_type}, match={engine.beat_match}")

Low-level usage (if DLL is not in default path):
    from neurocardiokit.osea._osea_bridge import load_osea
    lib = load_osea("path/to/osea.dll")
"""

import ctypes
import os
import sys
import json
from ctypes import c_int, POINTER, byref

_GITHUB_REPO = "guangyubin-png/NeuroCardioKit"

# ---------------------------------------------------------------------------
# Annotation type mapping
# ---------------------------------------------------------------------------

ANNOTYPE_LABELS = {
    0: "NOTQRS", 1: "NORMAL", 2: "LBBB", 3: "RBBB", 4: "ABERR",
    5: "PVC", 6: "FUSION", 7: "NPC", 8: "APC", 9: "SVPB",
    10: "VESC", 11: "NESC", 12: "PACE", 13: "UNKNOWN", 14: "NOISE",
    16: "ARFCT", 18: "STCH", 19: "TCH", 20: "SYSTOLE",
    21: "DIASTOLE", 22: "NOTE", 23: "MEASURE",
    25: "BBB", 26: "PACESP", 28: "RHYTHM",
    30: "LEARN", 31: "FLWAV", 32: "VFON", 33: "VFOFF",
    34: "AESC", 35: "SVESC", 37: "NAPC", 38: "PFUS",
    39: "PQ", 40: "JPT", 41: "RONT",
}

BEAT_CLASS_NAMES = {
    1: "N",   # NORMAL
    5: "V",   # PVC
    13: "Q",  # UNKNOWN
}


def _osea_lib_name():
    """Return the platform-appropriate OSEA shared library name."""
    if sys.platform == "win32":
        return "osea.dll"
    elif sys.platform == "darwin":
        return "osea.dylib"
    else:
        return "osea.so"


def _find_osea_dll():
    """Locate the OSEA shared library — local first, then auto-download from GitHub Release."""
    here = os.path.dirname(os.path.abspath(__file__))
    libname = _osea_lib_name()
    candidates = [
        os.path.join(here, libname),
        os.path.join(here, "..", "..", libname),
        libname,
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)

    # Auto-download from GitHub Release
    downloaded = _download_osea_lib(os.path.join(here, libname))
    if downloaded:
        return downloaded

    # Fallback: let ctypes search PATH
    return libname


def _download_osea_lib(target_path):
    """Download the platform-appropriate OSEA library from the latest GitHub Release."""
    api_url = f"https://api.github.com/repos/{_GITHUB_REPO}/releases/latest"
    libname = _osea_lib_name()

    try:
        import urllib.request
        import ssl

        # Try verified SSL first, fallback to unverified (corporate proxies, etc.)
        ctx = None
        for attempt in [None, ssl._create_unverified_context()]:
            try:
                req = urllib.request.Request(api_url)
                req.add_header("Accept", "application/vnd.github+json")
                req.add_header("User-Agent", "NeuroCardioKit")
                with urllib.request.urlopen(req, timeout=15, context=attempt) as resp:
                    release = json.loads(resp.read())
                ctx = attempt
                break
            except Exception:
                if attempt is not None:
                    raise

        # Find the right asset for this platform
        asset_url = None
        for asset in release.get("assets", []):
            if asset["name"] == libname:
                asset_url = asset["browser_download_url"]
                break

        if not asset_url:
            print(f"[neurocardiokit] No {libname} found in latest GitHub Release")
            return None

        # Download the library (urlretrieve doesn't support ssl_context, use urlopen)
        print(f"[neurocardiokit] Downloading {libname} ...")
        req2 = urllib.request.Request(asset_url)
        with urllib.request.urlopen(req2, timeout=60, context=ctx) as resp:
            with open(target_path, "wb") as f:
                f.write(resp.read())
        os.chmod(target_path, 0o755)
        print(f"[neurocardiokit] {libname} saved")
        return os.path.abspath(target_path)

    except Exception as e:
        print(f"[neurocardiokit] Auto-download failed: {e}")
        return None


_DLL_LOAD_ERROR = """
OSEA shared library (osea.dll / osea.so / osea.dylib) not found.

To compile it:
  1. Install a C++ compiler:
     - Windows: conda install -c conda-forge m2w64-gcc
     - Linux:   sudo apt install build-essential
     - macOS:   xcode-select --install
  2. Run: python -m neurocardiokit.osea.build_osea
  Or:  pip install -e .  (from the NeuroCardioKit directory)
"""


def load_osea(dll_path=None):
    """
    Load the OSEA shared library.

    Parameters
    ----------
    dll_path : str or None
        Path to osea.dll. If None, search standard locations.

    Returns
    -------
    ctypes.CDLL

    Raises
    ------
    OSError
        If the DLL cannot be found or loaded.
    """
    if dll_path is None:
        dll_path = _find_osea_dll()

    if not os.path.exists(dll_path) and not dll_path.startswith(("lib", "osea")):
        pass  # Let ctypes try the system path
    elif not os.path.exists(dll_path):
        raise OSError(_DLL_LOAD_ERROR.strip())

    try:
        lib = ctypes.CDLL(dll_path)
    except OSError as e:
        raise OSError(
            f"Failed to load OSEA library: {e}\n{_DLL_LOAD_ERROR.strip()}"
        ) from None

    # Configure function signatures
    lib.osea_reset.argtypes = []
    lib.osea_reset.restype = None

    lib.osea_detect.argtypes = [c_int, POINTER(c_int), POINTER(c_int)]
    lib.osea_detect.restype = c_int

    lib.osea_qrs_det.argtypes = [c_int, c_int]
    lib.osea_qrs_det.restype = c_int

    return lib


class OseaEngine:
    """
    High-level Python wrapper for OSEA beat detection and classification.

    Feeds ECG samples one-by-one and collects detection results.

    Parameters
    ----------
    dll_path : str or None
        Path to osea.dll. Auto-detected if None.

    Examples
    --------
    >>> engine = OseaEngine()
    >>> engine.reset()
    >>> for sample in ecg_array:
    ...     delay = engine.detect(sample)
    ...     if delay > 0:
    ...         print(f"{engine.beat_type} ({ANNOTYPE_LABELS.get(engine.beat_type, '?')})")
    """

    def __init__(self, dll_path=None):
        self._lib = load_osea(dll_path)
        self._beat_type = c_int(0)
        self._beat_match = c_int(0)
        self._detection_count = 0
        self._detections = []  # list of (sample_index, delay, beat_type, beat_match)

    # -- public API ----------------------------------------------------------

    def reset(self):
        """Reset the engine for a new ECG signal."""
        self._lib.osea_reset()
        self._beat_type.value = 0
        self._beat_match.value = 0
        self._detection_count = 0
        self._detections.clear()

    def detect(self, ecg_sample):
        """
        Process one ECG sample.

        Parameters
        ----------
        ecg_sample : int
            ECG sample at 200 Hz. Should be normalized: 200 ADC units = 1 mV,
            with baseline ~0.

        Returns
        -------
        delay : int
            0 if no beat detected.
            >0 = samples since R-wave peak of the detected beat.
        """
        delay = self._lib.osea_detect(ecg_sample, byref(self._beat_type), byref(self._beat_match))
        if delay > 0:
            self._detection_count += 1
            self._detections.append((
                self._detection_count, delay,
                self._beat_type.value, self._beat_match.value
            ))
        return delay

    def detect_qrs_only(self, datum, init=False):
        """
        Low-level QRS detection without classification.

        Parameters
        ----------
        datum : int
            ECG sample.
        init : bool
            If True, reset QRS detector state.

        Returns
        -------
        delay : int
        """
        return self._lib.osea_qrs_det(datum, 1 if init else 0)

    # -- properties ----------------------------------------------------------

    @property
    def beat_type(self):
        """Most recent beat annotation type (NORMAL=1, PVC=5, etc.)."""
        return self._beat_type.value

    @property
    def beat_match(self):
        """Most recent template match index (0-7)."""
        return self._beat_match.value

    @property
    def beat_type_label(self):
        """Human-readable label for the most recent beat type."""
        return ANNOTYPE_LABELS.get(self._beat_type.value, f"CODE_{self._beat_type.value}")

    @property
    def detection_count(self):
        """Total number of beats detected since last reset."""
        return self._detection_count

    @property
    def detections(self):
        """List of all detections: [(count, delay, beat_type, beat_match), ...]."""
        return self._detections

    # -- batch processing ----------------------------------------------------

    def process_array(self, ecg_array):
        """
        Process an entire numpy array or list of ECG samples.

        Parameters
        ----------
        ecg_array : array-like
            1-D array of ECG samples (200 Hz).

        Returns
        -------
        detections : list of dict
            [{'index': sample_index, 'delay': delay, 'type': beat_type,
              'type_label': label, 'match': beat_match}, ...]
        """
        self.reset()
        results = []
        for idx, sample in enumerate(ecg_array):
            delay = self.detect(int(sample))
            if delay > 0:
                results.append({
                    'index': idx,
                    'delay': delay,
                    'type': self._beat_type.value,
                    'type_label': ANNOTYPE_LABELS.get(self._beat_type.value, '?'),
                    'match': self._beat_match.value,
                })
        return results

    def detect_rpeaks(self, ecg_mv, fs):
        """
        Detect R-peak positions from an ECG signal at any sampling rate.

        Handles resampling to 200 Hz, unit conversion (mV → ADC),
        and delay correction automatically.

        Parameters
        ----------
        ecg_mv : array-like
            1-D ECG signal in mV.
        fs : float
            Sampling rate of the input signal (Hz).

        Returns
        -------
        rpos : ndarray (int)
            R-peak sample indices in the **original** sampling rate,
            delay-corrected. All detected beats.
        anntype : ndarray (int)
            Beat type code for each R-peak (1=NORMAL, 5=PVC, etc.).
            Same length as rpos.
        """
        import numpy as np
        from scipy.interpolate import interp1d

        ecg_mv = np.asarray(ecg_mv, dtype=np.float64)
        ecg_mv = ecg_mv - np.mean(ecg_mv)  # center baseline

        if fs != 200.0:
            n_dst = int(len(ecg_mv) * 200.0 / fs)
            t_src = np.arange(len(ecg_mv)) / fs
            t_dst = np.arange(n_dst) / 200.0
            f_interp = interp1d(t_src, ecg_mv, kind='linear', copy=False,
                                assume_sorted=True)
            ecg_200 = f_interp(t_dst)
        else:
            ecg_200 = ecg_mv

        # mV → ADC (200 units/mV), center baseline again after conversion
        ecg_adc = ecg_200 * 200.0
        ecg_adc -= np.mean(ecg_adc)

        beats = self.process_array(ecg_adc.astype(int))

        # Map to original sampling rate with delay correction
        scale = fs / 200.0
        rpos = np.array([int((b['index'] - b['delay']) * scale)
                         for b in beats], dtype=int)
        anntype = np.array([b['type'] for b in beats], dtype=int)

        return rpos, anntype

    def process_array_qrs_only(self, ecg_array):
        """
        Process array with QRS detection only (no classification).

        Returns
        -------
        list of dict: [{'index': sample_index, 'delay': delay}, ...]
        """
        self.detect_qrs_only(0, init=True)
        results = []
        for idx, sample in enumerate(ecg_array):
            delay = self.detect_qrs_only(int(sample), init=False)
            if delay > 0:
                results.append({'index': idx, 'delay': delay})
        return results
