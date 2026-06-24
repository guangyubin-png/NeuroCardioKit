"""
OSEA - Open Source ECG Analysis

Hamilton-Tompkins QRS detection and beat classification for 200 Hz ECG data.

Usage:
    from neurocardiokit.osea import OseaEngine

    engine = OseaEngine()
    engine.reset()
    for sample in ecg_data:
        delay = engine.detect(sample)
        if delay > 0:
            print(f"Beat: {engine.beat_type_label}")
"""

from neurocardiokit.osea._osea_bridge import (
    OseaEngine,
    load_osea,
    ANNOTYPE_LABELS,
    BEAT_CLASS_NAMES,
)
