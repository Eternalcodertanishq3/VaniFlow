"""
DEPRECATED: Wav2Lip has been replaced by MuseTalk (MIT licensed).

Wav2Lip uses a research-only license that is not commercially safe.
MuseTalk (Tencent/Lyra Lab) is MIT-licensed and provides equivalent
lip-sync capabilities.

See: vaaniflow.lipsync.musetalk_generator
"""

import warnings

warnings.warn(
    "Wav2LipGenerator is deprecated and has been replaced by MuseTalkGenerator. "
    "Import from vaaniflow.lipsync.musetalk_generator instead.",
    DeprecationWarning,
    stacklevel=2,
)


class Wav2LipGenerator:
    """Deprecated — use MuseTalkGenerator instead."""

    def __init__(self, **kwargs):  # noqa: ANN003
        raise NotImplementedError(
            "Wav2LipGenerator has been replaced by MuseTalkGenerator. "
            "Import from vaaniflow.lipsync.musetalk_generator instead."
        )
