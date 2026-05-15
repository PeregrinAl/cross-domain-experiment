"""Feature extraction and embedding representations."""

from nstad_bench.representations.base import BaseKernel, BaseRepresentation
from nstad_bench.representations.carla_ssl import CARLA_SSL
from nstad_bench.representations.cwt_morlet import CWT_Morlet
from nstad_bench.representations.log_stft import LogSTFT
from nstad_bench.representations.raw_signal import RawSignal

__all__ = [
    "BaseRepresentation",
    "BaseKernel",
    "RawSignal",
    "LogSTFT",
    "CWT_Morlet",
    "CARLA_SSL",
]
