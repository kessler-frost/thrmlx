"""MLX-native Ising-model sampling."""

from thrmlx.model import Ising
from thrmlx.sampling import sample
from thrmlx.schedule import Clamp, SamplingSchedule

__version__ = "0.1.0"

__all__ = ["Clamp", "Ising", "SamplingSchedule", "__version__", "sample"]
