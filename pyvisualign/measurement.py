from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


@dataclass
class Measurement:
    """A measurement line with its calculated length."""
    start: npt.NDArray[np.float64]
    end: npt.NDArray[np.float64]
    length: float
