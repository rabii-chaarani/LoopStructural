from LoopStructural.interpolators import FiniteDifferenceInterpolator as FDI
from LoopStructural.interpolators import PiecewiseLinearInterpolator as PLI
from LoopStructural.interpolators import P2Interpolator as P2
from LoopStructural.interpolators import SurfeRBFInterpolator as surfer

import numpy as np

class Interpolate:
    
    def __init__(self, origin = None, maximum=None):
        self._origin = origin
        self._maximum = maximum
        self._gradient_constraints = None
        self._gradient_norm_constraints = None
        self._value_constraints = None
        self._interface_constraints = None

    @property
    def gradient_constraints(self) -> np.ndarray:
        return self._gradient_constraints

    @gradient_constraints.setter
    def gradient_constraints(self, value: np.ndarray):
        self._gradient_constraints = value

    @property
    def gradient_norm_constraints(self) -> np.ndarray:
        return self._gradient_norm_constraints

    @gradient_norm_constraints.setter
    def gradient_norm_constraints(self, value: np.ndarray):
        self._gradient_norm_constraints = value

    @property
    def value_constraints(self) -> np.ndarray:
        return self._value_constraints

    @value_constraints.setter
    def value_constraints(self, value: np.ndarray):
        self._value_constraints = value

    @property
    def interface_constraints(self) -> np.ndarray:
        return self._interface_constraints

    @interface_constraints.setter
    def interface_constraints(self, value: np.ndarray):
        self._interface_constraints = value   
    
    @property
    def origin(self) -> np.ndarray:
        return self._origin
    
    @origin.setter
    def origin(self, origin: np.ndarray) -> None:
        if isinstance(origin, np.ndarray):
            self._origin = origin
        else:
            raise TypeError("origin must be a numpy.ndarray")

    @property
    def maximum(self) -> np.ndarray:
        return self._maximum

    @maximum.setter
    def maximum(self, maximum: np.ndarray) -> None:
        if isinstance(maximum, np.ndarray):
            self._maximum = maximum
        else:
            raise TypeError("maximum must be a numpy.ndarray")

    
    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        """Evaluate the interpolator as a list of locations.

        Parameters
        ----------
        xyz : numpy.ndarray
            Nx3 array of points to evaluate the interpolator.
        """
        return self.interpolator.evaluate_value(xyz)