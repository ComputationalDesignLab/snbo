import numpy as np
from .analytical_problems import *
from .rover import Rover
from .mujoco_env import HalfCheetah

class TestProblem():

    def __init__(self, func_name, dim, input_norm=False, negate=False):
        """
            Class for defining and evaluating a test problem

            Note: it is assumed that objective function is to be minimized

            Parameters
            ----------
            func_name: name of the test problem, str

            dim: number of dimensions, int

            input_norm: whether the given input is normalized to [0,1] space or not. If
                true, then the input will be converted to original space first, then
                function will be evaluated, bool, default=False
            
            negate: whether to negate the function value or not, set this to True when
                optimizer requries a maximization problem, bool, default=False
        """

        assert isinstance(func_name, str), "function name should be a string"
        assert isinstance(dim, int), "dimension should be an integer"
        assert isinstance(input_norm, bool), "input_norm should be a boolean value"

        if func_name.lower() == "hartmann6d":
            self.func = Hartmann6D()

        elif func_name.lower() == "ackley":
            self.func = Ackley(dim)

        elif func_name.lower() == "levy":
            self.func = Levy(dim)
        
        elif func_name.lower() == "rastrigin":
            self.func = Rastrigin(dim)

        elif func_name.lower() == "rover":
            self.func = Rover(dim)

        elif func_name.lower() == "halfcheetah":
            self.func = HalfCheetah()

        else:
            raise ValueError("Incorrect test function name, possible values are: 'hartmann6d', 'ackley', 'levy', 'rastrigin', 'rover', 'halfcheetah'")

        self.input_norm = input_norm
        self.negate = negate
        self.dim = self.func.dim
        self.lb = self.func.lb
        self.ub = self.func.ub

    def __call__(self, x):
        """
            Evalaute the function for given input x

            Parameters
            ----------
            x: np.ndarray
                1D/2D numpy array of shape (dim,) or (n_samples,dim)

            Returns
            -------
            y: np.ndarray
                1D/2D numpy array of shape (n_samples,1) or (1,) 
                containing the function value for each input sample
        """

        assert isinstance(x, np.ndarray), "Input x must be a numpy array"

        ndim = x.ndim

        if ndim == 1:
            x = x.reshape(1,-1)

        if self.input_norm:
            x = self.lb + x * (self.ub - self.lb)

        # assert np.all(x <= self.ub) and np.all(x >= self.lb), "Input is outside the bounds"

        y = self.func(x) 
        
        if self.negate:
            y = -y

        y = y.reshape(-1,1)
        
        if ndim == 1:
            y = y.reshape(-1,)
        
        return y

    def getUnnormalizedX(self, x):
        """
            Method to get unnormalized X for given normalized input x.
            Note: it is assumed that x is normalized to [0,1] space

            Parameters
            ----------
            x: np.ndarray
                1D/2D numpy array of shape (dim,) or (n_samples,dim) containing normalized input
        """

        if self.input_norm:
            return self.lb + x * (self.ub - self.lb)
        else:
            raise ValueError("Input is not normalized, cannot get unnormalized X")
