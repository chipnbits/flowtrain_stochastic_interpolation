import warnings
from abc import ABC, abstractmethod

import torch

"""
Implementation of various methods focusing on the mathematical framework from [1]. See Section 4 "Spatially linear interpolants", pg. 27 of [1] for more details.

A Stochastic Interpolator handles the retrieval of optimization objectives for a given interpolant. 
Interpolants are defined by their alpha, beta, and gamma functions and if they are one-sided or two-sided.
One-sided interpolants exclude the latent parameter for noise Z, instead treating the initial point X0 as the noise distribution

    Includes the following interpolants:
    - Linear (one-sided and two-sided)
    - Trig (one-sided and two-sided)
    - Enc-Dec (two-sided)
    - SBDM (one-sided)
    - Mirror (two-sided)

 References
    ----------
    [1] Albergo, Michael S., Nicholas M. Boffi, and Eric Vanden-Eijnden. “Stochastic Interpolants: A Unifying Framework for Flows and Diffusions.” arXiv, November 6, 2023. http://arxiv.org/abs/2303.08797.

"""


def reshape_time(func):
    """
    Decorator function that reshapes the time input tensor to match the dimensions of X0.
    """

    def wrapper(self, T, X0, *args, **kwargs):
        # Ensure T is reshaped correctly: X = Nxd1xd2x... -> T = Nx1x1x...xd1
        if T.dim() == 1:
            # New shape: first dimension is unchanged, the rest are ones to match the dimensions of X0
            shape = [T.size(0)] + [1] * (X0.dim() - 1)
            T = T.view(*shape)
        return func(self, T, X0, *args, **kwargs)

    return wrapper


class StochasticInterpolator:
    """A class for managing stochastic interpolants, requires an interpolant object.
    See Section 4 "Spatially linear interpolants" of [1] for more details.
    """

    def __init__(self, interpolant):
        self.interp = interpolant

    def __repr__(self):
        return f"StochasticInterpolator({self.interp})"

    def __str__(self):
        return f"StochasticInterpolator({self.interp})"

    def one_sided_handler(func):
        """
        This decorator is used to handle one-sided interpolants in the `Interpolants` class.
        Ensures that latent variable Z is not accidentally passed to one-sided interpolants and
        that latent Z is provided for two-sided interpolants.
        """

        def wrapper(self, T, X0, X1, Z=None, *args, **kwargs):
            # Check if Z is needed and provided
            if not self.interp.one_sided and Z is None:
                raise ValueError("Z must be provided for two-sided interpolants")
            # Warn if Z is provided but the interpolant is one-sided
            if self.interp.one_sided and Z is not None:
                warnings.warn(
                    "Z was provided for a one-sided interpolant which does not use it",
                    UserWarning,
                )
            return func(self, T, X0, X1, Z, *args, **kwargs)

        return wrapper

    @reshape_time
    @one_sided_handler
    def flow_objective(self, T, X0, X1, Z=None):
        """
        Sample the interpolant at a given time T with end points X0, X1.
        For two-sided interpolants, a noise vector Z must be provided, for onesided, no Z is needed.
        Returns the required parameters for the flow objective, see equation (2.13) in [1].

        Parameters
        ----------
        T : Tensor, shape (N,)
            A 1D tensor of time values for each of N samples
        X0 : Tensor, shape (N, C, spatial_dims...)
            The starting N sample points from distribution rho_0
        X1 : Tensor, shape (N, C, spatial_dims...)
            The ending N sample points from distribution rho_1
        Z : Tensor, shape (N, C, spatial_dims...), optional
            The latent noise tensor, required for two-sided interpolants.

        Returns
        ----------
        XT : Tensor, shape (N, C, spatial_dims...)
            The interpolated points at time T conditioned on X0, X1, and Z
            See equation (2.1) in [1]
        BT : Tensor, shape (N, C, spatial_dims...)
            The flow velocity tensor at time T conditioned on X0, X1, and Z
            See equation (2.10) in [1]

        Raises:
            AssertionError: If the shapes of X0 and X1 do not match, or if the shape of Z does not match X0 and X1.
        """
        # Ensure that all required inputs have the correct shape
        assert X0.shape == X1.shape, "Shapes of X0 and X1 must match"
        if Z is not None:
            assert Z.shape == X0.shape, "Shape of Z must match X0 and X1"

        # Compute interpolated points and their derivatives
        XT = self.get_XT(T, X0, X1, Z)
        BT = self.get_BT(T, X0, X1, Z)
        return XT, BT

    @reshape_time
    @one_sided_handler
    def denoising_objective(self, T, X0, X1, Z=None):
        """
        Compute the denoising objective for the interpolant at a given time T with end points X0, X1.
        Returns the required parameters for the denoising objective (2.19) in [1].

        Parameters
        ----------
        T : Tensor, shape (N,)
            A 1D tensor of time values for each of N samples
        X0 : Tensor, shape (N, C, spatial_dims...)
            The starting N sample points from distribution rho_0
        X1 : Tensor, shape (N, C, spatial_dims...)
            The ending N sample points from distribution rho_1
        Z : Tensor, shape (N, C,spatial_dims...), optional
            The latent noise tensor, required for two-sided interpolants.

        Returns
        ----------
        XT : Tensor, shape (N, C, spatial_dims...)
            The interpolated points at time T conditioned on X0, X1, and Z
            See equation (2.1) in [1]
        Z : Tensor, shape (N, C, spatial_dims...)
            The denoising objective tensor at time T
            See equation (2.19) in [1]

        Raises
        ----------
        AssertionError: If the shapes of X0, X1, and Z (if provided) do not match.
        """

        XT = self.get_XT(T, X0, X1, Z)
        if self.interp.one_sided:
            Z = X0
        return XT, Z

    @reshape_time
    @one_sided_handler
    def get_XT(self, T, X0, X1, Z=None):
        """
        Get tensor of xt values for sampled interpolant parameters.

        Parameters
        ----------
        T : Tensor, shape (N,)
            A 1D tensor of time values for each of N samples
        X0 : Tensor, shape (N, C, spatial_dims...)
            The starting N sample points from distribution rho_0
        X1 : Tensor, shape (N, C, spatial_dims...)
            The ending N sample points from distribution rho_1
        Z : Tensor, shape (N, C, spatial_dims...), optional
            The latent noise tensor, required for two-sided interpolants.

        Returns
        ----------
        XT : Tensor, shape (N, C, spatial_dims...)
            The interpolated points at time T conditioned on X0, X1, and Z
        """
        alpha = self.interp.alpha(T)
        beta = self.interp.beta(T)
        XT = alpha * X0 + beta * X1

        if Z is not None:
            gamma = self.interp.gamma(T)
            XT += gamma * Z
        return XT

    @reshape_time
    @one_sided_handler
    def get_BT(self, T, X0, X1, Z=None):
        """
        Get the flow velocity tensor for sampled interpolant parameters

        Parameters
        ----------
        T : Tensor, shape (N,)
            A 1D tensor of time values for each of N samples
        X0 : Tensor, shape (N, C, spatial_dims...)
            The starting N sample points from distribution rho_0
        X1 : Tensor, shape (N, C, spatial_dims...)
            The ending N sample points from distribution rho_1
        Z : Tensor, shape (N, C, spatial_dims...), optional
            The latent noise tensor, required for two-sided interpolants.

        Returns
        ----------
        BT : Tensor, shape (N, C, H, W)
            The velocity at time T at position XT conditioned on X0, X1, and Z
        """
        alpha_dot = self.interp.alpha_dot(T)
        beta_dot = self.interp.beta_dot(T)
        BT = alpha_dot * X0 + beta_dot * X1

        if Z is not None:
            gamma_dot = self.interp.gamma_dot(T)
            BT += gamma_dot * Z
        return BT

    @reshape_time
    def get_BT_from_score(self, T, VT, ST):
        """
        Get the flow velocity tensor for sampled interpolant parameters
        """
        gamma = self.interp.gamma(T)
        gamma_dot = self.interp.gamma_dot(T)
        return VT - gamma_dot * gamma * ST

    @reshape_time
    def get_ST(self, T, Z):
        """
        Get the score of distribution for sampled interpolant parameters.
        See equation (2.14) in [1].

        Parameters
        ----------
        T : Tensor, shape (N,)
            A 1D tensor of time values for each of N samples
        Z : Tensor, shape (N, C, spatial_dims...), optional
            The latent noise tensor, required for two-sided interpolants.

        Returns
        ----------
        ST : Tensor, shape (N, C, spatial_dims...)
            The score at time T at position XT conditioned on X0, X1, and Z
        """
        if self.interp.one_sided:
            # One sided case with gaussian rho_0 uses alpha for score
            gamma = self.interp.alpha(T)
        else:
            # Two sided case with latent Z uses gamma for score
            gamma = self.interp.gamma(T)
        return -(gamma ** (-1)) * Z

    @reshape_time
    def get_VT(self, T, X0, X1):
        """
        Get the velocity tensor for sampled interpolant parameters.

        Parameters
        ----------
        T : Tensor, shape (N,)
            A 1D tensor of time values for each of N samples
        X0 : Tensor, shape (N, C, spatial_dims...)
            The starting N sample points from distribution rho_0
        X1 : Tensor, shape (N, C, spatial_dims...)
            The ending N sample points from distribution rho_1

        Returns
        ----------
        VT : Tensor, shape (N, C, spatial_dims...)
            The mean velocity at time T at position XT conditioned on X0, X1
        """
        alpha_dot = self.interp.alpha_dot(T)
        beta_dot = self.interp.beta_dot(T)
        return alpha_dot * X0 + beta_dot * X1


class BaseInterpolant(ABC):
    """
    Stochastic Interpolant class for defining linear interpolants based on alpha, beta, gamma schemes.
    """

    def __init__(self, one_sided=False):
        self.one_sided = one_sided

    def __repr__(self):
        return f"{type(self).__name__}(one_sided={self.one_sided})"

    def __str__(self):
        return f"{type(self).__name__}(one_sided={self.one_sided})"

    @abstractmethod
    def alpha(self, t):
        """
        Compute the alpha value of the interpolant at time t.

        Args:
            t (float): Time value.

        Returns:
            float: The alpha value.
        """
        pass

    @abstractmethod
    def beta(self, t):
        """
        Compute the beta value of the interpolant at time t.

        Args:
            t (float): Time value.

        Returns:
            float: The beta value.
        """
        pass

    @abstractmethod
    def gamma(self, t):
        """
        Compute the gamma value of the interpolant at time t.

        Args:
            t (float): Time value.

        Returns:
            float: The gamma value.
        """
        pass

    @abstractmethod
    def alpha_dot(self, t):
        """
        Compute the derivative of alpha with respect to time at time t.

        Args:
            t (float): Time value.

        Returns:
            float: The derivative of alpha.
        """
        pass

    @abstractmethod
    def beta_dot(self, t):
        """
        Compute the derivative of beta with respect to time at time t.

        Args:
            t (float): Time value.

        Returns:
            float: The derivative of beta.
        """
        pass

    @abstractmethod
    def gamma_dot(self, t):
        """
        Compute the derivative of gamma with respect to time at time t.

        Args:
            t (float): Time value.

        Returns:
            float: The derivative of gamma.
        """
        pass

    def is_one_sided(self):
        """
        Check if the interpolant is one-sided.

        Returns:
            bool: True if the interpolant is one-sided, False otherwise.
        """
        return self.one_sided


class LinearInterpolant(BaseInterpolant):
    """Linear interpolant:
    Params: gamma_a (float)
    - alpha(t) = 1 - t
    - beta(t) = t
    - gamma(t) = sqrt(gamma_a*t*(1-t))
    """

    def __init__(self, one_sided=False, gamma_a=2.0):
        super().__init__(one_sided)
        self.gamma_a = gamma_a

    def alpha(self, t):
        return 1 - t

    def beta(self, t):
        return t

    def gamma(self, t):
        if self.one_sided:
            return torch.zeros_like(t)
        return torch.sqrt(self.gamma_a * t * (1 - t))

    def alpha_dot(self, t):
        return -torch.ones_like(t)

    def beta_dot(self, t):
        return torch.ones_like(t)

    def gamma_dot(self, t):
        if self.one_sided:
            return torch.zeros_like(t)
        a = self.gamma_a
        return 0.5 * a * (1 - 2 * t) / torch.sqrt(a * t * (1 - t))


class TrigInterpolant(BaseInterpolant):
    """Trig interpolant:
    Params: one_sided (bool)
            gamma_a (float)
    - alpha(t) = cos(pi*t/2)
    - beta(t) = sin(pi*t/2)
    - gamma(t) = sqrt(gamma_a*t*(1-t))
    """

    def __init__(self, one_sided=False, gamma_a=2.0):
        super().__init__(one_sided)
        self.gamma_a = gamma_a

    def alpha(self, t):
        return torch.cos(torch.pi * t / 2)

    def beta(self, t):
        return torch.sin(torch.pi * t / 2)

    def gamma(self, t):
        if self.one_sided:
            return torch.zeros_like(t)
        return torch.sqrt(self.gamma_a * t * (1 - t))

    def alpha_dot(self, t):
        return -torch.pi / 2 * torch.sin(torch.pi * t / 2)

    def beta_dot(self, t):
        return torch.pi / 2 * torch.cos(torch.pi * t / 2)

    def gamma_dot(self, t):
        if self.one_sided:
            return torch.zeros_like(t)
        a = self.gamma_a
        return 0.5 * a * (1 - 2 * t) / torch.sqrt(a * t * (1 - t))


class EncDecInterpolant(BaseInterpolant):
    """Enc-Dec interpolant:
    Params: None
    - alpha(t) = cos^2(pi*t) if t < 1/2, otherwise 0
    - beta(t) = cos^2(pi*t) if t > 1/2, otherwise 0
    - gamma(t) = sin^2(pi*t)
    """

    def __init__(self):
        super().__init__(one_sided=False)

    def alpha(self, t):
        return torch.where(t < 0.5, torch.cos(torch.pi * t) ** 2, torch.zeros_like(t))

    def beta(self, t):
        return torch.where(t > 0.5, torch.cos(torch.pi * t) ** 2, torch.zeros_like(t))

    def gamma(self, t):
        return torch.sin(torch.pi * t) ** 2

    def alpha_dot(self, t):
        return torch.where(
            t < 0.5, -torch.pi * torch.sin(2 * torch.pi * t), torch.zeros_like(t)
        )

    def beta_dot(self, t):
        return torch.where(
            t > 0.5, -torch.pi * torch.sin(2 * torch.pi * t), torch.zeros_like(t)
        )

    def gamma_dot(self, t):
        return torch.pi * torch.sin(2 * torch.pi * t)


class SBDMInterpolant(BaseInterpolant):
    """SBDM interpolant:
    Params: None
    - alpha(t) = sqrt(1 - t^2)
    - beta(t) = t
    - gamma(t) = 0
    """

    def __init__(self):
        super().__init__(one_sided=True)

    def alpha(self, t):
        return torch.sqrt(1 - t**2)

    def beta(self, t):
        return t

    def gamma(self, t):
        return torch.zeros_like(t)

    def alpha_dot(self, t):
        return -t / torch.sqrt(1 - t**2)

    def beta_dot(self, t):
        return torch.ones_like(t)

    def gamma_dot(self, t):
        return torch.zeros_like(t)


class MirrorInterpolant(BaseInterpolant):
    """Mirror interpolant:
    Params: gamma_a (float)
    - alpha(t) = 0
    - beta(t) = 1
    - gamma(t) = sqrt(gamma_a*t*(1-t))
    """

    def __init__(self, gamma_a=2.0):
        super().__init__(one_sided=False)
        self.gamma_a = gamma_a

    def alpha(self, t):
        return torch.zeros_like(t)

    def beta(self, t):
        return torch.ones_like(t)

    def gamma(self, t):
        a = self.gamma_a
        return torch.sqrt(a * t * (1 - t))

    def alpha_dot(self, t):
        return torch.zeros_like(t)

    def beta_dot(self, t):
        return torch.zeros_like(t)

    def gamma_dot(self, t):
        a = self.gamma_a
        return 0.5 * a * (1 - 2 * t) / torch.sqrt(a * t * (1 - t))


if __name__ == "__main__":
    """Plot the interpolants and derivatives for visual verification"""
    import matplotlib.pyplot as plt

    one_sided = True
    interpolants = [
        LinearInterpolant(one_sided=one_sided),
        TrigInterpolant(one_sided=one_sided),
        EncDecInterpolant(),
        SBDMInterpolant(),
        MirrorInterpolant(),
    ]
    fig, axs = plt.subplots(2, 5)
    for i in range(5):
        interpolator = StochasticInterpolator(interpolant=interpolants[i])
        t = torch.linspace(0, 1, 100)
        alpha = interpolator.interp.alpha(t)
        beta = interpolator.interp.beta(t)
        gamma = interpolator.interp.gamma(t)
        axs[0, i].plot(t, alpha, label="alpha")
        axs[0, i].plot(t, beta, label="beta")
        axs[0, i].plot(t, gamma, label="gamma")
        axs[0, i].legend()
        axs[0, i].set_title(type(interpolants[i]).__name__)

    for i in range(5):
        interpolator = StochasticInterpolator(interpolant=interpolants[i])
        t = torch.linspace(0, 1, 100)
        alpha = interpolator.interp.alpha_dot(t)
        beta = interpolator.interp.beta_dot(t)
        gamma = interpolator.interp.gamma_dot(t)
        axs[1, i].plot(t, alpha, label="alpha")
        axs[1, i].plot(t, beta, label="beta")
        axs[1, i].plot(t, gamma, label="gamma")
        axs[1, i].legend()
        # Set title
        axs[1, i].set_title(type(interpolants[i]).__name__)
    plt.show()
