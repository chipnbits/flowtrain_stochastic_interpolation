""" Collection of solving mechanisms for the stoch-interp model.

 References
    ----------
    [1] Albergo, Michael S., Nicholas M. Boffi, and Eric Vanden-Eijnden. “Stochastic Interpolants: A Unifying Framework for Flows and Diffusions.” arXiv, November 6, 2023. http://arxiv.org/abs/2303.08797.
"""

import torch
from torchdiffeq import odeint

from flowtrain.interpolation import BaseInterpolant


class ODEFlowSolver:
    """
    An adaptive dopri5 based ODE Solver class for torch data and models.
    Provides integration of the velocity field b_t = model(Xt, T)
    for a given initial condition Xt and time T.

    Parameters
    ----------
    model : torch.nn.Module
        A torch model that returns the time derivative of the ODE. model(Xt, T) should return dx/dt or a batch
    atol : float
        Absolute error tolerance for the adaptive ODE solver
    rtol : float
        Relative error tolerance for the adaptive ODE solver

    References
    ----------
    based on Equation (2.32) in [1]

    """

    def __init__(self, model, atol=1e-6, rtol=1e-6):
        self.model = model
        self.atol = atol
        self.rtol = rtol

    def solve(self, X0, frozen_mask=None, t0=0.0, tf=1.0, n_steps=32):
        """
        Solve the ODE for the given initial condition, time points, and mask.

        Parameters
        ----------
        X0 : torch.Tensor
            The initial condition for the ODE solver
        frozen_mask : torch.Tensor
            The mask to apply for frozen dimensions: dx/dt = 0
        t0 : float
            The initial time point for the ODE solver
        tf : float
            The final time point for the ODE solver
        n_steps : int
            The number of steps for the ODE solver
        """

        # Create the time vector and ensure it's on the same device as X0
        t = torch.linspace(t0, tf, n_steps).to(X0.device)

        # Ensure X0 is in the right shape (add batch dimension if necessary)
        if len(X0.shape) == 3:
            X0 = X0.unsqueeze(0)

        # Define the ODE function, considering batch and time broadcasting
        def ode_func(t, XT):
            with torch.no_grad():
                T = torch.full((XT.size(0),), t.item(), device=XT.device)
                # Calculate a value for dxdt
                dxdt = self.model(XT, T)
                if frozen_mask is not None:
                    # freeze masked values
                    dxdt[..., frozen_mask] = 0
                return dxdt

        # Solve the ODE
        return odeint(ode_func, X0, t, atol=self.atol, rtol=self.rtol, method="dopri5")


class ODEOneSidedDenoisingSolver:
    """
    Adaptive dopri5 ODE Solver class for torch data and modeled **denoising** objective.
    This solver is used for solving one sided stochastic interpolation problems
    based on a learned denoiser.

    Parameters
    ----------
    model : torch.nn.Module
        A torch model that returns the time derivative of the ODE. model(Xt, T) should return dx/dt or a batch
    interpolant : BaseInterpolant
        The one-sided interpolant used to train the objective, i.e. LinearInterpolant(one_sided=True)
    atol : float
        Absolute error tolerance for the adaptive ODE solver
    rtol : float
        Relative error tolerance for the adaptive ODE solver

    References
    ----------
    based on ODE Equation (6.7) in [1]
    """

    def __init__(self, model, interpolant: BaseInterpolant, atol=1e-6, rtol=1e-6):
        self.model = model
        self.interp = interpolant
        self.atol = atol
        self.rtol = rtol
        self.model
        
        self.validate_interpolant(interpolant)

    def validate_interpolant(self, interpolant):
        # Validate that the interpolant is one-sided and of the correct type
        assert isinstance(
            interpolant, BaseInterpolant
        ), "ODEOneSidedDenoisingSolver requires a BaseInterpolant"
        assert (
            self.interp.is_one_sided()
        ), "ODEOneSidedDenoisingSolver requires a one-sided interpolant"

    def solve(self, X0, t0=0.0, tf=1.0, n_steps=32):
        """Solve the ODE for the given initial condition and time points."""

        # Create the time vector and ensure it's on the same device as X0
        t = torch.linspace(t0, tf, n_steps, dtype=torch.float64).to(X0.device)

        # Ensure X0 is in the right shape (add batch dimension if necessary)
        if len(X0.shape) == 3:
            X0 = X0.unsqueeze(0)

        # Define the ODE function, considering batch and time broadcasting
        def ode_func(t, XT):
            # Create a vector from a scalar
            T = torch.full((XT.size(0),), t.item(), device=XT.device)

            with torch.no_grad():  # Ensure no gradients are computed
                eta_z = self.model(XT, T)
                # Time is same scross batch, a scalar input will suffice
                alpha = self.interp.alpha(t)
                beta = self.interp.beta(t)
                alpha_dot = self.interp.alpha_dot(t)
                beta_dot = self.interp.beta_dot(t)
                # Equation (6.7) from [1]
                dxdt = alpha_dot * eta_z + (beta_dot / beta) * (XT - alpha * eta_z)
                return dxdt

            # TODO: It is suggested to use (5.14) discrete equation to finish the final denoising step. e.g. .99 -> 1.00

        return odeint(ode_func, X0, t, atol=self.atol, rtol=self.rtol, method="dopri5")


# TODO: SDE solving method is missing something. It is not clear what Wt is supposed to be from equation (6.7) in [1]
# This SDE and the ODE class can be combined into one, with the ODE solution bein a special case for epsilon = 0.
class SDEOneSidedDenoisingSolver:
    """SDE Solver class for torch data and models. For generating samples using the denoise objective.

    Parameters
    ----------
    model : torch.nn.Module
        A torch model that returns the time derivative of the ODE. model(Xt, T) should return dx/dt or a batch
    interpolant : BaseInterpolant
        The one-sided nterpolant used to train the objective, i.e. LinearInterpolant(one_sided=True)

    References
    ----------
    based on SDE Equation (6.7) in [1]
    """

    def __init__(self, model, interpolant, epsilon, atol=1e-6, rtol=1e-6):
        self.model = model
        self.interp = interpolant
        # allow for epsilon to be passed function or scalar
        if callable(epsilon):
            self.epsilon = epsilon
        else:
            self.epsilon = lambda t: epsilon
        self.atol = atol
        self.rtol = rtol
        self.model

    def solve(self, X0, t0=0.0, tf=1.0, n_steps=32):
        """Solve the ODE for the given initial condition and time points."""
        assert (
            self.interp.one_sided
        ), "ODEOneSidedDenoisingSolver requires a one-sided interpolant"

        # Create the time vector and ensure it's on the same device as X0
        t = torch.linspace(t0, tf, n_steps).to(X0.device)

        # Ensure X0 is in the right shape (add batch dimension if necessary)
        if len(X0.shape) == 3:
            X0 = X0.unsqueeze(0)

        def ode_func(t, XT):
            # Create a vector from a scalar
            T = torch.full((XT.size(0),), t.item(), device=XT.device)

            with torch.no_grad():  # Ensure no gradients are computed
                eta_z = self.model(XT, T)
                # Time is same scross batch, a scalar input will suffice
                alpha = self.interp.alpha(t)
                beta = self.interp.beta(t)
                alpha_dot = self.interp.alpha_dot(t)
                beta_dot = self.interp.beta_dot(t)
                # Equation 6.7 denoiser is all you need Albergo, et.al
                # “Stochastic Interpolants: A Unifying Framework for Flows and Diffusions.”
                # arXiv, November 6, 2023. http://arxiv.org/abs/2303.08797.
                dxdt = alpha_dot * eta_z + beta_dot / beta * (XT - alpha * eta_z)

                # Additonal non-deterministic term for SDE based off of epsilon value.
                score = -eta_z / alpha
                eps = self.epsilon(t)
                stochastic_term = eps * score + torch.randn_like(XT) * torch.sqrt(
                    2 * eps
                )

                return dxdt + stochastic_term

        # TODO: Eldad suggest using a simpler solver for the ODE
        # Solve the ODE
        return odeint(
            ode_func, X0, t, atol=self.atol, rtol=self.rtol, method="adaptive_heun"
        )


def odeSol_RK4(x0, model, nsteps=100, Tf=1.0):
    """A simple RK4 Solver by Eldad Haber"""

    traj = torch.zeros(nsteps, *x0.shape, device=x0.device)
    traj[0, ...] = x0
    t = torch.zeros(x0.shape[0], device=x0.device)
    with torch.no_grad():
        h = Tf / nsteps
        for i in range(nsteps - 1):
            xt = traj[i, ...]
            k1 = model(xt, t)
            k2 = model(xt + h * k1 / 2, t + h / 2)
            k3 = model(xt + h * k2 / 2, t + h / 2)
            k4 = model(xt + h * k3, t + h)

            traj_next = traj[i, ...] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            traj[i + 1, ...] = traj_next
            t = t + h
            # print('%3.2e     %3.2e'%(t[0], (traj[i+1, :, :, :]-traj[0, :, :, :]).norm()))
    return traj
