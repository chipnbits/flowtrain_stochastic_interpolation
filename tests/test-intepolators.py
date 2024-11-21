from flowtrain.interpolation import *
import matplotlib.pyplot as plt
import torch


"""
Plot the interpolants and derivatives for visual verification
Should match the plots from the paper, shown in paper-vals.png
"""
import matplotlib.pyplot as plt

one_sided = True
interpolants = [
    LinearInterpolant(one_sided=one_sided),
    TrigInterpolant(one_sided=one_sided),
    EncDecInterpolant(),
    SBDMInterpolant(),
    MirrorInterpolant(),
]
fig, axs = plt.subplots(2, 5, figsize=(15, 10)) 
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

# Adjust layout and display
plt.tight_layout()
plt.show()