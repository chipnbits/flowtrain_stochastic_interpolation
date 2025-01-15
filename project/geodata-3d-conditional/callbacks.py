import os
import time as clock

import torch
import utils
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.core import LightningModule
from lightning.pytorch.loggers import WandbLogger

import torch.nn.functional as F

from flowtrain.solvers import ODEFlowSolver
from functools import partial

class InferenceCallback(Callback):
    """
    Callback to run inference every n epochs and log the results to WandB.
    Logs summary statistics and 2D slices of the inference results.

    Parameters
    ----------
    save_dir : str
        Directory to save the inference images.
    every_n_epochs : int
        Number of epochs to wait before running inference.
    n_samples : int
        Number of samples to generate for inference.
    n_steps : int
        For adaptive ODE, this is number of steps to save/return.
    t0 : float
        Initial time for the ODE solver. Note that for stochastic interp, small distance from t=0 is better
    tf : float
        Final time for the ODE solver.
    """

    def __init__(self, save_dir, every_n_epochs=10, n_samples=16, n_steps=32, tf=0.999, seed=123):
        super().__init__()
        self.save_dir = save_dir
        self.every_n_epochs = every_n_epochs
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.t0 = 0.001
        self.tf = tf
        # Randomization control
        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed)

    def on_train_epoch_end(self, trainer, pl_module):
        """Handle spacing of epochs and usage of EMA weights for inference."""
        epoch = trainer.current_epoch  # check if on an inference testing epoch
        if epoch % self.every_n_epochs == 0:
            # Use EMA weights if available
            if hasattr(trainer, "ema_callback"):
                trainer.ema_callback.apply_ema_weights(pl_module)
                print("Using EMA weights for inference.")
                self.run_inference(pl_module, epoch, trainer.logger)
                trainer.ema_callback.restore_original_weights(pl_module)
            else:
                self.run_inference(pl_module, epoch, trainer.logger)

    @torch.no_grad()
    def run_inference(self, pl_module: LightningModule, epoch: int, logger: WandbLogger, ATb=None, uncertainty=True):
        print(f"Running inference for epoch {epoch}")
        net = pl_module.net
        emb = pl_module.embedding
        was_training = net.training
        net.eval()
        emb.eval()
        
        self.generator.manual_seed(self.seed)
        X0 = torch.randn(self.n_samples, pl_module.embedding_dim, *pl_module.data_shape, generator=self.generator).to(pl_module.device)
        if ATb is None:
            ATb = torch.zeros_like(X0)
        assert ATb.shape == X0.shape, "ATb must have the same shape as X0 (model data input)"    
        
        # Wrapper function to align the call signature
        def dxdt_cond(x, time, *args, **kwargs):
            return net.forward(x, ATb=ATb, time=time, *args, **kwargs)       
        solver = ODEFlowSolver(model=dxdt_cond, rtol=1e-6)    
        start = clock.time()
        solution = solver.solve(X0, t0=self.t0, tf=self.tf, n_steps=self.n_steps)
        stop = clock.time()
        sol_tf = solution[-1].detach()
        logits = pl_module.decode(sol_tf, return_logits=True).cpu()
        sol_tf = torch.argmax(logits, dim=1)
        

        if uncertainty:
            # Apply softmax to the logits along the class dimension to get probabilities
            probabilities = F.softmax(logits, dim=1)  # shape: [B, C, X, Y, Z]            
            # Sort probabilities along the class dimension
            sorted_probs, _ = torch.sort(probabilities, dim=1, descending=True)            
            # Prominence: difference between the highest and second-highest probability
            prominence = (sorted_probs[:, 0, :, :, :] - sorted_probs[:, 1, :, :, :]).cpu()  # shape: [B, X,Y,Z]            

        
        # -------- Image Processing ----- #
        # Ensure software rendering and offscreen mode
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
        os.environ.pop('DISPLAY', None)

        # Prepare to log a batch of images
        image_files_2d = []
        prominence_files = []        
        image_files_3d = []
        for i in range(self.n_samples):
            # Save 2D images
            save_path_2d = os.path.join(
                self.save_dir, f"infigeo-inferencecb-{epoch:04d}-2d-sample-{i}.png"
            )
            utils.plot_2d_slices(sol_tf[i], save_path=save_path_2d)
            image_files_2d.append(save_path_2d)
            
            if uncertainty:
                # Save prominence maps
                save_path_prominence = os.path.join(
                    self.save_dir, f"infigeo-inferencecb-{epoch:04d}-prominence-sample-{i}.png"
                )
                utils.plot_2d_slices(prominence[i], save_path=save_path_prominence)
                prominence_files.append(save_path_prominence)

            # TODO: Requires xserver or usage of https://docs.pyvista.org/api/utilities/_autosummary/pyvista.start_xvfb.html
            # Cant access without admin privileges
            
            
            # # Save 3D images
            # save_path_3d = os.path.join(
            #     self.save_dir, f"infigeo-inferencecb-{epoch:04d}-3dmodel-sample-{i}.png"
            # )
            # utils.plot_static_views(tensor=sol_tf[i], save_path=save_path_3d)
            # image_files_3d.append(save_path_3d)

        # Log both 2D and 3D images separately in WandB
        images_2d_to_log = {}
        for i, image_path in enumerate(image_files_2d):
            try:
                # Add retry mechanism
                for _ in range(3):  # Attempt three times
                    try:
                        images_2d_to_log[f"2D Sample {i}"] = wandb.Image(image_path)
                        break  # Exit loop if successful
                    except OSError as e:
                        print(f"Retrying image {image_path} due to error: {e}")
                        clock.sleep(0.5)  # Wait before retrying
            except Exception as e:
                print(f"Error logging image {image_path}: {e}")
                
        images_3d_to_log = {
            f"3D Sample {i}": wandb.Image(image_path) for i, image_path in enumerate(image_files_3d)
        }
        
        prominence_to_log = {}
        if uncertainty:
            for i, image_path in enumerate(prominence_files):
                try:
                    for _ in range(3):  # Retry up to three times
                        try:
                            prominence_to_log[f"Prominence Heatmap {i}"] = wandb.Image(image_path)
                            break
                        except OSError as e:
                            print(f"Retrying prominence image {image_path} due to error: {e}")
                            clock.sleep(0.5)
                except Exception as e:
                    print(f"Error logging prominence image {image_path}: {e}")

        logger.experiment.log(
            {
                f"Inference Images": {
                    **images_2d_to_log,
                    **images_3d_to_log,
                    **prominence_to_log,
                },  # Combine 2D and 3D logs
                "Inference Info": {
                    "time_to_solve": stop - start,
                },
            }
        )

        if was_training:
            net.train()
            emb.train()

    def run_manual_inference(self, trainer, pl_module, epoch=0):
        # Ensure EMA weights are used for generating inference outputs if available
        if hasattr(trainer, "ema_callback"):
            trainer.ema_callback.apply_ema_weights(pl_module)
        pl_module.to(torch.device("cuda"))
        self.run_inference(pl_module, epoch, trainer.logger)
        # Restore weights if needed
        if hasattr(trainer, "ema_callback"):
            trainer.ema_callback.restore_original_weights(pl_module)
            
# TODO: Implement EMA Callback, this has not been recently tested            
class EMACallback(Callback):
    """
    Optionally implement EMA Callback to track a shadow set of Exponential Moving Average weights.
    
    """

    def __init__(self, decay=0.9999, start_step=15000, update_every=1, update_on_cpu=False):
        super().__init__()
        self.decay = decay
        self.start_step = start_step
        self.update_every = update_every
        self.update_on_cpu = update_on_cpu
        
        self.shadow = {}
        self.backup = {}
        self.step = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.step += 1
        if self.step < self.start_step:
            return  # Do not apply EMA updates too early
        
        if self.step % self.update_every != 0:
            return  # Update less frequently if desired

        # Update the EMA weights
        with torch.no_grad():
            alpha = self.decay
            for name, param in pl_module.named_parameters():
                if not param.requires_grad:
                    continue

                # Initialize shadow if not present
                if name not in self.shadow:
                    if self.update_on_cpu:
                        self.shadow[name] = param.data.detach().cpu().clone()
                    else:
                        self.shadow[name] = param.data.detach().clone()  # Stays on GPU
                else:
                    if self.update_on_cpu:
                        shadow_cpu = self.shadow[name]
                        param_cpu = param.data.detach().cpu()
                        shadow_cpu = alpha * shadow_cpu + (1.0 - alpha) * param_cpu
                        self.shadow[name] = shadow_cpu
                    else:
                        shadow_gpu = self.shadow[name]  
                        shadow_gpu = alpha * shadow_gpu + (1.0 - alpha) * param.data
                        self.shadow[name] = shadow_gpu

    def on_train_end(self, trainer, pl_module):
        """
        apply EMA weights permanently at the end of training.
        """
        self.apply_ema_weights(pl_module)

    def apply_ema_weights(self, pl_module):
        """
        Save the current weights to self.backup, then replace model params with EMA weights.
        """
        self.backup.clear()
        for name, param in pl_module.named_parameters():
            self.backup[name] = param.data.detach().cpu().clone()

            if name in self.shadow:
                param.data.copy_(self.shadow[name].to(param.device))

    def restore_original_weights(self, pl_module):
        """
        Restore the original (non-EMA) weights that were saved in _backup_and_apply_ema.
        """
        for name, param in pl_module.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name].to(param.device))

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """
        Save the shadow dictionary to the checkpoint so it can be reloaded upon resume.
        """
        checkpoint["ema_shadow"] = {
            name: shadow.clone().cpu() if self.update_on_cpu else shadow.clone()
            for name, shadow in self.shadow.items()
        }
        checkpoint["ema_update_on_cpu"] = self.update_on_cpu

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        """
        Restore the shadow dictionary if it exists in the checkpoint.
        """
        if "ema_shadow" in checkpoint:
            self.shadow = {
                name: shadow.clone().to("cpu" if self.update_on_cpu else pl_module.device)
                for name, shadow in checkpoint["ema_shadow"].items()
            }
        if "ema_update_on_cpu" in checkpoint:
            self.update_on_cpu = checkpoint["ema_update_on_cpu"]
