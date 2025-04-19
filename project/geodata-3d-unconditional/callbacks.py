import os
import time as time

import torch
import utils
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.core import LightningModule
from lightning.pytorch.loggers import WandbLogger

import torch.nn.functional as F

from flowtrain.solvers import ODEFlowSolver

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
                self.run_inference(pl_module, epoch, trainer.logger)
                trainer.ema_callback.restore_original_weights(pl_module)
            else:
                self.run_inference(pl_module, epoch, trainer.logger)

    @torch.no_grad()
    def run_inference(self, pl_module: LightningModule, epoch: int, logger: WandbLogger, uncertainty=True):
        print(f"Running inference for epoch {epoch}")
        net = pl_module.net
        emb = pl_module.embedding
        was_training = net.training
        net.eval()
        emb.eval()
        
        solver = ODEFlowSolver(model=net, rtol=1e-6)
        self.generator.manual_seed(self.seed)
        X0 = torch.randn(self.n_samples, pl_module.embedding_dim, *pl_module.data_shape, generator=self.generator).to(pl_module.device)
        start = time.time()
        solution = solver.solve(X0, t0=self.t0, tf=self.tf, n_steps=self.n_steps)
        stop = time.time()
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
                        time.sleep(0.5)  # Wait before retrying
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
                            time.sleep(0.5)
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

    def __init__(self, decay=0.9999, start_step=15000):
        super().__init__()
        self.decay = decay
        self.start_step = start_step
        self.shadow = {}
        self.step = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.step += 1
        if self.step >= self.start_step:
            # Update the EMA weights
            with torch.no_grad():
                alpha = self.decay
                for name, param in pl_module.named_parameters():
                    if param.requires_grad:
                        if name in self.shadow:
                            self.shadow[name] = alpha * self.shadow[name] + (1 - alpha) * param.data
                        else:
                            self.shadow[name] = param.data.clone()

    def on_validation_start(self, trainer, pl_module):
        # Apply EMA weights for validation
        self.apply_ema_weights(pl_module)

    def on_validation_end(self, trainer, pl_module):
        # Restore original weights after validation
        self.restore_original_weights(pl_module)

    def apply_ema_weights(self, pl_module):
        for name, param in pl_module.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore_original_weights(self, pl_module):
        for name, param in pl_module.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def on_train_end(self, trainer, pl_module):
        # Optionally apply EMA weights as final model weights
        self.apply_ema_weights(pl_module)
