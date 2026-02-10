"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import copy

import torch
from torchdiffeq import odeint
from torchcfm.optimal_transport import *
class FlowMatchingInterpolant:
    """Interpolant for simple Gaussian flow matching.

    - Constructs noisy samples from clean samples during training.
    - Implements sampling loop from random noise (or prior) during inference.
    - Also supports classifier-free guidance for DiT denoisers.

    Adapted from: https://github.com/microsoft/protein-frame-flow

    Args:
        min_t (float): Minimum time step to sample during training.
        corrupt (bool): Whether to corrupt samples during training.
        num_timesteps (int): Number of timesteps to integrate over.
        self_condition (bool): Whether to use self-conditioning during denoising.
        self_condition_prob (float): Probability of using self-conditioning during training.
        device (str): Device to run on.
    """

    def __init__(
        self,
        min_t: int = 1e-2,
        corrupt: bool = True,
        num_timesteps: int = 100,
        self_condition: bool = False,
        self_condition_prob: float = 0.5,
        device: str = "cpu",
    ):
        self.min_t = min_t
        self.corrupt = corrupt
        self.num_timesteps = num_timesteps
        self.self_condition = self_condition
        self.self_condition_prob = self_condition_prob
        self.device = device
        self.OTplan = OTPlanSampler("exact")
    def _sample_t(self, batch_size):
        t = torch.rand(batch_size, device=self.device)
        return t * (1 - 2 * self.min_t) + self.min_t

    def _centered_gaussian(self, batch_size, num_tokens, emb_dim=3):
        noise = torch.randn(batch_size, num_tokens, emb_dim, device=self.device)
        return noise - torch.mean(noise, dim=-2, keepdims=True)

    def _corrupt_x(self, x_1, t, token_mask, diffuse_mask):
        # x_0 = self._centered_gaussian(*x_1.shape)
        # x_0, x_1 = self.OTplan.sample_plan(x_0 * diffuse_mask[..., None], x_1 * diffuse_mask[..., None])
        # x_t = (1 - t[..., None]) * x_0 + t[..., None] * x_1
        # x_t = x_t * diffuse_mask[..., None] + x_1 * (~diffuse_mask[..., None])

        x_0 = self._centered_gaussian(*x_1.shape)
        pi = self.OTplan.get_map(
            x_0 * diffuse_mask[..., None], 
            x_1 * diffuse_mask[..., None]
        )
        i, j = self.OTplan.sample_map(pi, x_0.shape[0], replace=True)
        batch_size = x_0.shape[0]
        x_0_indices = torch.arange(batch_size, device=x_0.device, dtype=torch.long)        
        i_torch = torch.from_numpy(i).to(x_0.device)
        j_torch = torch.from_numpy(j).to(x_0.device)
        
        for target_idx in range(batch_size):
            mask = (j_torch == target_idx)
            if mask.any():
                x_0_indices[target_idx] = i_torch[mask][0]
                        
        x_0_matched = x_0[x_0_indices]
        

    
        x_t = (1 - t[..., None]) * x_0_matched + t[..., None] * x_1
        x_t = x_t * diffuse_mask[..., None] + x_1 * (~diffuse_mask[..., None])
        
        
        return x_t * token_mask[..., None]

    def corrupt_batch(self, batch):
        """Corrupts a batch of data by sampling a time t and interpolating to noisy samples.

        Args:
            batch (dict): Batch of clean data with keys:
                - x_1 (torch.Tensor): Clean data tensor.
                - token_mask (torch.Tensor): True if valid token, False if padding.
                - diffuse_mask (torch.Tensor): True if diffusion is to be performed, False if fixed during denoising.
        """
        noisy_batch = copy.deepcopy(batch)

        # [B, N, d]
        x_1 = batch["x_1"]

        # [B, N]
        token_mask = batch["token_mask"]
        diffuse_mask = batch["diffuse_mask"]
        batch_size, _ = diffuse_mask.shape

        # [B, 1]
        t = self._sample_t(batch_size)[:, None]
        noisy_batch["t"] = t

        # Apply corruptions
        # if self.corrupt:
        #     x_t, x_0, x_1 = self._corrupt_x(x_1, t, token_mask, diffuse_mask)
        # else:
        #     x_t = x_1
        x_t = self._corrupt_x(x_1, t, token_mask, diffuse_mask)
        if torch.any(torch.isnan(x_t)):
            raise ValueError("NaN in x_t during corruption")
        noisy_batch["x_t"] = x_t

        return noisy_batch

    def _x_vector_field(self, t, x_1, x_t):
        # Add numerical stability for t close to 1
        eps = 1e-5
        return (x_1 - x_t) / (1 - t + eps)

    def _x_euler_step(self, d_t, t, x_1, x_t):
        assert d_t > 0
        x_vf = self._x_vector_field(t, x_1, x_t)
        return x_t + x_vf * d_t

    def sample(
        self,
        batch_size,
        num_tokens,
        emb_dim,
        model,
        dataset_idx=None,
        spacegroup=None,
        num_timesteps=None,
        x_0=None,
        x_1=None,
        token_mask=None,
        token_idx=None,
    ):
        """Generates new samples of a specified (B, N, d) using denoiser model.

        Args:
            batch_size (int): Number of samples to generate.
            num_tokens (int): Number of tokens in each sample.
            emb_dim (int): Dimension of each token.
            model (nn.Module): Denoiser model to use.
            dataset_idx (torch.Tensor): Dataset index, used for classifier-free guidance. (B, 1)
            spacegroup (torch.Tensor): Spacegroup, used for classifier-free guidance. (B, 1)
            num_timesteps (int): Number of timesteps to integrate over.
            x_0 (torch.Tensor): Initial sample to start from.
            x_1 (torch.Tensor): Final sample to end at.
            token_mask (torch.Tensor): Mask for valid tokens.
            token_idx (torch.Tensor): Index of each token.

        Returns:
            Dict with keys:
                tokens_traj (list): List of generated samples at each timestep.
                clean_traj (list): List of denoised samples at each timestep.
        """
        # Set-up initial prior samples
        if x_0 is None:
            x_0 = self._centered_gaussian(batch_size, num_tokens, emb_dim)
        if token_mask is None:
            token_mask = torch.ones(batch_size, num_tokens, device=self.device).bool()
        if token_idx is None:
            token_idx = torch.arange(num_tokens, device=self.device, dtype=torch.float32)[
                None
            ].repeat(batch_size, 1)

        # Set-up time
        if num_timesteps is None:
            num_timesteps = self.num_timesteps
        ts = torch.linspace(self.min_t, 1.0, num_timesteps)
        t_1 = ts[0]

        tokens_traj = [x_0]
        clean_traj = []
        x_sc = None
        for t_2 in ts[1:]:
            # Run denoiser model
            x_t_1 = tokens_traj[-1]
            if self.corrupt:
                x = x_t_1
            else:
                if x_1 is None:
                    raise ValueError("Must provide x_1 if not corrupting.")
                x = x_1
            t = torch.ones((batch_size, 1), device=self.device) * t_1
            d_t = t_2 - t_1

            """Default"""
            # Run denoiser model
            with torch.no_grad():
                pred_x_1 = model(
                    x, t, token_mask, x_sc
                )

            # Process model output
            clean_traj.append(pred_x_1)
            if self.self_condition:
                x_sc = pred_x_1
            
            """EDit"""
            # # Run denoiser model
            # with torch.no_grad():
            #     x_sc = model(
            #         x, t, token_mask, None
            #     )

            #     pred_x_1 = model(
            #         x, t, token_mask, x_sc
            #     )
                
            # # Process model output
            # clean_traj.append(pred_x_1)
            # # if self.self_condition:
            # #     x_sc = pred_x_1
                

            # Take reverse step
            x_t_2 = self._x_euler_step(d_t, t_1, pred_x_1, x_t_1)

            tokens_traj.append(x_t_2)
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        x_t_1 = tokens_traj[-1]
        if self.corrupt:
            x = x_t_1
        else:
            if x_1 is None:
                raise ValueError("Must provide x_1 if not corrupting.")
            x = x_1
        t = torch.ones((batch_size, 1), device=self.device) * t_1
        with torch.no_grad():
            pred_x_1 = model(
                x, t, token_mask, x_sc
            )
        clean_traj.append(pred_x_1)
        tokens_traj.append(pred_x_1)

        return {"tokens_traj": tokens_traj, "clean_traj": clean_traj}

    def sample_with_dopri5(
        self,
        batch_size,
        num_tokens,
        emb_dim,
        model,
        # self_condition args are excluded due to complexity with odeint structure.
        num_timesteps=None,
        token_mask=None,
    ):
        """
        Generate samples using Dopri5 ODE solver.
        """
        # --- Initial setup ---
        self.device = next(model.parameters()).device
        x_0 = self._centered_gaussian(batch_size, num_tokens, emb_dim)
        if token_mask is None:
            token_mask = torch.ones(batch_size, num_tokens, device=self.device).bool()

        if num_timesteps is None:
            num_timesteps = self.num_timesteps
        ts = torch.linspace(self.min_t, 1.0, num_timesteps, device=self.device)

        # --- 2. Define ODE Dynamics function (located inside sample method) ---
        # This class replaces the core logic of the original for loop.
        class ODEFunct(torch.nn.Module):
            def __init__(self, model, token_mask):
                super().__init__()
                self.model = model
                self.token_mask = token_mask
                # self_condition needs to pass state between steps,
                # which is complex in simple odeint structure.
                # Thus, x_sc is fixed to None in this example.

            def forward(self, t, x_t):
                # odeint treats t as scalar, convert to tensor for model input
                t_input = torch.ones(x_t.shape[0], 1, device=x_t.device) * t

                # Call model to predict original x_1
                with torch.no_grad():
                    pred_x_1 = self.model(x_t, t_input, token_mask, x_sc=None)

                # Compute Flow Matching vector field (velocity): v = (x_1 - x_t) / (1 - t)
                velocity = (pred_x_1 - x_t) / (1. - t + 1e-5)  # Prevent division by zero
                return velocity

        ode_func = ODEFunct(model, token_mask)

        # --- 3. Call ODE solver (replaces original for loop with this single line) ---
        tokens_traj_tensor = odeint(
            ode_func,
            x_0,
            ts,
            method='dopri5',
            atol=1e-5,
            rtol=1e-5
        )

        # --- Process results ---
        tokens_traj = [x for x in tokens_traj_tensor]

        # Recompute clean_traj based on generated trajectory
        clean_traj = []
        with torch.no_grad():
            for i, t in enumerate(ts):
                x_t = tokens_traj[i]
                t_input = torch.ones(x_t.shape[0], 1, device=x_t.device) * t
                pred_x_1 = model(x_t, t_input, token_mask, x_sc=None)
                clean_traj.append(pred_x_1)

        return {"tokens_traj": tokens_traj, "clean_traj": clean_traj}
    
    def sample_with_classifier_free_guidance(
        self,
        batch_size,
        num_tokens,
        emb_dim,
        model,
        dataset_idx,
        spacegroup,
        cfg_scale=4.0,
        num_timesteps=None,
        x_0=None,
        x_1=None,
        token_mask=None,
        token_idx=None,
    ):
        """Generates new samples of a specified (B, N, d) using denoiser model with classifier-free
        guidance.

        To be used with DiT denoisers, which use a different forward pass signature.

        Args:
            batch_size (int): Number of samples to generate: B.
            num_tokens (int): Max number of tokens in each sample: N.
            emb_dim (int): Dimension of each token: d.
            model (nn.Module): Denoiser model to use.
            dataset_idx (torch.Tensor): Dataset index, used for classifier-free guidance. (B, 1)
            spacegroup (torch.Tensor): Spacegroup, used for classifier-free guidance. (B, 1)
            cfg_scale (float): Scale factor for classifier-free guidance.
            num_timesteps (int): Number of timesteps to integrate over.
            x_0 (torch.Tensor): Initial sample to start from. (B, N, d)
            x_1 (torch.Tensor): Final sample to end at. (B, N, d)
            token_mask (torch.Tensor): Mask for valid tokens. (B, N)
            token_idx (torch.Tensor): Index of each token.

        Returns:
            Dict with keys:
                tokens_traj (list): List of generated samples at each timestep.
                clean_traj (list): List of denoised samples at each timestep.
        """
        assert torch.all(dataset_idx > 0), "Dataset 0 -> null class"

        # Set-up initial prior samples
        if x_0 is None:
            x_0 = self._centered_gaussian(batch_size, num_tokens, emb_dim)
        if token_mask is None:
            token_mask = torch.ones(batch_size, num_tokens, device=self.device).bool()
        if token_idx is None:
            token_idx = torch.arange(num_tokens, device=self.device, dtype=torch.float32)[
                None
            ].repeat(batch_size, 1)

        # Set-up classifier-free guidance
        x_0 = torch.cat([x_0, x_0], dim=0)  # (2B, N, d)
        dataset_idx_null = torch.zeros_like(dataset_idx)
        dataset_idx = torch.cat([dataset_idx, dataset_idx_null], dim=0)  # (2B, 1)
        spacegroup_null = torch.zeros_like(spacegroup)
        spacegroup = torch.cat([spacegroup, spacegroup_null], dim=0)  # (2B, 1)
        token_mask = torch.cat([token_mask, token_mask], dim=0)  # (2B, N)

        # Set-up time
        if num_timesteps is None:
            num_timesteps = self.num_timesteps
        ts = torch.linspace(self.min_t, 1.0, num_timesteps)
        t_1 = ts[0]

        tokens_traj = [x_0]
        clean_traj = []
        x_sc = None
        for t_2 in ts[1:]:
            # Set-up input to denoiser
            x_t_1 = tokens_traj[-1]
            if self.corrupt:
                x = x_t_1
            else:
                if x_1 is None:
                    raise ValueError("Must provide x_1 if not corrupting.")
                x = torch.cat([x_1, x_1], dim=0)  # (2B, N, d)
            t = torch.ones((2 * batch_size, 1), device=self.device) * t_1
            d_t = t_2 - t_1

            # Run denoiser model
            with torch.no_grad():
                pred_x_1 = model(
                    x, t, token_mask, x_sc
                )

            # Process model output
            clean_traj.append(pred_x_1.chunk(2, dim=0)[0])  # Remove null class samples
            if self.self_condition:
                x_sc = pred_x_1

            # Take reverse step
            x_t_2 = self._x_euler_step(d_t, t_1, pred_x_1, x_t_1)

            tokens_traj.append(x_t_2)
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        x_t_1 = tokens_traj[-1]
        if self.corrupt:
            x = x_t_1
        else:
            if x_1 is None:
                raise ValueError("Must provide x_1 if not corrupting.")
            x = x_1
        t = torch.ones((2 * batch_size, 1), device=self.device) * t_1
        with torch.no_grad():
            pred_x_1 = model(
                x, t, token_mask, x_sc
            )
        clean_traj.append(pred_x_1.chunk(2, dim=0)[0])  # Remove null class samples
        tokens_traj.append(pred_x_1)

        return {"tokens_traj": tokens_traj, "clean_traj": clean_traj}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_timesteps={self.num_timesteps}, self_condition={self.self_condition})"