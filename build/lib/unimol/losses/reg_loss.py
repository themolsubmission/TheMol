# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
import math
def generate_random_rotation_matrix(batch_size, device='cpu'):
    """
    Generate random 3D rotation matrices for each sample in the batch.
    
    Args:
        batch_size: Number of samples in the batch
        device: Device to put the tensor on
    
    Returns:
        rotation_matrices: [batch_size, 3, 3] random rotation matrices
    """
    # Generate random quaternions for efficient rotation matrix generation
    u = torch.rand(batch_size, device=device)
    v = torch.rand(batch_size, device=device) * 2 * math.pi
    w = torch.rand(batch_size, device=device) * 2 * math.pi

    # Convert to quaternion components
    q0 = torch.sqrt(1 - u) * torch.sin(v)
    q1 = torch.sqrt(1 - u) * torch.cos(v)
    q2 = torch.sqrt(u) * torch.sin(w)
    q3 = torch.sqrt(u) * torch.cos(w)

    # Convert quaternion to rotation matrix
    rotation_matrices = torch.zeros(batch_size, 3, 3, device=device)

    rotation_matrices[:, 0, 0] = 1 - 2 * (q2**2 + q3**2)
    rotation_matrices[:, 0, 1] = 2 * (q1*q2 - q0*q3)
    rotation_matrices[:, 0, 2] = 2 * (q1*q3 + q0*q2)

    rotation_matrices[:, 1, 0] = 2 * (q1*q2 + q0*q3)
    rotation_matrices[:, 1, 1] = 1 - 2 * (q1**2 + q3**2)
    rotation_matrices[:, 1, 2] = 2 * (q2*q3 - q0*q1)

    rotation_matrices[:, 2, 0] = 2 * (q1*q3 - q0*q2)
    rotation_matrices[:, 2, 1] = 2 * (q2*q3 + q0*q1)
    rotation_matrices[:, 2, 2] = 1 - 2 * (q1**2 + q2**2)

    return rotation_matrices

def apply_random_rotation_to_target_coords(sample, target_key, apply_prob=0.5):
    """
    Apply random rotation to target coordinates.
    
    Args:
        sample: Sample dictionary containing molecular data
        target_key: Key for target data (typically "target")
        apply_prob: Probability of applying rotation (default: 0.5)
    
    Returns:
        Modified sample with rotated target coordinates
    """
    # Only apply rotation with given probability
    if torch.rand(1).item() > apply_prob:
        return sample
    input_key = "net_input"
    #coords = sample[target_key]['coord_target']  # [batch_size, max_atoms, 3]
    coords = sample[input_key]['src_coord']
    batch_size, max_atoms, _ = coords.shape

    # Generate random rotation matrices for each sample in batch
    rotation_matrices = generate_random_rotation_matrix(batch_size, device=coords.device)
    #rotation_matrices = generate_random_rotation_matrix(batch_size, device=coords.device)
    # Apply rotation to coordinates
    # coords: [B, N, 3], rotation_matrices: [B, 3, 3]
    # We need to do batch matrix multiplication
    rotated_coords = torch.bmm(coords, rotation_matrices.transpose(-1, -2))  # [B, N, 3]
    #rotated_coords_src = torch.bmm(coords_src, rotation_matrices.transpose(-1, -2))  # [B, N, 3]
    # Update the target coordinates in sample
    #sample[target_key]['coord_target'] = rotated_coords
    sample[input_key]['src_coord'] = rotated_coords
    return sample

@register_loss("finetune_mse")
class FinetuneMSELoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        input_key = "net_input"
        sample["net_input"]['src_bond_type'] = None
        sample[input_key]['src_tokens'][sample[input_key]['src_tokens'] == 1] = 0 
        sample[input_key]['src_tokens'][sample[input_key]['src_tokens'] == 2] = 0 
        sample[input_key]['src_distance'][:,0,:] = 0
        sample[input_key]['src_distance'][:,:,0] = 0
        sample = apply_random_rotation_to_target_coords(sample, None, apply_prob=0.7)
        net_output = model(
            **sample["net_input"],
            #features_only=False,
            classification_head_name=self.args.classification_head_name,
            mode="ae_only", encoder_masked_tokens=None
        )
        reg_output = net_output[0]
        loss = self.compute_loss(model, reg_output, sample, reduce=reduce)
        sample_size = sample["target"]["finetune_target"].size(0)
        if not self.training:
            if self.task.mean and self.task.std:
                targets_mean = torch.tensor(self.task.mean, device=reg_output.device)
                targets_std = torch.tensor(self.task.std, device=reg_output.device)
                reg_output = reg_output * targets_std + targets_mean
                #denor_targets = (sample["target"]["finetune_target"] - targets_mean) / targets_std
                denor_targets = sample["target"]["finetune_target"]
            logging_output = {
                "loss": loss.data,
                "predict": reg_output.view(-1, self.args.num_classes).data,
                "target": denor_targets
                .view(-1, self.args.num_classes)
                .data,
                "smi_name": sample["smi_name"],
                "sample_size": sample_size,
                "num_task": self.args.num_classes,
                "conf_size": self.args.conf_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        predicts = net_output.view(-1, self.args.num_classes).float()
        targets = (
            sample["target"]["finetune_target"].view(-1, self.args.num_classes).float()
        )
        if self.task.mean and self.task.std:
            targets_mean = torch.tensor(self.task.mean, device=targets.device)
            targets_std = torch.tensor(self.task.std, device=targets.device)
            targets = (targets - targets_mean) / targets_std
        loss = F.mse_loss(
            predicts,
            targets,
            reduction="sum" if reduce else "none",
        )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            predicts = torch.cat([log.get("predict") for log in logging_outputs], dim=0)
            if predicts.size(-1) == 1:
                # single label regression task, add aggregate acc and loss score
                targets = torch.cat(
                    [log.get("target", 0) for log in logging_outputs], dim=0
                )
                smi_list = [
                    item for log in logging_outputs for item in log.get("smi_name")
                ]
                df = pd.DataFrame(
                    {
                        "predict": predicts.view(-1).cpu(),
                        "target": targets.view(-1).cpu(),
                        "smi": smi_list,
                    }
                )
                mae = np.abs(df["predict"] - df["target"]).mean()
                mse = ((df["predict"] - df["target"]) ** 2).mean()
                df = df.groupby("smi").mean()
                agg_mae = np.abs(df["predict"] - df["target"]).mean()
                agg_mse = ((df["predict"] - df["target"]) ** 2).mean()

                # Calculate Pearson and Spearman correlation on aggregated predictions
                # This is the correct way: calculate on all data at once, not batch-wise average
                agg_pcc, _ = pearsonr(df["predict"].values, df["target"].values)
                agg_spearman, _ = spearmanr(df["predict"].values, df["target"].values)


                # try:
                #     agg_pcc, _ = pearsonr(df["predict"].values, df["target"].values)
                #     agg_spearman, _ = spearmanr(df["predict"].values, df["target"].values)
                # except:
                #     # Handle edge cases (e.g., constant predictions/targets)
                #     agg_pcc = 0.0
                #     agg_spearman = 0.0

                metrics.log_scalar(f"{split}_mae", mae, sample_size, round=3)
                metrics.log_scalar(f"{split}_mse", mse, sample_size, round=3)
                metrics.log_scalar(f"{split}_agg_mae", agg_mae, sample_size, round=3)
                metrics.log_scalar(f"{split}_agg_mse", agg_mse, sample_size, round=3)
                metrics.log_scalar(
                    f"{split}_agg_rmse", np.sqrt(agg_mse), sample_size, round=4
                )
                metrics.log_scalar(f"{split}_agg_pcc", agg_pcc, sample_size, round=4)
                metrics.log_scalar(f"{split}_agg_spearman", agg_spearman, sample_size, round=4)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train


@register_loss("finetune_mae")
class FinetuneMAELoss(FinetuneMSELoss):
    def __init__(self, task):
        super().__init__(task)

    def compute_loss(self, model, net_output, sample, reduce=True):
        predicts = net_output.view(-1, self.args.num_classes).float()
        targets = (
            sample["target"]["finetune_target"].view(-1, self.args.num_classes).float()
        )
        if self.task.mean and self.task.std:
            targets_mean = torch.tensor(self.task.mean, device=targets.device)
            targets_std = torch.tensor(self.task.std, device=targets.device)
            targets = (targets - targets_mean) / targets_std
        loss = F.l1_loss(
            predicts,
            targets,
            reduction="sum" if reduce else "none",
        )
        return loss


@register_loss("finetune_smooth_mae")
class FinetuneSmoothMAELoss(FinetuneMSELoss):
    def __init__(self, task):
        super().__init__(task)

    def compute_loss(self, model, net_output, sample, reduce=True):
        predicts = net_output.view(-1, self.args.num_classes).float()
        targets = (
            sample["target"]["finetune_target"].view(-1, self.args.num_classes).float()
        )
        if self.task.mean and self.task.std:
            targets_mean = torch.tensor(self.task.mean, device=targets.device)
            targets_std = torch.tensor(self.task.std, device=targets.device)
            targets = (targets - targets_mean) / targets_std
        loss = F.smooth_l1_loss(
            predicts,
            targets,
            reduction="sum" if reduce else "none",
        )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        """Aggregate logging outputs from data parallel training."""
        if not logging_outputs:
            return
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        if sample_size == 0:
            return
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            num_task = logging_outputs[0].get("num_task", 0)
            conf_size = logging_outputs[0].get("conf_size", 0)
            #print(torch.cat([log.get("target", 0) for log in logging_outputs], dim=0).shape)
            y_true = (
                torch.cat([log.get("target", 0) for log in logging_outputs], dim=0)
                #.view(-1, conf_size, num_task)
                .view(-1, conf_size, num_task)
                .cpu()
                .numpy()
                .mean(axis=1)
            )
            y_pred = (
                torch.cat([log.get("predict") for log in logging_outputs], dim=0)
                .view(-1, conf_size, num_task)
                .cpu()
                .numpy()
                .mean(axis=1)
            )
            agg_mae = np.abs(y_pred - y_true).mean(axis=0).mean(axis=0)
            agg_mse = ((y_pred - y_true) ** 2).mean(axis=0).mean(axis=0)
            agg_rmse = np.sqrt(agg_mse)
            
            # Calculate Pearson and Spearman correlation on aggregated predictions
            # This is the correct way: calculate on all data at once, not batch-wise average
            try:
                # Flatten in case of multi-task, or use first task if single task
                y_pred_flat = y_pred.flatten() if num_task == 1 else y_pred[:, 0]
                y_true_flat = y_true.flatten() if num_task == 1 else y_true[:, 0]
                agg_pcc, _ = pearsonr(y_pred_flat, y_true_flat)
                agg_spearman, _ = spearmanr(y_pred_flat, y_true_flat)
            except Exception as e:
                # Handle edge cases (e.g., constant predictions/targets, insufficient data)
                agg_pcc = 0.0
                agg_spearman = 0.0

            # try:
            #     agg_pcc, _ = pearsonr(df["predict"].values, df["target"].values)
            #     agg_spearman, _ = spearmanr(df["predict"].values, df["target"].values)
            # except:
            #     # Handle edge cases (e.g., constant predictions/targets)
            #     agg_pcc = 0.0
            #     agg_spearman = 0.0
                    
            metrics.log_scalar(f"{split}_agg_mae", agg_mae, sample_size, round=4)
            metrics.log_scalar(f"{split}_agg_mse", agg_mse, sample_size, round=4)
            metrics.log_scalar(f"{split}_agg_rmse", agg_rmse, sample_size, round=4)
            metrics.log_scalar(f"{split}_agg_pcc", agg_pcc, sample_size, round=4)
            metrics.log_scalar(f"{split}_agg_spearman", agg_spearman, sample_size, round=4)

@register_loss("finetune_mse_pocket")
class FinetuneMSEPocketLoss(FinetuneMSELoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.args.classification_head_name,
        )
        reg_output = net_output[0]
        loss = self.compute_loss(model, reg_output, sample, reduce=reduce)
        sample_size = sample["target"]["finetune_target"].size(0)
        if not self.training:
            if self.task.mean and self.task.std:
                targets_mean = torch.tensor(self.task.mean, device=reg_output.device)
                targets_std = torch.tensor(self.task.std, device=reg_output.device)
                reg_output = reg_output * targets_std + targets_mean
            logging_output = {
                "loss": loss.data,
                "predict": reg_output.view(-1, self.args.num_classes).data,
                "target": sample["target"]["finetune_target"]
                .view(-1, self.args.num_classes)
                .data,
                "sample_size": sample_size,
                "num_task": self.args.num_classes,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            predicts = torch.cat([log.get("predict") for log in logging_outputs], dim=0)
            if predicts.size(-1) == 1:
                # single label regression task
                targets = torch.cat(
                    [log.get("target", 0) for log in logging_outputs], dim=0
                )
                df = pd.DataFrame(
                    {
                        "predict": predicts.view(-1).cpu(),
                        "target": targets.view(-1).cpu(),
                    }
                )
                mse = ((df["predict"] - df["target"]) ** 2).mean()
                metrics.log_scalar(f"{split}_mse", mse, sample_size, round=3)
                metrics.log_scalar(f"{split}_rmse", np.sqrt(mse), sample_size, round=4)
