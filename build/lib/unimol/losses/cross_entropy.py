# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
import pandas as pd
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
from unicore.losses.cross_entropy import CrossEntropyLoss
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
import numpy as np
import warnings

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

@register_loss("finetune_cross_entropy")
class FinetuneCrossEntropyLoss(CrossEntropyLoss):
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
            #features_only=True,
            classification_head_name=self.args.classification_head_name,
            mode="ae_only", encoder_masked_tokens=None
        )
        logit_output = net_output[0]
        loss = self.compute_loss(model, logit_output, sample, reduce=reduce)
        sample_size = sample["target"]["finetune_target"].size(0)
        if not self.training:
            probs = F.softmax(logit_output.float(), dim=-1).view(
                -1, logit_output.size(-1)
            )
            logging_output = {
                "loss": loss.data,
                "prob": probs.data,
                "target": sample["target"]["finetune_target"].view(-1).data,
                "smi_name": sample["smi_name"],
                "sample_size": sample_size,
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
        lprobs = F.log_softmax(net_output.float(), dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        targets = sample["target"]["finetune_target"].view(-1)
        loss = F.nll_loss(
            lprobs,
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
            acc_sum = sum(
                sum(log.get("prob").argmax(dim=-1) == log.get("target"))
                for log in logging_outputs
            )
            probs = torch.cat([log.get("prob") for log in logging_outputs], dim=0)
            metrics.log_scalar(
                f"{split}_acc", acc_sum / sample_size, sample_size, round=3
            )
            if probs.size(-1) == 2:
                # binary classification task, add auc score
                targets = torch.cat(
                    [log.get("target", 0) for log in logging_outputs], dim=0
                )
                smi_list = [
                    item for log in logging_outputs for item in log.get("smi_name")
                ]
                df = pd.DataFrame(
                    {
                        "probs": probs[:, 1].cpu(),
                        "targets": targets.cpu(),
                        "smi": smi_list,
                    }
                )
                auc = roc_auc_score(df["targets"], df["probs"])
                df = df.groupby("smi").mean()
                # Round averaged targets to binary values (0 or 1) to handle duplicate SMILES
                df["targets"] = df["targets"].round().astype(int)
                agg_auc = roc_auc_score(df["targets"], df["probs"])
                agg_auprc  = average_precision_score(df["targets"], df["probs"])
                metrics.log_scalar(f"{split}_auc", auc, sample_size, round=3)
                metrics.log_scalar(f"{split}_agg_auc", agg_auc, sample_size, round=4)
                metrics.log_scalar(f"{split}_agg_auprc", agg_auprc, sample_size, round=4)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train


@register_loss("multi_task_BCE")
class MultiTaskBCELoss(CrossEntropyLoss):
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
            #masked_tokens=None,
            #features_only=True,
            classification_head_name=self.args.classification_head_name,
            mode="ae_only", encoder_masked_tokens=None
        )
        logit_output = net_output[0]
        is_valid = sample["target"]["finetune_target"] > -0.5
        loss = self.compute_loss(
            model, logit_output, sample, reduce=reduce, is_valid=is_valid
        )
        sample_size = sample["target"]["finetune_target"].size(0)
        if not self.training:
            probs = torch.sigmoid(logit_output.float()).view(-1, logit_output.size(-1))
            logging_output = {
                "loss": loss.data,
                "prob": probs.data,
                "target": sample["target"]["finetune_target"].view(-1).data,
                "num_task": self.args.num_classes,
                "sample_size": sample_size,
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

    def compute_loss(self, model, net_output, sample, reduce=True, is_valid=None):
        pred = net_output[is_valid].float()
        targets = sample["target"]["finetune_target"][is_valid].float()
        loss = F.binary_cross_entropy_with_logits(
            pred,
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
            agg_auc_list = []
            num_task = logging_outputs[0].get("num_task", 0)
            conf_size = logging_outputs[0].get("conf_size", 0)
            y_true = (
                torch.cat([log.get("target", 0) for log in logging_outputs], dim=0)
                .view(-1, conf_size, num_task)
                .cpu()
                .numpy()
                .mean(axis=1)
            )
            y_pred = (
                torch.cat([log.get("prob") for log in logging_outputs], dim=0)
                .view(-1, conf_size, num_task)
                .cpu()
                .numpy()
                .mean(axis=1)
            )
            # [test_size, num_classes] = [test_size * conf_size, num_classes].mean(axis=1)
            for i in range(y_true.shape[1]):
                # AUC is only defined when there is at least one positive data.
                if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
                    # ignore nan values
                    is_labeled = y_true[:, i] > -0.5
                    agg_auc_list.append(
                        roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
                    )
            if len(agg_auc_list) < y_true.shape[1]:
                warnings.warn("Some target is missing!")
            if len(agg_auc_list) == 0:
                raise RuntimeError(
                    "No positively labeled data available. Cannot compute Average Precision."
                )
            agg_auc = sum(agg_auc_list) / len(agg_auc_list)
            metrics.log_scalar(f"{split}_agg_auc", agg_auc, sample_size, round=4)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train


@register_loss("finetune_cross_entropy_pocket")
class FinetuneCrossEntropyPocketLoss(FinetuneCrossEntropyLoss):
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
        logit_output = net_output[0]
        loss = self.compute_loss(model, logit_output, sample, reduce=reduce)
        sample_size = sample["target"]["finetune_target"].size(0)
        if not self.training:
            probs = F.softmax(logit_output.float(), dim=-1).view(
                -1, logit_output.size(-1)
            )
            logging_output = {
                "loss": loss.data,
                "prob": probs.data,
                "target": sample["target"]["finetune_target"].view(-1).data,
                "sample_size": sample_size,
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
            acc_sum = sum(
                sum(log.get("prob").argmax(dim=-1) == log.get("target"))
                for log in logging_outputs
            )
            metrics.log_scalar(
                f"{split}_acc", acc_sum / sample_size, sample_size, round=3
            )
            preds = (
                torch.cat(
                    [log.get("prob").argmax(dim=-1) for log in logging_outputs], dim=0
                )
                .cpu()
                .numpy()
            )
            targets = (
                torch.cat([log.get("target", 0) for log in logging_outputs], dim=0)
                .cpu()
                .numpy()
            )
            metrics.log_scalar(f"{split}_pre", precision_score(targets, preds), round=3)
            metrics.log_scalar(f"{split}_rec", recall_score(targets, preds), round=3)
            metrics.log_scalar(
                f"{split}_f1", f1_score(targets, preds), sample_size, round=3
            )
