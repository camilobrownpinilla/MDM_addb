import torch
from data.AddBiomechanicsDataset import AddBiomechanicsDataset, OutputDataKeys, InputDataKeys
from typing import Dict, List, Optional
import numpy as np
import os

## Necessary?
components = {
    0: "left-x",
    1: "left-y",
    2: "left-z",
    3: "right-x",
    4: "right-y",
    5: "right-z"
}
wrench_components = {
    0: "left-moment-x",
    1: "left-moment-y",
    2: "left-moment-z",
    3: "left-force-x",
    4: "left-force-y",
    5: "left-force-z",
    6: "right-moment-x",
    7: "right-moment-y",
    8: "right-moment-z",
    9: "right-force-x",
    10: "right-force-y",
    11: "right-force-z"
}

class RegressionLossEvaluator:
    dataset: AddBiomechanicsDataset

    losses: List[torch.Tensor]
    force_losses: List[torch.Tensor]
    moment_losses: List[torch.Tensor]
    wrench_losses: List[torch.Tensor]
    cop_losses: List[torch.Tensor]

    force_reported_metrics: List[float]
    moment_reported_metrics: List[float]
    cop_reported_metrics: List[float]
    wrench_reported_metrics: List[float]
    tau_reported_metrics: List[float]
    com_acc_reported_metrics: List[float]

    def __init__(self, dataset: AddBiomechanicsDataset, split: str):
        self.dataset = dataset
        self.split = split

        # Aggregating losses across batches for dev set evaluation
        self.losses = []
        # self.force_losses = []
        # self.moment_losses = []
        # self.wrench_losses = []
        # self.cop_losses = []
        self.acc_losses = []
        self.vel_losses = []
        self.pos_losses  = []
        self.contact_losses = []
        

        """
        Change for evalutation later
        """
        # Aggregating reported metrics for dev set evaluation
        self.force_reported_metrics = []
        self.moment_reported_metrics = []
        self.cop_reported_metrics = []
        self.wrench_reported_metrics = []
        self.wrench_moment_reported_metrics = []
        self.tau_reported_metrics = []
        self.com_acc_reported_metrics = []

    @staticmethod
    def get_squared_diff_mean_vector(output_tensor: torch.Tensor, label_tensor: torch.Tensor) -> torch.Tensor:
        if output_tensor.shape != label_tensor.shape:
            print(f'Label tensor: {label_tensor.shape}\n Output tensor: {output_tensor.shape}')
            raise ValueError('Output and label tensors must have the same shape')
        if len(output_tensor.shape) != 3:
            raise ValueError('Output and label tensors must be 3-dimensional')
        if output_tensor.shape[0] * output_tensor.shape[1] * output_tensor.shape[2] == 0:
            raise ValueError('Output and label tensors must not be empty')
        force_diff = (output_tensor - label_tensor)
        force_loss = torch.mean(force_diff ** 2, dim=(0,1))
        return force_loss
    
    @staticmethod
    def get_mean_norm_error(output_tensor: torch.Tensor, label_tensor: torch.Tensor, vec_size: int = 3) -> torch.Tensor:
        if output_tensor.shape != label_tensor.shape:
            raise ValueError('Output and label tensors must have the same shape')
        if len(output_tensor.shape) != 3:
            raise ValueError('Output and label tensors must be 3-dimensional')
        if output_tensor.shape[0] * output_tensor.shape[1] * output_tensor.shape[2] == 0:
            raise ValueError('Output and label tensors must not be empty')
        if output_tensor.shape[-1] % vec_size != 0:
            raise ValueError('Tensors must have a final dimension divisible by vec_size=' + str(vec_size))

        diffs = output_tensor - label_tensor

        # Reshape the tensor so that the last dimension is split into chunks of `vec_size`
        reshaped_tensor = diffs.view(diffs.shape[0], diffs.shape[1], -1, vec_size)

        # Compute the norm over the last dimension
        norms = torch.norm(reshaped_tensor[:,-1:,:,:], dim=3)

        # Compute the mean norm over all the dimensions
        mean_norm = torch.mean(norms)

        return mean_norm
    
    @staticmethod
    def get_com_acc_error(output_force_tensor: torch.Tensor, label_force_tensor: torch.Tensor) -> torch.Tensor:
        if output_force_tensor.shape != label_force_tensor.shape:
            raise ValueError('Output and label tensors must have the same shape')
        if len(output_force_tensor.shape) != 3:
            raise ValueError('Output and label tensors must be 3-dimensional')
        if output_force_tensor.shape[0] * output_force_tensor.shape[1] * output_force_tensor.shape[2] == 0:
            raise ValueError('Output and label tensors must not be empty')
        if output_force_tensor.shape[-1] != 6:
            raise ValueError('Output and label tensors must have a 6 dimensional final dimension')

        # Compute the mean norm over all the dimensions
        output_force_sum = output_force_tensor[:, :, :3] + output_force_tensor[:, :, 3:]
        label_force_sum = label_force_tensor[:, :, :3] + label_force_tensor[:, :, 3:]

        return RegressionLossEvaluator.get_mean_norm_error(output_force_sum, label_force_sum, vec_size=3)
    
    def __call__(self,
                 inputs: Dict[str, torch.Tensor],
                 outputs: Dict[str, torch.Tensor],
                 labels: Dict[str, torch.Tensor],
                 batch_subject_indices: List[int],
                 batch_trial_indices: List[int],
                 compute_report: bool = False,
                 log_reports_to_wandb: bool = False,
                 analyze: bool = False,
                 plot_path_root: str = 'outputs/plots') -> torch.Tensor:

        ############################################################################
        # Step 1: Compute the loss
        ############################################################################

        # 1.1. Compute the force loss, as a single vector of length 3*N
        pos_loss = RegressionLossEvaluator.get_squared_diff_mean_vector(
            outputs[OutputDataKeys.POS],
            labels[OutputDataKeys.POS]
        )
        self.pos_losses.append(pos_loss)

        com_pos_loss = RegressionLossEvaluator.get_squared_diff_mean_vector(
            outputs[OutputDataKeys.COM_POS],
            labels[OutputDataKeys.COM_POS]
        )
        self.pos_losses.append(com_pos_loss)

        vel_loss = RegressionLossEvaluator.get_squared_diff_mean_vector(
            outputs[OutputDataKeys.VEL],
            labels[OutputDataKeys.VEL]
        )
        self.vel_losses.append(vel_loss)

        com_vel_loss = RegressionLossEvaluator.get_squared_diff_mean_vector(
            outputs[OutputDataKeys.COM_VEL],
            labels[OutputDataKeys.COM_VEL]
        )
        self.vel_losses.append(com_vel_loss)

        acc_loss = RegressionLossEvaluator.get_squared_diff_mean_vector(
            outputs[OutputDataKeys.ACC],
            labels[OutputDataKeys.ACC]
        )
        self.acc_losses.append(acc_loss)


        com_acc_loss = RegressionLossEvaluator.get_squared_diff_mean_vector(
            outputs[OutputDataKeys.COM_ACC],
            labels[OutputDataKeys.COM_ACC]
        )
        self.acc_losses.append(com_acc_loss)

        contact_loss = RegressionLossEvaluator.get_squared_diff_mean_vector(
            outputs[OutputDataKeys.CONTACT],
            labels[OutputDataKeys.CONTACT]
        )
        self.contact_losses.append(contact_loss)
        
        loss = (torch.sum(pos_loss) +
                torch.sum(com_pos_loss) +
                torch.sum(vel_loss) +
                torch.sum(com_vel_loss) +
                torch.sum(acc_loss) +
                torch.sum(com_acc_loss) +
                torch.sum(contact_loss))

        self.losses.append(loss)

        return loss