from pyexpat import model
from random import shuffle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
from PIL import Image, ImageFile
from tqdm import tqdm
import time
import yaml
import click

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch import Tensor, seed
import optuna
from torch.optim import Optimizer, Adam, SGD
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision import datasets

from pytorch_lightning.callbacks import Callback
from pytorch_lightning import loggers as pl_loggers 
from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR

import sys
sys.path.append('/home/v1xjian2/BDL/Bayesian_DL')
from swafa.utils import (
    get_callback_epoch_range,
    vectorise_weights,
    vectorise_gradients,
    get_weight_dimension,
    set_weights,
    normalise_gradient,
)
from swafa.fa import OnlineGradientFactorAnalysis

# Method in 'Benchmarking Bayesian Deep Learning on Biabetic Retinopathy Detection Tasks' 2022.
def predictive_distributions_entropy(dist: torch.tensor, base = 'log2', eps = 1e-4):
    """
    Args:
        dist: list of predictive distributions which need calculating the entropy, of shape (n_test_samples, n_classes)

    Returns:
        entropies for each test point, of shape (n_test_samples, )

    """
    # for numerical stability, replace 0 by a small number
    dist[dist == 0] = eps
    if base == 'log2':
        return -torch.sum(dist * torch.log2(dist), dim = -1)


# Method in 'On Stein Variational Neural Network Ensembles' 2021
def model_disagreement_score(dists: torch.tensor):

    """
    Args:
        dists: List of predictive distributions (from different samples of variational distribution)
        of shape(n_test_samples, n_bma_samples, n_classes)

    Returns:
        model disagreement score for the test inputs (n_test_samples, )
    """
    mean_dist = torch.mean(dists, axis=1) # (n_test_samples, n_classes)
    mean_dist = mean_dist.unsqueeze(dim=1).repeat(1, dists.shape[1], 1)
    scores = (dists - mean_dist)**2
    return torch.mean(scores, axis = [1,2])


def evaluation_function(pred_list, output_softmax_list, target_list, model_disag_score_list, pred_entropy_list, thresholds):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
    '''
    Args:
        pred_list: List of predicted class for test samples.
        output_softmax_list: List of (ensembled) softmax predictions for test samples, which are averaged from n_bma_samples.
        target_list: List of labels of test samples.
        model_disag_score_list: List of model disagreement scores.
        pred_entropy_list: List of entropies for predictive distributions.
        thresholds: List of threshold we want to use in uncertainty guided prediction.

    Returns:
        All metrices we want, stored in a dirctionary.
        The metrices are:   
            1. Test accuracy
            2. Precision
            3. Recall
            4. AU-ROC
            5. AU-PRC
            6. F1 score
        These metrices are tested for (pure) prediction and for (different level of) uncertainty guided prediction.
    '''
    # Sort, small -> large
    pred_entropy_sorted_indices = [pred_entropy_list.index(x) for x in sorted(pred_entropy_list)]
    model_disag_score_indices = [model_disag_score_list.index(x) for x in sorted(model_disag_score_list)]

    if 1 not in thresholds:
        thresholds += [1]

    metric_dirct = {}
    for threshold in thresholds:
        index =  int(np.floor(len(pred_list) * threshold))
        curr_accuracy = accuracy_score(target_list[:index], pred_list[:index])
        curr_precision = precision_score(target_list[:index], pred_list[:index])
        curr_recall = recall_score(target_list[:index], pred_list[:index])
        curr_f1 = f1_score(target_list[:index], pred_list[:index])
        curr_auroc = roc_auc_score(target_list[:index], output_softmax_list[:index])
        curr_auprc = average_precision_score(target_list[:index], output_softmax_list[:index])

        jj = int(np.floor(threshold * 100))
        exec(f'metric_dirct["accuracy_{jj}"] = round(curr_accuracy,3)')
        exec(f'metric_dirct["precision_{jj}"] = round(curr_precision,3)')
        exec(f'metric_dirct["recall_{jj}"] = round(curr_recall,3)')
        exec(f'metric_dirct["f1_{jj}"] = round(curr_f1,3)')
        exec(f'metric_dirct["auroc_{jj}"] = round(curr_auroc,3)')
        exec(f'metric_dirct["auprc_{jj}"] = round(curr_auprc,3)')

    return metric_dirct


def delete_and_clear(list):

    try:
        for item in list:
            item = None

    except Exception: # pass all the errors
        pass

    torch.cuda.empty_cache()

class RetinopathyDataset(Dataset):

    def __init__(self, csv_file, image_folder, transform=None):
        
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.data.loc[idx, 'id_code'] + '.png')
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.data.loc[idx, 'target'])

        return image, label


class ConvolutionalNet(LightningModule):

    """
    A simple convolutional neural network with a classification output.

    Implements functionality which allows it to be used with a PyTorch Lightning Trainer.

    Args:
        net: The CNN network architecture.
        optimiser_class: The class of the PyTorch optimiser to use for training the neural network.
        optimiser_kwargs: Keyword arguments for the optimiser class.
        loss_fn: The PyTorch loss function to use for training the model. Will be applied to the un-activated outputs
            of the neural network.
        loss_multiplier: A constant with which to multiply the loss of each batch. Useful if an estimate of the total
            loss over the full dataset is needed.
        random_seed: The random seed for initialising the weights of the neural network. If None, won't be reproducible.

    """

    def __init__(self, Net: nn.Module, optimiser_class: Optimizer = Adam, optimiser_kwargs: Optional[dict] = None,
                 loss_fn: nn.Module = nn.CrossEntropyLoss(), loss_multiplier: float = 1.0, random_seed: Optional[int] = None):
        super().__init__()
        if random_seed is not None:
            torch.manual_seed(random_seed)

        self.net = Net
        self.optimiser_class = optimiser_class
        self.optimiser_kwargs = optimiser_kwargs or dict(lr=1e-3)
        self.loss_fn = loss_fn
        self.loss_multiplier = loss_multiplier

        
    @staticmethod
    def _identity_fn(X: Tensor) -> Tensor:
        """
        An function which returns the input unchanged.

        Args:
            X: A Tensor of any shape.

        Returns:
            Exactly the same as the unput.
        """
        return X

    def forward(self, X: Tensor) -> Tensor:
        """
        Run the forward pass of the neural network.

        Args:
            X: Input features. Of shape (n_samples, n_features).

        Returns:
            Neural network outputs. Of shape (n_samples,).
        """
        return self.net(X)

    def configure_optimizers(self) -> Optimizer:
        """
        Initialise the optimiser which will be used to train the neural network.

        Returns:
            The initialised optimiser
        """
        return self.optimiser_class(self.net.parameters(), **self.optimiser_kwargs)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Compute the training loss for a single batch of data.

        Args:
            batch: (X, y), where X is the input features of shape (batch_size, n_features) and y is the outputs of shape
                (batch_size,).
            batch_idx: The index of the batch relative to the current epoch.

        Returns:
            The batch training loss. Of shape (1,).
        """
        loss, correct_rate = self._step(batch, batch_idx)
        self.log('train/loss', loss, prog_bar=True, logger=True)
        self.log('train/acc', correct_rate, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Compute the validation loss for a single batch of data.

        Args:
            batch: (X, y), where X is the input features of shape (batch_size, n_features) and y is the outputs of shape
                (batch_size,).
            batch_idx: The index of the batch relative to the current epoch.

        Returns:
            The batch validation loss. Of shape (1,).
        """
        loss, correct_rate = self._step(batch, batch_idx)
        self.log('valid/loss', loss, prog_bar=True, logger=True)
        self.log('valid/acc', correct_rate, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Compute the test loss for a single batch of data.

        Args:
            batch: (X, y), where X is the input features of shape (batch_size, n_features) and y is the outputs of shape
                (batch_size,).
            batch_idx: The index of the batch relative to the current epoch.

        Returns:
            The batch test loss. Of shape (1,).
        """
        loss, correct_rate = self._step(batch, batch_idx)
        self.log('test/loss', loss, prog_bar=True, logger=True)
        self.log('test/acc', correct_rate, prog_bar=True, logger=True)
        return loss

    def _step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Compute the loss for a single batch of data.

        Args:
            batch: (X, y), where X is the input features of shape (batch_size, n_features) and y is the outputs of shape
                (batch_size,).
            batch_idx: The index of the batch relative to the current epoch.

        Returns:
            The batch loss. Of shape (1,).
        """
        X, y = batch
        y_hat = self(X)
        _, pred = torch.max(y_hat, 1)    
        correct_tensor = pred.eq(y.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(correct_tensor.cpu().numpy())
        correct_rate = np.mean(1.0 * correct)
        return self.loss_fn(y_hat, y) * self.loss_multiplier, correct_rate

    '''
    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None) -> Tensor:
        """
        Predict the outputs for a single batch of data.

        Args:
            batch: (X, y), where X is the input features of shape (batch_size, n_features) and y is the outputs of shape
                (batch_size,).
            batch_idx: The index of the batch relative to the current epoch.
            dataloader_idx: The index of the dataloader (may be more than one) from which the batch was sampled.

        Returns:
            The activated outputs. Of shape (batch_size,).
        """
        return self(batch[0])
    '''

    def validation_epoch_end(self, step_losses: List[Tensor]) -> Dict[str, Tensor]:
        """
        Compute the average validation loss over all batches.

        Log the loss under the name 'epoch_val_loss'.

        Args:
            step_losses: The validation loss for each individual batch. Each one of shape (1,).

        Returns:
            A dict of the form {'epoch_val_loss': loss}, where loss is the average validation loss, of shape (1,).
        """
        loss = self._average_loss(step_losses)
        self.logger.experiment.add_scalar("valid/average loss", loss, self.current_epoch)
        metrics = dict(epoch_val_loss=loss)
        self.log_dict(metrics)
        return metrics

    def test_epoch_end(self, step_losses: List[Tensor]) -> Dict[str, Tensor]:
        """
        Compute the average test loss over all batches.

        Log the loss under the name 'epoch_test_loss'.

        Args:
            step_losses: The test loss for each individual batch. Each one of shape (1,).

        Returns:
            A dict of the form {'epoch_test_loss': loss}, where loss is the average test loss, of shape (1,).
        """
        loss = self._average_loss(step_losses)
        self.logger.experiment.add_scalar("test/average loss", loss, self.current_epoch)

    @staticmethod
    def _average_loss(step_losses: List[Tensor]) -> Tensor:
        """
        Compute the average of all losses.

        Args:
            step_losses: Individual losses. Each one of shape (1,).

        Returns:
            The average loss. Of shape (1,).
        """
        return torch.stack(step_losses).mean()


class FactorAnalysisVariationalInferenceCallback(Callback):
    """
    A callback which can be used with a PyTorch Lightning Trainer to learn the parameters of a factor analysis
    variational distribution of a model's weights.

    The parameters are updated to minimise the Kullback-Leibler divergence between the variational distribution and the
    true posterior of the model's weights. This is done via stochastic gradient descent.

    See [1] for full details of the algorithm.

    Args:
        init_c: The initial value of the bias vector in FA (i.e. self.c).
        latent_dim: The latent dimension of the factor analysis model used as the variational distribution.
        precision: The precision of the prior of the true posterior.
        n_gradients_per_update: The number of mini-batch gradients to use to form the expectation of the true gradient
            for each parameter update.
        optimiser_class: The class of the optimiser to use for gradient updates.
        bias_optimiser_kwargs: Keyword arguments for the optimiser which updates the bias term of the factor analysis
            variational distribution. If not given, will default to dict(lr=1e-3).
        factors_optimiser_kwargs: Keyword arguments for the optimiser which updates the factor loading matrix of the
            factor analysis variational distribution. If not given, will default to dict(lr=1e-3).
        noise_optimiser_kwargs: Keyword arguments for the optimiser which updates the logarithm of the diagonal entries
            of the Gaussian noise covariance matrix of the factor analysis variational distribution. If not given, will
            default to dict(lr=1e-3).
        max_grad_norm: Optional maximum norm for gradients which are used to update the parameters of the variational
            distribution.
        device: The device (CPU or GPU) on which to perform the computation. If None, uses the device for the default
            tensor type.
        random_seed: The random seed for reproducibility.

    Attributes:
        weight_dim: An integer specifying the total number of weights in the model. Note that this is computed when the
            model is fit for the first time.
        c: The bias term of the factor analysis variational distribution. A Tensor of shape (weight_dim, 1).
        F: The factor loading matrix of the factor analysis variational distribution. A Tensor of shape
            (weight_dim, latent_dim).
        diag_psi: The diagonal entries of the Gaussian noise covariance matrix of the factor analysis variational
            distribution. A Tensor of shape (weight_dim, 1).

    References:
        [1] Scott Brownlie. Extending the Bayesian Deep Learning Method MultiSWAG. MSc Thesis, University of Edinburgh,
            2021.
    """

    def __init__(self, init_c: Tensor, latent_dim: int, precision: float, n_gradients_per_update: int = 1,
                 optimiser_class: Optimizer = SGD, bias_optimiser_kwargs: Optional[dict] = None,
                 factors_optimiser_kwargs: Optional[dict] = None, noise_optimiser_kwargs: Optional[dict] = None,
                 max_grad_norm: Optional[float] = None, device: Optional[torch.device] = None,
                 random_seed: Optional[int] = None, epochs_at_stage_1: int = 35, increase_resistance: int = 20):
                 
        self.init_c = init_c
        self.latent_dim = latent_dim
        self.precision = precision
        self.n_gradients_per_update = n_gradients_per_update
        self.optimiser_class = optimiser_class
        self.bias_optimiser_kwargs = bias_optimiser_kwargs or dict(lr=1e-3)
        self.factors_optimiser_kwargs = factors_optimiser_kwargs or dict(lr=1e-3)
        self.noise_optimiser_kwargs = noise_optimiser_kwargs or dict(lr=1e-3)
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.random_seed = random_seed
        self.epochs_at_stage_1 = epochs_at_stage_1
        self.increase_resistance = increase_resistance # The smaller, the quicker

        self.weight_dim = None
        self.c = None
        self.F = None
        self.diag_psi = None

        self._I = torch.eye(latent_dim, device=device)
        self._log_diag_psi = None
        self._h = None
        self._z = None
        self._sqrt_diag_psi_dot_z = None
        self._A = None
        self._B = None
        self._C = None
        self._var_grad_wrt_F = None
        self._var_grad_wrt_log_diag_psi = None
        self._prior_grad_wrt_c = None
        self._prior_grad_wrt_F = None
        self._prior_grad_wrt_log_diag_psi = None

        self._optimiser = None
        self._scheduler = None
        self._batch_counter = 0
        self._epoch_counter = 0

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        """
        Called when fit begins.

        If parameters of variational distribution have not already been initialised, initialise them and the optimiser
        which will update them.

        Args:
            trainer: A PyTorch Lightning Trainer which trains the model.
            pl_module: The model being trained.
        """
        if self.weight_dim is None:
            self.weight_dim = get_weight_dimension(pl_module)
            self._init_variational_params()
            self._update_expected_gradients()
            self._init_optimisers()
            self._init_schedulers()        

    def on_batch_start(self, trainer: Trainer, pl_module: LightningModule):
        """
        Called when the training batch begins.

        Sample weight vector from the variational distribution and use it to set the weights of the neural network.

        Args:
            trainer: A PyTorch Lightning Trainer which trains the model.
            pl_module: The model being trained.
        """
        weights = self.sample_weight_vector()
        set_weights(pl_module, weights)

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule):
        """
        Called after loss.backward() and before optimisers are stepped.

        Use the back propagated gradient of the network's loss wrt the network's weights to compute the gradient wrt
        the parameters of the variational distribution. Accumulate these gradients.

        Periodically, use the accumulated gradients to approximate the expected gradients and update the parameters of
        the variational distribution.

        Args:
            trainer: A PyTorch Lightning Trainer which trains the model.
            pl_module: The model being trained.
        """
        grad_weights = vectorise_gradients(pl_module)[:, None]
        self._accumulate_gradients(grad_weights)

        self._batch_counter += 1
        if self._batch_counter % self.n_gradients_per_update == 0:
            self._update_variational_params()
            self._update_expected_gradients()
            
    def _init_variational_params(self):
        """
        Initialise the parameters of the factor analysis variational distribution.
        """
        fa = OnlineGradientFactorAnalysis(
            observation_dim=self.weight_dim,
            latent_dim=self.latent_dim,
            device=self.device,
            random_seed=self.random_seed,
        )
        
        #self.c = Variable(fa.c.data, requires_grad=False)
        self.c = Variable(self.init_c, requires_grad=False)  # we will compute our own gradients
        self.F = Variable(fa.F.data * 0.00001, requires_grad=False)
        self.diag_psi = fa.diag_psi * 0.00001
        self._log_diag_psi = Variable(torch.log(self.diag_psi), requires_grad=False)

        self.c.grad = torch.zeros_like(self.c.data, device=self.device)
        self.F.grad = torch.zeros_like(self.F.data, device=self.device)
        self._log_diag_psi.grad = torch.zeros_like(self._log_diag_psi.data, device=self.device)

    def _init_optimisers(self):
        """
        Initialise the optimiser which will be used to update the parameters of the variational distribution.
        """
        self._optimiser_c = self.optimiser_class([self.c], **self.bias_optimiser_kwargs)
        self._optimiser_F = self.optimiser_class([self.F], **self.factors_optimiser_kwargs)
        self._optimiser_log_diag_psi = self.optimiser_class([self._log_diag_psi], **self.noise_optimiser_kwargs)

    def _init_schedulers(self):
        """
        Initialise the schedulers which will control the learning rate for updating parameters of the variational distribution.
        """
        self._scheduler_c = ExponentialLR(self._optimiser_c, 0.95)
        
        m = torch.nn.Sigmoid()
        lr_lambda_factors = lambda epoch: 0.0 if epoch < self.epochs_at_stage_1 else self.factors_optimiser_kwargs['lr'] * float(m( torch.tensor((epoch - self.increase_resistance)) ) )
        lr_lambda_noise = lambda epoch: 0.0 if epoch < self.epochs_at_stage_1 else self.noise_optimiser_kwargs['lr'] * float(m( torch.tensor((epoch - self.increase_resistance)) ) )
        
        self._scheduler_F = LambdaLR(self._optimiser_F, lr_lambda_factors)
        self._scheduler_log_diag_psi = LambdaLR(self._optimiser_log_diag_psi, lr_lambda_noise)

    def on_train_epoch_end(self, lightning_module, outputs: None, a: None):
        """
        Schedule learning rate for the next epoch.
        """
        self._scheduler_c.step()
        self._scheduler_F.step()
        self._scheduler_log_diag_psi.step()
        self._epoch_counter += 1

    def sample_weight_vector(self) -> Tensor:
        """
        Generate a single sample of the neural network's weight vector from the variational distribution.

        Returns:
            Sample of shape (self.weight_dim,).
        """
        self._h = torch.normal(torch.zeros(self.latent_dim, device=self.device),
                               torch.ones(self.latent_dim, device=self.device))[:, None]
        self._z = torch.normal(torch.zeros(self.weight_dim, device=self.device),
                               torch.ones(self.weight_dim, device=self.device))[:, None]
        self._sqrt_diag_psi_dot_z = torch.sqrt(self.diag_psi) * self._z
        return (self.F.mm(self._h) + self.c + self._sqrt_diag_psi_dot_z).squeeze(dim=1)

    def _accumulate_gradients(self, grad_weights: Tensor):
        """
        Accumulate gradients wrt the parameters of the variational distribution.

        Args:
            grad_weights: The back propagated gradient of the network's loss wrt the network's weights. Of shape
                (self.weight_dim, 1).
        """
        self.c.grad += self._compute_gradient_wrt_c(grad_weights)
        if self._epoch_counter >= self.epochs_at_stage_1:
            self.F.grad += self._compute_gradient_wrt_F(grad_weights)
            self._log_diag_psi.grad += self._compute_gradient_wrt_log_diag_psi(grad_weights)

    def _compute_gradient_wrt_c(self, grad_weights: Tensor) -> Tensor:
        """
        Compute the gradient of the variational objective wrt the bias term of the factor analysis variational
        distribution.

        Args:
            grad_weights: The back propagated gradient of the network's loss wrt the network's weights. Of shape
                (self.weight_dim, 1).

        Returns:
            The gradient of the variational objective wrt the bias term of the factor analysis variational
            distribution. Of shape (self.weight_dim, 1).
        """
        return -self._prior_grad_wrt_c + grad_weights

    def _compute_gradient_wrt_F(self, grad_weights: Tensor) -> Tensor:
        """
        Compute the gradient of the variational objective wrt the factors matrix of the factor analysis variational
        distribution.

        Args:
            grad_weights: The back propagated gradient of the network's loss wrt the network's weights. Of shape
                (self.weight_dim, 1).

        Returns:
            The gradient of the variational objective wrt the factors matrix of the factor analysis variational
            distribution. Of shape (self.weight_dim, self.latent_dim).
        """
        loss_grad = self._compute_loss_gradient_wrt_F(grad_weights)

        return self._var_grad_wrt_F - self._prior_grad_wrt_F + loss_grad

    def _compute_loss_gradient_wrt_F(self, grad_weights: Tensor) -> Tensor:
        """
        Compute the gradient of the network's loss wrt the factors matrix.

        Args:
            grad_weights: The back propagated gradient of the network's loss wrt the network's weights. Of shape
                (self.weight_dim, 1).

        Returns:
            The gradient of the network's loss wrt the factors matrix. Of shape (self.weight_dim, self.latent_dim).
        """
        return grad_weights.mm(self._h.t())

    def _compute_gradient_wrt_log_diag_psi(self, grad_weights: Tensor) -> Tensor:
        """
        Compute the gradient of the variational objective wrt the logarithm of the diagonal of the noise covariance
        matrix of the factor analysis variational distribution.

        Args:
            grad_weights: The back propagated gradient of the network's loss wrt the network's weights. Of shape
                (self.weight_dim, 1).

        Returns:
            The gradient of the variational objective wrt the logarithm of the diagonal of the noise covariance
            matrix of the factor analysis variational distribution. Of shape (self.weight_dim, 1).
        """
        loss_grad = self._compute_loss_gradient_wrt_log_diag_psi(grad_weights)

        return self._var_grad_wrt_log_diag_psi - self._prior_grad_wrt_log_diag_psi + loss_grad

    def _compute_loss_gradient_wrt_log_diag_psi(self, grad_weights: Tensor) -> Tensor:
        """
        Compute the gradient of the network's loss wrt the logarithm of the diagonal of the noise covariance matrix.

        Args:
            grad_weights: The back propagated gradient of the network's loss wrt the network's weights. Of shape
                (self.weight_dim, 1).

        Returns:
            The gradient of the network's loss wrt the logarithm of the diagonal of the noise covariance matrix. Of
            shape (self.weight_dim, 1).
        """
        return 0.5 * grad_weights * self._sqrt_diag_psi_dot_z

    def _update_variational_params(self):
        """
        Update the parameters of the factor analysis variational distribution.

        This is done by using the accumulated gradients to approximate the expected gradients and then performing a
        gradient step.

        After performing the updates, the gradients are reset to zero.
        """
        self._average_and_normalise_gradient(self.c)
        self._optimiser_c.step()
        self._optimiser_c.zero_grad()

        if self._epoch_counter >= self.epochs_at_stage_1:
            self._average_and_normalise_gradient(self.F)
            self._optimiser_F.step()
            self._optimiser_F.zero_grad()

            self._average_and_normalise_gradient(self._log_diag_psi)
            self._optimiser_log_diag_psi.step()
            self._optimiser_log_diag_psi.zero_grad()
            self.diag_psi = torch.exp(self._log_diag_psi)

    def _average_and_normalise_gradient(self, var: Variable):
        """
        Average the gradients accumulated in the variable by dividing by self.n_gradients_per_update and normalise if
        required.

        Args:
            var: The variable whose gradient to average and normalise.
        """
        var.grad /= self.n_gradients_per_update

        if self.max_grad_norm is not None:
            var.grad = normalise_gradient(var.grad, self.max_grad_norm)

    def _update_expected_gradients(self):
        """
        Update the expected gradients used in the algorithm which do not depend on the sampled network weights.
        """
        self._update_prior_gradient_wrt_c()
        if self._epoch_counter >= self.epochs_at_stage_1 or self._epoch_counter == 0:
            self._update_A()
            self._update_B()
            self._update_C()
            self._update_variational_gradient_wrt_F()
            self._update_variational_gradient_wrt_log_diag_psi()
            self._update_prior_gradient_wrt_F()
            self._update_prior_gradient_wrt_log_diag_psi()

    def _update_A(self):
        """
        Update A = psi^(-1) * F.
        """
        diag_inv_psi = 1 / self.diag_psi
        self._A = diag_inv_psi * self.F

    def _update_B(self):
        """
        Update B = Ft * A.
        """
        self._B = self.F.t().mm(self._A)

    def _update_C(self):
        """
        Update C = A * (I + B)^(-1).
        """
        inv_term = torch.linalg.inv(self._I + self._B)
        self._C = self._A.mm(inv_term)

    def _update_variational_gradient_wrt_F(self):
        """
        Update d(variational distribution) / d(F) = C * Bt - A
        """
        self._var_grad_wrt_F = self._C.mm(self._B.t()) - self._A

    def _update_variational_gradient_wrt_log_diag_psi(self):
        """
        Update d(variational distribution) / d(log diag psi) = 0.5 * sum(C dot A, dim=1) dot diag_psi - 0.5
        """
        sum_term = (self._C * self._A).sum(dim=1, keepdims=True)
        self._var_grad_wrt_log_diag_psi = 0.5 * sum_term * self.diag_psi - 0.5

    def _update_prior_gradient_wrt_c(self):
        """
        Update d(prior distribution) / d(c) = -precision * c
        """
        self._prior_grad_wrt_c = -self.precision * self.c

    def _update_prior_gradient_wrt_F(self):
        """
        Update d(prior distribution) / d(F) = -precision * F
        """
        self._prior_grad_wrt_F = -self.precision * self.F

    def _update_prior_gradient_wrt_log_diag_psi(self):
        """
        Update d(prior distribution) / d(log diag psi) = -0.5 * precision * diag_psi
        """
        self._prior_grad_wrt_log_diag_psi = -0.5 * self.precision * self.diag_psi

    def get_variational_mean(self) -> Tensor:
        """
        Get the mean of the factor analysis variational distribution.

        Returns:
            The mean vector. Of shape (self.weight_dim,).
        """
        return self.c.squeeze()

    def get_variational_covariance(self) -> Tensor:
        """
        Get the full covariance matrix of the factor analysis variational distribution.

        Note: if the network dimension is large, this may result in a memory error.

        Returns:
            The covariance matrix. Of shape (self.weight_dim, self.weight_dim).
        """
        psi = torch.diag(self.diag_psi.squeeze())
        return self.F.mm(self.F.t()) + psi


class Objective:
    """
    An objective function which can be used in an Optuna study to optimise the hyperparameters of the resnet model
    """

    def __init__(
        self,
        all_data,
        train_idx,
        valid_idx,
        test_idx,
        latent_dim: int,
        n_gradients_per_update: int,
        max_grad_norm: float,
        batch_size: int,
        n_epochs: int,
        learning_rate_range: List[float],
        prior_precision_range: List[float],
        n_bma_samples: int,
        random_seed: Optional[int] = None,
        device = "cuda:0",
        epochs_at_stage_1 = 3,
        increase_resistance = 10,
        log_dir = '/home/v1xjian2/BDL/Bayesian_DL/pl_logs/CNN_undertest',
        threshold = 0.8
    ):
        """
        Args:
            epochs_at_stage_1: the number of epochs during which we only train bias parameter c in variational distribution.
            increase_resistance: the bias term in sigma used for coefficient computation, ceof = sigma(epoch - increase_resistance).
            log_dir: the path to store log files generated by tensorboard.
            threshold: the number of machine-made predictions with top uncertainty.
        """
        self._all_data = all_data
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        self.latent_dim = latent_dim
        self.n_gradients_per_update = n_gradients_per_update
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate_range = learning_rate_range
        self.prior_precision_range = prior_precision_range
        self.n_bma_samples = n_bma_samples
        self.random_seed = random_seed
        self.device = device
        self.epochs_at_stage_1 = epochs_at_stage_1
        self.increase_resistance = increase_resistance
        self.log_dir = log_dir
        self.threshold = threshold

        self.train_loader = torch.utils.data.DataLoader(self._all_data, batch_size=self.batch_size, sampler=SubsetRandomSampler(self.train_idx))
        self.val_loader   = torch.utils.data.DataLoader(self._all_data, batch_size=self.batch_size, sampler=SubsetRandomSampler(self.valid_idx))
        self.test_loader  = torch.utils.data.DataLoader(self._all_data, batch_size=self.batch_size, sampler=SubsetRandomSampler(self.test_idx))

        self.use_gpu = not self.device=="cpu"
    
    def __call__(self, trial: optuna.Trial):
        """
        Sample hyperparameters and run train_and_test.
        Args:
            trial: An optuna trial from which to sample hyperparameters.
        Returns:
            Validation accuracy. Higher is better (maximisation).
        """
        print('Allocated Memory:')
        print(torch.cuda.memory_allocated())
        learning_rate   = trial.suggest_loguniform('learning_rate',   *self.learning_rate_range)
        prior_precision = trial.suggest_loguniform('prior_precision', *self.prior_precision_range)

        acc, _ , _ = self.train_and_test(learning_rate, prior_precision, test=False) # here we only run on valid_set, no test_set needed

        return acc
    
    def train_and_test(self, learning_rate, prior_precision, test:bool = False):
        """
        Args:
            learning_rate: learning rate used for training.
            prior_precision: prior_precision used in training. 
            test: If false, test on the validation set; if true, test on the test set. 
        
        Return:
            Metrices from test function.
        """
        model, callbacks = self.train(learning_rate, prior_precision)

        if test == False:
            val_acc, val_loss, val_metric_dirct = self.test(model, callbacks, self.val_loader)
            return val_acc, val_loss, val_metric_dirct

        elif test == True:
            test_acc, test_loss, test_metric_dirct = self.test(model, callbacks, self.test_loader)
            return test_acc, test_loss, test_metric_dirct
        

    def train(self, learning_rate, prior_precision):

        """
        Args:
            learning_rate: learning rate used for training.
            prior_precision: prior_precision used in training.    

        Return:
            Model and Callback after training.
        """

        resnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

        num_ftrs = resnet_model.fc.in_features
        resnet_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 2),
        )

        use_gpu = self.use_gpu

        model = ConvolutionalNet(
            Net = resnet_model,
            optimiser_class = Adam,
            optimiser_kwargs = dict(lr=learning_rate),
            loss_fn = nn.CrossEntropyLoss(),
            loss_multiplier = len(self.train_idx),
            random_seed = self.random_seed,
        )


        init_c = vectorise_weights(model).reshape(-1,1).to(self.device) # use the default init values from pytorch

        callbacks = FactorAnalysisVariationalInferenceCallback(
            init_c = init_c,
            latent_dim = self.latent_dim,
            precision = prior_precision,
            n_gradients_per_update = self.n_gradients_per_update,
            optimiser_class = Adam,
            bias_optimiser_kwargs = dict(lr=learning_rate),
            factors_optimiser_kwargs = dict(lr=learning_rate),
            noise_optimiser_kwargs = dict(lr=learning_rate),
            max_grad_norm = self.max_grad_norm,
            random_seed = self.random_seed,
            device = self.device,
            epochs_at_stage_1 = self.epochs_at_stage_1,
            increase_resistance = self.increase_resistance, 
        )

        tb_logger = pl_loggers.TensorBoardLogger(save_dir = self.log_dir, name=f'lr:{learning_rate}_precision_{prior_precision}_epoch_{self.n_epochs}_seed_{self.random_seed}')

        trainer = Trainer(
            max_epochs = self.n_epochs, 
            callbacks = [callbacks, LearningRateMonitor()], 
            gpus=1 if self.use_gpu else 0,
            logger=tb_logger,
        )

        trainer.fit(model, self.train_loader, self.val_loader)
        
        return model, callbacks


    def test(self, model, callbacks, test_loader):
        """
        Args:
            model: the trained model, is an instance of class ConvolutionalNet.
            callbacks: the trained callback, is an instance of class FactorAnalysisVariationalInferenceCallback.
            test_loader: the dataloader which we would like to evaluate, has two choices, dataloader for valid set or test set. 

        Returns:
            evaliation metrices, acc (float), loss (float) and a metric dictionary containing other metrices we want.
        """

        mm = torch.nn.Softmax(dim=1)
        pred_entropy_list = []
        model_disag_score_list = []
        pred_list = []
        output_softmax_list = []
        target_list = []

        # track test loss
        test_loss = 0.0

        model.eval()
        with torch.no_grad():
            # iterate over test data
            for data, target in tqdm(test_loader):
                # move tensors to GPU if CUDA is available
                if self.use_gpu:
                    data, target = data.cuda(), target.cuda()

                # get posterior predictive mean for predicting
                target_list.extend(target.float().tolist())
                output_softmax = torch.zeros(data.size(0), 2).to(self.device)
                output_softmax_store = torch.tensor([]).to(self.device)

                for i in range(self.n_bma_samples):

                    curr_weights = callbacks.sample_weight_vector()
                    set_weights(model, torch.tensor(curr_weights).reshape(-1))
                    model.to(self.device)
                    # forward pass: compute predicted outputs by passing inputs to the model
                    curr_output = model(data)
                    curr_output_softmax = mm(curr_output) # (data.size(0),2)
                    output_softmax += curr_output_softmax
                    output_softmax_store = torch.cat( (curr_output_softmax.reshape(data.size(0),1,2), output_softmax_store), dim = 1 ) 

                output_softmax = output_softmax / self.n_bma_samples # (data.size(0),2)
                # Predicted probabilities for class 1.
                output_softmax_list += output_softmax[:,1].tolist()
                # Uncertainty measurement
                pred_entropy_list += (predictive_distributions_entropy(output_softmax)).tolist()
                model_disag_score_list += (model_disagreement_score(output_softmax_store)).tolist()

                # calculate the batch loss
                loss = -sum([torch.log2(output_softmax)[i][target[i]] for i in range(data.size(0))]) # cross entropy value
                # update test loss 
                test_loss += loss.data
                # convert output probabilities to predicted class
                _, pred = torch.max(output_softmax, 1)  
                pred_list += pred.tolist()

        test_loss = test_loss/len(test_loader.dataset)  

        metric_dirct = evaluation_function(pred_list=pred_list, output_softmax_list=output_softmax_list, target_list=target_list, \
                        model_disag_score_list=model_disag_score_list, pred_entropy_list=pred_entropy_list, thresholds=self.threshold)      
        
        
        delete_and_clear([model, callbacks, pred_list, output_softmax_list, target_list, model_disag_score_list, pred_entropy_list])

        return round(float(metric_dirct['accuracy_100']), 3), round(float(test_loss), 3), metric_dirct


def run_experiment(
    all_data,
    latent_dim: int,
    n_gradients_per_update: int,
    max_grad_norm: float,
    batch_size: int,
    n_epochs: int,
    learning_rate_range: List[float],
    prior_precision_range: List[float],
    n_bma_samples: int,
    test: bool,
    n_hyperparameter_trials: int = 20,
    n_data_splits: int = 10,
    valid_size: float = 0.2,
    test_size: float = 0.3,
    random_seed: Optional[int] = None,
    device = "cuda:0",
    epochs_at_stage_1 = 3,
    increase_resistance = 10,
    log_dir = '/home/v1xjian2/BDL/Bayesian_DL/pl_logs/CNN_undertest',
    threshold = [0.8],
): 
    """
    Args:
        n_data_splits: the number of train/valid/test splits.
        valid_size: the proportion of data used for validation. 
        test_size: the proportion of data used for testing.
        n_hyperparameter_trials: number of trails for selecting hyperparameter settings (on valid dataset).
        learning_rate_range: range for selecting learning rate used for training.
        prior_precision_range: range for selecting prior_precision used in training. 
        test: If false, test on the validation set; if true, test on the test set. 
        epochs_at_stage_1: the number of epochs during which we only train bias parameter c in variational distribution.
        increase_resistance: the bias term in sigma used for coefficient computation, ceof = sigma(epoch - increase_resistance).
        log_dir: the path to store log files generated by tensorboard.
        threshold: the proportion of machine-made predictions with top uncertainty.

    Returns:
        results and aggregated results, both stored in pd.DataFrame
    """
    num_data = len(all_data)

    indices = list(range(num_data))
    
    results = []

    for i in range(n_data_splits):
        print(f'\nRunning data split {i + 1} of {n_data_splits}...\n')
        np.random.seed()
        np.random.shuffle(indices)
        split1 = int(np.floor(valid_size * num_data))
        split2 = int(np.floor(test_size * num_data))
        train_idx, valid_idx, test_idx = indices[split1+split2:], indices[:split1], indices[split1:split1+split2]
        start_time = time.time()

        trial_results = run_trial(
            all_data = all_data,
            train_idx=train_idx,
            valid_idx=valid_idx,
            test_idx = test_idx,
            latent_dim=latent_dim,
            n_gradients_per_update=n_gradients_per_update,
            max_grad_norm=max_grad_norm,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate_range=learning_rate_range,
            prior_precision_range=prior_precision_range,
            n_bma_samples=n_bma_samples,
            n_hyperparameter_trials=n_hyperparameter_trials,
            random_seed=random_seed,
            device=device,
            epochs_at_stage_1=epochs_at_stage_1,
            increase_resistance=increase_resistance,
            log_dir=log_dir,
            threshold=threshold,
            test = test
        )

        end_time = time.time()
        trial_results['runtime'] = round(end_time - start_time, 3)
        results.append(trial_results)

    results = pd.DataFrame(results)
    agg_results = aggregate_results(results)

    return results, agg_results


def run_trial(
    all_data,
    train_idx,
    valid_idx,
    test_idx,
    latent_dim,
    n_gradients_per_update,
    max_grad_norm,
    batch_size,
    n_epochs,
    learning_rate_range,
    prior_precision_range,
    n_bma_samples,
    n_hyperparameter_trials,
    random_seed,
    device,
    epochs_at_stage_1,
    increase_resistance,
    log_dir,
    threshold,
    test,
):
    """
    Args:
        all_data: contains all data: train, valid and test.
        train_idx: pick these indices of samples to form train set.
        valid_idx: pick these indices of samples to form valid set.
        test_idx:  pick these indices of samples to form test set.
        learning_rate_range: range for selecting learning rate used for training.
        prior_precision_range: range for selecting prior_precision used in training. 
        n_hyperparameter_trials: number of trails for selecting hyperparameter settings (for each train/valid split).
        epochs_at_stage_1: the number of epochs during which we only train bias parameter c in variational distribution.
        increase_resistance: the bias term in sigma used for coefficient computation, ceof = sigma(epoch - increase_resistance).
        log_dir: the path to store log files generated by tensorboard.
        threshold: the number of machine-made predictions with top uncertainty.
        test: If false, test on the validation set; if true, test on the test set. 
    
    Returns:
        results stored in a dict.
    """

    objective = Objective(
        all_data=all_data,
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        latent_dim=latent_dim,
        n_gradients_per_update=n_gradients_per_update,
        max_grad_norm=max_grad_norm,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate_range=learning_rate_range,
        prior_precision_range=prior_precision_range,
        n_bma_samples=n_bma_samples,
        random_seed=random_seed,
        device=device,
        epochs_at_stage_1=epochs_at_stage_1,
        increase_resistance=increase_resistance,
        log_dir=log_dir,
        threshold=threshold
    )

    sampler = optuna.samplers.RandomSampler(seed=random_seed)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(objective, n_trials=n_hyperparameter_trials, gc_after_trial=True)

    learning_rate = study.best_params['learning_rate']
    prior_precision = study.best_params['prior_precision']

    # Train and evaluate (valid set) with the best hyper-parameter setting, record results. 
    # TODO: Too computational complex!
    val_acc, val_loss, val_metric_dirct = objective.train_and_test(learning_rate, prior_precision)

    results = dict(val_acc=val_acc, val_loss=val_loss)
    # results.update(val_metric_dirct)

    if not test:
        return results
    
    # Train and evaluate (test set) with the best hyper-parameter setting, record results. 
    test_acc, test_loss, test_metric_dirct = objective.train_and_test(learning_rate, prior_precision, test=test)

    results['test_acc'] = test_acc
    results['test_loss'] = test_loss
    results.update(test_metric_dirct)

    return results # dirctionary

def aggregate_results(results:pd.DataFrame) -> pd.DataFrame:

    """
    Compute the mean and standard error of each column.
    Args:
        results: Un-aggregated results, of shape (n_rows, n_columns).
    Returns:
        Aggregated results, of shape (n_columns, 5). Columns are the mean, standard error, media, max and min, and
        indices are the columns of the input.
    """
    means = results.mean()
    standard_errors = results.sem()
    medians = results.median()
    maximums = results.max()
    minimums = results.min()

    agg_results = pd.concat([means, standard_errors, medians, maximums, minimums], axis=1)
    agg_results.columns = ['mean', 'se', 'median', 'max', 'min']

    return agg_results

@click.command()
@click.option('--results-output-dir', type=str, help='The directory path to save the results of the experiment')

def main(results_output_dir:str):
    
    with open('/home/v1xjian2/BDL/Bayesian_DL/blindness_detection/blindness_detection_VIFA_params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)['resnet']


    if params['experiment_model'] == 'resnet18' and params['experiment_dataset'] == 'blindness':
        transform = transforms.Compose([
            transforms.Resize((256, 256)), # default mode: BILINEAR
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # pre-processing for resnet 
        ])

        all_data = RetinopathyDataset(csv_file=params['data_csv_path'], image_folder=params['data_image_folder'], transform=transform)

    results, agg_results = run_experiment(
        all_data = all_data,
        latent_dim = params['latent_dim'],
        n_gradients_per_update = params['n_gradients_per_update'],
        max_grad_norm = params['max_grad_norm'],
        batch_size = params['batch_size'],
        n_epochs = params['n_epochs'],
        learning_rate_range = params['learning_rate_range'],
        prior_precision_range = params['prior_precision_range'],
        n_bma_samples = params['n_bma_samples'],
        test = params['test'],
        n_hyperparameter_trials = params['n_hyperparameter_trials'],
        n_data_splits = params['n_data_splits'],
        valid_size = params['valid_size'],
        test_size = params['test_size'],
        random_seed = params['random_seed'],
        device = "cuda:0" if torch.cuda.is_available() else 'cpu',
        epochs_at_stage_1 = params['epochs_at_stage_1'],
        increase_resistance = params['increase_resistance'],
        log_dir = params['log_dir'],
        threshold = params['threshold'],
    )

    Path(results_output_dir).mkdir(parents=True, exist_ok = True)
    results.to_csv(os.path.join(results_output_dir, 'epoch: %2d results.csv' %  (params['n_epochs'])))
    agg_results.to_csv(os.path.join(results_output_dir, 'epoch: %2d aggregate_results.csv' %  (params['n_epochs'])))

if __name__ == '__main__':
    main()