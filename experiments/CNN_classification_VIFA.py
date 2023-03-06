from typing import Dict, List, Optional, Tuple
from copy import deepcopy
import os

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor, seed
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from pytorch_lightning import LightningModule
from torch.optim import Optimizer, Adam, SGD
import sys
sys.path.append('/home/v1xjian2/BDL/Bayesian_DL')
from pytorch_lightning import loggers as pl_loggers 
from pytorch_lightning import Trainer
from swafa.utils import (
    get_callback_epoch_range,
    vectorise_weights,
    vectorise_gradients,
    get_weight_dimension,
    set_weights,
    normalise_gradient,
)
from swafa.fa import OnlineGradientFactorAnalysis
from torch.autograd import Variable

# Here we modify this callback so that it can use initial model parameter value as the initial value of self.c
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import ExponentialLR
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

    def __init__(self, init_c: Tensor, scheduler_class, latent_dim: int, precision: float, n_gradients_per_update: int = 1,
                 optimiser_class: Optimizer = SGD, bias_optimiser_kwargs: Optional[dict] = None,
                 factors_optimiser_kwargs: Optional[dict] = None, noise_optimiser_kwargs: Optional[dict] = None,
                 max_grad_norm: Optional[float] = None, device: Optional[torch.device] = None,
                 random_seed: Optional[int] = None):
                 
        self.init_c = init_c
        self.latent_dim = latent_dim
        self.precision = precision
        self.n_gradients_per_update = n_gradients_per_update
        self.optimiser_class = optimiser_class
        self.scheduler_class = scheduler_class
        self.bias_optimiser_kwargs = bias_optimiser_kwargs or dict(lr=1e-3)
        self.factors_optimiser_kwargs = factors_optimiser_kwargs or dict(lr=1e-3)
        self.noise_optimiser_kwargs = noise_optimiser_kwargs or dict(lr=1e-3)
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.random_seed = random_seed

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
            self._init_optimiser()
            self._init_scheduler()

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
        self.F = Variable(fa.F.data, requires_grad=False)
        self.diag_psi = fa.diag_psi
        self._log_diag_psi = Variable(torch.log(self.diag_psi), requires_grad=False)

        self.c.grad = torch.zeros_like(self.c.data, device=self.device)
        self.F.grad = torch.zeros_like(self.F.data, device=self.device)
        self._log_diag_psi.grad = torch.zeros_like(self._log_diag_psi.data, device=self.device)

    def _init_optimiser(self):
        """
        Initialise the optimiser which will be used to update the parameters of the variational distribution.
        """
        self._optimiser = self.optimiser_class(
            [
                {'params': [self.c], **self.bias_optimiser_kwargs},
                {'params': [self.F], **self.factors_optimiser_kwargs},
                {'params': [self._log_diag_psi], **self.noise_optimiser_kwargs},
            ],
        )
    def _init_scheduler(self):
        """
        Initialise the scheduler which will controls the learning rate for updating parameters of the variational distribution.
        """
        if self.scheduler_class[0] == 'ExponentialLR':
            self._scheduler = ExponentialLR(self._optimiser, self.scheduler_class[1]) # set last_epoch here

    def on_train_epoch_end(self, lightning_module, outputs: None, a: None):
        """
        Schedule learning rate for the next epoch.
        """
        self._scheduler.step()

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
        self._average_and_normalise_gradient(self.F)
        self._average_and_normalise_gradient(self._log_diag_psi)

        self._optimiser.step()
        self._optimiser.zero_grad()

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
        self._update_A()
        self._update_B()
        self._update_C()
        self._update_variational_gradient_wrt_F()
        self._update_variational_gradient_wrt_log_diag_psi()
        self._update_prior_gradient_wrt_c()
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

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

class SimpleNet(nn.Module):
  def __init__(self):
    super(SimpleNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)
  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x


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
                 loss_fn: nn.Module = nn.MSELoss(), loss_multiplier: float = 1.0, random_seed: Optional[int] = None):
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
        self.log('train/correctness_rate', correct_rate, prog_bar=True, logger=True)
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
        self.log('valid/correctness_rate', correct_rate, prog_bar=True, logger=True)
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
        self.log('test/correctness_rate', correct_rate, prog_bar=True, logger=True)
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
        # compare predictions to true label
        correct_tensor = pred.eq(y.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not use_gpu else np.squeeze(correct_tensor.cpu().numpy())
        correct_rate = np.mean(1.0 * correct)
        # Test whether the loss computation is correct
        # print('y_hat', y_hat)
        # print('y', y)
        # print('self.loss_fn(y_hat, y)', self.loss_fn(y_hat, y))
        # tmp_loss = 0
        # for k in range(2):
        #    tmp_loss += torch.log(torch.sum(torch.exp(y_hat[k]))) - torch.log(torch.exp(y_hat[k][y[k]]))
        # print('to verify', tmp_loss)
        # print(stop)
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



### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Load data
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 32
# percentage of training set to use as validation
valid_size = 0.2

# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# choose the training and test datasets
train_data = datasets.CIFAR10(root='/home/v1xjian2/BDL/Bayesian_DL/datasets/cifar-10', train=True,
                              download=False, transform=transform)
test_data = datasets.CIFAR10(root='/home/v1xjian2/BDL/Bayesian_DL/datasets/cifar-10', train=False,
                             download=False, transform=transform)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers) # pin_memory=True
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
    sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)

print('data being loaded!')
# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']   



### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Hyperparameteres
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


random_seed = 987
weight_prior_precision = 0.01
latent_dim = 1
n_gradients_per_update = 12
optimiser_class = Adam
learning_rate = 1e-4
bias_optimiser_kwargs = dict(lr=learning_rate)
factor_optimiser_kwargs = dict(lr=learning_rate)
noise_optimiser_kwargs = dict(lr=learning_rate)


n_samples = (1 - valid_size) * train_data.data.shape[0] # size of train_dataloader
n_epochs = 6000
max_grad_norm = 10 # default = 10
n_bma_samples = 5

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Training
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


SimpleNet_train = SimpleNet()

model = ConvolutionalNet(
    Net = SimpleNet_train,
    optimiser_class = optimiser_class,
    optimiser_kwargs = bias_optimiser_kwargs,
    loss_fn = nn.CrossEntropyLoss(),
    loss_multiplier = n_samples,
    random_seed = random_seed,
)

init_c = vectorise_weights(model).reshape(-1,1).to(device) # use the default init values from pytorch

callbacks = FactorAnalysisVariationalInferenceCallback( # if FixD, use its default fixed dig value; noise_optimiser_kwargs actually not in use
    init_c = init_c,
    scheduler_class = ['ExponentialLR', 0.95],
    latent_dim = latent_dim,
    precision = weight_prior_precision,
    n_gradients_per_update = n_gradients_per_update,
    optimiser_class = optimiser_class,
    bias_optimiser_kwargs = bias_optimiser_kwargs,
    factors_optimiser_kwargs = factor_optimiser_kwargs,
    noise_optimiser_kwargs = noise_optimiser_kwargs,
    max_grad_norm = max_grad_norm,
    random_seed = random_seed,
    device=device,
)

tb_logger = pl_loggers.TensorBoardLogger(save_dir = '/home/v1xjian2/BDL/Bayesian_DL/pl_logs/CNN_undertest', name=f'lr:{learning_rate}_precision_{weight_prior_precision}_epoch_{n_epochs}_seed_{random_seed}')

trainer = Trainer(
    max_epochs = n_epochs, 
    callbacks = [callbacks, LearningRateMonitor()], 
    gpus=1 if use_gpu else 0,
    logger=tb_logger
    )

callbacks.on_fit_start(trainer = trainer, pl_module = model)
variational_mean_init = deepcopy(Tensor.cpu(callbacks.c).numpy())
diag_psi_init = deepcopy(Tensor.cpu(callbacks.diag_psi).numpy())
F_init = deepcopy(Tensor.cpu(callbacks.F).numpy())

trainer.fit(model, train_loader, valid_loader)

variational_mean_after = deepcopy(Tensor.cpu(callbacks.c).numpy())
diag_psi_after = deepcopy(Tensor.cpu(callbacks.diag_psi).numpy())
F_after = deepcopy(Tensor.cpu(callbacks.F).numpy())

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Testing
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

# Sample a weight vector
weights1 = callbacks.sample_weight_vector()
set_weights(model, torch.tensor(weights1).reshape(-1))


# track test loss
test_loss = 0.0
loss_fn = nn.CrossEntropyLoss()
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval()
# iterate over test data
for data, target in test_loader:
    # move tensors to GPU if CUDA is available
    if use_gpu:
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the batch loss
    loss = loss_fn(output.data, target) # nn.CrossEntropyLoss
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)    
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not use_gpu else np.squeeze(correct_tensor.cpu().numpy())
    #print(correct)
    # calculate test accuracy for each object class
    for i in range(len(correct)):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# average test loss
test_loss = test_loss/len(test_loader.dataset)

folder_path = '/home/v1xjian2/BDL/Bayesian_DL/experiments/cnn_results'
file_name = f'results:lr:{learning_rate}_precision_{weight_prior_precision}_epoch_{n_epochs}_seed_{random_seed}'
file_path = os.path.join(folder_path, file_name)

with open(file_path, 'w') as file:
    file.write('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(10):
        if class_total[i] > 0:
            file.write('Test Accuracy of %5s: %2d%% (%2d/%2d)\n' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:   
            file.write('Test Accuracy of %5s: N/A (no training examples)\n' % (classes[i]))

    file.write('\nTest Accuracy (Overall): %2d%% (%2d/%2d)\n' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))


##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
