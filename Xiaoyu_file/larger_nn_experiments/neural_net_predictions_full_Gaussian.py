from ast import Str
import logging
import os
from pathlib import Path
import time
from typing import Dict, List, Optional
import warnings

import click
import numpy as np
import optuna
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
import yaml

import sys
sys.path.append('/home/s2174572/BDL/Bayesian_DL')

from experiments.utils.factory import ACTIVATION_FACTORY
from swafa.new_callbacks import FactorAnalysisVariationalInferenceCallback
from swafa.models import FeedForwardGaussianNet
from swafa.utils import set_weights

# turn off annoying pytorch lightning logging and warnings
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')


class Objective:
    """
    An objective function which can be used in an Optuna study to optimise the hyperparameters of a Gaussian neural
    network.

    The performance of each hyperparameter configuration is estimated via cross-validation. In each fold, the training
    data is used to approximate the posterior distribution of the weights of the neural network via the VIFA algorithm.
    Then the approximate posterior is used to compute the average validation marginal log-likelihood.

    The hyperparameters which are tuned are the learning rate with which to update the parameters of the posterior, the
    precision of the prior of the posterior and the precision of the additive noise distribution of the targets.
    Hyperparameter values are sampled from log-uniform distributions. The user must define the hyperparameter ranges.

    Note: since the primary metric is the average marginal log-likelihood, this objective should be MAXIMISED.

    Args:
        dataset: The features and targets to use to perform cross-validation, of shape (n_rows, n_features + 1). Target
            should be in final column.
        n_cv_folds: The number of cross-validation folds.
        latent_dim: The latent dimension of the factor analysis model used to approximate the posterior.
        n_gradients_per_update: The number of mini-batch gradients to use to form the expectation of the true gradient
            for each parameter update of the posterior.
        max_grad_norm: Maximum norm for gradients which are used to update the parameters of the posterior.
        batch_size: The batch size to use while training.
        n_epochs: The number of training epochs.
        learning_rate_range: The minimum and maximum values of the hyperparameter range of the learning rate with which
            to update the parameters of the posterior.
        prior_precision_range: The minimum and maximum values of the hyperparameter range of the precision of the prior
            of the posterior.
        noise_precision_range: The minimum and maximum values of the hyperparameter range of the precision of the
            additive noise distribution of the targets.
        n_bma_samples: The number of samples in each Bayesian model averaging when testing.
        hidden_dims: The dimension of each hidden layer in the neural network. hidden_dims[i] is the dimension of the
            i-th hidden layer. If None, the input will be connected directly to the output.
        hidden_activation_fn: The activation function to apply to the output of each hidden layer. If None, will be set
            to the identity activation function.
        random_seed: The random seed to use when initialising the parameters of the posterior.

    Attributes:
        k_fold: (KFold) Defines the split of each cross-validation fold.
    """
    def __init__(
            self,
            dataset: pd.DataFrame,
            n_cv_folds: int,
            latent_dim: int,
            n_gradients_per_update: int,
            max_grad_norm: float,
            batch_size: int,
            n_epochs: int,
            learning_rate_range: List[float],
            prior_precision_range: List[float],
            noise_precision_range: List[float],
            n_bma_samples: int,
            hidden_dims: Optional[List[int]] = None,
            hidden_activation_fn: Optional[torch.nn.Module] = None,
            random_seed: Optional[int] = None,
            log_path: Str = None,
    ):
        self._dataset = dataset
        self.latent_dim = latent_dim
        self.n_gradients_per_update = n_gradients_per_update
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate_range = learning_rate_range
        self.prior_precision_range = prior_precision_range
        self.noise_precision_range = noise_precision_range
        self.n_bma_samples = n_bma_samples
        self.hidden_dims = hidden_dims
        self.hidden_activation_fn = hidden_activation_fn
        self.random_seed = random_seed
        self.log_path = log_path

        self.k_fold = KFold(n_splits=n_cv_folds)

    @property
    def dataset(self) -> pd.DataFrame:
        return self._dataset

    @dataset.setter
    def dataset(self, value: pd.DataFrame):
        self._dataset = value

    def __call__(self, trial: optuna.Trial):
        """
        Sample hyperparameters and run cross-validation.

        Args:
            trial: An optuna trial from which to sample hyperparameters.

        Returns:
            The average marginal log-likelihood over the cross-validation folds. Higher is better (maximisation).
        """
        learning_rate = trial.suggest_loguniform('learning_rate', *self.learning_rate_range)
        prior_precision = trial.suggest_loguniform('prior_precision', *self.prior_precision_range)
        noise_precision = trial.suggest_loguniform('noise_precision', *self.noise_precision_range)

        cv_mll, _ = self.cross_validate(learning_rate, prior_precision, noise_precision)

        return cv_mll

    def cross_validate(self, learning_rate: float, prior_precision: float, noise_precision: float) -> (float, float):
        """
        Cross-validate a neural network for the given hyperparameters.

        In each fold, use the training data to approximate the posterior distribution of the weights of a neural network
        via the VIFA algorithm. Then use the posterior to compute metrics for the test data.

        Args:
            learning_rate: The learning rate with which to update the parameters of the VIFA callback.
            prior_precision: The precision of the prior of the posterior.
            noise_precision: The precision of the additive noise distribution of the targets.

        Returns:
            The average marginal log-likelihood and root mean squared error across all validation folds.
        """
        mll_list = []
        rmse_list = []
        for train_index, test_index in self.k_fold.split(self.dataset):
            mll, rmse = self.train_and_test(train_index, test_index, learning_rate, prior_precision, noise_precision)
            mll_list.append(mll)
            rmse_list.append(rmse)

        return np.mean(mll_list), np.mean(rmse_list)

    def train_and_test(self, train_index: np.ndarray, test_index: np.ndarray, learning_rate: float,
                       prior_precision: float, noise_precision: float) -> (float, float):
        """
        Train and test a neural network for the given train and test indices and hyperparameters.

        Using the training data, approximate the posterior distribution of the weights of a neural network via the VIFA
        algorithm. Then use the posterior to compute metrics for the test data.

        Args:
            train_index: Train row indices of self.dataset, of shape (n_train,).
            test_index: Test row indices of self.dataset, of shape (n_test,).
            learning_rate: The learning rate with which to update the parameters of the VIFA callback.
            prior_precision: The precision of the prior of the posterior.
            noise_precision: The precision of the additive noise distribution of the targets.

        Returns:
            The average marginal log-likelihood and root mean squared error of the model relative to the true targets of
            the test data.
        """
        train_dataset = self.dataset.iloc[train_index, :]
        test_dataset = self.dataset.iloc[test_index, :]

        X_train, y_train, scaler = self.fit_transform_features_and_targets(train_dataset)

        X_test, y_test = self.transform_features_and_targets(test_dataset, scaler)

        y_mean = scaler.mean_[-1]
        y_scale = scaler.scale_[-1]

        standardised_noise_precision = self.standardise_noise_precision(noise_precision, y_scale)

        model, variational_callback = self.train_model(
            X_train, y_train, learning_rate, prior_precision, standardised_noise_precision,
        )

        mll, rmse = self.test_model(model, variational_callback, X_test, y_test, y_mean, y_scale, noise_precision)

        return mll, rmse

    def fit_transform_features_and_targets(self, dataset: pd.DataFrame) -> (Tensor, Tensor, StandardScaler):
        """
        Fit a standard scaler to the given dataset and use it to scale the data.

        It is assumed that the target is in the final column of the dataset and all other columns are features.

        Args:
            dataset: Features and targets, of shape (n_rows, n_features + 1). Target should be in final column.

        Returns:
            The transformed features of shape (n_rows, n_features), and transformed targets of shape (n_rows,) and the
            scaler used to transform the data.
        """
        scaler = StandardScaler()
        scaler.fit(dataset.values)

        X, y = self.transform_features_and_targets(dataset, scaler)

        return X, y, scaler

    @staticmethod
    def transform_features_and_targets(dataset: pd.DataFrame, scaler: StandardScaler) -> (Tensor, Tensor):
        """
        Transform features and targets in the given dataset using the given scaler.

        It is assumed that the target is in the final column of the dataset and all other columns are features.

        Args:
            dataset: Features and targets, of shape (n_rows, n_features + 1). Target should be in final column.
            scaler: A scaler which has already been fit to training data.

        Returns:
            The transformed features of shape (n_rows, n_features), and transformed targets of shape (n_rows,).
        """
        standardised_dataset = scaler.transform(dataset.values)

        X = torch.from_numpy(standardised_dataset[:, :-1]).float()
        y = torch.from_numpy(standardised_dataset[:, -1]).float()

        return X, y

    @staticmethod
    def standardise_noise_precision(noise_precision: float, y_scale: float) -> float:
        """
        Suppose the original targets were standardised by dividing them by sigma. Then the variance of the standardised
        noise should be old_variance / sigma^2. Hence, the precision of the standardised noise is
        sigma^2 / old_variance = sigma^2 * precision.

        Args:
            noise_precision: The precision of the additive noise distribution of the non-standardised targets.
            y_scale: The standard deviation of the non-standardised training target.

        Returns:
            The precision of the additive noise distribution of the standardised targets.
        """
        return noise_precision * (y_scale ** 2)

    def train_model(self, X: Tensor, y: Tensor, learning_rate: float, prior_precision: float, noise_precision: float,
                    ) -> (FeedForwardGaussianNet, FactorAnalysisVariationalInferenceCallback):
        """
        Given the input data, approximate the posterior distribution of the weights of a neural network via the VIFA
        algorithm.

        Args:
            X: The features, of shape (n_rows, n_features).
            y: The targets, of shape (n_rows,).
            learning_rate: The learning rate with which to update the parameters of the VIFA callback.
            prior_precision: The precision of the prior of the posterior.
            noise_precision: The precision of the additive noise distribution of the targets.

        Returns:
            The model and the callback. The callback can be used to sample weight vectors for the model from the
            approximate posterior.
        """
        n_samples, n_features = X.shape

        optimiser_kwargs = dict(lr=learning_rate)

        model = FeedForwardGaussianNet(
            input_dim=n_features,
            hidden_dims=self.hidden_dims,
            hidden_activation_fn=self.hidden_activation_fn,
            loss_multiplier=n_samples,
            target_variance=1 / noise_precision,
            random_seed=self.random_seed,
        )

        use_gpu = torch.cuda.is_available()

        variational_callback = FactorAnalysisVariationalInferenceCallback(
            latent_dim=self.latent_dim,
            precision=prior_precision,
            n_gradients_per_update=self.n_gradients_per_update,
            optimiser_class=torch.optim.Adam,
            bias_optimiser_kwargs=optimiser_kwargs,
            factors_optimiser_kwargs=optimiser_kwargs,
            noise_optimiser_kwargs=optimiser_kwargs,
            max_grad_norm=self.max_grad_norm,
            random_seed=self.random_seed,
            device=torch.device('cuda:0') if use_gpu else None,
        )

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, drop_last=True, shuffle=True)

        tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"/home/s2174572/BDL/Bayesian_DL/larger_NN_experiments/new_pl_log/neural_net_predictions_full_Gaussian/{self.log_path}") # Check bugs

        trainer = Trainer(
            max_epochs=self.n_epochs,
            callbacks=variational_callback,
            gpus=1 if use_gpu else 0,
            logger=tb_logger,
            # accelerator='gpu' if use_gpu else 'cpu',
            # enable_progress_bar=False,
        )

        trainer.fit(model, dataloader)

        return model, variational_callback

    def test_model(self, model: FeedForwardGaussianNet,
                   variational_callback: FactorAnalysisVariationalInferenceCallback,
                   X: Tensor, y: Tensor, y_mean: float, y_scale: float, noise_precision: float) -> (float, float):
        """
        Use the given model and variational callback to compute the marginal log-likelihood and root mean square error.

        Note: it is assumed that the model was fit to a standardised target variable (zero mean and unit standard
        deviation). However, metrics for non-standardised predictions will be computed.

        Args:
            model: The model to use to make predictions.
            variational_callback: A variational callback which can be used to sample weight vectors for the model.
            X: The features for which to make predictions, of shape (n_rows, n_features).
            y: The standardised training targets, of shape (n_rows,)
            y_mean: The mean of the non-standardised training target.
            y_scale: The standard deviation of the non-standardised training target.
            noise_precision: The un-standardised precision of the additive noise distribution of the targets.

        Returns:
            The average marginal log-likelihood and root mean squared error of the model relative to the
            non-standardised targets.
        """
        y_original = self.de_standardise_target(y, y_mean, y_scale)

        y_preds = self.sample_predictions(model, variational_callback, X, y_mean, y_scale)

        rmse = self.compute_rmse(y_original, y_preds)

        mll = self.compute_marginal_log_likelihood(y_original, y_preds, noise_precision)

        return mll, rmse

    def sample_predictions(self, model: FeedForwardGaussianNet,
                           variational_callback: FactorAnalysisVariationalInferenceCallback, X: Tensor, y_mean: float,
                           y_scale: float) -> (Tensor, Tensor):
        """
        Use the given model and variational callback to sample predictions for each input.

        A total of self.n_bma_samples predictions are made for each input using different weight vectors sampled from
        the variational callback.

        Note: it is assumed that the model was fit to a standardised target variable (zero mean and unit standard
        deviation). However, non-standardised predictions will be returned.

        Args:
            model: The model to use to make predictions.
            variational_callback: A variational callback which can be used to sample weight vectors for the model.
            X: The features for which to make predictions, of shape (n_rows, n_features).
            y_mean: The mean of the non-standardised training target.
            y_scale: The standard deviation of the non-standardised training target.

        Returns:
            The self.n_bma_samples predictions for each input, stored in a tensor of shape (n_rows, self.n_bma_samples).
        """
        return torch.hstack(
            [
                self.predict(model, variational_callback, X, y_mean, y_scale).unsqueeze(dim=1)
                for _ in range(self.n_bma_samples)
            ]
        )

    @staticmethod
    def compute_rmse(y: Tensor, y_preds: Tensor) -> float:
        """
        Compute the root mean square error given the true targets and the predicted targets.

        Args:
            y: The true targets, of shape (n_rows,).
            y_preds: Multiple predictions for each target, of shape (n_rows, n_predictions).

        Returns:
            The root mean squared error between the true targets and the mean of the predicted targets.
        """
        device = y_preds.device

        mu = y_preds.mean(dim=1)
        mse_fn = torch.nn.MSELoss(reduction='mean')

        return mse_fn(mu, y.to(device)).sqrt().item()

    @staticmethod
    def compute_marginal_log_likelihood(y: Tensor, y_preds: Tensor, noise_precision: float) -> float:
        """
        Compute the average marginal log-likelihood given the true targets and the predicted targets.

        Args:
            y: The true targets, of shape (n_rows,).
            y_preds: Multiple predictions for each target, of shape (n_rows, n_predictions).
            noise_precision: The precision of the additive noise distribution of the targets.

        Returns:
            The average marginal log-likelihood.
        """
        device = y_preds.device

        n_predictions = y_preds.shape[1]
        var = torch.ones_like(y, device=device) * (1 / noise_precision)

        nll_fn = torch.nn.GaussianNLLLoss(reduction='none', full=True)
        lls = -torch.hstack(
            [nll_fn(y_preds[:, i], y.to(device), var).unsqueeze(dim=1) for i in range(n_predictions)]
        )

        return torch.logsumexp(lls, dim=1).mean().item() - np.log(n_predictions)

    def predict(self, model: FeedForwardGaussianNet, variational_callback: FactorAnalysisVariationalInferenceCallback,
                X: Tensor, y_mean: float, y_scale: float) -> Tensor:
        """
        Sample a model weight vector from the variational callback and use it to make predictions for the given data.

        Note: it is assumed that the model was fit to a standardised target variable (zero mean and unit standard
        deviation). However, non-standardised predictions will be returned.

        Args:
            model: The model to use to make predictions.
            variational_callback: A variational callback which can be used to sample weight vectors for the model.
            X: The features to make predictions for, of shape (n_rows, n_features).
            y_mean: The mean of the non-standardised training target.
            y_scale: The standard deviation of the non-standardised training target.

        Returns:
            The predictions, of shape (n_rows,). Note that these are non-standardised.
        """
        weights = variational_callback.sample_weight_vector()
        set_weights(model, weights)

        if torch.cuda.is_available():
            model.cuda()

        device = next(model.parameters()).device
        y_pred, _ = model(X.to(device))

        return self.de_standardise_target(y_pred, y_mean, y_scale)

    @staticmethod
    def de_standardise_target(y: Tensor, y_mean: float, y_scale: float) -> Tensor:
        """
        De-standardise the target variable by multiplying by the scaling factor and adding the mean.

        Args:
            y: The standardised target, of shape (n_rows,)
            y_mean: The mean of the non-standardised training target.
            y_scale: The standard deviation of the non-standardised training target.

        Returns:
            The non-standardised target, of shape (n_rows,).
        """
        return y * y_scale + y_mean

    def compute_bayesian_model_average(self, model: FeedForwardGaussianNet,
                                       variational_callback: FactorAnalysisVariationalInferenceCallback,
                                       X: Tensor, y_mean: float, y_scale: float) -> (Tensor, Tensor):
        """
        Use the given model and variational callback to compute a Bayesian model average for each input.

        The Bayesian model average is constructed by making self.n_bma_samples predictions for each input - using
        different weight vectors sampled from the variational callback - and then computing the mean and variance of the
        predictions.

        Note: it is assumed that the model was fit to a standardised target variable (zero mean and unit standard
        deviation). However, non-standardised predictions will be returned.

        Args:
            model: The model to use to make predictions.
            variational_callback: A variational callback which can be used to sample weight vectors for the model.
            X: The features for which to compute the Bayesian model average, of shape (n_rows, n_features).
            y_mean: The mean of the non-standardised training target.
            y_scale: The standard deviation of the non-standardised training target.

        Returns:
            The mean and variance of the predictions for each input, both of shape (n_rows,). Note that these are
            non-standardised.
        """
        ys = torch.hstack(
            [
                self.predict(model, variational_callback, X, y_mean, y_scale).unsqueeze(dim=1)
                for _ in range(self.n_bma_samples)
            ]
        )

        return ys.mean(dim=1), ys.var(dim=1)


def run_experiment(
        dataset: pd.DataFrame,
        n_train_test_splits: int,
        train_fraction: float,
        n_hyperparameter_trials: int,
        n_cv_folds: int,
        latent_dim: int,
        n_gradients_per_update: int,
        max_grad_norm: float,
        batch_size: int,
        n_epochs: int,
        learning_rate_range: List[float],
        prior_precision_range: List[float],
        noise_precision_range: List[float],
        n_bma_samples: int,
        hidden_dims: Optional[List[int]] = None,
        hidden_activation_fn: Optional[torch.nn.Module] = None,
        data_split_random_seed: Optional[int] = None,
        test: bool = False,
        log_path: Str = None,

) -> (pd.DataFrame, pd.DataFrame):
    """
    Run several trials for different train/test splits of the dataset.

    In each trial, using the training data only, run a study to select the best hyperpararameters of an approximate
    posterior distribution of the weights of a neural network trained via the VIFA algorithm. Optionally, use the best
    hyperparameters to fit the posterior to all the training data and compute metrics on the test set.

    The hyperparameters which are tuned are the learning rate with which to update the parameters of the posterior, the
    precision of the prior of the posterior and the precision of the additive noise distribution of the targets.
    Hyperparameter values are sampled from log-uniform distributions. The user must define the hyperparameter ranges.

    Return the mean and standard error of each metric across all trials.

    Args:
        dataset: The features and targets to use to perform training, cross-validation and testing, of shape
            (n_rows, n_features + 1). Target should be in final column.
        n_train_test_splits: The number of random splits of the dataset. For each split, run a hyperparameter study and
            (optionally) test the best set of hyperparameters.
        train_fraction: The fraction of the dataset to include in the training set of each split.
        n_hyperparameter_trials: The number of rounds of hyperparameter optimisation in each study.
        n_cv_folds: The number of cross-validation folds in each hyperparameter trial.
        latent_dim: The latent dimension of the factor analysis model used to approximate the posterior.
        n_gradients_per_update: The number of mini-batch gradients to use to form the expectation of the true gradient
            for each parameter update of the posterior.
        max_grad_norm: Maximum norm for gradients which are used to update the parameters of the posterior.
        batch_size: The batch size to use while training.
        n_epochs: The number of training epochs.
        learning_rate_range: The minimum and maximum values of the hyperparameter range of the learning rate with which
            to update the parameters of the posterior.
        prior_precision_range: The minimum and maximum values of the hyperparameter range of the precision of the prior
            of the posterior.
        noise_precision_range: The minimum and maximum values of the hyperparameter range of the precision of the
            additive noise distribution of the targets.
        n_bma_samples: The number of samples in each Bayesian model averaging when testing.
        hidden_dims: The dimension of each hidden layer in the neural network. hidden_dims[i] is the dimension of the
            i-th hidden layer. If None, the input will be connected directly to the output.
        hidden_activation_fn: The activation function to apply to the output of each hidden layer. If None, will be set
            to the identity activation function.
        data_split_random_seed: The random seed to use to construct the train/test splits.
        test: Whether or not to compute test results.

    Returns:
        results: The average cross-validated marginal log-likelihood (val_mll) and root mean squared error (val_rmse)
            corresponding to the best hyperparameters as well as the runtime for each separate train-test split. Also,
            if test=True, the test marginal log-likelihood (test_mll) and test root mean squared error (test_rmse) for
            each train-test split will also be returned.
        agg_results: The mean, standard error, median, max and min of the average cross-validated marginal
            log-likelihood (val_mll) and root mean squared error (val_rmse) corresponding to the best hyperparameters,
            as well as the runtime. Also, if test=True, the same metrics for the test marginal log-likelihood (test_mll)
            and test root mean squared error (test_rmse) will also be returned.
    """
    np.random.seed(data_split_random_seed)
    train_test_indices = [train_test_split(dataset, train_fraction) for _ in range(n_train_test_splits)]

    results = []
    for i, train_test in enumerate(train_test_indices):
        print(f'Running train/test split {i + 1} of {n_train_test_splits}...')
        train_index, test_index = train_test

        start_time = time.time()

        trial_results = run_trial(
            dataset=dataset,
            train_index=train_index,
            test_index=test_index,
            n_hyperparameter_trials=n_hyperparameter_trials,
            n_cv_folds=n_cv_folds,
            latent_dim=latent_dim,
            n_gradients_per_update=n_gradients_per_update,
            max_grad_norm=max_grad_norm,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate_range=learning_rate_range,
            prior_precision_range=prior_precision_range,
            noise_precision_range=noise_precision_range,
            n_bma_samples=n_bma_samples,
            hidden_dims=hidden_dims,
            hidden_activation_fn=hidden_activation_fn,
            model_random_seed=i,
            test=test,
            log_path=log_path,
        )

        end_time = time.time()
        trial_results['runtime'] = end_time - start_time

        results.append(trial_results)

    results = pd.DataFrame(results)

    agg_results = aggregate_results(results)

    return results, agg_results


def run_trial(
        dataset: pd.DataFrame,
        train_index: np.ndarray,
        test_index: np.ndarray,
        n_hyperparameter_trials: int,
        n_cv_folds: int,
        latent_dim: int,
        n_gradients_per_update: int,
        max_grad_norm: float,
        batch_size: int,
        n_epochs: int,
        learning_rate_range: List[float],
        prior_precision_range: List[float],
        noise_precision_range: List[float],
        n_bma_samples: int,
        hidden_dims: Optional[List[int]] = None,
        hidden_activation_fn: Optional[torch.nn.Module] = None,
        model_random_seed: Optional[int] = None,
        test: bool = False,
        log_path: Str = None,

) -> Dict[str, float]:
    """
    Run a hyperparameter study and testing (optional) for the given train/test split of the dataset.

    In each trial, using the training data only, run a study to select the best hyperpararameters of an approximate
    posterior distribution of the weights of a neural network trained via the VIFA algorithm. Optionally, use the best
    hyperparameters to fit the posterior to all the training data and compute metrics on the test set.

    The hyperparameters which are tuned are the learning rate with which to update the parameters of the posterior, the
    precision of the prior of the posterior and the precision of the additive noise distribution of the targets.
    Hyperparameter values are sampled from log-uniform distributions. The user must define the hyperparameter ranges.

    Args:
        dataset: The features and targets to use to perform training, cross-validation and testing, of shape
            (n_rows, n_features + 1). Target should be in final column.
        train_index: Train row indices of the dataset, of shape (n_train,).
        test_index: Test row indices of the dataset, of shape (n_test,).
        n_hyperparameter_trials: The number of rounds of hyperparameter optimisation.
        n_cv_folds: The number of cross-validation folds.
        latent_dim: The latent dimension of the factor analysis model used to approximate the posterior.
        n_gradients_per_update: The number of mini-batch gradients to use to form the expectation of the true gradient
            for each parameter update of the posterior.
        max_grad_norm: Maximum norm for gradients which are used to update the parameters of the posterior.
        batch_size: The batch size to use while training.
        n_epochs: The number of training epochs.
        learning_rate_range: The minimum and maximum values of the hyperparameter range of the learning rate with which
            to update the parameters of the posterior.
        prior_precision_range: The minimum and maximum values of the hyperparameter range of the precision of the prior
            of the posterior.
        noise_precision_range: The minimum and maximum values of the hyperparameter range of the precision of the
            additive noise distribution of the targets.
        n_bma_samples: The number of samples in each Bayesian model averaging when testing.
        hidden_dims: The dimension of each hidden layer in the neural network. hidden_dims[i] is the dimension of the
            i-th hidden layer. If None, the input will be connected directly to the output.
        hidden_activation_fn: The activation function to apply to the output of each hidden layer. If None, will be set
            to the identity activation function.
        model_random_seed: The random seed to use when initialising the parameters of the posterior.
        test: Whether or not to compute test results after running cross-validation.

    Returns:
        The average cross-validated marginal log-likelihood (val_mll) and root mean squared error (val_rmse)
        corresponding to the best hyperparameters. Also, if test=True, the average test marginal log-likelihood
        (test_ll) and test root mean squared error (test_rmse).
    """
    train_dataset = dataset.iloc[train_index, :]

    objective = Objective(
        dataset=train_dataset,
        n_cv_folds=n_cv_folds,
        latent_dim=latent_dim,
        n_gradients_per_update=n_gradients_per_update,
        max_grad_norm=max_grad_norm,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate_range=learning_rate_range,
        prior_precision_range=prior_precision_range,
        noise_precision_range=noise_precision_range,
        n_bma_samples=n_bma_samples,
        hidden_dims=hidden_dims,
        hidden_activation_fn=hidden_activation_fn,
        random_seed=model_random_seed,
        log_path=log_path,
    )

    sampler = optuna.samplers.RandomSampler(seed=model_random_seed)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(objective, n_trials=n_hyperparameter_trials)

    learning_rate = study.best_params['learning_rate']
    prior_precision = study.best_params['prior_precision']
    noise_precision = study.best_params['noise_precision']

    val_mll, val_rmse = objective.cross_validate(learning_rate, prior_precision, noise_precision)

    results = dict(val_mll=val_mll, val_rmse=val_rmse)

    if not test:
        return results

    objective.dataset = dataset

    test_mll, test_rmse = objective.train_and_test(
        train_index, test_index, learning_rate, prior_precision, noise_precision,
    )

    results['test_mll'] = test_mll
    results['test_rmse'] = test_rmse

    return results


def train_test_split(dataset: pd.DataFrame, train_fraction: float) -> (np.ndarray, np.ndarray):
    """
    Sample train and test indices for the given dataset.

    Args:
        dataset: The dataset for which to get train and test indices. Of shape (n_rows, n_columns).
        train_fraction: The fraction of rows to include in the training set. The remaining fraction will go in the test
            set.

    Returns:
        Train and test indices corresponding to rows of the dataset.
    """
    n_samples = dataset.shape[0]
    permutation = np.random.choice(range(n_samples), n_samples, replace=False)
    end_train = round(n_samples * train_fraction)
    train_index = permutation[:end_train]
    test_index = permutation[end_train:]

    return train_index, test_index


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
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
@click.option('--dataset-label', type=str, help='Label for the dataset. Used to retrieve parameters')
@click.option('--dataset-input-path', type=str, help='The parquet file path to load the dataset')
@click.option('--results-output-dir', type=str, help='The directory path to save the results of the experiment')
def main(dataset_label: str, dataset_input_path: str, results_output_dir: str):
    """
    Run neural network prediction experiment for the given dataset.
    """
    with open("/home/s2174572/BDL/Bayesian_DL/params_new.yaml", 'r') as fd: # new refers to larger NN
        params = yaml.safe_load(fd)['neural_net_predictions']

    dataset_params = params['datasets'][dataset_label]

    # dataset = pd.read_parquet(dataset_input_path)
    dataset = pd.read_csv(dataset_input_path)

    print(f'Running experiment for {dataset_label} dataset...')
    results, agg_results = run_experiment(
        dataset=dataset,
        n_train_test_splits=params['n_train_test_splits'],
        train_fraction=params['train_fraction'],
        n_cv_folds=params['n_cv_folds'],
        n_hyperparameter_trials=params['n_hyperparameter_trials'],
        latent_dim=dataset_params['latent_dim'],
        n_gradients_per_update=dataset_params['n_gradients_per_update'],
        max_grad_norm=dataset_params['max_grad_norm'],
        batch_size=dataset_params['batch_size'],
        n_epochs=dataset_params['n_epochs'],
        learning_rate_range=dataset_params['learning_rate_range'],
        prior_precision_range=dataset_params['prior_precision_range'],
        noise_precision_range=dataset_params['noise_precision_range'],
        n_bma_samples=dataset_params['n_bma_samples'],
        hidden_dims=params['hidden_dims'],
        hidden_activation_fn=ACTIVATION_FACTORY[params['hidden_activation_fn']],
        data_split_random_seed=params['data_split_random_seed'],
        test=params['test'],
        log_path=dataset_label,
    )

    Path(results_output_dir).mkdir(parents=True, exist_ok=True)
    results.to_csv(os.path.join(results_output_dir, 'results.csv'))
    agg_results.to_csv(os.path.join(results_output_dir, 'aggregate_results.csv'))


if __name__ == '__main__':
    main()
