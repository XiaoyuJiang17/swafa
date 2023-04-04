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

def evaluation_function(pred_list, output_softmax_list, target_list):

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
    '''
    Args:
        pred_list: List of predicted class for test samples.
        output_softmax_list: List of softmax predictions for test samples.
        target_list: List of labels of test samples.

    Returns:
        All metrices we want, stored in a dirctionary.
        The metrices are:   
            1. Test accuracy
            2. Precision
            3. Recall
            4. AU-ROC
            5. AU-PRC
            6. F1 score
    '''

    metric_dirct = {}
    curr_accuracy = accuracy_score(target_list, pred_list)
    curr_precision = precision_score(target_list, pred_list)
    curr_recall = recall_score(target_list, pred_list)
    curr_f1 = f1_score(target_list, pred_list)
    curr_auroc = roc_auc_score(target_list, output_softmax_list)
    curr_auprc = average_precision_score(target_list, output_softmax_list)

    metric_dirct["accuracy"] = round(curr_accuracy,3)
    metric_dirct["precision"] = round(curr_precision,3)
    metric_dirct["recall"] = round(curr_recall,3)
    metric_dirct["f1"] = round(curr_f1,3)
    metric_dirct["auroc"] = round(curr_auroc,3)
    metric_dirct["auprc"] = round(curr_auprc,3)

    return metric_dirct


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

    def configure_optimizers(self) -> Dict:
        """
        Initialise the optimiser and lr_scheduler which will be used to train the neural network.

        Returns:
            The initialised optimiser, lr_scheduler
        """
        optimizer = self.optimiser_class(self.net.parameters(), **self.optimiser_kwargs)

        scheduler = {
            'scheduler': ExponentialLR(optimizer, gamma=0.95),
            'interval': 'epoch', 
            'frequency': 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

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
        random_seed: Optional[int] = None,
        device = "cuda:0",
        log_dir = '/home/v1xjian2/BDL/Bayesian_DL/pl_logs/CNN_undertest',
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
        self.random_seed = random_seed
        self.device = device
        self.log_dir = log_dir

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
        learning_rate   = trial.suggest_loguniform('learning_rate',   *self.learning_rate_range)
        prior_precision = trial.suggest_loguniform('prior_precision', *self.prior_precision_range)

        acc, _ , _ = self.train_and_test(learning_rate, prior_precision) # here we only run on valid_set, no test_set needed

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
        model = self.train(learning_rate, prior_precision)

        if test == False:
            val_acc, val_loss, val_metric_dirct = self.test(model, self.val_loader)
            return val_acc, val_loss, val_metric_dirct

        elif test == True:
            test_acc, test_loss, test_metric_dirct = self.test(model, self.test_loader)
            return test_acc, test_loss, test_metric_dirct
        

    def train(self, learning_rate, prior_precision):

        """
        Args:
            learning_rate: learning rate used for training.
            prior_precision: prior_precision used in training.    

        Return:
            Model after training.
        """

        resnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

        num_ftrs = resnet_model.fc.in_features
        resnet_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 2),
        )

        

        model = ConvolutionalNet(
            Net = resnet_model,
            optimiser_class = Adam,
            optimiser_kwargs = dict(lr=learning_rate),
            loss_fn = nn.CrossEntropyLoss(),
            loss_multiplier = len(self.train_idx),
            random_seed = self.random_seed,
        )

        if self.use_gpu:
            model = model.cuda()
            
        tb_logger = pl_loggers.TensorBoardLogger(save_dir = self.log_dir, name=f'lr:{learning_rate}_precision_{prior_precision}_epoch_{self.n_epochs}_seed_{self.random_seed}')

        trainer = Trainer(
            max_epochs = self.n_epochs, 
            gradient_clip_val = self.max_grad_norm,
            accumulate_grad_batches = self.n_gradients_per_update,
            gpus=1 if self.use_gpu else 0,
            logger=tb_logger,
        )

        trainer.fit(model, self.train_loader, self.val_loader)

        return model


    def test(self, model, test_loader):
        """
        Args:
            model: the trained model, is an instance of class ConvolutionalNet.
            test_loader: the dataloader which we would like to evaluate, has two choices, dataloader for valid set or test set. 

        Returns:
            evaliation metrices, acc (float), loss (float) and a metric dictionary containing other metrices we want.
        """

        mm = torch.nn.Softmax(dim=1)
        
        pred_list = []
        output_softmax_list = [] # Only store output probabilities for class 1.
        target_list = []

        # track test loss
        test_loss = 0.0

        if self.use_gpu:
            model = model.cuda()
            
        model.eval()
        with torch.no_grad():
            # iterate over test data
            for data, target in tqdm(test_loader):
                # move tensors to GPU if CUDA is available
                if self.use_gpu:
                    data, target = data.cuda(), target.cuda()

                target_list.extend(target.float().tolist())

                output_softmax = mm(model(data))
                # Only store output probabilities for class 1.
                output_softmax_list.extend(output_softmax[:,1].tolist())

                # calculate the batch loss
                loss = -sum([torch.log2(output_softmax)[i][target[i]] for i in range(data.size(0))]) # cross entropy value
                # update test loss 
                test_loss += loss.data
                # convert output probabilities to predicted class
                _, pred = torch.max(output_softmax, 1)  
                pred_list += pred.tolist()

        test_loss = test_loss/len(test_loader.dataset)  

        metric_dirct = evaluation_function(pred_list=pred_list, output_softmax_list=output_softmax_list, target_list=target_list)

        return round(float(metric_dirct['accuracy']), 3), round(float(test_loss), 3), metric_dirct


def run_experiment(
    all_data,
    latent_dim: int,
    n_gradients_per_update: int,
    max_grad_norm: float,
    batch_size: int,
    n_epochs: int,
    learning_rate_range: List[float],
    prior_precision_range: List[float],
    test: bool,
    n_hyperparameter_trials: int = 20,
    n_data_splits: int = 10,
    valid_size: float = 0.2,
    test_size: float = 0.3,
    random_seed: Optional[int] = None,
    device = "cuda:0",
    log_dir = '/home/v1xjian2/BDL/Bayesian_DL/pl_logs/CNN_undertest',
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
        log_dir: the path to store log files generated by tensorboard.

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
            n_hyperparameter_trials=n_hyperparameter_trials,
            random_seed=random_seed,
            device=device,
            log_dir=log_dir,
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
    n_hyperparameter_trials,
    random_seed,
    device,
    log_dir,
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
        log_dir: the path to store log files generated by tensorboard.
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
        random_seed=random_seed,
        device=device,
        log_dir=log_dir,
    )

    sampler = optuna.samplers.RandomSampler(seed=random_seed)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(objective, n_trials=n_hyperparameter_trials)

    learning_rate = study.best_params['learning_rate']
    prior_precision = study.best_params['prior_precision']

    # Train and evaluate (valid set) with the best hyper-parameter setting, record results. 
    # TODO: Too computational complex!
    val_acc, val_loss, val_metric_dirct = objective.train_and_test(learning_rate, prior_precision)

    results = dict(val_acc=val_acc, val_loss=val_loss)

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
    
    with open('/home/v1xjian2/BDL/Bayesian_DL/blindness_detection/blindness_detection_SGD_params.yaml', 'r') as fd:
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
        test = params['test'],
        n_hyperparameter_trials = params['n_hyperparameter_trials'],
        n_data_splits = params['n_data_splits'],
        valid_size = params['valid_size'],
        test_size = params['test_size'],
        random_seed = params['random_seed'],
        device = "cuda:0" if torch.cuda.is_available() else 'cpu',
        log_dir = params['log_dir'],
    )

    Path(results_output_dir).mkdir(parents=True, exist_ok = True)
    results.to_csv(os.path.join(results_output_dir, 'epoch: %2d results.csv' %  (params['n_epochs'])))
    agg_results.to_csv(os.path.join(results_output_dir, 'epoch: %2d aggregate_results.csv' %  (params['n_epochs'])))

if __name__ == '__main__':
    main()