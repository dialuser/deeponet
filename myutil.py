import torch.nn as nn
import torch
import numpy as np
from torch import Tensor

from typing import Optional, Union, Tuple
from typing import Dict, List

from modulus.domain.validator import Validator
from modulus.domain.constraint import Constraint
from modulus.utils.io.vtk import grid_to_vtk
from modulus.utils.io import GridValidatorPlotter
from modulus.graph import Graph
from modulus.key import Key
from modulus.node import Node
from modulus.constants import TF_SUMMARY
from modulus.distributed import DistributedManager
from modulus.dataset import Dataset, DictGridDataset
from modulus.domain.validator.discrete import GridValidator
from modulus.loss import Loss
from modulus.hydra import to_absolute_path, instantiate_arch, ModulusConfig, to_yaml
import pandas as pd

import os, sys

#https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629

import pickle as pkl

class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(
                torch.max(
                   (q-1) * errors, 
                   q * errors
            ).unsqueeze(1))
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss

#from https://github.com/Nixtla/neuralforecast/blob/103c9860c701cbcdadfb278a23d7ced6db447a1b/neuralforecast/losses/pytorch.py#L21
def _divide_no_nan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Auxiliary funtion to handle divide by 0
    """
    div = a / b
    div[div != div] = 0.0
    div[div == float("inf")] = 0.0
    return div


class SMAPE(torch.nn.Module):
    def __init__(self):
        super(SMAPE, self).__init__()

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
    ):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies date stamps per serie to consider in loss.<br>
        **Returns:**<br>
        `smape`: tensor (single value).
        """
        if mask is None:
            mask = torch.ones_like(y_hat)

        delta_y = torch.abs((y - y_hat))
        scale = torch.abs(y) + torch.abs(y_hat)
        smape = _divide_no_nan(delta_y, scale)
        smape = smape * mask
        smape = 2 * torch.mean(smape)
        return smape

class RMSE(torch.nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()    
    
    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
    ):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies date stamps per serie to consider in loss.<br>
        **Returns:**<br>
        `rmse`: tensor (single value).
        """
        if mask is None:
            mask = torch.ones_like(y_hat)

        mse = (y - y_hat) ** 2
        mse = mask * mse
        mse = torch.mean(mse)
        mse = torch.sqrt(mse)
        return mse

class NRMSE(torch.nn.Module):
    def __init__(self):
        super(NRMSE, self).__init__()    
    
    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
    ):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies date stamps per serie to consider in loss.<br>
        **Returns:**<br>
        `rmse`: tensor (single value).
        """
        if mask is None:
            mask = torch.ones_like(y_hat)

        mse = (y - y_hat) ** 2
        mse = mask * mse
        mse = torch.mean(mse)
        mse = torch.sqrt(mse)
        #calculate range of observations
        mse = mse/(torch.max(y)-torch.min(y))
        return mse

class LpLoss(object):
    """from https://github.com/gegewen/ufno/blob/main/lploss.py
    """
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


class Gaussian(nn.Module):

    def __init__(self, hidden_size, output_size):
        '''
        Gaussian Likelihood Supports Continuous Data
        Args:
        input_size (int): hidden h_{i,t} column size
        output_size (int): embedding size
        '''
        super(Gaussian, self).__init__()
        self.mu_layer = nn.Linear(hidden_size, output_size)
        self.sigma_layer = nn.Linear(hidden_size, output_size)

        # initialize weights
        # nn.init.xavier_uniform_(self.mu_layer.weight)
        # nn.init.xavier_uniform_(self.sigma_layer.weight)
    
    def forward(self, h):
        _, hidden_size = h.size()
        sigma_t = torch.log(1 + torch.exp(self.sigma_layer(h))) + 1e-6
        sigma_t = sigma_t.squeeze(0)
        mu_t = self.mu_layer(h).squeeze(0)
        return mu_t, sigma_t


class GaussianLikelihoodLoss(nn.Module):
    '''
    Gaussian Liklihood Loss
    Args:
    z (tensor): true observations, shape (num_ts, num_periods)
    mu (tensor): mean, shape (num_ts, num_periods)
    sigma (tensor): standard deviation, shape (num_ts, num_periods)

    likelihood: 
    (2 pi sigma^2)^(-1/2) exp(-(z - mu)^2 / (2 sigma^2))

    log likelihood:
    -1/2 * (log (2 pi) + 2 * log (sigma)) - (z - mu)^2 / (2 sigma^2)
    '''
    def __init__(self, size_average=True, reduction=True):
        super(GaussianLikelihoodLoss, self).__init__()
        self.reduction = reduction
        self.size_average = size_average

    def forward(self, z, mu, sigma):
        negative_likelihood = torch.log(sigma + 1) + (z - mu) ** 2 / (2 * sigma ** 2) + 6
        if self.reduction:
            if self.size_average:
                return negative_likelihood.mean()
            else:
                return torch.sum(negative_likelihood)


class MyGridValidator(GridValidator):
    """Data-driven grid field validator

    Parameters
    ----------
    nodes : List[Node]
        List of Modulus Nodes to unroll graph with.
    dataset: Dataset
        dataset which contains invar and true outvar examples
    batch_size : int, optional
            Batch size used when running validation, by default 100
    plotter : GridValidatorPlotter
        Modulus plotter for showing results in tensorboard.
    requires_grad : bool = False
        If automatic differentiation is needed for computing results.
    num_workers : int, optional
        Number of dataloader workers, by default 0
    """

    def __init__(
        self,
        nodes: List[Node],
        dataset: Dataset,
        batch_size: int = 100,
        plotter: GridValidatorPlotter = None,
        requires_grad: bool = False,
        num_workers: int = 0,
    ):

        # get dataset and dataloader
        self.dataset = dataset
        self.dataloader = Constraint.get_dataloader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            distributed=False,
            infinite=False,
        )

        # construct model from nodes
        self.model = Graph(
            nodes,
            Key.convert_list(self.dataset.invar_keys),
            Key.convert_list(self.dataset.outvar_keys),
        )
        self.manager = DistributedManager()
        self.device = self.manager.device
        self.model.to(self.device)

        # set foward method
        self.requires_grad = requires_grad
        self.forward = self.forward_grad if requires_grad else self.forward_nograd

        # set plotter
        self.plotter = plotter

    def save_results(self, name, results_dir, writer, save_filetypes, step):

        invar_cpu = {key: [] for key in self.dataset.invar_keys}
        true_outvar_cpu = {key: [] for key in self.dataset.outvar_keys}
        pred_outvar_cpu = {key: [] for key in self.dataset.outvar_keys}
        # Loop through mini-batches
        for i, (invar0, true_outvar0, lambda_weighting) in enumerate(self.dataloader):
            # Move data to device (may need gradients in future, if so requires_grad=True)
            
            invar = Constraint._set_device(
                invar0, device=self.device, requires_grad=self.requires_grad
            )
            true_outvar = Constraint._set_device(
                true_outvar0, device=self.device, requires_grad=self.requires_grad
            )
            pred_outvar = self.forward(invar)

            # Collect minibatch info into cpu dictionaries
            invar_cpu = {
                key: value + [invar[key].cpu().detach()]
                for key, value in invar_cpu.items()
            }
            true_outvar_cpu = {
                key: value + [true_outvar[key].cpu().detach()]
                for key, value in true_outvar_cpu.items()
            }
            pred_outvar_cpu = {
                key: value + [pred_outvar[key].cpu().detach()]
                for key, value in pred_outvar_cpu.items()
            }

        # Concat mini-batch tensors
        invar_cpu = {key: torch.cat(value) for key, value in invar_cpu.items()}

        true_outvar_cpu = {
            key: torch.cat(value) for key, value in true_outvar_cpu.items()
        }

        pred_outvar_cpu = {
            key: torch.cat(value) for key, value in pred_outvar_cpu.items()
        }
        # compute losses on cpu
        losses = MyGridValidator._l2_relative_error(true_outvar_cpu, pred_outvar_cpu)
        print ('validation losses', losses['l2_relative_error_u'].item())
        # convert to numpy arrays
        invar = {k: v.numpy() for k, v in invar_cpu.items()}
        true_outvar = {k: v.numpy() for k, v in true_outvar_cpu.items()}
        pred_outvar = {k: v.numpy() for k, v in pred_outvar_cpu.items()}

        # save batch to vtk file TODO clean this up after graph unroll stuff
        named_true_outvar = {"true_" + k: v for k, v in true_outvar.items()}
        named_pred_outvar = {"pred_" + k: v for k, v in pred_outvar.items()}
        
        # save batch to vtk/npz file TODO clean this up after graph unroll stuff
        if "np" in save_filetypes:
            np.savez(
                results_dir + name, {**invar, **named_true_outvar, **named_pred_outvar}
            )
        if "vtk" in save_filetypes:

            grid_to_vtk(
                {**invar, **named_true_outvar, **named_pred_outvar}, results_dir + name
            )
        # add tensorboard plots
        if self.plotter is not None:
            self.plotter._add_figures(
                "Validators",
                name,
                results_dir,
                writer,
                step,
                invar,
                true_outvar,
                pred_outvar,
            )

        # add tensorboard scalars
        for k, loss in losses.items():
            if TF_SUMMARY:
                writer.add_scalar("val/" + name + "/" + k, loss, step, new_style=True)
            else:
                writer.add_scalar(
                    "Validators/" + name + "/" + k, loss, step, new_style=True
                )
        return losses

    @staticmethod
    def _l2_relative_error(true_var, pred_var):  # TODO replace with metric classes
        new_var = {}
        for key in true_var.keys():
            new_var["l2_relative_error_" + str(key)] = torch.sqrt(
                torch.sum(
                    torch.square(torch.reshape(true_var[key], (-1, 1)) - pred_var[key])
                )/torch.var(true_var[key])
            )
        return new_var


class MyPointwiseLossNorm(Loss):
    """
    L-p loss function for pointwise data
    Computes the p-th order loss of each output tensor

    Parameters
    ----------
    ord : int
        Order of the loss. For example, `ord=2` would be the L2 loss.
    """

    def __init__(self, ord: int = 2):
        super().__init__()
        self.ord: int = ord

    @staticmethod
    def _loss(
        invar: Dict[str, Tensor],
        pred_outvar: Dict[str, Tensor],
        true_outvar: Dict[str, Tensor],
        lambda_weighting: Dict[str, Tensor],
        step: int,
        ord: float,
    ) -> Dict[str, Tensor]:
        losses = {}
        for key, value in pred_outvar.items():
            l = lambda_weighting[key] * torch.abs(
                pred_outvar[key] - true_outvar[key]
            ).pow(ord)
            if "area" in invar.keys():
                l *= invar["area"]
            losses[key] = l.sum()
        return losses

    def forward(
        self,
        invar: Dict[str, Tensor],
        pred_outvar: Dict[str, Tensor],
        true_outvar: Dict[str, Tensor],
        lambda_weighting: Dict[str, Tensor],
        step: int,
    ) -> Dict[str, Tensor]:
        return MyPointwiseLossNorm._loss(
            invar, pred_outvar, true_outvar, lambda_weighting, step, self.ord
        )

def loadUSGSObs(gageid, startDate, endDate, returnDF=False):
    """This pkl file was generated by running getUSGSdata.py in conda mympi !!!
    """    
    data = pkl.load(open(to_absolute_path('data/usgs_data.pkl'), 'rb'))
    valid_stations = data.keys() 
    assert (gageid in valid_stations)
    df = data[gageid]
    df = df[(df.index >= pd.to_datetime(startDate).tz_localize('UTC')) 
             & (df.index <= pd.to_datetime(endDate).tz_localize('UTC'))]


    if returnDF:
        return df
    else:
        return df['Q'].values #in [m3/s]

def getEnsembleUSGSData(usgsDict, gageid, startDate, endDate, returnDF= False):
    """This pkl file was generated by running ats/readensemble.py
    """        
    valid_stations = usgsDict.keys() 
    assert (gageid in valid_stations)
    df = usgsDict[gageid]
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[(df.index >= pd.to_datetime(startDate).tz_localize('UTC')) 
             & (df.index <= pd.to_datetime(endDate).tz_localize('UTC'))]

    if returnDF:
        return df
    else:
        return df['Q'].values #in [m3/s]



def transformQ(arr, imethod=1):
    if imethod == 1:
        return np.log(arr+1e-4) 

def inverseTransformQ_simple(arr):
        return np.exp(arr) - 1e-4

def inverseTransformQ(scaler, arr, imethod=1):
    """inverse transform the Q
    """
    mu,std = scaler
    if imethod == 1:
        return np.exp(arr*std+mu)-1e-4

def inverseTransformQ2(basescaler, AR,  predu=None, dqscaler=None, imethod=1):
    """inverse transform the Q
    """
    mu,std = basescaler
    if imethod == 1:
        return np.exp(AR*std+mu)-1e-4
    
    elif imethod == 2:
        arr = []
        id_test = dqscaler['id_test']
        for ix,id in enumerate(id_test):
            minDQ,maxDQ = dqscaler[id]
            arr.append(0.5*(AR+1.0)*(maxDQ-minDQ)+minDQ)
        arr = np.concatenate(arr)
        predu = predu.squeeze(-1).cpu().detach().numpy()

        return np.exp((arr + predu)*std+mu)-1e-4


def printParams(paramValues):
    paramNames = [
    "priestley_taylor_alpha-canopy",
    "priestley_taylor_alpha-bare ground",
    "priestley_taylor_alpha-snow",
    "priestley_taylor_alpha-transpiration",
    "snowmelt_rate",
    "snowmelt_degree_diff",
    "manning_n",
    "perm_NRCS-295484",
    "perm_NRCS-295105",
    "perm_NRCS-295545",
    "perm_NRCS-295064",
    "perm_NRCS-295142",
    "perm_fractured_bedrock"]
    for param, val in zip(paramNames,paramValues):
        print (f"{param}, {val:6.3f}")

def setLossFun(itype):
    if itype == 1:
        return torch.nn.MSELoss()
    elif itype == 2:
        return LpLoss(p=1, d=2)
    elif itype == 3:
        return LpLoss(p=2, d=2)
    elif itype == 4:
        return RMSE()
    elif itype == 5:
        return NRMSE()
    elif itype == 6:
        return SMAPE()
        