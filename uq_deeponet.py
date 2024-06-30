#from https://github.com/cmoyacal/DeepONet-Grid-UQ/blob/master/src/models/nns.py

import torch
import torch.nn as nn

from typing import Any
from torch.utils.data import Dataset,DataLoader
from torch import distributions
import numpy as np
import os,sys
from tqdm import tqdm
import torch.nn.functional as F


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def RRMSE(y_true, y_pred):
    """
    Compute the Relative Mean Square Error (RMSE) between y_true and y_pred.

    Parameters:
        y_true (torch.Tensor): Ground truth values.
        y_pred (torch.Tensor): Predicted values.

    Returns:
        torch.Tensor: RMSE value.
    """
    mse  = torch.mean((y_true - y_pred)**2)
    rmse = torch.sqrt(mse)
    
    # Calculate relative RMSE
    rrmse = rmse / torch.sqrt(torch.mean(y_true**2))
    
    return rrmse

class DPODataset(Dataset):
    def __init__(self, U, Utarget, Tout, Z, learnDelta, mode='train'):
        self.u, self.s, self.y, self.z, self.mode = U, Utarget, Tout, Z, mode        
        self.learnDelta = learnDelta

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        u = torch.FloatTensor(self.u[idx])
        y = torch.FloatTensor(self.y[idx])
        s = torch.FloatTensor(self.s[idx])
        if self.learnDelta:
            if self.mode == 'test':
                #only need to return simu data during testing
                z = torch.FloatTensor(self.z[idx])
                return u,y,s,z 
            else:
                return u,y,s 
        else:
            return u, y,s
    
# MLP
class MLP(nn.Module):
    def __init__(self, layer_size: list, activation: str) -> None:
        super(MLP, self).__init__()
        self.net = nn.ModuleList()
        for k in range(len(layer_size) - 2):
            self.net.append(nn.Linear(layer_size[k], layer_size[k+1], bias=True))
            self.net.append(get_activation(activation))
        self.net.append(nn.Linear(layer_size[-2], layer_size[-1], bias=True))
        self.net.apply(self._init_weights)
    
    def _init_weights(self, m: Any) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for k in range(len(self.net)):
            y = self.net[k](y)
        return y
    
# modified MLP
class modified_MLP_original(nn.Module):
    def __init__(self, layer_size: list, activation: str) -> None:
        super(modified_MLP_original, self).__init__()
        self.net = nn.ModuleList()
        self.U = nn.Sequential(nn.Linear(layer_size[0], layer_size[1], bias=True))
        self.V = nn.Sequential(nn.Linear(layer_size[0], layer_size[1], bias=True))
        self.activation = get_activation(activation)
        for k in range(len(layer_size) - 1):
            self.net.append(nn.Linear(layer_size[k], layer_size[k+1], bias=True))
        self.net.apply(self._init_weights)
        self.U.apply(self._init_weights)
        self.V.apply(self._init_weights)
            
    def _init_weights(self, m: Any) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        u = self.activation(self.U(x))
        v = self.activation(self.V(x))

        for k in range(len(self.net) - 1):
            y = self.net[k](x)
            y = self.activation(y)
            x = y * u + (1 - y) * v
        y = self.net[-1](x)

        return y


# modified MLP
class modified_MLP(nn.Module):
    def __init__(self, layer_size: list, activation: str) -> None:
        super(modified_MLP, self).__init__()
        self.net = nn.ModuleList()
        self.U = nn.Sequential(nn.Linear(layer_size[0], layer_size[1], bias=True))
        self.V = nn.Sequential(nn.Linear(layer_size[0], layer_size[1], bias=True))
        self.activation = get_activation(activation)
        for k in range(len(layer_size) - 1):
            self.net.append(nn.Linear(layer_size[k], layer_size[k+1], bias=True))
        self.net.apply(self._init_weights)
        self.U.apply(self._init_weights)
        self.V.apply(self._init_weights)

    def _init_weights(self, m: Any) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        u = self.activation(self.U(x))
        v = self.activation(self.V(x))

        for k in range(len(self.net) - 1):
            y = self.net[k](x)
            y = self.activation(y)
            x = y * u + (1 - y) * v
        y = self.net[-1](x)

        return y


# get activation function from str
def get_activation(identifier: str) -> Any:
    """get activation function."""
    return{
            "elu": nn.ELU(),
            "relu": nn.ReLU(),
            "selu": nn.SELU(),
            "sigmoid": nn.Sigmoid(),
            "leaky": nn.LeakyReLU(),
            "tanh": nn.Tanh(),
            "sin": sin_act(),
            "softplus": nn.Softplus(),
            "Rrelu": nn.RReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "Mish": nn.Mish(),
    }[identifier]

# sin activation function
class sin_act(nn.Module):
    def __init__(self):
        super(sin_act, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)

# vanilla DeepONet
class DeepONet(nn.Module):
    def __init__(self, branch: dict, trunk: dict, use_bias: bool=True) -> None:
        super(DeepONet, self).__init__()
        if branch["type"] == "MLP":
            self.branch = MLP(branch["layer_size"], branch["activation"])
        elif branch["type"] == "modified":
            self.branch = modified_MLP(branch["layer_size"], branch["activation"])
        # trunk
        if trunk["type"] == "MLP":
            self.trunk = MLP(trunk["layer_size"], trunk["activation"])
        elif trunk["type"] == "modified":
            self.trunk = modified_MLP(trunk["layer_size"], trunk["activation"])

        self.use_bias  = use_bias

        if use_bias:
            self.tau = nn.Parameter(torch.rand(1), requires_grad=True) 

    def forward(self, x: list) -> torch.Tensor:
        u, y = x
        B = self.branch(u)
        T = self.trunk(y)
        s = torch.einsum("bi, bi -> b", B, T)
        s = torch.unsqueeze(s, dim=-1)
        
        return s + self.tau if self.use_bias else s


# probabilistic DeepONet
class prob_DeepONet(nn.Module):
    def __init__(self, branch: dict, trunk: dict, use_bias: bool=True, dropoutRate:float =0.25) -> None:
        super(prob_DeepONet, self).__init__()
        if branch["type"] == "MLP":
            self.branch = MLP(branch["layer_size"][:-2], branch["activation"])
        elif branch["type"] == "modified":
            self.branch = modified_MLP(branch["layer_size"][:-2], branch["activation"])
        # trunk
        if trunk["type"] == "MLP":
            self.trunk = MLP(trunk["layer_size"][:-2], trunk["activation"])
        elif trunk["type"] == "modified":
            self.trunk = modified_MLP(trunk["layer_size"][:-2], trunk["activation"])

        self.dropout_rate = dropoutRate
        


        self.use_bias  = use_bias

        if use_bias: 
            self.bias_mu = nn.Parameter(torch.rand(1), requires_grad=True)
            self.bias_std = nn.Parameter(torch.rand(1), requires_grad=True)

        self.branch_mu = nn.Sequential(
            get_activation(branch["activation"]), 
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(branch["layer_size"][-3], branch["layer_size"][-2], bias=True),
            get_activation(branch["activation"]),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(branch["layer_size"][-2], branch["layer_size"][-1], bias=True)
            )
        
        self.branch_std = nn.Sequential(
            get_activation(branch["activation"]), 
            nn.Linear(branch["layer_size"][-3], branch["layer_size"][-2], bias=True),
            get_activation(branch["activation"]),
            nn.Linear(branch["layer_size"][-2], branch["layer_size"][-1], bias=True)
            )

        self.trunk_mu = nn.Sequential(
            get_activation(trunk["activation"]), 
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(trunk["layer_size"][-3], trunk["layer_size"][-2], bias=True),
            get_activation(trunk["activation"]),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(trunk["layer_size"][-2], trunk["layer_size"][-1], bias=True)
            )
        
        self.trunk_std = nn.Sequential(
            get_activation(trunk["activation"]), 
            nn.Linear(trunk["layer_size"][-3], trunk["layer_size"][-2], bias=True),
            get_activation(trunk["activation"]),
            nn.Linear(trunk["layer_size"][-2], trunk["layer_size"][-1], bias=True)
            )

        self.branch_mu.apply(self._init_weights)
        self.branch_std.apply(self._init_weights)
        self.trunk_mu.apply(self._init_weights)
        self.trunk_std.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

    def forward(self, x: list) -> list:
        u, y = x
        #asun: collapse the sequence dimension
        nb, nseq, nf = u.shape
        u = u.view(nb, -1)
        
        b = self.branch(u)

        t = self.trunk(y)

        # branch prediction and UQ
        b_mu = self.branch_mu(b)
        b_std = self.branch_std(b)
        # trunk prediction and UQ
        t_mu = self.trunk_mu(t)
        t_std = self.trunk_std(t)

        # dot product
        mu = torch.einsum("bi, bi -> b", b_mu, t_mu)
        mu = torch.unsqueeze(mu, dim=-1)            

        log_std = torch.einsum("bi, bi -> b", b_std, t_std)
        log_std = torch.unsqueeze(log_std, dim=-1)
        if self.use_bias:
            mu += self.bias_mu
            log_std += self.bias_std

        return (mu, log_std)
    
# compute metrics for N trajectories
def compute_metrics(s_true: list, s_pred: list, metrics: list, verbose: bool=False)-> list:
    out = []
    for m in metrics:
        temp = []
        for k in range(len(s_true)):
            temp.append(m(s_true[k], s_pred[k]))
        out.append(
            [
                np.round(100 * np.max(temp), decimals=5),
                np.round(100 * np.min(temp), decimals=5),
                np.round(100 * np.mean(temp), decimals=5),
            ]
        )
        del temp
    if verbose:
        try:
            print("l1-relative errors: max={:.3f}, min={:.3f}, mean={:.3f}".format(out[0][0], out[0][1], out[0][2]))
            print("l2-relative errors: max={:.3f}, min={:.3f}, mean={:.3f}".format(out[1][0], out[1][1], out[1][2]))
        except:
            print("not the correct metrics")
    return out

# test
from myutil import inverseTransformQ2

def test(net: torch.nn, loader: DataLoader, scaler: Any, learnDelta:bool=False, dqscaler:Any=None) -> Any:
    net.eval()
    mean = []
    std = []
    targets = []
    for abatch in loader:
        if learnDelta:
          u,y,u_true,predu = abatch
        else:
          u,y,u_true = abatch  

        with torch.no_grad():
            u,y = u.to(device),y.to(device)
            #!!!!!!!!!!!!revisit this 
            u = u.squeeze(-1)
            mean_k, log_std_k = net((u,y))
            std_k = torch.exp(log_std_k)
        if learnDelta:
            mean.append(inverseTransformQ2(scaler, mean_k.cpu().detach().numpy(), predu, dqscaler=dqscaler,imethod=2))
            std.append(std_k.cpu().detach().numpy())
            targets.append(inverseTransformQ2(scaler, u_true.cpu().detach().numpy().squeeze(-1), predu, dqscaler=dqscaler, imethod=2))
        else:
            mean.append(inverseTransformQ2(scaler, mean_k.cpu().detach().numpy(), imethod=1))
            std.append(std_k.cpu().detach().numpy())
            targets.append(inverseTransformQ2(scaler, u_true.cpu().detach().numpy().squeeze(-1), imethod=1))
    #convert to numpy array
    mean = np.concatenate(mean, axis=0)
    std = np.concatenate(std, axis=0)
    targets = np.concatenate(targets, axis=0)

    return mean, std, targets

def probabilistic_train(
    model: torch.nn,
    datasets: Any,
    params: dict,
    scheduler_params: dict=None,
    verbose: bool=False,
    loss_history: list=None,
    test_data: list=None,
    model_path: str="best-model.pth",
    metrics: list=None,
    reTrain: bool = False,
    scaler: list = None,
    learn_delta: bool = False,
    dqscaler: Any = None
    ) -> Any:
    
    ## step 1: unpack test data and losses
    #u_test, s_test = test_data
    #L1_history, L2_history = loss_history
    model = model.to(device)
    testLoader   = DataLoader(datasets['test'], 
                              batch_size=params["batch size"], 
                              shuffle=False,
                              drop_last =  False,
                              pin_memory = True)

    if reTrain:
        model.train()
        if verbose:
            print("\n***** Probabilistic Training for {} epochs and using {} data samples*****\n".format(params["epochs"], len(datasets['train'])))

        # ## step 2: split the dataset
        # n_train = int(0.9 * dataset.len)
        # n_val = dataset.len - n_train
        # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])

        ## step 3: load the torch dataset
        #trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=params["batch size"], shuffle=True)
        #valloader = torch.utils.data.DataLoader(val_dataset, batch_size=n_val, shuffle=True)
        trainloader = DataLoader(datasets['train'], batch_size=params["batch size"], 
                                 shuffle=True, 
                                 drop_last =  False,
                                 pin_memory = True)
        valloader   = DataLoader(datasets['val'], batch_size=params["batch size"], 
                                 shuffle=False,
                                 drop_last =  False,
                                 pin_memory = True)
        
        ## step 4: build the optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning rate"])
        if scheduler_params is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=scheduler_params["patience"], 
                                                                verbose=verbose, factor=scheduler_params["factor"])
        else:
            scheduler = None

        ## step 5: define best values, logger and pbar
        best = {}
        best["prob loss"] = np.Inf

        # logger
        logger = {}
        logger["prob loss"] = []
        pbar = tqdm(range(params["epochs"]), file=sys.stdout)

        #loss_fn = F.mse_loss
        loss_fn = RRMSE
        ## step 6: training loop
        for epoch in pbar:
            model.train()
            epoch_loss = 0

            for abatch in trainloader:
                x_batch, y_batch, x_target = abatch
                x_batch,y_batch,x_target = x_batch.to(device),y_batch.to(device),x_target.to(device)

                #!!!!!!!!!!!!revisit this 
                x_batch = x_batch.squeeze(-1)
                ## batch training
                # step a: forward pass
                mean_pred, log_std_pred = model((x_batch,y_batch))

                # step b: compute loss
                #dist = distributions.Normal(mean_pred, torch.exp(log_std_pred))
                #loss = -dist.log_prob(y_batch).mean()
                loss = loss_fn(mean_pred, x_target.squeeze(-1))
                # step c: compute gradients and backpropagate
                optimizer.zero_grad()
                loss.backward()

                # log batch loss
                optimizer.step()
                epoch_loss += loss.detach().cpu().numpy().squeeze()
            try:
                avg_epoch_loss = epoch_loss / len(trainloader)
            except ZeroDivisionError as e:
                print("error: ", e, "batch size larger than number of training examples")

            # log epoch loss
            logger["prob loss"].append(avg_epoch_loss)

            # scheduler
            if scheduler is not None:
                if epoch % params["eval every"] == 0:
                    with torch.no_grad():
                        epoch_val_loss = 0
                        model.eval()
                        for abatch in valloader:
                            x_val_batch, y_val_batch, x_val_target = abatch
                            ## batch validation
                            x_val_batch,y_val_batch,x_val_target = x_val_batch.to(device),y_val_batch.to(device),x_val_target.to(device)
                            #!!!!!!!!!!!!revisit this 
                            x_val_batch = x_val_batch.squeeze(-1)
                            # step a: forward pass without computing gradients
                            mean_val_pred, log_std_val_pred = model((x_val_batch,y_val_batch))

                            # step b: compute validation loss
                            #val_dist = distributions.Normal(mean_val_pred, torch.exp(log_std_val_pred))
                            #val_loss = -val_dist.log_prob(y_val_batch).mean()
                            val_loss = loss_fn(mean_val_pred, x_val_target.squeeze(-1))
                            epoch_val_loss += val_loss.detach().cpu().numpy().squeeze()
                        try:
                            avg_epoch_val_loss = epoch_val_loss / len(valloader)
                        except ZeroDivisionError as e:
                            print("error: ", e, "batch size larger than number of training examples")    
                    ## take a scheduler step
                    scheduler.step(avg_epoch_val_loss)

            #metrics_state = compute_metrics(targets, pred, metrics, verbose=False)
            #L1_history = update_metrics_history(L1_history, metrics_state[0])
            #L2_history = update_metrics_history(L2_history, metrics_state[1])

            if epoch % params["print every"] == 0 or epoch + 1 == params["epochs"]:
                if avg_epoch_val_loss < best["prob loss"]:
                    best["prob loss"] = avg_epoch_val_loss
                    print ('saving best model ....')
                    torch.save(model.state_dict(), model_path)
                
                pbar.set_postfix(
                    {
                        'Train-Loss': avg_epoch_loss,
                        'Best-Loss': best["prob loss"],
                        #'L1-[max, min, mean]': metrics_state[0], 
                        #'L2-[max, min, mean]': metrics_state[1],      
                    })
                # testing
                if learn_delta:
                    pred, std, targets = test(model, testLoader, scaler, learn_delta, dqscaler) 
                else:
                    pred, std, targets = test(model, testLoader, scaler)
                print ('max pred', np.max(pred))

            #del metrics_state
    
    model.load_state_dict(torch.load(model_path))
        
    return model
    