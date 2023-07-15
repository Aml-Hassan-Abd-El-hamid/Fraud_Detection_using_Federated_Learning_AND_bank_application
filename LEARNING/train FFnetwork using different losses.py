import torch
import numpy as np

from tqdm import tqdm

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from model import *
from loss import * 
from utils import show_metrics,clean_data

import rich.traceback as traceback
from rich.console import Console

import typer

from typing import List

from enum import Enum


app = typer.Typer()
console = Console(record=True)
error_console = Console(stderr=True)
traceback.install(show_locals=False)


device="cpu"#parmeter_to_be_taken_in
num_epochs=1#parmeter_to_be_taken_in

def back_prob(loss, optimizer):
    loss.retain_grad()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
#training the network
from tqdm import tqdm
def train_roc( model, train_dl,optimizer,y_train,epochs=1,epoch_gamma=.2,device=device):
    last_epoch_y_pred = torch.ones(y_train.size ).to(device)
    last_epoch_y_t    = torch.from_numpy(y_train).to(device)
    model.train()
    for epoch in tqdm(range(epochs),desc="train", position=0, leave=True):
        epoch_y_pred=[]
        epoch_y_t=[]
        for xb, yb in train_dl:
            preds=model(xb)
            loss = roc_star_loss(yb.unsqueeze(1),preds,epoch_gamma, last_epoch_y_t, last_epoch_y_pred)
            back_prob(loss ,optimizer)
            epoch_y_pred.extend(preds)
            epoch_y_t.extend(yb)
        last_epoch_y_pred = torch.tensor(epoch_y_pred).to(device)
        last_epoch_y_t = torch.tensor(epoch_y_t).to(device)
        epoch_gamma = epoch_update_gamma(last_epoch_y_t, last_epoch_y_pred, epoch,device=device)
        
def train(model, train_loader, loss_func, optimizer,num_epochs=1):
    model.train()
    for epoch in tqdm(range(num_epochs)):
        for data, target in train_loader:
            output=model.forward(data.to(device))
            loss = loss_func(output.squeeze(),target.float())#.to(torch.int64))
            back_prob(loss ,optimizer)

def test(model, test_loader):
    model.eval()
    y_test_pred=[]
    y_test_true=[]
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            if output.shape[-1]!=1:
                _, output = output.max(1)

            y_test_pred+=output.detach().numpy().tolist()
            y_test_true+=target.numpy().tolist()
            
    y_test_pred=np.array(y_test_pred)
    y_test_true=np.array(y_test_true)
    if output.shape[-1]!=1:
         show_metrics(y_test_true, y_test_pred)
    else:
        y_test_pred [y_test_pred>=0.5] =1.0
        y_test_pred [y_test_pred<0.5] =0.0
        y_test_pred=np.squeeze(y_test_pred)
        show_metrics(y_test_true, y_test_pred)

class Loss_Func(Enum):
    Roc_Star="Roc_Star"
    Focal_Loss="Focal_Loss"
    LDAMLoss = "LDAMLoss"
    BCE = "BCE"

def define_opt(optimizer_name,model,learning_rate,momentum):
    if optimizer_name=="SGD":
            optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer_name=="Adam":
            optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate, momentum=momentum)
    else: 
        raise ValueError("You need to choose SGD or Adam as an optimizer")
    
    return optimizer

def load_data(data_path,target,drop,batch_size):
    x, y= clean_data(data_path,target,drop)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy (y))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,drop_last=True)
    return loader,x.shape[-1],y


@app.command()
def run(train_data_path:str = typer.Argument(...),
        train_target:str= typer.Option("Class",help="name of the column that contains the labels in the training dataset"),
        train_drop:List[str]= typer.Option(['Class','Time','Unnamed: 0'],help="list of columns to be dropped from the training dataset"),
        test_data_path:str = typer.Argument(...),
        test_target:str= typer.Option("Class",help="name of the column that contains the labels in the test dataset"),
        test_drop:List[str]= typer.Option(['Class','Time','Unnamed: 0'],help="list of columns to be dropped from the test dataset"),
        batch_size:int=typer.Option(32, help="the size of the batch for trainig and test data loader"),
        learning_rate:float=typer.Option(.001, help="learning rate"),
        optimizer_name:str= typer.Option("SGD",help="optimizer name to be used in the tranig, can be SGD or Adam"),
        momentum:float=typer.Option(.9,help="only used if the selected optimizer is SGD"),
        loss_func_name:Loss_Func=typer.Option("Focal_Loss",help="the loss function that'll be used for trainig, can be one of 4: [\"Focal_Loss\",\"Roc_Star\",\"LDAMLoss\",\"BCE\"]"),
        gamma:float=typer.Option(2,help="only used if the loss function is Focal Loss"),
        pos_weight:float=typer.Option(5,help="wight of the positive class, only used if the loss function is Binary Cross Entropy with Logits"),
        ):
    """
    Testing Focal_Loss, Roc_Star, LDAMLoss , BCEWithLogits loss function on higly imblanced data
    """
    
    train_dl,input_shape,y_train=load_data(train_data_path,train_target,train_drop,batch_size)
    test_dl,_,_=load_data(test_data_path,test_target,test_drop,batch_size)
    
    if loss_func_name.value=="LDAMLoss":
        model=Net(input_shape,out_shape=2,sig=False)
        optimizer=define_opt(optimizer_name,model,learning_rate,momentum)
        cls_num_list = [y_train[y_train==1].size, y_train[y_train==0].size]
        per_cls_weights=torch.FloatTensor(cls_num_list / np.sum(cls_num_list) * len(cls_num_list)).to(device)#torch.tensor([0.008, 2.2])
        loss_func = LDAMLoss(device, cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights)
        train(model, train_dl, loss_func, optimizer)
    else:
        model=Net(input_shape)
        optimizer=define_opt(optimizer_name,model,learning_rate,momentum)
        if loss_func_name.value=="Roc_Star":
            train_roc( model, train_dl,optimizer,y_train=y_train)
        else:
            if loss_func_name.value=="Focal_Loss":
                loss_func=Focal_Loss(gamma)
            elif loss_func_name.value=="BCE":
                print("hello")
                loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
            else:
                raise ValueError("The chosen loss fuction need to be one of the follwing: [\"Focal_Loss\",\"Roc_Star\",\"LDAMLoss\",\"BCE\"] ")
            train(model, train_dl, loss_func, optimizer)

    test(model, test_dl)
    console.log("If you want to save the model just say y or Y")
    save=input()
    if save == "y" or save=="Y":
        torch.save(model,"model.pth")

if __name__ == "__main__":
    app()
