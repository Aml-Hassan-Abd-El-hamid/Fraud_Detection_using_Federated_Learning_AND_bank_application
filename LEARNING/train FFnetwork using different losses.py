import torch
import numpy as np

from tqdm import tqdm

from loss import * 
from utils import show_metrics,clean_data,train,train_roc,back_prob,test,load_data

import rich.traceback as traceback
from rich.console import Console

import typer

from typing import List

from enum import Enum


app = typer.Typer()
console = Console(record=True)
error_console = Console(stderr=True)
traceback.install(show_locals=False)


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

@app.command()
def run(train_data_path:str = typer.Argument(...),
        train_target:str= typer.Option("Class",help="name of the column that contains the labels in the training dataset"),
        device:str= typer.Option("cpu",help="name of the device that is available for training, can be cpu or cuda"),
        num_epochs:int= typer.Option(1,help="number of training epochs"),
        train_drop:List[str]= typer.Option(['Class','Time','Unnamed: 0'],help="list of columns to be dropped from the training dataset"),
        test_data_path:str = typer.Argument(...),
        test_target:str= typer.Option("Class",help="name of the column that contains the labels in the test dataset"),
        test_drop:List[str]= typer.Option(['Class','Time','Unnamed: 0'],help="list of columns to be dropped from the test dataset"),
        batch_size:int=typer.Option(32, help="the size of the batch for training and test data loader"),
        learning_rate:float=typer.Option(.001, help="learning rate"),
        optimizer_name:str= typer.Option("SGD",help="optimizer name to be used in the training, can be SGD or Adam"),
        momentum:float=typer.Option(.9,help="only used if the selected optimizer is SGD"),
        loss_func_name:Loss_Func=typer.Option("Focal_Loss",help="the loss function that'll be used for training, can be one of 4: [\"Focal_Loss\",\"Roc_Star\",\"LDAMLoss\",\"BCE\"]"),
        gamma:float=typer.Option(2,help="only used if the loss function is Focal Loss"),
        pos_weight:float=typer.Option(5,help="wight of the positive class, only used if the loss function is Binary Cross Entropy with Logits"),
        ):
    """
    Testing Focal_Loss, Roc_Star, LDAMLoss, BCEWithLogits loss function on highly imbalanced data
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
                raise ValueError("The chosen loss function need to be one of the following: [\"Focal_Loss\",\"Roc_Star\",\"LDAMLoss\",\"BCE\"] ")
            train(model, train_dl, loss_func, optimizer)

    test(model, test_dl)
    console.log("If you want to save the model just say y or Y")
    save=input()
    if save == "y" or save=="Y":
        torch.save(model,"model.pth")

if __name__ == "__main__":
    app()
