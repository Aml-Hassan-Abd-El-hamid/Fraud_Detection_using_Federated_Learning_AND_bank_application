from collections import Counter
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score,precision_score,confusion_matrix, accuracy_score,ConfusionMatrixDisplay,average_precision_score,roc_auc_score, f1_score
from sklearn.preprocessing import MinMaxScaler, minmax_scale
import torch
from tqdm import tqdm
device = "CPU"

def back_prob(loss, optimizer):
    """
    perform backpropgation
    Args:
        loss: the loss function to be used in the training
        optimizer: the optimizer to be used in the training
    """
    loss.retain_grad()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
def train_roc( model, train_dl,optimizer,y_train,epochs=1,epoch_gamma=.2,device=device):
    """
    train the model network with the roc_star loss function
    Args:
      model: the network to be trained
      train_loader: a PyTorch loader of the training dataset
      optimizer: the optimizer to be used in the training
      y_train: training labels
    """
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
    """
    train the model network with different loss functions
    Args:
      model: the network to be trained
      train_loader: a PyTorch loader of the training dataset
      loss_func: the loss function to be used in the training
      optimizer: the optimizer to be used in the training
    """
    model.train()
    for epoch in tqdm(range(num_epochs)):
        for data, target in train_loader:
            output=model.forward(data.to(device))
            loss = loss_func(output.squeeze(),target.float())#.to(torch.int64))
            back_prob(loss ,optimizer)

def test(model, test_loader):
    """
    test the model and show its performance
    Args:
      model: the network to be tested
      test_loader: a PyTorch loader of the test dataset
    """
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

def show_metrics(target,results):
    """
    show different metrics and display confusion matrix
    Args:
      target: the true value of the labels and should be numpy array or a list
      results: the predicted value of the labels and should be numpy array or a list
    """
    print("f1_score = ",f1_score(target, results))
    print("roc_auc_score = ",roc_auc_score(target, results))
    print("average_precision_score",average_precision_score(target, results))
    print("Precision : ",precision_score(target, results))
    print("Recall : ",recall_score(target, results))
    print("accuracy_score : ",accuracy_score(target, results))
    cm=confusion_matrix(target, results)
    tn, fp, fn, tp =cm.ravel()    
    print("tn",tn, "fp",fp, "fn",fn,"tp", tp )
    disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels="cm")
    disp.plot()
    plt.show()

def clean_data(file,target_col,to_drop):
    """
    conduct initial cleaning and pre-processing
    Args:
        file: the path to the csv data file
        target: the column that contains the labels
        to_drop: the columns to be dropped from the features  
    Return:
        x: NumPy array of shape (rows, no.of features)
        y: NumPy array of shape (rows,) contains the labels
    """
    df=pd.read_csv(file)
    x = df.drop(to_drop, axis=1).values
    y = df[target_col].values
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    return x , y

def load_data(data_path,target,drop,batch_size):
    """
    load data using PyTorch
    Args:
        data_path: the path to the CSV data file
        target: the column that contains the labels
        drop: the columns to be dropped from the features  
    Return:
        loader: Pytorch Data Loader
        network_input_shape: no.of training features
        y: labels
    """
    x, y= clean_data(data_path,target,drop)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy (y))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,drop_last=True)
    network_input_shape=x.shape[-1]
    return loader,network_input_shape,y

#choose randomly n samples from x and y, and y are tensors
def chose_randomly_n_samples(x,y,n=2):
    """
    chose randomly an assigned number of samples
    Args:
        x: tensor that samples will be selected from
        y: tensor that samples will be selected from
        n: the number of samples
    Return:
        x[idxs]: tensor that contains the selected samples
        y[idxs]: tensor that contains the selected samples
    """
    l=list(range(x.shape[0]))
    idxs=random.sample(l, n)
    return x[idxs],y[idxs]


def pro_data(data):
    """
    that function is used to prepare data from numpy array to be handled as a 3-channel tensor, it was made to pass the tubular data
    to pre-trained ResNet networks that only deal with images. 
    Args:
        data: the data that need to be transformed, NumPy array of shape (no.of rows, no.of features)
    Return:
        data: the data after transformation, tensor of shape (no.of rows, 3, 1,no.of features)
    """
    min_max_scaler = MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    data = torch.FloatTensor(data)
    data = data.reshape(data.shape[0],1,1,data.shape[1]).expand(-1,3,-1,-1)
    return data


