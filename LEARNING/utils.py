from collections import Counter
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score,precision_score,confusion_matrix, accuracy_score,ConfusionMatrixDisplay,average_precision_score,roc_auc_score, f1_score
from sklearn.preprocessing import MinMaxScaler, minmax_scale
import torch
from tqdm import tqdm
device = "cpu"

def show_metrics(target,results):
    """
    show different metrics and display confusion matrix
    Args:
      target:the true value of the labels and should be numpy array or a list
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
    conduct intial cleaning and pre-processing
    Args:
        file: the path to the csv data file
        target: the column that contains the labels
        results: the predicted value of the labels and should be numpy array or a list 
    Return:
        x: numpy array of shape (rows, no.of features)
        y: numpy array of shape (rows,) contains the labels
    """
    df=pd.read_csv(file)
    print(df.columns)
    x = df.drop(to_drop, axis=1).values
    y = df[target_col].values
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    return x , y


#choose randomly n samples from x and y, xabd y are tensors
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
    that function is used to prepare data from numpy array to be handled as 3-channel tensor, it was made to pass the tublar data
    to pre-trained ResNet networks that only deal with images. 
    Args:
        data: the data with that need to be transformed, numpy array of shape (no.of rows, no.of features)
    Return:
        data: the data after transformtion, tensor of shape (no.of rows, ,3, 1,no.of features)
    """
    min_max_scaler = MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    data = torch.FloatTensor(data)
    data = data.reshape(data.shape[0],1,1,data.shape[1]).expand(-1,3,-1,-1)
    return data


