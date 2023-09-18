import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
import datetime
import glob
import pickle
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification.f_beta import F1Score
import random
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler,WeightedRandomSampler

from model import Mobilenet_reg,extractor_model
random.seed(42)

print(os.listdir('Dataset/'))

df=pd.read_csv(r'Dataset/train.csv')
test_df=pd.read_csv(r'Dataset/test.csv')

class gtos_dataset(Dataset):
    def __init__(self,df,transform=None):
        self.df=df
        self.transform=transform
        self.total_classes=list(np.unique(self.df['class']))
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        df_temp=self.df.iloc[idx]
        image_path=df_temp['path']
        image=Image.open(image_path)
        image=image.convert('RGB')
        image=np.array(image)
        image=self.transform(image=image)['image']
        return{
        'image':image,
        'roughness':torch.tensor(df_temp['roughness'],dtype=torch.float),
        'sliperiness':torch.tensor(df_temp['sliperiness'],dtype=torch.float),
        'class':F.one_hot(torch.tensor(self.total_classes.index(df_temp['class']),dtype=torch.long),num_classes=len(self.total_classes)).float(),
        'class_name':df_temp['class']

        }


def return_class_weights(df):
    target=df['class']
    value_weights=1/df['class'].value_counts().sort_index().values
    samples_weight = torch.from_numpy(value_weights)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, replacement=True, num_samples=int(len(df)*1.5))
    return sampler

from sklearn.model_selection import train_test_split
train_df,val_df=train_test_split(df,test_size=0.2,random_state=42)

train_loader=DataLoader(gtos_dataset(train_df,transform=A.Compose([
    A.Normalize(),
    A.RandomCrop(128,128,p=0.15),
    A.PadIfNeeded(min_height=256,min_width=256,p=1.0),
    A.HorizontalFlip(p=0.25),
    A.VerticalFlip(p=0.25),
    A.RandomBrightnessContrast(p=0.1),
    A.pytorch.ToTensorV2()])),batch_size=8,shuffle=False,sampler=return_class_weights(train_df))

val_loader=DataLoader(gtos_dataset(val_df,transform=A.Compose([
    A.Normalize(),
    A.pytorch.ToTensorV2()
])),batch_size=8,shuffle=False,sampler=return_class_weights(val_df))


test_loader=DataLoader(gtos_dataset(test_df,transform=A.Compose([
    A.Normalize(),
    A.pytorch.ToTensorV2()
])),batch_size=8,sampler=return_class_weights(test_df))

accuracy = MulticlassAccuracy(num_classes=len(np.unique(df['class'])),top_k=1)
F1 = F1Score(task='multiclass',num_classes=len(np.unique(df['class'])), average='macro')

def train_one_iter(model,train_loader,optimizer,lossfn,lossfn2,device):
    model.train()
    train_loss=0
    train_acc=0
    train_f1=0
    train_mse=0
    for idx,i in tqdm(enumerate(train_loader),total=len(train_loader)):
        optimizer.zero_grad()
        image=i['image'].to(device)
        roughness=i['roughness'].to(device)
        sliperiness=i['sliperiness'].to(device)
        class_target=i['class'].to(device)
        class_pred,regression_pred=model(image)
        loss1=lossfn(class_pred,class_target.argmax(dim=1))
        loss2=lossfn2(regression_pred,torch.stack([roughness,sliperiness],dim=1))
        loss=loss1+loss2
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
        train_acc+=accuracy(class_pred,class_target.argmax(dim=1)).item()
        train_f1+=F1(class_pred,class_target.argmax(dim=1)).item()
        train_mse+=loss2.item()

        if idx%100==0:
            print(f'loss:{loss.item()}')
            print(f'accuracy:{accuracy(class_pred,class_target.argmax(dim=1)).item()}')
            print(f'f1:{F1(class_pred,class_target.argmax(dim=1)).item()}')
            print(f'mse:{loss2.item()}')
    return train_loss/len(train_loader),train_acc/len(train_loader),train_f1/len(train_loader),train_mse/len(train_loader)


def test_one_iter(model,test_loader,lossfn,lossfn2,device):
    model.eval()
    test_loss=0
    test_acc=0
    test_f1=0
    test_mse=0
    with torch.no_grad():
        for idx,i in tqdm(enumerate(test_loader),total=len(test_loader)):
            image=i['image'].to(device)
            roughness=i['roughness'].to(device)
            sliperiness=i['sliperiness'].to(device)
            class_target=i['class'].to(device)
            class_pred,regression_pred=model(image)
            loss1=lossfn(class_pred,class_target.argmax(dim=1))
            loss2=lossfn2(regression_pred,torch.stack([roughness,sliperiness],dim=1))
            loss=loss1+loss2
            test_loss+=loss.item()
            test_acc+=accuracy(class_pred,class_target.argmax(dim=1)).item()
            test_f1+=F1(class_pred,class_target.argmax(dim=1)).item()
            test_mse+=loss2.item()

            if idx%100==0:
                print(f'loss:{loss.item()}')
                print(f'accuracy:{accuracy(class_pred,class_target.argmax(dim=1)).item()}')
                print(f'f1:{F1(class_pred,class_target.argmax(dim=1)).item()}')
                print(f'mse:{loss2.item()}')
    return test_loss/len(test_loader),test_acc/len(test_loader),test_f1/len(test_loader),test_mse/len(test_loader)

def training_loop(model,optimizer,test_loader,train_loader,lossfn,lossfn2,device,epochs):
    history={
        'train_loss':[],
        'train_acc':[],
        'train_f1':[],
        'train_mse':[],
        'test_loss':[],
        'test_acc':[],
        'test_f1':[],
        'test_mse':[]
    }
    os.makedirs('classifier',exist_ok=True)
    writer=SummaryWriter('runs/terrain_classification')

    for epoch in range(epochs):

        train_loss,train_acc,train_f1,train_mse=train_one_iter(model,train_loader,optimizer,lossfn,lossfn2,device)
        test_loss,test_acc,test_f1,test_mse=test_one_iter(model,test_loader,lossfn,lossfn2,device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['train_mse'].append(train_mse)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_f1'].append(test_f1)
        history['test_mse'].append(test_mse)

        writer.add_scalar('train_loss',train_loss,epoch)
        writer.add_scalar('train_acc',train_acc,epoch)
        writer.add_scalar('train_f1',train_f1,epoch)
        writer.add_scalar('train_mse',train_mse,epoch)
        writer.add_scalar('test_loss',test_loss,epoch)
        writer.add_scalar('test_acc',test_acc,epoch)
        writer.add_scalar('test_f1',test_f1,epoch)
        writer.add_scalar('test_mse',test_mse,epoch)

        print(f'Epoch {epoch+1}/{epochs} train_loss:{train_loss} train_acc:{train_acc} train_f1:{train_f1} train_mse:{train_mse} test_loss:{test_loss} test_acc:{test_acc} test_f1:{test_f1} test_mse:{test_mse}')

        torch.save(model.state_dict(),f'classifier/epoch_{epoch+1}.pth')

        if test_loss<=min(history['test_loss']):
            torch.save(model.state_dict(),'classifier/best.pth')
        
    return history


# main function to train the model
if __name__=='__main__':
    new_model=Mobilenet_reg(extractor_model)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    new_model.to(device)
    optimizer=torch.optim.AdamW(new_model.parameters(),lr=1e-4)
    lossfn=nn.CrossEntropyLoss()
    lossfn2=nn.MSELoss()
    history=training_loop(new_model,optimizer,test_loader,train_loader,lossfn,lossfn2,device,epochs=10)
    with open('classifier/history.pkl','wb') as f:
        pickle.dump(history,f)