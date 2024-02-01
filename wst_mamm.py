import torch
import torchaudio
import matplotlib.pyplot as plt
import pandas as pd
from zipfile import ZipFile 
import torch
if(torch.cuda.is_available()):
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
import os
import numpy as np
from kymatio.torch import Scattering1D

path = ''
file = ZipFile('wss.zip', "r")

def findOccurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]


test=[]
for item in file.namelist():
    pos=findOccurrences(item,'/')
    test.append(item[pos[0]+1:pos[1]])
classes_set=list(set(test))

y_labs = []
data_full_list=[]
srate_full_list=[]
for item in file.namelist():
        pos=findOccurrences(item,'/')
        name_class=item[pos[0]+1:pos[1]]
        file_op=file.open(item,'r')

        mf = torchaudio.info(file_op)
        file_op=file.open(item,'r')
        if mf.bits_per_sample in [16,24,32]:

            x, sr = torchaudio.load(file_op)
            data_full_list.append(x)
            y_labs.append(name_class)
            srate_full_list.append(sr)
            

signal_len = np.array([x.shape[1]/srate_full_list[k] for (k,x) in enumerate(data_full_list)])
avg = np.mean(signal_len)
sd = np.std(signal_len)
idx = np.where(signal_len <= avg+100*sd)[0]
selected_items = signal_len[idx]
from torch.nn.functional import pad
def cutter(X,cut_point): 
    cut_list = []
    cut_point = int(cut_point)
    j = 0

    for x in X:
        n_len = x.shape[1]
        add_pts = cut_point-n_len

        if (n_len<= cut_point):
            pp_left = int(add_pts/2)
            pp_right = add_pts - pp_left
            cut_list.append(pad(x, (pp_left,pp_right)))

        else :

            center_time = int(n_len/2)
            pp_left = int(cut_point-center_time)
            pp_right = cut_point - pp_left
            cut_list.append(x[:,center_time-pp_left: center_time+ pp_right])
        j += 1

    return torch.cat(cut_list)

y_sel = np.array(y_labs)[idx]
data_sel = [data_full_list[j] for j in idx]

lens = [x.shape[1] for x in data_sel]
cut_point= 8000
X_cut = cutter(data_sel, cut_point)
dict = {}

for name in set(y_labs):
    dict[name] = np.where(y_sel == name)

keep_idx = []
thrs = 50

for k in dict.keys():
    els = len(dict[k][0])
    if els > thrs:
        keep_idx.append(dict[k][0])

keep = set()


for l in keep_idx:
    keep = keep.union(set(l))
kp = list(keep)
XC = X_cut[kp, :]

y = y_sel[kp]


def standardize(X):
    st = torch.std(X, dim=1, keepdim=True)
    mn = torch.mean(X, dim=1, keepdim=True)
    return (X - mn) / st


X = standardize(XC)
X = XC

df_X = pd.DataFrame(X, columns=[f'col_{i}' for i in range(X.shape[1])])


df_X['y'] = y
df_X_no_duplicates = df_X.drop_duplicates(subset=df_X.iloc[:, 0:cut_point])
X = torch.from_numpy(df_X_no_duplicates.iloc[:, 0:8000].values).to(device)
y = df_X_no_duplicates.iloc[:, -1].values


batches = [64, 128, 256]
JQ = [(7, 10), (6, 16), (2,20)]


batch_size = batches[2]
J, Q = JQ[2]
T = X.shape[1]

scattering=Scattering1D(J,T,Q)
scattering.cuda()
SX=scattering(X)



meta = scattering.meta()
order0 = np.where(meta['order'] == 0)
order1 = np.where(meta['order'] == 1)
order2 = np.where(meta['order'] == 2)


def median_norm(X):
    md = torch.median(X)
    sn = torch.std(X)
    return (X - md) / sn





SX_med = SX
for i in range(SX.shape[0]):
    SX_med[i][order0] = median_norm(SX[i][order0])
    SX_med[i][order1] = median_norm(SX[i][order1])
    SX_med[i][order2] = median_norm(SX[i][order2])
SX_med = SX_med[:, 1:, :]  

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset



from sklearn.preprocessing import LabelEncoder

lbe = LabelEncoder()
y_trc = torch.as_tensor(lbe.fit_transform(y))
XX_tr, XX_test, y_trXX, y_testXX = train_test_split(SX_med, y_trc, test_size=.25, stratify=y)

train_dataset = TensorDataset(XX_tr, y_trXX)
val_dataset = TensorDataset(XX_test, y_testXX)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels=1, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 64, layers[2], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



model = ResNet(BasicBlock, [2, 2, 2], in_channels=1, num_classes=32).to(device)

criterion = nn.CrossEntropyLoss()
learning_rate = .01
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,amsgrad= True, weight_decay= .001 )
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
n_total_steps = len(train_dataloader)
num_epochs = 200


loss_train = []
acc_train = []
acc_eval = []
loss_eval = []
for epoch in range(num_epochs):

    loss_ep_train = 0
    n_samples = 0
    n_correct = 0
    for i, (x, labels) in enumerate(train_dataloader):

        x = x.unsqueeze(1).to(device)

        labels = labels.to(device)

        
        outputs = model(x)
        loss = criterion(outputs, labels)

    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_ep_train += loss.item()
        _, predictions = torch.max(outputs, 1)

        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()



        if (i + 1) % 10 == 0:
            print(f'epoch: {epoch + 1}, step: {i + 1}/{n_total_steps}, loss:{loss.item():.4f}, ')

    acc_tr = 100 * n_correct / n_samples
    acc_train.append(acc_tr)
    loss_train.append(loss_ep_train/len(train_dataloader))

    loss_ep_eval = 0

    with torch.no_grad():

        n_correct = 0
        n_samples = 0

        for x, labels in val_dataloader:
            x = x.unsqueeze(1).to(device)

            labels = labels.to(device)
            outputs = model(x)
            lossvv = criterion(outputs, labels)

            _, predictions = torch.max(outputs, 1)

            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()
            loss_ep_eval += lossvv.item()

        acc = 100 * n_correct / n_samples

    acc_eval.append(acc)
    loss_eval.append(loss_ep_eval/len(val_dataloader))

    print(f' validation accuracy = {acc}')

res = np.array([loss_train, loss_eval, acc_train, acc_eval])


namefile = 'WST_JQ'+str(J)+','+str(Q)+'_batch'+str(batch_size)
np.save(namefile, res)





from sklearn.metrics import roc_auc_score
yp = []
ytr = []
y_prob = []
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for x, labels in val_dataloader:
        x = x.unsqueeze(1).to(device)

        labels = labels.to(device)
        outputs = model(x)
        pr_out = torch.softmax(outputs, dim = 1)

        proba, predictions = torch.max(pr_out, 1)

        yp.append(predictions.cpu().numpy())
        ytr.append(labels.cpu().numpy())
        y_prob.append(pr_out.cpu().numpy())
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100 * n_correct / n_samples


from sklearn.metrics import classification_report, confusion_matrix

ypred_np = np.hstack(yp)
ytrue_np = np.hstack(ytr)
prob_np = np.vstack(y_prob)

c_rep = classification_report(lbe.inverse_transform(ytrue_np), lbe.inverse_transform(ypred_np))
auc_ovr = roc_auc_score(ytrue_np, prob_np, multi_class= 'ovr')
auc_ovo = roc_auc_score(ytrue_np, prob_np, multi_class= 'ovr')
with open(f'classification_report_{namefile}', 'w') as file:
    file.write(c_rep)
    file.write(f'auc_ovr = {auc_ovr}\n\n')
    file.write(f'auc_ovo = {auc_ovo}')

            
