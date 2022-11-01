import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from torch_geometric.nn import MessagePassing, RGCNConv, GATConv
from torch.utils.data import Dataset, DataLoader
from math import sqrt
import math
import numpy as np


def get_metric(probs, labels):
    # hit 123
    hit1 = 0
    hit2 = 0
    hit3 = 0
    for i in range(len(labels)):
        temp = probs[i].clone()
        if torch.argmax(temp) == labels[i]:
            hit1 += 1
            hit2 += 1
            hit3 += 1
            continue
        temp[torch.argmax(temp)] = 0
        if torch.argmax(temp) == labels[i]:
            hit2 += 1
            hit3 += 1
            continue
        temp[torch.argmax(temp)] = 0
        if torch.argmax(temp) == labels[i]:
            hit3 += 1
            continue
    hit1 = hit1 / len(labels)
    hit2 = hit2 / len(labels)
    hit3 = hit3 / len(labels)
    # F1 and accs
    TP = [0,0,0,0,0]
    TN = [0,0,0,0,0]
    FP = [0,0,0,0,0]
    FN = [0,0,0,0,0]
    for i in range(len(labels)):
        temp = probs[i]
        if torch.argmax(temp) == labels[i]:
            TP[labels[i]] += 1
            for j in range(5):
                if not j == labels[i]:
                    TN[j] += 1
        else:
            FP[torch.argmax(temp)] += 1
            FN[labels[i]] += 1
            for j in range(5):
                if not j == torch.argmax(temp) and not j == labels[i]:
                    TN[j] += 1
    
    precision = [TP[i] / max(TP[i] + FP[i], 1) for i in range(5)]
    recall = [TP[i] / max(TP[i] + FN[i], 1) for i in range(5)]
    F1 = [2 * precision[i] * recall[i] / max(precision[i] + recall[i], 1) for i in range(5)]
    #macro_precision = sum(precision) / 5
    #macro_recall = sum(recall) / 5
    macro_F1 = sum(F1) / 5
    micro_precision = sum(TP) / (sum(TP) + sum(FP))
    micro_recall = sum(TP) / (sum(TP) + sum(FN))
    assert (micro_precision == micro_recall)
    assert (micro_precision == hit1)
    micro_F1 = micro_precision
    return {'hit1':hit1, 'hit2':hit2, 'hit3':hit3, 'micro_F1':micro_F1, 'macro_F1':macro_F1}
        

class PSPDataset(Dataset):
    def __init__(self, batch_size, name):  # name = train/dev
        path = './'
        self.edge_index = torch.load(path + 'edge_index.pt')
        self.edge_type = torch.load(path + 'edge_type.pt')
        self.node_features = torch.load(path + 'PSPnode_roberta.pt').squeeze(1)

        self.liberal_id = torch.tensor(torch.load(path + 'liberal_id.pt'))
        self.liberal_label = torch.tensor(torch.load(path + 'liberal_label.pt'))
        self.conservative_id = torch.tensor(torch.load(path + 'conservative_id.pt'))
        self.conservative_label = torch.tensor(torch.load(path + 'conservative_label.pt'))
        self.liberal_split = torch.load(path + 'liberal_split.pt')
        self.conservative_split = torch.load(path + 'conservative_split.pt')

        self.batch_size = batch_size
        self.name = name
        if self.name == 'train':
            self.length = 5 * int(846 / self.batch_size)
            self.liberal_id = self.liberal_id[self.liberal_split[0]]
            self.liberal_label = self.liberal_label[self.liberal_split[0]]
            self.conservative_id = self.conservative_id[self.conservative_split[0]]
            self.conservative_label = self.conservative_label[self.conservative_split[0]]
        else:
            self.length = 1
            self.liberal_id = self.liberal_id[self.liberal_split[2]]
            self.liberal_label = self.liberal_label[self.liberal_split[2]]
            self.conservative_id = self.conservative_id[self.conservative_split[2]]
            self.conservative_label = self.conservative_label[self.conservative_split[2]]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {'edge_index':self.edge_index, 'edge_type':self.edge_type, 'node_features':self.node_features, 'liberal_id':self.liberal_id,
                'liberal_label':self.liberal_label, 'conservative_id':self.conservative_id, 'conservative_label':self.conservative_label, 'batch_size':self.batch_size}

# gated RGCN in a nutshell
# in_channels: text encoding dimension, out_channels: dim for each node rep, num_relations
# Input: node_features:torch.size([node_cnt,in_channels]), query_features = torch.size([in_channels]) (MISSING IN DATA FILE)
# edge_index = torch.size([[headlist],[taillist]]), edge_type = torch.size([typelist])
# Output: node representation of torch.size([node_cnt, out_channels])
class GatedRGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations):
        super(GatedRGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.RGCN1 = RGCNConv(in_channels = out_channels, out_channels = out_channels, num_relations = num_relations)
        self.attention_layer = nn.Linear(2 * out_channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        nn.init.xavier_uniform_(self.attention_layer.weight, gain=nn.init.calculate_gain('sigmoid'))

    def forward(self, node_features, edge_index, edge_type):

        #layer 1
        #print(node_features.size())
        #print(edge_index.size())
        #print(edge_type.size())
        u_0 = self.RGCN1(node_features, edge_index, edge_type)
        a_1 = self.sigmoid(self.attention_layer(torch.cat((u_0, node_features),dim=1)))
        h_1 = self.tanh(u_0) * a_1 + node_features * (1 - a_1)

        return h_1

class PSPDetector(pl.LightningModule):
    def __init__(self, in_channels, out_channels, dropout, batch_size, num_relations, negative_sample, negative_sample_weight):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_size = batch_size
        self.num_relations = num_relations
        self.negative_sample = negative_sample
        self.negative_sample_weight = negative_sample_weight

        self.linear_before_RGCN = nn.Linear(self.in_channels, self.out_channels)
        self.GatedRGCN = GatedRGCN(self.out_channels, self.out_channels, self.num_relations)
        self.linear_liberal = nn.Linear(self.out_channels, 5)
        self.linear_conservative = nn.Linear(self.out_channels, 5)

        torch.nn.init.kaiming_uniform(self.linear_before_RGCN.weight, nonlinearity='relu')

        self.dropout_layer = nn.Dropout(dropout)
        self.CELoss = nn.CrossEntropyLoss()
        self.KLDivLoss = nn.KLDivLoss(size_average=False, reduction='sum')
        self.relu = nn.ReLU()

    def forward(self, x):
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), weight_decay=1e-5)
        # optimizer = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=0.9)
        return optimizer

    def training_step(self, train_batch, batch_idx):

        edge_index = train_batch['edge_index'][0]
        edge_type = train_batch['edge_type'][0]
        node_features = train_batch['node_features'][0]
        liberal_id = train_batch['liberal_id'][0]
        liberal_label = train_batch['liberal_label'][0]
        conservative_id = train_batch['conservative_id'][0]
        conservative_label = train_batch['conservative_label'][0]
        batch_size = train_batch['batch_size'][0]

        node_features = self.dropout_layer(self.relu(self.linear_before_RGCN(node_features)))
        node_features = self.dropout_layer(self.relu(self.GatedRGCN(node_features, edge_index, edge_type)))
        node_features = self.dropout_layer(self.relu(self.GatedRGCN(node_features, edge_index, edge_type)))
        liberal_prob = self.linear_liberal(node_features)
        conservative_prob = self.linear_conservative(node_features)

        #loss 1: proximity
        loss1 = 0
        sample_edge_list = list(np.random.permutation(len(edge_type))[:batch_size])
        fro = list(edge_index[0][sample_edge_list])
        to = list(edge_index[1][sample_edge_list])
        for i in range(len(fro)):
            loss1 += -torch.log(torch.sigmoid(torch.dot(node_features[fro[i]], node_features[to[i]])))
            for j in range(self.negative_sample):
                another = np.random.randint(0, len(node_features))
                loss1 += -self.negative_sample_weight * (torch.log(torch.sigmoid(-1 * torch.dot(node_features[fro[i]], node_features[another]))))

        #loss 2: reciprocal
        loss2 = 0
        sample_node_list = list(np.random.permutation(len(node_features))[:batch_size])
        lib = liberal_prob[sample_node_list]
        con = conservative_prob[sample_node_list]
        # probs
        #lib = torch.softmax(lib, dim=1)
        #con = torch.softmax(con, dim=1)
        con_flip = torch.index_select(con, 1, torch.LongTensor([4,3,2,1,0]).cuda()) # 1d convolution operation, maximize the result
        lib_flip = torch.index_select(lib, 1, torch.LongTensor([4,3,2,1,0]).cuda())
        #loss2 = -torch.sum(torch.sum(torch.log(torch.sigmoid(-lib * con))))
        
        loss2 = self.CELoss(lib, torch.argmax(con_flip, dim=1)) + self.CELoss(con, torch.argmax(lib_flip, dim=1))

        #loss2 = self.KLDivLoss(torch.nn.functional.log_softmax(lib, -1), torch.nn.functional.softmax(con_flip, dim=-1))

        #loss 3: liberal stance
        loss3 = 0
        sample_liberal_list = list(np.random.permutation(len(liberal_id))[:batch_size])
        sample_liberal_ids = list(liberal_id[sample_liberal_list])
        loss3 = self.CELoss(liberal_prob[sample_liberal_ids], torch.tensor(liberal_label)[sample_liberal_list].long())

        #liberal_pred = torch.argmax(liberal_prob[sample_liberal_ids], dim = 1)
        #liberal_truth = torch.tensor(liberal_label)[sample_liberal_list].long()
        
        #correct = 0
        #for k in range(len(liberal_pred)):
        #    if liberal_pred[k] == liberal_truth[k]:
        #        correct += 1
        #liberal_acc = correct / len(liberal_pred)

        #loss 4: conservative stance
        loss4 = 0
        sample_conservative_list = list(np.random.permutation(len(conservative_id))[:batch_size])
        sample_conservative_ids = list(conservative_id[sample_conservative_list])
        loss4 = self.CELoss(conservative_prob[sample_conservative_ids], torch.tensor(conservative_label)[sample_conservative_list].long())

        #conservative_pred = torch.argmax(conservative_prob[sample_conservative_ids], dim=1)
        #conservative_truth = torch.tensor(conservative_label)[sample_conservative_list].long()
        #correct = 0
        #for k in range(len(conservative_pred)):
        #    if conservative_pred[k] == conservative_truth[k]:
        #        correct += 1
        #conservative_acc = correct / len(conservative_pred)

        self.log('train_loss1', 0.01 * loss1.item())
        self.log('train_loss2', 0.2 * loss2.item())
        self.log('train_loss3', loss3.item())
        self.log('train_loss4', loss4.item())
        #self.log('train_lib_acc', liberal_acc)
        #self.log('train_con_acc', conservative_acc)
        #self.log('train_overall_f1', 2 * liberal_acc * conservative_acc / (liberal_acc + conservative_acc))

        return 0.01 * loss1 + 0.2 * loss2 + loss3 + loss4

    def validation_step(self, val_batch, batch_idx):
        edge_index = val_batch['edge_index'][0]
        edge_type = val_batch['edge_type'][0]
        node_features = val_batch['node_features'][0]
        liberal_id = val_batch['liberal_id'][0]
        liberal_label = val_batch['liberal_label'][0]
        conservative_id = val_batch['conservative_id'][0]
        conservative_label = val_batch['conservative_label'][0]
        batch_size = val_batch['batch_size'][0]
        
        #print(liberal_id)

        node_features = self.relu(self.linear_before_RGCN(node_features))
        node_features = self.relu(self.GatedRGCN(node_features, edge_index, edge_type))
        node_features = self.relu(self.GatedRGCN(node_features, edge_index, edge_type))
        liberal_prob = self.linear_liberal(node_features)
        conservative_prob = self.linear_conservative(node_features)

        # loss 3: liberal stance
        loss3 = 0
        sample_liberal_list = list(liberal_id)
        loss3 = self.CELoss(liberal_prob[sample_liberal_list], torch.tensor(liberal_label).long())
        
        liberal_metric = get_metric(liberal_prob[sample_liberal_list], torch.tensor(liberal_label).long())

        #liberal_pred = torch.argmax(liberal_prob[sample_liberal_list], dim=1)
        #liberal_truth = torch.tensor(liberal_label).long()
        #correct = 0
        #for k in range(len(liberal_pred)):
        #    if liberal_pred[k] == liberal_truth[k]:
        #        correct += 1
        #liberal_acc = correct / len(liberal_pred)

        # loss 4: conservative stance
        loss4 = 0
        sample_conservative_list = list(conservative_id)
        loss4 = self.CELoss(conservative_prob[sample_conservative_list],
                            torch.tensor(conservative_label).long())
                            
        conservative_metric = get_metric(conservative_prob[sample_conservative_list], torch.tensor(conservative_label).long())

        #conservative_pred = torch.argmax(conservative_prob[sample_conservative_list], dim=1)
        #conservative_truth = torch.tensor(conservative_label).long()
        #correct = 0
        #for k in range(len(conservative_pred)):
        #    if conservative_pred[k] == conservative_truth[k]:
        #        correct += 1
        #conservative_acc = correct / len(conservative_pred)

        self.log('val_lib_hit1', liberal_metric['hit1'])
        self.log('val_lib_hit2', liberal_metric['hit2'])
        self.log('val_lib_hit3', liberal_metric['hit3'])
        self.log('val_lib_micro_F1', liberal_metric['micro_F1'])
        self.log('val_lib_macro_F1', liberal_metric['macro_F1'])
        self.log('val_con_hit1', conservative_metric['hit1'])
        self.log('val_con_hit2', conservative_metric['hit2'])
        self.log('val_con_hit3', conservative_metric['hit3'])
        self.log('val_con_micro_F1', conservative_metric['micro_F1'])
        self.log('val_con_macro_F1', conservative_metric['macro_F1'])
        self.log('val_overall_acc', (liberal_metric['hit1'] * len(liberal_id) + conservative_metric['hit1'] * len(conservative_id)) / (len(liberal_id) + len(conservative_id)))
        self.log('val_overall_micro_F1', 2 * liberal_metric['micro_F1'] * conservative_metric['micro_F1'] / (liberal_metric['micro_F1'] + conservative_metric['micro_F1']))
        self.log('val_overall_macro_F1', 2 * liberal_metric['macro_F1'] * conservative_metric['macro_F1'] / (liberal_metric['macro_F1'] + conservative_metric['macro_F1']))
        
    def test_step(self, test_batch, batch_idx):
        edge_index = test_batch['edge_index'][0]
        edge_type = test_batch['edge_type'][0]
        node_features = test_batch['node_features'][0]
        liberal_id = test_batch['liberal_id'][0]
        liberal_label = test_batch['liberal_label'][0]
        conservative_id = test_batch['conservative_id'][0]
        conservative_label = test_batch['conservative_label'][0]
        batch_size = test_batch['batch_size'][0]
        
        #print(liberal_id)

        node_features = self.relu(self.linear_before_RGCN(node_features))
        node_features = self.relu(self.GatedRGCN(node_features, edge_index, edge_type))
        node_features = self.relu(self.GatedRGCN(node_features, edge_index, edge_type))
        torch.save(node_features, 'representation.pt')


# data
dataset1 = PSPDataset(batch_size=64,
                           name='train')  # batch_size governs (xxxx / 64) batches of training
dataset2 = PSPDataset(batch_size=1,
                           name='dev')  # batch_size governs nothing
dataset3 = PSPDataset(batch_size=1,
                           name='test')  # batch_size governs nothing

train_loader = DataLoader(dataset1, batch_size=1)  # always should be 1
val_loader = DataLoader(dataset2, batch_size=1)  # always should be 1
test_loader = DataLoader(dataset3, batch_size=1)

# model
model = PSPDetector(in_channels=768, out_channels=512, dropout=0.5,
                      batch_size=64, num_relations = 5, negative_sample=2, negative_sample_weight=0.1)  # batch_size governs each batch how many nodes

# training
trainer = pl.Trainer(gpus=1, num_nodes=1, precision=16, max_epochs = 100)
print('training begins')
trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)

