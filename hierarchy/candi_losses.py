import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
from monai.losses import DiceLoss, DiceCELoss


class TreeTripletLoss(nn.Module):
    def __init__(self, num_classes):
        super(TreeTripletLoss, self).__init__()
        dist_mat = np.load('./hiera_dist_CANDI/hiera_dist_CANDI.npy')
        self.dist_mat = torch.from_numpy(dist_mat).float().cuda()
        self.num_classes = num_classes

    def forward(self, feats, labels, max_triplet=200):
        batch_size = feats.shape[0]
        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3], feats.shape[4]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        labels = labels.view(-1)
        feats = feats.permute(0, 2, 3, 4, 1)
        feats = feats.contiguous().view(-1, feats.shape[-1])
        
        triplet_loss = 0
        exist_classes = torch.unique(labels)
        exist_classes = [x for x in exist_classes if x != 0]
        class_count = 0
        
        for ii in exist_classes:
            index_anchor = labels == ii
            # print(exist_classes)
            anchor_dist = self.dist_mat[ii, torch.tensor(exist_classes).cuda()]
            anchor_dist[anchor_dist == 0] = 256
            min_dist = torch.min(anchor_dist)
            min_dist_idx = (anchor_dist == min_dist).nonzero(as_tuple=True)[0]
            index_pos = sum(labels==exist_classes[int(idx)] for idx in min_dist_idx).bool()
            index_neg = (~index_anchor) & (~index_pos) & (labels != 0)
            
            min_size = min(torch.sum(index_anchor), torch.sum(index_pos), torch.sum(index_neg), max_triplet)
            feats_anchor = feats[index_anchor][:min_size]
            feats_pos = feats[index_pos][:min_size]
            feats_neg = feats[index_neg][:min_size]

            labels_neg = labels[index_neg][:min_size]
            labels_pos = labels[index_pos][:min_size]
            dist_neg = self.dist_mat[ii, labels_neg]
            dist_pos = self.dist_mat[ii, labels_pos]

            distance = torch.zeros(min_size,2).cuda()
            distance[:,0:1] = 1 - (feats_anchor*feats_pos).sum(1, True) 
            distance[:,1:2] = 1 - (feats_anchor*feats_neg).sum(1, True) 
            
            # margin always 0.1 + (4-2)/4 since the hierarchy is three level
            # TODO: should include label of pos is the same as anchor, i.e. margin=0.1
            margin = 0.1*torch.ones(min_size).cuda() + (dist_neg - dist_pos) / 8
            
            tl = distance[:,0] - distance[:,1] + margin
            tl = F.relu(tl)

            if tl.size(0) > 0:
                triplet_loss += tl.mean()
                class_count += 1

        if class_count == 0:
            return None, torch.tensor([0]).cuda()

        triplet_loss /= class_count

        return triplet_loss


class DiceCETreeTripletLossCANDI(nn.Module):
    def __init__(self,
                 num_classes,
                 total_epochs,
                 to_onehot_y=True,
                 softmax=True):
        super(DiceCETreeTripletLossCANDI, self).__init__()
        self.dl = DiceCELoss(to_onehot_y=to_onehot_y, softmax=softmax)
        self.num_classes = num_classes
        self.triplet = TreeTripletLoss(num_classes)
        self.total_epochs = total_epochs

    def forward(self,
                outputs,
                label,
                epoch=None):
        cls_score, embed = outputs[0], outputs[1]
        label_s = label.squeeze(1).long()
        cls_score_down = cls_score[:,:self.num_classes]
        dice_loss_down = self.dl(cls_score_down, label)
        triplet_loss = self.triplet(embed, label_s)

        w = 1/4*(1+torch.cos(torch.tensor((epoch-self.total_epochs)/self.total_epochs*math.pi)))

        loss = dice_loss_down + w * triplet_loss
        return loss, triplet_loss, dice_loss_down