import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contiguous_params import ContiguousParams

from .knowformer import Knowformer
from .utils import get_ranks, get_norms, get_scores
from torch.nn.modules.loss import _WeightedLoss

class LabelSmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = LabelSmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                                                              self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss

class MultiLossLayer(nn.Module):
    def __init__(self, loss_num):
        super(MultiLossLayer, self).__init__()
        self.loss_num = loss_num
        self.log_vars = nn.Parameter(torch.zeros(self.loss_num, ), requires_grad=True)

    def forward(self, loss_list):
        assert len(loss_list) == self.loss_num
        precision = torch.exp(-self.log_vars)
        loss = 0
        for i in range(self.loss_num):
            loss += precision[i] * loss_list[i] + self.log_vars[i]
        return loss

class HyperMono(nn.Module):
    def __init__(self, args: dict, bert_encoder: Knowformer):
        super(HyperMono, self).__init__()

        self.device = torch.device(args['device'])
        self.ent_neighbor_loss_weight = args['ent_neighbor_loss_weight']
        self.add_ent_neighbors = True if args['add_ent_neighbors'] == 'True' else False
        self.add_rel_neighbors = True if args['add_rel_neighbors'] == 'True' else False
        self.lr = args['kge_lr']
        self.weight_decay = args['weight_decay']
        self.loss_train_ablation = args['loss_train_ablation']
        self.loss_valid_ablation = args['loss_valid_ablation']

        self.entity_begin_idx = args['entity_begin_idx']
        self.entity_end_idx = args['entity_end_idx']
        self.use_extra_encoder = args['extra_encoder']

        self.ablation_mode = args['ablation_mode']

        self.bert_encoder = bert_encoder
        self.loss_fc = LabelSmoothCrossEntropyLoss(smoothing=args['kge_label_smoothing'])
        self.multi_loss = MultiLossLayer(loss_num=2)

    def forward(self, batch_data):
        output = self.link_prediction(batch_data)
        return output['loss'], output['rank']

    def training_step(self, batch):
        loss, _ = self.forward(batch)
        return loss

    def training_epoch_end(self, outputs):
        return np.round(np.mean([loss.item() for loss in outputs]), 4)

    def validation_step(self, batch):
        output = self.link_prediction_validation(batch)
        loss, rank = output['loss'], output['rank']
        return loss.item(), rank

    def validation_epoch_end(self, outputs):
        loss, rank = list(), list()
        for batch_loss, batch_rank in outputs:
            loss.append(batch_loss)
            rank += batch_rank
        loss = np.mean(loss)
        scores = get_scores(rank, loss)
        return scores

    def link_prediction_validation(self, batch):
        # 1. prepare input data
        tri_input = batch['tri_seq']['input_ids'].to(self.device)
        tri_qual_input = batch['tri_qual_seq']['input_ids'].to(self.device)
        if self.add_ent_neighbors:
            ent_context_input = [t['input_ids'].to(self.device) for t in batch['ent_neighbors']]
            ent_qual_context_input = [t['input_ids'].to(self.device) for t in batch['ent_qual_neighbors']]
        else:
            ent_context_input = None
            ent_qual_context_input = None
        if self.add_rel_neighbors:
            rel_context_input = [t['input_ids'].to(self.device) for t in batch['rel_qual_neighbors']]
        else:
            rel_context_input = None
        labels = batch['labels'].to(self.device)
        neighbor_labels = batch['neighbor_labels'].to(self.device)
        entity_filters = batch['entity_filters'].to(self.device)
        qual_filters = batch['qual_filters'].to(self.device)
        mask_pos = batch['mask_pos'].to(self.device)

        # 2. get output from model
        output = self.bert_encoder(tri_input, tri_qual_input, ent_context_input, ent_qual_context_input, rel_context_input, mask_pos, self.use_extra_encoder)

        # 3. compute loss and rank
        logits_from_triple = output['tri_without_neighbors']
        logits_from_triple_context = output['tri_with_neighbors']
        logits_from_local_ent_neighbors = output['tri_local_neighbors']
        logits_from_global_ent_neighbors = output['tri_global_neighbors']
        logits_from_local_global_ent_neighbors = output['tri_local_global_neighbors']

        logits_from_triple_qual = output['tri_qual_without_neighbors']
        logits_from_triple_qual_context = output['tri_qual_with_neighbors']
        logits_from_local_ent_qual_neighbors = output['tri_qual_local_neighbors']
        logits_from_global_ent_qual_neighbors = output['tri_qual_global_neighbors']
        logits_from_local_global_ent_qual_neighbors = output['tri_qual_local_global_neighbors']

        # 3.1 loss from the triple
        tri_loss = self.loss_fc(logits_from_triple, labels + self.entity_begin_idx)
        tri_context_loss = self.loss_fc(logits_from_triple_context, labels + self.entity_begin_idx)
        if self.ablation_mode == 'normal':
            ent_neighbors_loss = self.loss_fc(logits_from_local_global_ent_neighbors, neighbor_labels + self.entity_begin_idx)
        elif self.ablation_mode == 'local':
            ent_neighbors_loss = self.loss_fc(logits_from_local_ent_neighbors, neighbor_labels + self.entity_begin_idx)
        elif self.ablation_mode == 'global':
            ent_neighbors_loss = self.loss_fc(logits_from_global_ent_neighbors, neighbor_labels + self.entity_begin_idx)
        tri_merge_loss = tri_loss + tri_context_loss + self.ent_neighbor_loss_weight * ent_neighbors_loss
        tri_merge_logits = logits_from_triple[:, self.entity_begin_idx: self.entity_end_idx] + logits_from_triple_context[:, self.entity_begin_idx: self.entity_end_idx]
        tri_merge_rank = get_ranks(F.softmax(tri_merge_logits, dim=-1), labels, entity_filters)

        # 3.2 loss from the triple with qualifier
        tri_qual_loss = self.loss_fc(logits_from_triple_qual, labels + self.entity_begin_idx)
        tri_qual_context_loss = self.loss_fc(logits_from_triple_qual_context, labels + self.entity_begin_idx)
        if self.ablation_mode == 'normal':
            ent_qual_neighbors_loss = self.loss_fc(logits_from_local_global_ent_qual_neighbors, neighbor_labels + self.entity_begin_idx)
        elif self.ablation_mode == 'local':
            ent_qual_neighbors_loss = self.loss_fc(logits_from_local_ent_qual_neighbors, neighbor_labels + self.entity_begin_idx)
        elif self.ablation_mode == 'global':
            ent_qual_neighbors_loss = self.loss_fc(logits_from_global_ent_qual_neighbors, neighbor_labels + self.entity_begin_idx)
        tri_qual_merge_loss = tri_qual_loss + tri_qual_context_loss + self.ent_neighbor_loss_weight * ent_qual_neighbors_loss
        tri_qual_merge_logits = logits_from_triple_qual[:, self.entity_begin_idx: self.entity_end_idx] + logits_from_triple_qual_context[:, self.entity_begin_idx: self.entity_end_idx]
        tri_qual_merge_rank = get_ranks(F.softmax(tri_qual_merge_logits, dim=-1), labels, qual_filters)

        if self.loss_valid_ablation == 'tri':
            return {'loss': tri_merge_loss, 'rank': tri_merge_rank, 'logits': tri_merge_logits}
        elif self.loss_valid_ablation == 'qual':
            return {'loss': tri_qual_merge_loss, 'rank': tri_qual_merge_rank, 'logits': tri_qual_merge_logits}
        elif self.loss_valid_ablation == 'tri_qual':
            tri_qual_all_loss = self.multi_loss([tri_merge_loss, tri_qual_merge_loss])
            tri_qual_all_logits = tri_merge_logits + tri_qual_merge_logits
            tri_qual_all_rank = get_ranks(F.softmax(tri_qual_all_logits, dim=-1), labels, qual_filters)
            return {'loss': tri_qual_all_loss, 'rank': tri_qual_all_rank, 'logits': tri_qual_all_logits}

    def link_prediction(self, batch):
        # 1. prepare the input data
        tri_input = batch['tri_seq']['input_ids'].to(self.device)
        tri_qual_input = batch['tri_qual_seq']['input_ids'].to(self.device)
        if self.add_ent_neighbors:
            ent_context_input = [t['input_ids'].to(self.device) for t in batch['ent_neighbors']]
            ent_qual_context_input = [t['input_ids'].to(self.device) for t in batch['ent_qual_neighbors']]
        else:
            ent_context_input = None
            ent_qual_context_input = None
        if self.add_rel_neighbors:
            rel_context_input = [t['input_ids'].to(self.device) for t in batch['rel_qual_neighbors']]
        else:
            rel_context_input = None
        labels = batch['labels'].to(self.device)
        neighbor_labels = batch['neighbor_labels'].to(self.device)
        entity_filters = batch['entity_filters'].to(self.device)
        qual_filters = batch['qual_filters'].to(self.device)
        mask_pos = batch['mask_pos'].to(self.device)

        # 2. encode
        output = self.bert_encoder(tri_input, tri_qual_input, ent_context_input, ent_qual_context_input, rel_context_input, mask_pos, self.use_extra_encoder)
        logits_from_triple = output['tri_without_neighbors']
        logits_from_triple_context = output['tri_with_neighbors']
        logits_from_local_ent_neighbors = output['tri_local_neighbors']
        logits_from_global_ent_neighbors = output['tri_global_neighbors']
        logits_from_local_global_ent_neighbors = output['tri_local_global_neighbors']

        logits_from_triple_qual = output['tri_qual_without_neighbors']
        logits_from_triple_qual_context = output['tri_qual_with_neighbors']
        logits_from_local_ent_qual_neighbors = output['tri_qual_local_neighbors']
        logits_from_global_ent_qual_neighbors = output['tri_qual_global_neighbors']
        logits_from_local_global_ent_qual_neighbors = output['tri_qual_local_global_neighbors']

        # 3. compute lossed
        # 3.1 loss from the triple
        tri_loss = self.loss_fc(logits_from_triple, labels + self.entity_begin_idx)
        tri_context_loss = self.loss_fc(logits_from_triple_context, labels + self.entity_begin_idx)
        if self.ablation_mode == 'normal':
            ent_neighbors_loss = self.loss_fc(logits_from_local_global_ent_neighbors, neighbor_labels + self.entity_begin_idx)
        elif self.ablation_mode == 'local':
            ent_neighbors_loss = self.loss_fc(logits_from_local_ent_neighbors, neighbor_labels + self.entity_begin_idx)
        elif self.ablation_mode == 'global':
            ent_neighbors_loss = self.loss_fc(logits_from_global_ent_neighbors, neighbor_labels + self.entity_begin_idx)
        tri_merge_loss = tri_loss + tri_context_loss + self.ent_neighbor_loss_weight * ent_neighbors_loss
        tri_merge_logits = logits_from_triple[:, self.entity_begin_idx: self.entity_end_idx] + logits_from_triple_context[:, self.entity_begin_idx: self.entity_end_idx]
        tri_merge_rank = get_ranks(F.softmax(tri_merge_logits, dim=-1), labels, entity_filters)

        # 3.2 loss from the triple with qualifier
        tri_qual_loss = self.loss_fc(logits_from_triple_qual, labels + self.entity_begin_idx)
        tri_qual_context_loss = self.loss_fc(logits_from_triple_qual_context, labels + self.entity_begin_idx)
        if self.ablation_mode == 'normal':
            ent_qual_neighbors_loss = self.loss_fc(logits_from_local_global_ent_qual_neighbors, neighbor_labels + self.entity_begin_idx)
        elif self.ablation_mode == 'local':
            ent_qual_neighbors_loss = self.loss_fc(logits_from_local_ent_qual_neighbors, neighbor_labels + self.entity_begin_idx)
        elif self.ablation_mode == 'global':
            ent_qual_neighbors_loss = self.loss_fc(logits_from_global_ent_qual_neighbors, neighbor_labels + self.entity_begin_idx)
        tri_qual_merge_loss = tri_qual_loss + tri_qual_context_loss + self.ent_neighbor_loss_weight * ent_qual_neighbors_loss
        tri_qual_merge_logits = logits_from_triple_qual[:, self.entity_begin_idx: self.entity_end_idx] + logits_from_triple_qual_context[:, self.entity_begin_idx: self.entity_end_idx]
        tri_qual_merge_rank = get_ranks(F.softmax(tri_qual_merge_logits, dim=-1), labels, qual_filters)

        if self.loss_train_ablation == 'tri':
            return {'loss': tri_merge_loss, 'rank': tri_merge_rank, 'logits': tri_merge_logits}
        elif self.loss_train_ablation == 'qual':
            return {'loss': tri_qual_merge_loss, 'rank': tri_qual_merge_rank, 'logits': tri_qual_merge_logits}
        elif self.loss_train_ablation == 'tri_qual':
            tri_qual_all_loss = self.multi_loss([tri_merge_loss, tri_qual_merge_loss])
            tri_qual_all_logits = tri_merge_logits + tri_qual_merge_logits
            tri_qual_all_rank = get_ranks(F.softmax(tri_qual_all_logits, dim=-1), labels, qual_filters)
            return {'loss': tri_qual_all_loss, 'rank': tri_qual_all_rank, 'logits': tri_qual_all_logits}

    def configure_optimizers(self, max_step):
        opt = torch.optim.AdamW(ContiguousParams(self.bert_encoder.parameters()).contiguous(), lr=self.lr)
        scheduler = None
        return {'optimizer': opt, 'scheduler': scheduler}

    def get_parameters(self):
        decay_param = []
        no_decay_param = []
        for n, p in self.bert_encoder.named_parameters():
            if not p.requires_grad:
                continue
            if ('bias' in n) or ('LayerNorm.weight' in n):
                no_decay_param.append(p)
            else:
                decay_param.append(p)
        return [
            {'params': decay_param, 'weight_decay': 1e-2, 'lr': self.lr},
            {'params': no_decay_param, 'weight_decay': 0, 'lr': self.lr}
        ]

    def freeze(self):
        for n, p in self.bert_encoder.named_parameters():
            p.requires_grad = False

    def clip_grad_norm(self):
        norms = get_norms(self.bert_encoder.parameters()).item()
        info = f'grads for N-Former: {round(norms, 4)}'
        return info

    def grad_norm(self):
        norms = get_norms(self.bert_encoder.parameters()).item()
        return round(norms, 4)