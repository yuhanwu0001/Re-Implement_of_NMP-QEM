#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import TestDataset, TrainDataset, SingledirectionalOneShotIterator
import random
import pickle
import math
import collections
import itertools
import time
from tqdm import tqdm
import os

def Identity(x):
    return x

class BoxOffsetIntersection(nn.Module):
    
    def __init__(self, dim):
        super(BoxOffsetIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))
        layer1_mean = torch.mean(layer1_act, dim=0) 
        gate = torch.sigmoid(self.layer2(layer1_mean))
        offset, _ = torch.min(embeddings, dim=0)

        return offset * gate

class CenterIntersection(nn.Module):

    def __init__(self, dim):
        super(CenterIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings)) # (num_conj, dim)
        attention = F.softmax(self.layer2(layer1_act), dim=0) # (num_conj, dim)
        embedding = torch.sum(attention * embeddings, dim=0)

        return embedding

class BetaIntersection(nn.Module):

    def __init__(self, dim):
        super(BetaIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(2 * self.dim, 2 * self.dim)
        self.layer2 = nn.Linear(2 * self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, alpha_embeddings, beta_embeddings):
        all_embeddings = torch.cat([alpha_embeddings, beta_embeddings], dim=-1)
        layer1_act = F.relu(self.layer1(all_embeddings)) # (num_conj, batch_size, 2 * dim)
        attention = F.softmax(self.layer2(layer1_act), dim=0) # (num_conj, batch_size, dim)

        alpha_embedding = torch.sum(attention * alpha_embeddings, dim=0)
        beta_embedding = torch.sum(attention * beta_embeddings, dim=0)

        return alpha_embedding, beta_embedding

class BetaProjection(nn.Module):
    def __init__(self, entity_dim, relation_dim, hidden_dim, projection_regularizer, num_layers):
        super(BetaProjection, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim) # 1st layer
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim) # final layer
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)
        self.projection_regularizer = projection_regularizer

    def forward(self, e_embedding, r_embedding):
        x = torch.cat([e_embedding, r_embedding], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)
        x = self.projection_regularizer(x)

        return x

class NMP_QEM_Projection(nn.Module):
    def __init__(self, dim, num_layers):
        super(NMP_QEM_Projection, self).__init__()
        self.entity_dim = dim + 1
        self.relation_dim = dim
        self.hidden_dim = dim + 1
        self.num_layers = num_layers
        # concatenation 版本：
        # self.layer1 = nn.Linear(self.relation_dim + self.entity_dim, self.hidden_dim)
        # self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim)
        # for nl in range(2, self.num_layers + 1):
        #     setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        # for nl in range(self.num_layers + 1):
        #     nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)

        # sum 版本
        self.layer1 = nn.Linear(self.entity_dim, self.hidden_dim)
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim)
        for nl in range(2, self.num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(self.num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)

    def forward(self, query_embedding, relation_embedding):
        '''
            relation embedding: [batch_size, d]
            query embedding: [batch_size, 2K, d+1]
        '''
        # concatenation 版本
        # batch_size, doubleK, entity_dim = query_embedding.shape
        # # [batch_size, 1, d]
        # relation_embedding = relation_embedding.unsqueeze(1)
        # # [batch_size, 2K, d]
        # expanded_relation_embedding = relation_embedding.repeat(1, doubleK, 1)
        # # [batch_size, 2K, 2d+1]
        # x = torch.cat([query_embedding, expanded_relation_embedding], dim=-1)
        # # [batch_size, 2K, d+1]
        # for nl in range(1, self.num_layers + 1):
        #     x = F.relu(getattr(self, "layer{}".format(nl))(x))
        # x = self.layer0(x)

        # sum 版本：
        batch_size, doubleK, entity_dim = query_embedding.shape
        # [batch_size, 1, d]
        relation_embedding = relation_embedding.unsqueeze(1)
        weight_padding = np.array([1.0])
        relation_weight_padding = torch.Tensor(weight_padding).cuda()
        # [1, 1, 1]
        relation_weight_padding = relation_weight_padding.unsqueeze(0).unsqueeze(0)
        # [batch_size, 1, 1]
        relation_weight_padding = relation_weight_padding.repeat(batch_size, 1, 1)
        # [batch_size, 1, d+1]
        expanded_relation_embedding = torch.cat([relation_embedding, relation_weight_padding], dim=-1)
        # [batch_size, 2K, d+1]
        x = expanded_relation_embedding + query_embedding
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)

        # regularize for weight value and variance
        # [batch_size, K, d+1]
        query_mean_weight, query_variance = torch.chunk(x, 2, dim=1)
        # [batch_size, K, d], [batch_size, K, 1]
        query_mean, query_weight = torch.split(query_mean_weight, [self.relation_dim, 1], dim=-1)
        normalized_query_weight = F.softmax(query_weight, dim=1)
        # [batch_size, K, d+1]
        new_query_mean_weight = torch.cat([query_mean, normalized_query_weight], dim=-1)
        regularized_query_variance = F.relu(query_variance)
        # [batch_size, 2K, d+1]
        projected_query_embedding = torch.cat([new_query_mean_weight, regularized_query_variance], dim=1)
        return projected_query_embedding

class NMP_QEM_Negation(nn.Module):
    def __init__(self, dim, num_layers):
        super(NMP_QEM_Negation, self).__init__()
        self.entity_dim = dim + 1
        self.relation_dim = dim
        self.hidden_dim = dim + 1
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.entity_dim, self.hidden_dim)
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim)
        for nl in range(2, self.num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(self.num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)

    def forward(self, query_embedding):
        # [batch_size, 2K, d+1]
        x = query_embedding
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)

        # regularize for weight value and variance
        # [batch_size, K, d+1]
        query_mean_weight, query_variance = torch.chunk(x, 2, dim=1)
        # [batch_size, K, d], [batch_size, K, 1]
        query_mean, query_weight = torch.split(query_mean_weight, [self.relation_dim, 1], dim=-1)
        normalized_query_weight = F.softmax(query_weight, dim=1)
        # [batch_size, K, d+1]
        new_query_mean_weight = torch.cat([query_mean, normalized_query_weight], dim=-1)
        regularized_query_variance = F.relu(query_variance)
        # [batch_size, 2K, d+1]
        negationed_query_embedding = torch.cat([new_query_mean_weight, regularized_query_variance], dim=1)
        return negationed_query_embedding

class NMP_QEM_Intersection(nn.Module):
    def __init__(self, dim):
        super(NMP_QEM_Intersection, self).__init__()
        self.dim = dim
        self.hidden_size = self.dim
        self.WQ = nn.Linear(2 * (dim + 1), 2 * (dim + 1))
        self.WK = nn.Linear(2 * (dim + 1), 2 * (dim + 1))
        self.WV = nn.Linear(2 * (dim + 1), 2 * (dim + 1))

    def forward(self, stacked_embeddings):
        '''
        stacked_embeddings:
            2i: [2, batch_size, 2K, d+1]
            3i: [3, batch_size, 2K, d+1]
        embedding1/2/3: [batch_size, 2K, d+1]
        '''
        input_count, batch_size, doubleK, query_dim = stacked_embeddings.shape
        if input_count == 2:
            # [batch_size, 2K, d+1]
            embedding1, embedding2 = torch.chunk(stacked_embeddings, 2, dim=0)
            embedding1 = embedding1.squeeze(0)
            embedding2 = embedding2.squeeze(0)
        elif input_count == 3:
            # [batch_size, 2K, d+1]
            embedding1, embedding2, embedding3 = torch.chunk(stacked_embeddings, 3, dim=0)
            embedding1 = embedding1.squeeze(0)
            embedding2 = embedding2.squeeze(0)
            embedding3 = embedding3.squeeze(0)
        else:
            raise ValueError('intersection count %d not supported' % input_count)
        # [batch_size, K, d+1]
        embedding_mean_part1, embedding_variance_part1 = torch.chunk(embedding1, 2, dim=1)
        embedding_mean_part2, embedding_variance_part2 = torch.chunk(embedding2, 2, dim=1)
        # [batch_size, K, 2(d+1)]
        reshaped_embedding1 = torch.cat([embedding_mean_part1, embedding_variance_part1], dim=-1)
        reshaped_embedding2 = torch.cat([embedding_mean_part2, embedding_variance_part2], dim=-1)
        if input_count == 3:
            embedding_mean_part3, embedding_variance_part3 = torch.chunk(embedding3, 2, dim=1)
            # [batch_size, K, 2(d+1)]
            reshaped_embedding3 = torch.cat([embedding_mean_part3, embedding_variance_part3], dim=-1)
        Q1 = self.WQ(reshaped_embedding1)
        K1 = self.WK(reshaped_embedding1)
        V1 = self.WV(reshaped_embedding1)
        Q2 = self.WQ(reshaped_embedding2)
        K2 = self.WK(reshaped_embedding2)
        V2 = self.WV(reshaped_embedding2)
        # [batch_size, K, K]
        attention_score1 = torch.matmul(Q1, K2.permute(0, 2, 1))
        attention_score1 = attention_score1 / math.sqrt(self.hidden_size)
        attention_prob1 = nn.Softmax(dim=-1)(attention_score1)
        # [batch_size, K, 2(d+1)]
        attention_output1 = torch.matmul(attention_prob1, V2)

        attention_score2 = torch.matmul(Q2, K1.permute(0, 2, 1))
        attention_score2 = attention_score2 / math.sqrt(self.hidden_size)
        attention_prob2 = nn.Softmax(dim=-1)(attention_score2)
        attention_output2 = torch.matmul(attention_prob2, V1)
        # [batch_size, K, 2(d+1)]
        intersected_embedding = attention_output1 + attention_output2
        if input_count == 3:
            Q1 = self.WQ(intersected_embedding)
            K1 = self.WK(intersected_embedding)
            V1 = self.WV(intersected_embedding)
            Q2 = self.WQ(reshaped_embedding3)
            K2 = self.WK(reshaped_embedding3)
            V2 = self.WV(reshaped_embedding3)
            # [batch_size, K, K]
            attention_score1 = torch.matmul(Q1, K2.permute(0, 2, 1))
            attention_score1 = attention_score1 / math.sqrt(self.hidden_size)
            attention_prob1 = nn.Softmax(dim=-1)(attention_score1)
            # [batch_size, K, 2(d+1)]
            attention_output1 = torch.matmul(attention_prob1, V2)

            attention_score2 = torch.matmul(Q2, K1.permute(0, 2, 1))
            attention_score2 = attention_score2 / math.sqrt(self.hidden_size)
            attention_prob2 = nn.Softmax(dim=-1)(attention_score2)
            attention_output2 = torch.matmul(attention_prob2, V1)
            # [batch_size, K, 2(d+1)]
            intersected_embedding = attention_output1 + attention_output2
        # recover origin shape
        # [batch_size, K, d+1]
        intersected_mean_weight, intersected_variance = torch.chunk(intersected_embedding, 2, dim=-1)
        # [batch_size, 2K, d+1]
        intersected_embedding = torch.cat([intersected_mean_weight, intersected_variance], dim=1)
        return intersected_embedding

class Regularizer():
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)

class KGReasoning(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, gamma, 
                 test_batch_size=1,
                 box_mode=None, use_cuda=False,
                 query_name_dict=None, beta_mode=None,
                 component_count = 2):
        super(KGReasoning, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.use_cuda = use_cuda
        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1).cuda() if self.use_cuda else torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1) # used in test_step
        self.query_name_dict = query_name_dict

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim
        self.hidden_dim = hidden_dim

        self.component_count = component_count

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )

        self.center_net = NMP_QEM_Intersection(self.hidden_dim)
        self.projection_net = NMP_QEM_Projection(self.hidden_dim, 2)
        self.negation_net = NMP_QEM_Negation(self.hidden_dim, 2)

    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        return self.forward_multiGau(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)

    def embed_query_multiGau(self, queries, query_structure, idx):
        '''
                Iterative embed a batch of queries with same structure using BetaE
                queries: a flattened batch of queries
                '''
        all_relation_flag = True
        for ele in query_structure[
            -1]:  # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                # [batch_size, d]
                embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                batch_size, hidden_dim = embedding.shape
                weight_padding = np.array([1.0])
                weight_padding = torch.Tensor(weight_padding).cuda()
                weight_padding = weight_padding.unsqueeze(1)
                # [batch_size, 1]
                weight_padding = weight_padding.repeat(batch_size, 1)
                # [batch_size, d+1]
                embedding = torch.cat([embedding, weight_padding], dim=-1)
                embedding = embedding.unsqueeze(1)
                # [batch_size, 2K, d+1]
                embedding = embedding.repeat(1, self.component_count * 2, 1)
                idx += 1
            else:
                embedding, idx = self.embed_query_multiGau(queries, query_structure[0], idx)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    embedding = self.negation_net(embedding)
                else:
                    r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                    embedding = self.projection_net(embedding, r_embedding)
                idx += 1
        else:
            embedding_list = []
            for i in range(len(query_structure)):
                embedding, idx = self.embed_query_multiGau(queries, query_structure[i], idx)
                embedding_list.append(embedding)
            embedding = self.center_net(torch.stack(embedding_list))
        return embedding, idx

    def cal_logit_multiGau(self, entity_embedding, query_embedding):
        '''
        entity_embedding: [batch_size, d]
        query_embedding: [batch_size, 2K, d+1]
        '''
        # [batch_size, K, d+1]
        query_mean_weight, query_variance = torch.chunk(query_embedding, 2, dim=-2)
        # [batch_size, K, d], [batch_size, K, 1]
        query_mean, query_weight = torch.split(query_mean_weight, [self.hidden_dim, 1], dim=-1)
        # [batch_size, d]
        centroid_center = torch.sum(query_weight * query_mean, dim=-2)
        # [batch_size]
        logit = torch.norm(entity_embedding - centroid_center, p=1, dim=-1)
        return logit

    def forward_multiGau(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_idxs, all_embeddings = [], []
        all_union_idxs, all_union_embeddings = [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure] and 'DNF' in self.query_name_dict[query_structure]:
                embedding, _ = self.embed_query_multiGau(self.transform_union_query(batch_queries_dict[query_structure], query_structure),
                                                         self.transform_union_structure(query_structure), 0)
                all_union_embeddings.append(embedding)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
            else:
                embedding, _ = self.embed_query_multiGau(batch_queries_dict[query_structure], query_structure, 0)
                all_embeddings.append(embedding)
                all_idxs.extend(batch_idxs_dict[query_structure])

        if len(all_embeddings) > 0:
            all_embeddings = torch.cat(all_embeddings, dim=0).unsqueeze(1)
        if len(all_union_embeddings) > 0:
            all_union_embeddings = torch.cat(all_union_embeddings, dim=0).unsqueeze(1)
            all_union_embeddings = all_union_embeddings.view(all_union_embeddings.shape[0] // 2, 2, 1, self.component_count * 2, -1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]
        if type(positive_sample) != type(None):
            if len(all_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1)
                positive_logit = self.cal_logit_multiGau(positive_embedding, all_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)
            if len(all_union_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_union_logit = self.cal_logit_multiGau(positive_embedding, all_union_embeddings)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None
        if type(negative_sample) != type(None):
            if len(all_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size,
                                                                                                                                   negative_size,
                                                                                                                                   -1)
                negative_logit = self.cal_logit_multiGau(negative_embedding, all_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)
            if len(all_union_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_union.view(-1)).view(batch_size, 1,
                                                                                                                                 negative_size,
                                                                                                                                 -1)
                negative_union_logit = self.cal_logit_multiGau(negative_embedding, all_union_embeddings)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None
        return positive_logit, negative_logit, subsampling_weight, all_idxs + all_union_idxs

    def transform_union_query(self, queries, query_structure):
        '''
        transform 2u queries to two 1p queries
        transform up queries to two 2p queries
        '''
        if self.query_name_dict[query_structure] == '2u-DNF':
            queries = queries[:, :-1] # remove union -1
        elif self.query_name_dict[query_structure] == 'up-DNF':
            queries = torch.cat([torch.cat([queries[:, :2], queries[:, 5:6]], dim=1), torch.cat([queries[:, 2:4], queries[:, 5:6]], dim=1)], dim=1)
        queries = torch.reshape(queries, [queries.shape[0]*2, -1])
        return queries

    def transform_union_structure(self, query_structure):
        if self.query_name_dict[query_structure] == '2u-DNF':
            return ('e', ('r',))
        elif self.query_name_dict[query_structure] == 'up-DNF':
            return ('e', ('r', 'r'))

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step):
        model.train()
        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures = next(train_iterator)
        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)
        for i, query in enumerate(batch_queries): # group queries with same structure
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)
        for query_structure in batch_queries_dict:
            if args.cuda:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
            else:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        positive_logit, negative_logit, subsampling_weight, _ = model(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)

        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        loss.backward()
        optimizer.step()
        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
        }
        return log

    @staticmethod
    def test_step(model, easy_answers, hard_answers, args, test_dataloader, query_name_dict, save_result=False, save_str="", save_empty=False):
        model.eval()

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)

        with torch.no_grad():
            for negative_sample, queries, queries_unflatten, query_structures in tqdm(test_dataloader, disable=not args.print_on_screen):
                batch_queries_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)
                for i, query in enumerate(queries):
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idxs_dict[query_structures[i]].append(i)
                for query_structure in batch_queries_dict:
                    if args.cuda:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
                    else:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
                if args.cuda:
                    negative_sample = negative_sample.cuda()

                _, negative_logit, _, idxs = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict)
                queries_unflatten = [queries_unflatten[i] for i in idxs]
                query_structures = [query_structures[i] for i in idxs]
                argsort = torch.argsort(negative_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                if len(argsort) == args.test_batch_size: # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
                    ranking = ranking.scatter_(1, argsort, model.batch_entity_range) # achieve the ranking of all entities
                else: # otherwise, create a new torch Tensor for batch_entity_range
                    if args.cuda:
                        ranking = ranking.scatter_(1, 
                                                   argsort, 
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 
                                                                                                      1).cuda()
                                                   ) # achieve the ranking of all entities
                    else:
                        ranking = ranking.scatter_(1, 
                                                   argsort, 
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 
                                                                                                      1)
                                                   ) # achieve the ranking of all entities
                for idx, (i, query, query_structure) in enumerate(zip(argsort[:, 0], queries_unflatten, query_structures)):
                    hard_answer = hard_answers[query]
                    easy_answer = easy_answers[query]
                    num_hard = len(hard_answer)
                    num_easy = len(easy_answer)
                    assert len(hard_answer.intersection(easy_answer)) == 0
                    cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy
                    if args.cuda:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
                    else:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float)
                    cur_ranking = cur_ranking - answer_list + 1 # filtered setting
                    cur_ranking = cur_ranking[masks] # only take indices that belong to the hard answers

                    mrr = torch.mean(1./cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()

                    logs[query_structure].append({
                        'MRR': mrr,
                        'HITS1': h1,
                        'HITS3': h3,
                        'HITS10': h10,
                        'num_hard_answer': num_hard,
                    })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]])/len(logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        return metrics