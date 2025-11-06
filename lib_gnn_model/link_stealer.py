import logging
import os

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

torch.cuda.empty_cache()
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborSampler, NeighborLoader
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import negative_sampling, to_undirected
from torch.autograd import grad
import numpy as np

from lib_gnn_model.gcn.gcn_net_batch import GCNNet
from lib_gnn_model.gat.gat_net_batch import GATNet
from lib_utils import utils
from lib_gnn_model.node_classifier import NodeClassifier
from lib_gnn_model.gnn_base import GNNBase

from lib_utils.trend_feature import compute_trend_features
from lib_utils.distance import *

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

from .link_stealer_backbones import *

class LinkStealer(GNNBase):
    def __init__(self, args, shadow_data, attack_data=None):
        super(LinkStealer, self).__init__()

        self.args = args
        self.logger = logging.getLogger('link_stealer')
        self.lp_attack_model = args['lp_attack_model']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.determine_model(shadow_data)
        self.build_data(shadow_data, attack_data)

    def determine_model(self, shadow_data):
        x_dim, prob_dim = shadow_data.x.shape[1], shadow_data.unlearn_prob.shape[1]

        self.link_predictor = LinkPredictor(in_channels=64, hidden_channels=16, out_channels=1, num_layers=2, dropout=0.5)

        if self.args['attack_method'] == 'mia_gnn':
            self.model = MLPNet(prob_dim, hidden_feats=64)
        elif self.args['attack_method'] == 'transfer_lp':
            if self.lp_attack_model == 'SAGE':
                self.model = SAGE(x_dim, hidden_channels=64, out_channels=64, num_layers=2, dropout=0.5)
            elif self.lp_attack_model == 'MLP':
                self.model = MLPNet(x_dim, hidden_feats=64)
        elif self.args['attack_method'] == 'steal_link':
            self.model = EmptyBackbone()
            self.link_predictor = SimilarityPredictorV2(x_dim=x_dim, prob_dim=prob_dim, hidden_dim=64)
        elif self.args['attack_method'] == 'trend_mia':
            self.model = EmptyBackbone()
            trend_feat_dim = 2 * self.args['trend_k']
            self.link_predictor = TrendMIAPredictor(prob_dim, trend_feat_dim=trend_feat_dim, 
                                                    hidden_dim1=64, hidden_dim2=16, dropout=0.5)
        elif self.args['attack_method'] == 'trend_steal':
            self.model = EmptyBackbone()
            trend_feat_dim = 2 * self.args['trend_k']
            self.link_predictor = TrendSimilarityPredictor(x_dim=x_dim, prob_dim=prob_dim, 
                                                           hidden_dim=64, trend_feat_dim=trend_feat_dim)

        self.lr, self.decay = 0.01, 0.0001
    
    def __build_feature(self, shadow_data, attack_data):
        shadow_data = shadow_data.to(self.device)
        attack_data = attack_data.to(self.device)
        if self.args['attack_method'] in ['mia_gnn', 'group_attack']:
            train_x = shadow_data.unlearn_prob
            test_x = attack_data.unlearn_prob
        elif self.args['attack_method'] == 'trend_mia':
            train_x = shadow_data.unlearn_prob
            test_x = attack_data.unlearn_prob
        elif self.args['attack_method'] == 'trend_steal':
            train_x = torch.cat([shadow_data.x, 
                                 shadow_data.unlearn_prob,
                                 shadow_data.reference_prob], dim=1)
            test_x = torch.cat([attack_data.x, 
                                attack_data.unlearn_prob,
                                attack_data.reference_prob], dim=1)
        elif self.args['attack_method'] == 'transfer_lp':
            train_x = shadow_data.x
            test_x = attack_data.x
        elif self.args['attack_method'] == 'steal_link':
            train_x = torch.cat([shadow_data.x, 
                                 shadow_data.unlearn_prob,
                                 shadow_data.reference_prob], dim=1)
            test_x = torch.cat([attack_data.x, 
                                attack_data.unlearn_prob,
                                attack_data.reference_prob], dim=1)

        return train_x, test_x

    def __sample_positive_edges(self, data):
        pos_unlearn = data.removed_edges_und
            
        edge_remain = data.edge_index_unlearn
        unique_remain = edge_remain[:, edge_remain[0] < edge_remain[1]]
        selected_indices = torch.randperm(unique_remain.size(1), device=edge_remain.device)[:pos_unlearn.size(1) // 2]
        pos_norm = unique_remain[:, selected_indices]
        pos_norm_und = torch.cat([pos_norm, pos_norm[[1, 0]]], dim=1)
        
        return pos_unlearn, pos_norm_und
        
    def build_data(self, shadow_data, attack_data):
        train_x, test_x = self.__build_feature(shadow_data, attack_data)
        
        # Build train data
        self.train_data = Data(x=train_x, edge_index=shadow_data.edge_index_unlearn)
        pos_unlearn, pos_norm_und = self.__sample_positive_edges(shadow_data)
        if self.args['attack_method'] in ['trend_mia', 'trend_steal']:
            self.train_pos_edges = torch.cat([pos_unlearn, pos_norm_und], dim=1)
        else:
            self.train_pos_edges = pos_norm_und
        n_neg_samples = int(self.train_pos_edges.size(1) * self.args['attack_train_neg_ratio'])
        self.train_neg_edges = negative_sampling(edge_index=shadow_data.edge_index,
                                                num_nodes=shadow_data.num_nodes,
                                                num_neg_samples=n_neg_samples)
        
        # Build test data
        pos_unlearn, pos_norm_und = self.__sample_positive_edges(attack_data)
        self.test_pos_edges = torch.cat([pos_unlearn, pos_norm_und], dim=1)
        n_neg_samples = int(self.test_pos_edges.size(1) * self.args['attack_test_neg_ratio'])
        self.test_neg_edges = negative_sampling(edge_index=attack_data.edge_index,
                                                num_nodes=attack_data.num_nodes,
                                                num_neg_samples=n_neg_samples)
    
        test_edge_index = attack_data.edge_index_unlearn
        all_encode = attack_data.num_nodes * test_edge_index[0] + test_edge_index[1]
        pos_norm_encode = attack_data.num_nodes * pos_norm_und[0] + pos_norm_und[1]
        keep_mask = ~torch.isin(all_encode, pos_norm_encode)
        new_test_edge_index = test_edge_index[:, keep_mask]

        self.test_data = Data(x=test_x, edge_index=new_test_edge_index)
        
        # For trend attack, we append some extra features to the original features
        if self.args['attack_method'] in ['trend_mia', 'trend_steal']:
            # Compute Trend Features
            train_trend_f = compute_trend_features(self.train_data, k=self.args['trend_k'])
            test_trend_f = compute_trend_features(self.test_data, k=self.args['trend_k'])
            
            # Normalize Trend Features
            #scaler = StandardScaler()
            #scaler.fit(train_trend_f.cpu().numpy())
            #train_trend_f_norm = torch.from_numpy(scaler.transform(train_trend_f.cpu().numpy())).to(self.device)
            #test_trend_f_norm = torch.from_numpy(scaler.transform(test_trend_f.cpu().numpy())).to(self.device)
            train_trend_f_norm = train_trend_f
            test_trend_f_norm = test_trend_f

            # Get the final feature
            self.train_data.x = torch.cat([self.train_data.x, train_trend_f_norm], dim=1)
            self.test_data.x = torch.cat([self.test_data.x, test_trend_f_norm], dim=1)

        self.train_tot_edges = self.train_pos_edges.size(1)
        self.test_tot_edges = self.test_pos_edges.size(1)
        self.train_batch_size = 512

    def shadow_train(self):
        if self.args['attack_method'] == 'group_attack':
            self.shadow_get_threshold()
        else:
            self.shadow_train_gnn()
    
    @torch.no_grad()
    def attack_evaluate(self):
        if self.args['attack_method'] == 'group_attack':
            return self.attack_evaluate_threshold()
        else:
            return self.attack_evaluate_gnn()
    
    def shadow_train_gnn(self, model=None, link_predictor=None):
        self.logger.info("shadow link stealer training")

        if model is None:
            model = self.model
        if link_predictor is None:
            link_predictor = self.link_predictor

        model.train()
        link_predictor.train()
        model, link_predictor = model.to(self.device), link_predictor.to(self.device)
        self.train_data = self.train_data.to(self.device)
        self.train_pos_edges, self.train_neg_edges = self.train_pos_edges.to(self.device), self.train_neg_edges.to(self.device)

        # mid = self.train_pos_edges.shape[1] // 2
        # sim_0 = js_distance(self.train_data.x[self.train_pos_edges[0, :mid]],
        #                     self.train_data.x[self.train_pos_edges[1, :mid]]).cpu().numpy()
        # sim_1 = js_distance(self.train_data.x[self.train_pos_edges[0, mid:]],
        #                     self.train_data.x[self.train_pos_edges[1, mid:]]).cpu().numpy()
        # sim_2 = js_distance(self.train_data.x[self.train_neg_edges[0]],
        #                     self.train_data.x[self.train_neg_edges[1]]).cpu().numpy()
        # sim_1 = sim_1[~np.isnan(sim_1)]
        # sim_0 = sim_0[~np.isnan(sim_0)]
        # sim_2 = sim_2[~np.isnan(sim_2)]

        #input(f'({1 - np.average(sim_2):.4f} ± {np.std(sim_2):.4f},{1 - np.average(sim_0):.4f} ± {np.std(sim_0):.4f},{1 - np.average(sim_1):.4f} ± {np.std(sim_1):.4f})')

        optimizer = torch.optim.Adam([
            {'params': model.parameters(), 'lr': self.lr, 'weight_decay': self.decay},
            {'params': link_predictor.parameters(), 'lr': self.lr, 'weight_decay': self.decay}
        ])

        for epoch in range(100):
            self.logger.info('epoch %s' % (epoch,))

            total_loss = total_examples = 0
            for i in range(0, self.train_tot_edges, self.train_batch_size):
                pos_edges = self.train_pos_edges[:, i:i+self.train_batch_size]
                ratio = self.args['attack_train_neg_ratio']
                neg_edges = self.train_neg_edges[:, int(i * ratio):int((i+self.train_batch_size) * ratio)]
                target_nodes = torch.unique(torch.cat([pos_edges.T, neg_edges.T]).flatten()) # This can be accelerated
                
                pos_edges_mapped = torch.searchsorted(target_nodes, pos_edges)
                neg_edges_mapped = torch.searchsorted(target_nodes, neg_edges)
                #input(pos_edges_mapped)

                # These attack methods are not using neighbor aggergation
                if self.args['attack_method'] in ['steal_link', 'mia_gnn', 'trend_mia', 'trend_steal']:
                    loader = NeighborLoader(
                        self.train_data,
                        num_neighbors=[0],
                        batch_size=target_nodes.size(0),
                        input_nodes=target_nodes,
                        filter_per_worker=True
                    )
                elif self.args['attack_method'] == 'transfer_lp':
                    D = self.args['num_neighbors']
                    loader = NeighborLoader(
                        self.train_data,
                        num_neighbors=[D, D],
                        batch_size=target_nodes.size(0),
                        input_nodes=target_nodes,
                        filter_per_worker=True
                    )
                batch_data = next(iter(loader))
                
                optimizer.zero_grad()
                
                x = model(batch_data.x, batch_data.edge_index)
                pos_out = link_predictor(x[pos_edges_mapped[0]], x[pos_edges_mapped[1]])
                neg_out = link_predictor(x[neg_edges_mapped[0]], x[neg_edges_mapped[1]])
                
                pos_loss = -torch.log(pos_out + 1e-15).mean()
                neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

                loss = pos_loss + neg_loss
                loss.backward()

                #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                #torch.nn.utils.clip_grad_norm_(link_predictor.parameters(), 1.0)

                optimizer.step()

                num_examples = pos_out.size(0)
                total_loss += loss.item() * num_examples
                total_examples += num_examples

            self.logger.info(f'Train Loss: {total_loss / total_examples:.4f}')
    
    def __eval_once(self, y_pred, y_true, is_binary_predict=False):
        if not is_binary_predict:
            y_pred2 = y_pred > 0.5
        else:
            y_pred2 = y_pred

        return {
            'auc':       roc_auc_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred2),
            'recall':    recall_score(y_true, y_pred2),
            'f1':        f1_score(y_true, y_pred2),
        }
    
    @torch.no_grad()
    def __compute_metrics(self, pos_out, neg_out, is_binary_predict=False):
        num_pos = len(pos_out) 
        num_un_pos = num_pos // 2
        num_neg = len(neg_out) 
    
        # --- Group 1: unlearned vs. negatives ---
        g1_pred = torch.cat([pos_out[:num_un_pos], neg_out]).cpu().numpy()
        g1_true = np.concatenate([np.ones(num_un_pos), np.zeros(num_neg)])
        res_g1  = self.__eval_once(g1_pred, g1_true, is_binary_predict)

        # --- Group 2: normal vs. negatives ---
        g2_pred = torch.cat([pos_out[num_un_pos:], neg_out]).cpu().numpy()
        g2_true = np.concatenate([np.ones(num_un_pos), np.zeros(num_neg)])
        res_g2  = self.__eval_once(g2_pred, g2_true, is_binary_predict)

        # --- Combined: all positives vs. negatives ---
        all_pred = torch.cat([pos_out, neg_out]).cpu().numpy()
        all_true = np.concatenate([np.ones(num_pos), np.zeros(num_neg)])
        res_all  = self.__eval_once(all_pred, all_true, is_binary_predict)

        return res_g1, res_g2, res_all

    @torch.no_grad()
    def attack_evaluate_gnn(self, model=None, link_predictor=None):
        if model is None:
            model = self.model
        if link_predictor is None:
            link_predictor = self.link_predictor

        model.eval()
        link_predictor.eval()

        model, link_predictor = model.to(self.device), link_predictor.to(self.device)
        self.test_data = self.test_data.to(self.device)
        self.test_pos_edges, self.test_neg_edges = self.test_pos_edges.to(self.device), self.test_neg_edges.to(self.device)

        pos_edges = self.test_pos_edges
        neg_edges = self.test_neg_edges
        target_nodes = torch.unique(torch.cat([pos_edges.T, neg_edges.T]).flatten()) # This can be accelerated
                
        pos_edges_mapped = torch.searchsorted(target_nodes, pos_edges)
        neg_edges_mapped = torch.searchsorted(target_nodes, neg_edges)
        
        # These attack methods are not using neighbor aggergation
        if self.args['attack_method'] in ['steal_link', 'mia_gnn', 'trend_mia', 'trend_steal']:
            loader = NeighborLoader(
                self.test_data,
                num_neighbors=[0],
                batch_size=target_nodes.size(0),
                input_nodes=target_nodes,
                filter_per_worker=True
            )
        elif self.args['attack_method'] == 'transfer_lp':
            D = self.args['num_neighbors']
            loader = NeighborLoader(
                self.test_data,
                num_neighbors=[D, D],
                batch_size=target_nodes.size(0),
                input_nodes=target_nodes,
                filter_per_worker=True
            )
        batch_data = next(iter(loader))
                
        x = model(batch_data.x, batch_data.edge_index)
        pos_out = link_predictor(x[pos_edges_mapped[0]], x[pos_edges_mapped[1]])
        neg_out = link_predictor(x[neg_edges_mapped[0]], x[neg_edges_mapped[1]])

        return self.__compute_metrics(pos_out, neg_out, is_binary_predict=False)

    def shadow_get_threshold(self, search_interval=0.05, sim_func=cosine_distance):
        self.logger.info("shadow link stealer training")
        
        # Part 1: Assume that there is no group splitting, get an optimal threshold
        # We search the threshold based on the shadow data and binary classification AUC
        pos_scores = 1 - sim_func(self.train_data.x[self.train_pos_edges[0]], 
                                  self.train_data.x[self.train_pos_edges[1]]).cpu().numpy()
        neg_scores = 1 - sim_func(self.train_data.x[self.train_neg_edges[0]],
                                  self.train_data.x[self.train_neg_edges[1]]).cpu().numpy()
        #input((np.average(pos_scores), np.average(neg_scores), np.std(pos_scores), np.std(neg_scores)))
        
        tau_s_candidates = np.arange(0., 1., search_interval)
        aucs = []
        y_true = np.concatenate([np.ones(pos_scores.size), np.zeros(neg_scores.size)])
        y_scores = np.concatenate([pos_scores, neg_scores])
        for tau_s in tau_s_candidates:
            y_pred = (y_scores > tau_s).astype(int)
            auc = roc_auc_score(y_true, y_pred)
            aucs.append(auc)
        #input((tau_s_candidates, aucs))
        
        best_tau_s = tau_s_candidates[np.argmax(aucs)]
        self.tau_intra = best_tau_s
        self.logger.info('Best threshold: {:.4f}'.format(best_tau_s))
        self.logger.info('Best shadow AUC: {:.4f}'.format(np.max(aucs)))

        # Part 2: Compute homophily ratio (beta)
        labels = torch.argmax(self.train_data.x, dim=1)
        m = self.train_data.edge_index.size(1)
        hom_cnt = torch.sum(labels[self.train_data.edge_index[0]] == labels[self.train_data.edge_index[1]])
        beta = 1 - hom_cnt / m
        self.logger.info('Homophily ratio: {:.4f}'.format(beta))

        # Part 3: Compute g_inter and g_intra ratio
        train_edges = torch.cat([self.train_pos_edges, self.train_neg_edges], dim=1)
        g_inter_size = torch.sum(labels[train_edges[0]] == labels[train_edges[1]])
        alpha = g_inter_size / m
        self.logger.info('g_inter ratio: {:.4f}'.format(alpha))

        # Part 4: Compute the final threshold
        self.tau_inter = float(alpha * self.tau_intra + (1 - alpha) * beta)
        self.logger.info('Final threshold: tau_intra = {:.4f}, tau_inter = {:.4f}'.format(self.tau_intra, self.tau_inter))

        # Part 5: Test the thresholds on shadow data
        inter_indices = torch.where(labels[train_edges[0]] != labels[train_edges[1]])[0].cpu().numpy()
        intra_indices = torch.where(labels[train_edges[0]] == labels[train_edges[1]])[0].cpu().numpy()
        
        y_pred[inter_indices] = (y_scores[inter_indices] > self.tau_inter)
        y_pred[intra_indices] = (y_scores[intra_indices] > self.tau_intra)
        y_pred = y_pred.astype(int)
        auc = roc_auc_score(y_true, y_pred)

        self.logger.info('Final threshold test AUC: {:.4f}'.format(auc))

    def attack_evaluate_threshold(self, sim_func=cosine_distance):
        pos_scores = 1 - sim_func(self.test_data.x[self.test_pos_edges[0]], 
                                  self.test_data.x[self.test_pos_edges[1]]).cpu().numpy()
        neg_scores = 1 - sim_func(self.test_data.x[self.test_neg_edges[0]],
                                  self.test_data.x[self.test_neg_edges[1]]).cpu().numpy()
        
        y_true = np.concatenate([np.ones(pos_scores.size), np.zeros(neg_scores.size)])
        y_scores = np.concatenate([pos_scores, neg_scores])

        test_edges = torch.cat([self.test_pos_edges, self.test_neg_edges], dim=1)

        inter_indices = torch.where(self.test_data.x[test_edges[0]] != self.test_data.x[test_edges[1]])[0].cpu().numpy()
        intra_indices = torch.where(self.test_data.x[test_edges[0]] == self.test_data.x[test_edges[1]])[0].cpu().numpy()

        y_pred = np.zeros(y_scores.shape)
        y_pred[inter_indices] = (y_scores[inter_indices] > self.tau_inter)
        y_pred[intra_indices] = (y_scores[intra_indices] > self.tau_intra)
        y_pred = y_pred.astype(int)

        pos_out, neg_out = y_pred[:self.test_pos_edges.size(1)], y_pred[self.test_pos_edges.size(1):]
        pos_out, neg_out = torch.from_numpy(pos_out), torch.from_numpy(neg_out)

        return self.__compute_metrics(pos_out, neg_out, is_binary_predict=True)
