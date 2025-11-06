from cgi import test
import logging
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

torch.cuda.empty_cache()
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import NeighborSampler
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import is_undirected, to_undirected
import numpy as np

from exp.exp import Exp
from lib_gnn_model.gat.gat_net_batch import GATNet
from lib_gnn_model.gin.gin_net_batch import GINNet
from lib_gnn_model.gcn.gcn_net_batch import GCNNet
#from lib_gnn_model.graphsage.graphsage_net import SageNet
from lib_gnn_model.sgc.sgc_net_batch import SGCNet
from lib_gnn_model.node_classifier import NodeClassifier
from lib_gnn_model.link_stealer import LinkStealer
from lib_gnn_model.gnn_base import GNNBase
from parameter_parser import parameter_parser
from lib_utils import utils
from lib_utils.partition import DatasetPartitioner
from lib_unlearn.gif import GIF_Unlearn, GA_Unlearn


''' 1. Split the original dataset into shadow and attack datasets with random/clustering. 
       The shadow dataset and the attack dataset should be disjoint. 
    2. Train the shadow & attack model on the shadow dataset.
    3. Train the target model on the attack dataset.
    4. Transfer the attack model toattack the target model
'''
class ExpUnlearningInversion(Exp):
    def __init__(self, args):
        super(ExpUnlearningInversion, self).__init__(args)

        self.logger = logging.getLogger('ExpUnlearningInversion')
        
        self.load_data()
        self.num_feats = self.data.num_features

        ''' Part 1: Dataset splitting: shadow & attack (attack = train + test)
            Related Arguments:
                --partition_method --random_part_ratio --metis_parts --metis_shadow_parts --is_split --test_ratio
        '''

        self.shadow_attack_split()
        self.train_test_split()

        res_g1, res_g2, res_all = {}, {}, {}
        assert self.args['num_runs'] > 0, 'num_runs should be greater than 0'
        for run_id in range(self.args['num_runs']):
            ''' Part 2: Generate unlearning request for both shadow and attack dataset
            Related Arguments:
                --unlearn_task --unlearn_ratio
            '''

            self.unlearning_request(self.shadow_data, 'shadow', run_id)
            self.unlearning_request(self.attack_data, 'attack', run_id)

            ''' Part 3: Train shadow & attack victim models
            '''

            if self.args['is_gen_unlearned_probs']:
                self.target_model_name = self.args['target_model']
                self.determine_target_model()

                sha_time, _, sha_utime, _, sha_new_params = self.train_and_unlearn(0, self.shadow_data, 
                                                                                self.shadow_target_model, 
                                                                                evaluate_F1=False)
                if self.args['attack_method'] in ['steal_link', 'trend_steal']:
                    self.logger.info('Training shadow reference model for StealLink attack')
                    ref_time, _, ref_utime, _, ref_new_params = self.train_and_unlearn(0, self.shadow_data,
                                                                                    self.shadow_reference_model,
                                                                                    evaluate_F1=False)
                atk_time, orig_f1, atk_utime, u_f1, atk_new_params = self.train_and_unlearn(0, self.attack_data, 
                                                                                            self.attack_target_model, 
                                                                                            evaluate_F1=True)
                
                # The model parameters are equal to the unlearned parameters. To show this, execute the following code:
                #for idx, p in enumerate(self.attack_target_model.model.parameters()):
                #    input((p - atk_new_params[idx]).norm(2))

                self.logger.info("shadow victim model: training %s seconds, unlearning %s seconds" % (sha_time, sha_utime))
                if self.args['attack_method'] in ['steal_link', 'trend_steal']:
                    self.logger.info("shadow reference model: training %s seconds, unlearning %s seconds" % (ref_time, ref_utime))
                self.logger.info("attack victim model: training %s seconds, unlearning %s seconds" % (atk_time, atk_utime))
                self.logger.info("attack victim model: original F1 %s, unlearned F1 %s" % (orig_f1, u_f1))

                ''' Part 4: Create training (shadow dataset) and testing data (attack dataset) for the attack model
                '''

                self.shadow_data.unlearn_prob = self.shadow_target_model.generate_unlearn_probs(sha_new_params)
                if self.args['attack_method'] in ['steal_link', 'trend_steal']:
                    self.shadow_data.reference_prob = self.shadow_reference_model.generate_unlearn_probs(ref_new_params)
                    self.shadow_reference_model.data = self.attack_data
                    self.attack_data.reference_prob = self.shadow_reference_model.generate_unlearn_probs(ref_new_params)
                self.attack_data.unlearn_prob = self.attack_target_model.generate_unlearn_probs(atk_new_params)

                # Save the unlearned probabilities
                self.data_store.save_unlearn_prob(self.shadow_data, 'shadow', run_id)
                self.data_store.save_unlearn_prob(self.attack_data, 'attack', run_id)
            else:
                self.data_store.load_unlearn_prob(self.shadow_data, 'shadow', run_id)
                self.data_store.load_unlearn_prob(self.attack_data, 'attack', run_id)
                if self.args['attack_method'] in ['steal_link', 'trend_steal']:
                    # Please set is_gen_unlearned_probs to True to generate the reference probabilities
                    assert hasattr(self.shadow_data, 'reference_prob'), 'shadow data should have reference_prob'
            
            ''' Part 5: Attack model training and evaluation
            '''

            self.attack_model = LinkStealer(self.args, self.shadow_data, self.attack_data)
            if self.args['export_data']:
                self.export_data(run_id)
            self.attack_model.shadow_train()
            res_g1_run, res_g2_run, res_all_run = self.attack_model.attack_evaluate()

            if len(res_g1) == 0:
                for key in res_g1_run.keys():
                    res_g1[key] = np.array([res_g1_run[key]])
                    res_g2[key] = np.array([res_g2_run[key]])
                    res_all[key] = np.array([res_all_run[key]])
            else:
                for key in res_g1_run.keys():
                    res_g1[key] = np.append(res_g1[key], [res_g1_run[key]])
                    res_g2[key] = np.append(res_g2[key], [res_g2_run[key]])
                    res_all[key] = np.append(res_all[key], [res_all_run[key]])
            
        for key in res_g1.keys():
            res_g1[key] = f'{np.mean(res_g1[key]):.4f} ± {np.std(res_g1[key]):.4f}'
            res_g2[key] = f'{np.mean(res_g2[key]):.4f} ± {np.std(res_g2[key]):.4f}'
            res_all[key] = f'{np.mean(res_all[key]):.4f} ± {np.std(res_all[key]):.4f}'
        
        self.logger.info('Attack model evaluation:')
        self.logger.info(
            '  Group1 (unlearned) → AUC: {auc}, Prec: {precision}, Rec: {recall}, F1: {f1}'
            .format(**res_g1)
        )
        self.logger.info(
            '  Group2 (normal)    → AUC: {auc}, Prec: {precision}, Rec: {recall}, F1: {f1}'
            .format(**res_g2)
        )
        self.logger.info(
            '  Combined           → AUC: {auc}, Prec: {precision}, Rec: {recall}, F1: {f1}'
            .format(**res_all)
        )

    def load_data(self):
        self.data = self.data_store.load_raw_data()
    
    def determine_target_model(self):
        self.logger.info('target model: %s' % (self.args['target_model'],))
        num_classes = len(self.data.y.unique())

        self.shadow_target_model = NodeClassifier(self.num_feats, num_classes, self.args)
        self.attack_target_model = NodeClassifier(self.num_feats, num_classes, self.args)

        if self.args['attack_method'] in ['steal_link', 'trend_steal']:
            self.shadow_reference_model = NodeClassifier(self.num_feats, num_classes, self.args)
            self.shadow_reference_model.target_model = 'MLP'
            self.shadow_reference_model.model = self.shadow_reference_model.determine_model(
                                                    self.shadow_reference_model.num_feats, 
                                                    self.shadow_reference_model.num_classes
                                                )
            

    def shadow_attack_split(self):
        self.partitioner = DatasetPartitioner(self.data, self.args)

        if self.args['is_split']:
            self.shadow_data, self.attack_data, shadow_nodes, attack_nodes = self.partitioner.split()
            self.data_store.save_shadow_attack_split(shadow_nodes, attack_nodes)
        else:
            shadow_nodes, attack_nodes = self.data_store.load_shadow_attack_split()
            self.shadow_data, self.attack_data = self.partitioner._create_subgraphs(shadow_nodes, attack_nodes)
        
        self.shadow_data.num_nodes, self.shadow_data.num_edges = self.shadow_data.x.shape[0], self.shadow_data.edge_index.shape[1]
        self.attack_data.num_nodes, self.attack_data.num_edges = self.attack_data.x.shape[0], self.attack_data.edge_index.shape[1]
        self.shadow_data.train_mask = torch.ones(self.shadow_data.x.shape[0], dtype=torch.bool)
        self.shadow_data.test_mask = torch.zeros(self.shadow_data.x.shape[0], dtype=torch.bool)

    def train_test_split(self):
        # Only splits the attack dataset
        if self.args['is_split']:
            self.logger.info('splitting train/test data')
            # use the dataset's default split
            if self.data.name in ['ogbn-arxiv', 'ogbn-products']:
                # TODO: deal with predefined train/test splits
                pass
            else:
                self.train_indices, self.test_indices = train_test_split(np.arange((self.attack_data.num_nodes)), 
                                                                         test_size=self.args['test_ratio'], 
                                                                         random_state=100)
                
            self.data_store.save_train_test_split(self.train_indices, self.test_indices)
    
            self.attack_data.train_mask = torch.from_numpy(np.isin(np.arange(self.attack_data.num_nodes), self.train_indices))
            self.attack_data.test_mask = torch.from_numpy(np.isin(np.arange(self.attack_data.num_nodes), self.test_indices))
        else:
            self.train_indices, self.test_indices = self.data_store.load_train_test_split()

            self.attack_data.train_mask = torch.from_numpy(np.isin(np.arange(self.attack_data.num_nodes), self.train_indices))
            self.attack_data.test_mask = torch.from_numpy(np.isin(np.arange(self.attack_data.num_nodes), self.test_indices))

    def unlearning_request(self, data, save_name, run_id=0):
        self.logger.debug("Data used for unlearning request generation #.Nodes: %f, #.Edges: %f" % (
            data.num_nodes, data.num_edges))

        data.x_unlearn = data.x.clone()
        data.edge_index_unlearn = data.edge_index.clone()
        data.deleted_nodes = np.array([])     
        data.feature_nodes = np.array([])

        # Three stuffs to save/load: edge_index_unlearn, removed_edges_und, influence_nodes
        if self.args["is_gen_unlearn_request"]:
            edge_index = data.edge_index.cpu().numpy()
            unique_indices = np.where(edge_index[0] < edge_index[1])[0]

            if self.args["unlearn_task"] == 'edge':
                remove_indices = np.random.choice(
                    unique_indices,
                    int(unique_indices.shape[0] * self.args['unlearn_ratio']),
                    replace=False)
                remove_edges = edge_index[:, remove_indices]
                unique_nodes = np.unique(remove_edges)
            
                data.edge_index_unlearn = self.update_edge_index_unlearn(data, unique_nodes, remove_indices)
                #data.removed_edges = remove_edges
                data.removed_edges_und = to_undirected(torch.from_numpy(remove_edges))
                
            elif self.args["unlearn_task"] == 'node' or self.args["unlearn_task"] == 'feature':
                pass
                #TODO: Node/Feature unlearning, to be implemented for future works

            self.find_k_hops(data, unique_nodes) # Added data.influence_nodes

            # Save unlearn request
            self.data_store.save_unlearn_request(data, save_name, run_id)
        else:
            data = self.data_store.load_unlearn_request(data, save_name, run_id)
            
    def update_edge_index_unlearn(self, data, delete_nodes, delete_edge_index=None):
        edge_index = data.edge_index.cpu().numpy()

        unique_indices = np.where(edge_index[0] < edge_index[1])[0]
        unique_indices_not = np.where(edge_index[0] > edge_index[1])[0]

        if self.args["unlearn_task"] == 'edge':
            remain_indices = np.setdiff1d(unique_indices, delete_edge_index)
        else:
            unique_edge_index = edge_index[:, unique_indices]
            delete_edge_indices = np.logical_or(np.isin(unique_edge_index[0], delete_nodes),
                                                np.isin(unique_edge_index[1], delete_nodes))
            remain_indices = np.logical_not(delete_edge_indices)
            remain_indices = np.where(remain_indices == True)

        remain_encode = edge_index[0, remain_indices] * edge_index.shape[1] * 2 + edge_index[1, remain_indices]
        unique_encode_not = edge_index[1, unique_indices_not] * edge_index.shape[1] * 2 + edge_index[0, unique_indices_not]
        sort_indices = np.argsort(unique_encode_not)
        remain_indices_not = unique_indices_not[sort_indices[np.searchsorted(unique_encode_not, remain_encode, sorter=sort_indices)]]
        remain_indices = np.union1d(remain_indices, remain_indices_not)

        return torch.from_numpy(edge_index[:, remain_indices])
    
    def find_k_hops(self, data, unique_nodes, hops=2):
        edge_index = data.cpu().edge_index.numpy()
        
        ## finding influenced neighbors
        if self.args["unlearn_task"] == 'node':
            hops += 1
        influenced_nodes = unique_nodes
        for _ in range(hops):
            target_nodes_location = np.isin(edge_index[0], influenced_nodes)
            neighbor_nodes = edge_index[1, target_nodes_location]
            influenced_nodes = np.append(influenced_nodes, neighbor_nodes)
            influenced_nodes = np.unique(influenced_nodes)
        neighbor_nodes = np.setdiff1d(influenced_nodes, unique_nodes)
        if self.args["unlearn_task"] == 'feature':
            pass
            #TODO: implement feature unlearning
        if self.args["unlearn_task"] == 'node':
            pass
            #TODO: implement node unlearning
        if self.args["unlearn_task"] == 'edge':
            data.influence_nodes = influenced_nodes
            #data.deleted_nodes = np.array([])     
            #data.feature_nodes = np.array([])

    def train_and_unlearn(self, run, data, model, evaluate_F1=False):
        # For CEU, we need to add noise to the loss function here!
        run_training_time, result_tuple = self._train_model(run, model, data)

        #old_params = [p.detach().clone() for p in model.model.parameters() if p.requires_grad]
        if evaluate_F1:
            f1_score = self.evaluate(run)
        else:
            f1_score = -1
            
        # unlearning with GIF
        if self.args['method'] == 'GIF':
            unlearn_method = GIF_Unlearn(model, self.args)
            unlearning_time, f1_score_unlearning, new_params = unlearn_method.gif_approxi(result_tuple, evaluate_F1)
            #Check parameter change
            #for p1, p2 in zip(old_params, new_params):
            #    input((p1 - p2).norm(2))
        elif self.args['method'] == 'CEU':
            unlearn_method = GIF_Unlearn(model, self.args)
            unlearning_time, f1_score_unlearning, new_params = unlearn_method.gif_approxi(result_tuple, evaluate_F1)
        elif self.args['method'] == 'GA':
            unlearn_method = GA_Unlearn(model, self.args)
            unlearning_time, f1_score_unlearning, new_params = unlearn_method.unlearn()
        
        return run_training_time, f1_score, unlearning_time, f1_score_unlearning, new_params
 
    def _train_model(self, run, model, data):
        self.logger.info('training target models, run %s' % run)

        start_time = time.time()
        model.data = data
        res = model.train_model(
            (data.deleted_nodes, data.feature_nodes, data.influence_nodes))
        train_time = time.time() - start_time

        self.logger.info("Model training time: %s" % (train_time))

        return train_time, res
    
    def evaluate(self, run):
        self.logger.info('model evaluation')

        start_time = time.time()
        posterior = self.attack_target_model.posterior()
        test_f1 = f1_score(
            self.attack_data.y[self.attack_data['test_mask']].cpu().numpy(), 
            posterior.argmax(axis=1).cpu().numpy(), 
            average="micro"
        )

        evaluate_time = time.time() - start_time
        self.logger.info("Evaluation cost %s seconds." % evaluate_time)

        self.logger.info("Final Test F1: %s" % (test_f1,))
        return test_f1

    def export_data(self, run_id=0):
        self.logger.info('exporting data for external LP baselines')
        file_name = '_'.join([self.args['dataset_name'], 
                                      str(self.args['unlearn_ratio']),
                                      str(run_id)]) + '.pth'
        
        train_data = self.attack_model.train_data.detach().clone()
        test_data = self.attack_model.test_data.detach().clone()
        
        train_data.x = self.shadow_data.x
        test_data.x = self.attack_data.x

        train_data.train_pos_edges = self.attack_model.train_pos_edges
        train_data.train_neg_edges = self.attack_model.train_neg_edges
        test_data.test_pos_edges = self.attack_model.test_pos_edges
        test_data.test_neg_edges = self.attack_model.test_neg_edges

        save_dict = {'train': train_data,
                     'test': test_data}
        with open(file_name, 'wb') as f:
            torch.save(save_dict, f)