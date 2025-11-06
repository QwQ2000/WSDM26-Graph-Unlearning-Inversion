import torch
import torch.nn.functional as F
from lib_utils.distance import *

class EmptyBackbone(torch.nn.Module):
    def forward(self, x, edge_index):
        return x


class SimilarityPredictorV2(torch.nn.Module):
    def __init__(self, x_dim, prob_dim, hidden_dim=64):
        super(SimilarityPredictorV2, self).__init__()
        # The input for each branch is the element-wise similarity, which has dimension equal to the input dimension.
        self.xdim = x_dim
        self.probdim = prob_dim

        self.distance_list = [euclidean_distance, cityblock_distance, sqeuclidean_distance,
                              chebyshev_distance, canberra_distance, braycurtis_distance,
                              correlation_distance, cosine_distance]
        
        self.fc1 = torch.nn.Linear(len(self.distance_list) * 3 + 8, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
        self.dropout = 0.5

    def __compute_prob_metric(self, p_i, p_j, metric_type='entropy'):
        if metric_type == 'entropy':
            a = entropy(p_i)
            b = entropy(p_j)
        elif metric_type == 'kl_divergence':
            a = kl_divergence(p_i, p_j)
            b = kl_divergence(p_j, p_i)
        elif metric_type == 'js_divergence':
            a = js_distance(p_i, p_j)
            b = js_distance(p_j, p_i)
        else:
            raise ValueError("Unknown metric type: {}".format(metric_type))

        avg = (a + b) / 2
        had = a * b
        l1 = torch.abs(a - b)
        l2_sq = (a - b) * (a - b) 

        return torch.cat([avg.unsqueeze(1), had.unsqueeze(1), l1.unsqueeze(1), l2_sq.unsqueeze(1)], dim=1)

    def forward(self, x_i, x_j, return_logits=False):
        # Split each input into three parts: feature, GNN prob, MLP prob.
        # Assume x_i and x_j shape: (batch_size, xdim+probdim+probdim)
        f_i = x_i[:, :self.xdim]
        p1_i, p2_i = x_i[:, self.xdim:self.xdim + self.probdim], x_i[:, self.xdim + self.probdim:]
        f_j = x_j[:, :self.xdim]
        p1_j, p2_j = x_j[:, self.xdim:self.xdim + self.probdim], x_j[:, self.xdim + self.probdim:]

        # Compute similarity.
        sim_f = torch.cat([distance(f_i, f_j).unsqueeze(1) for distance in self.distance_list], dim=1)

        sim_p1 = torch.cat([distance(p1_i, p1_j).unsqueeze(1) for distance in self.distance_list], dim=1)
        sim_p2 = torch.cat([distance(p2_i, p2_j).unsqueeze(1) for distance in self.distance_list], dim=1)
        
        metric_p1 = self.__compute_prob_metric(p1_i, p1_j)
        metric_p2 = self.__compute_prob_metric(p2_i, p2_j)
        
        sim_concat = torch.cat([sim_f, sim_p1, sim_p2, metric_p1, metric_p2], dim=1)

        # Pass through 2-layer MLP.
        out = self.fc1(sim_concat)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.fc2(out)

        if return_logits:
            return out
        else:
            return torch.sigmoid(out)
    

class TrendMIAPredictor(torch.nn.Module):
    def __init__(self, prob_dim, trend_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(TrendMIAPredictor, self).__init__()
        
        self.wp1 = torch.nn.Linear(prob_dim, hidden_dim1)
        self.wp2 = torch.nn.Linear(hidden_dim1, hidden_dim1)
        
        self.wt1 = torch.nn.Linear(trend_feat_dim, hidden_dim1)
        self.wt2 = torch.nn.Linear(hidden_dim1, 1)
        
        self.concat_dim = hidden_dim1
        self.w1 = torch.nn.Linear(self.concat_dim, hidden_dim2)
        self.w2 = torch.nn.Linear(hidden_dim2, 1)
        
        self.wt = torch.nn.Linear(trend_feat_dim, 1)
        with torch.no_grad():
            #self.wt.weight.copy_(torch.tensor([[0.9, -0.9, -0.9, 0.9]]))
            self.wt.weight.copy_(torch.tensor([[0.5, -0.5, -0.5, 0.5]]))
        self.dropout = dropout

    def _mlp_compute(self, x, w1, w2):
        x = w1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = w2(x)
        return x
    
    def forward(self, x_i, x_j):
        # Split probability and trend features
        p_i, t_i = x_i[:, :self.wp1.in_features], x_i[:, self.wp1.in_features:]
        p_j, t_j = x_j[:, :self.wp1.in_features], x_j[:, self.wp1.in_features:]

        # x1: Prediction based on probability
        p_i_encode = self._mlp_compute(p_i, self.wp1, self.wp2)
        p_j_encode = self._mlp_compute(p_j, self.wp1, self.wp2)
        p_encode = p_i_encode * p_j_encode

        x1 = self._mlp_compute(p_encode, self.w1, self.w2)
        
        # x2: Refinement based on trend features
        x2 = self.wt(t_i) + self.wt(t_j)

        x = x1 + x2

        return torch.sigmoid(x)


class TrendSimilarityPredictor(torch.nn.Module):
    def __init__(self, x_dim, prob_dim, hidden_dim, trend_feat_dim):
        super(TrendSimilarityPredictor, self).__init__()
        
        self.sim_predictor = SimilarityPredictorV2(x_dim, prob_dim, hidden_dim)
        self.feat_dim = x_dim + prob_dim * 2

        self.wt = torch.nn.Linear(trend_feat_dim, 1)
        with torch.no_grad():
            #self.wt.weight.copy_(torch.tensor([[0.9, -0.9, -0.9, 0.9]]))
            self.wt.weight.copy_(torch.tensor([[0.7, -0.7, -0.7, 0.7]]))
    
    def forward(self, x_i, x_j):
        # Split original features and trend features
        f_i, t_i = x_i[:, :self.feat_dim], x_i[:, self.feat_dim:]
        f_j, t_j = x_j[:, :self.feat_dim], x_j[:, self.feat_dim:]
        
        # Preidction: original score + trend score refinement
        out = self.sim_predictor(f_i, f_j, return_logits=True) + self.wt(t_i) + self.wt(t_j)
        
        return torch.sigmoid(out)


from torch_geometric.nn import SAGEConv

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)
    

class MLPNet(torch.nn.Module):
    def __init__(self, num_feats, hidden_feats=32):
        super(MLPNet, self).__init__()
        #self.dropout = 0.5

        self.num_layers = 2

        self.lin1 = torch.nn.Linear(num_feats, hidden_feats)
        self.lin2 = torch.nn.Linear(hidden_feats, hidden_feats)
        self.dropout = 0.5

    def forward(self, x, _):
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        
        return x