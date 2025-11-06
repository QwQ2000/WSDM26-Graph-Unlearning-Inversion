import torch
import torch.nn.functional as F
from torch.nn import Linear


class MLPNet(torch.nn.Module):
    def __init__(self, num_feats, num_classes, num_layers=2, dropout=0.5, for_link_pred=False):
        super(MLPNet, self).__init__()
        self.dropout = dropout
        self.for_link_pred = for_link_pred

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(Linear(num_feats, 16))
        for _ in range(num_layers - 2):
            self.convs.append(Linear(16, 16))
        if for_link_pred == False:
            self.convs.append(Linear(16, num_classes))
        else:
            self.convs.append(Linear(16, 16))

    def forward(self, x, adjs):
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i](x_target)#((x, x_target), edge_index)

            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.for_link_pred == False:
            return F.log_softmax(x, dim=1)
        else:
            return x

    def forward_once(self, data):
        x = data.x 
        x = self.convs[0](x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[1](x) 

        if self.for_link_pred == False:
            return F.log_softmax(x, dim=1)
        else:
            return x

    def forward_once_unlearn(self, data):
        x = data.x 
        x = self.convs[0](x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[1](x) 
        
        if self.for_link_pred == False:
            return F.log_softmax(x, dim=1)
        else:
            return x

    def inference(self, x_all, subgraph_loader, device):
        for i in range(self.num_layers):
            xs = []

            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)

                x_target = x[:size[1]]
                x = self.convs[i](x_target)

                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all

    def reset_parameters(self):
        for i in range(self.num_layers):
            self.convs[i].reset_parameters()
