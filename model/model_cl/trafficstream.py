import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import k_hop_subgraph
from torch import autograd

from model import Model_cl
from model.model_cl.base_model import Backbone
from model.model_cl.ewc import EWC

class Model(Model_cl):
    def __init__(self, config, data):
        super().__init__(config)
        config.input_dim = self.dim_in
        config.output_dim = self.dim_out
        self.model = TrafficStream_Model(config)
        self.ewc = None
        self.data = data
        g = torch.tensor(self.data.load_graph(config.start_year), dtype=self.dtype, device=self.device)
        self.graph = g / (g.sum(1, keepdim=True) + 1e-6)
        self.cur_graph = self.graph
        self.loss = F.mse_loss

    def forward(self, *data):
        x = self.norm.norm(data[0])
        output = self.model(x, data[-1])
        output = self.norm.inorm(output)
        return output

    def calculate_loss(self, data, graph=None, ewc=True):
        if not hasattr(self, 'loss'):
            self.loss = self.masked_mse
        x, y = self.get_data(data, graph)
        output = self.forward(*x)
        loss = self.loss(output, y)
        if ewc and self.year > self.config.start_year:
            loss += self.ewc.compute_consolidation_loss()
        return loss

    def predict(self, data):
        x, y = self.get_data(data)
        output = self.forward(x[0], self.cur_graph)
        return output, y

    def get_data(self, data, graph=None):
        x, y, *_ = data
        x, y = x.cuda().to(torch.float32), y.cuda().to(torch.float32)
        g = graph if graph is not None else self.graph.to(x.dtype)
        return (x, g), y

    def prepare_for_new_task(self, **kwargs):
        node_list = list()

        # new nodes
        if self.config.increase:
            cur_node_size = self.config.num_nodes_year[self.year]
            pre_node_size = self.config.num_nodes_year[self.year - 1]
            node_list.extend(list(range(pre_node_size, cur_node_size)))

        pre_data = np.load(self.config.data_path + self.data.file_name + f'data_{self.year - 1}.npy')[-7*288:]
        cur_data = np.load(self.config.data_path + self.data.file_name + f'data_{self.year}.npy')[-7*288:]
        self.pre_graph = self.cur_graph
        cur_graph = torch.tensor(self.data.load_graph(self.year)).cuda().to(torch.float32)
        self.cur_graph = cur_graph / (cur_graph.sum(1, keepdim=True) + 1e-6)

        # Sample most 5% of the current graph node size
        topk = int(0.05 * self.config.num_nodes_year[self.year])

        # influence nodes
        if self.config.detect or self.config.replay:
            influence_node_score = self.influence_node_selection(pre_data, cur_data)

        if self.config.detect:  # Get the affected nodes
            influence_node_list = np.argpartition(np.asarray(influence_node_score), -topk)[-topk:]
            node_list.extend(list(influence_node_list))

        if self.config.replay:  # Get sampling node
            replay_node_list = self.replay_node_selection(influence_node_score, topk)  # Select the replay node
            node_list.extend(list(replay_node_list))

        node_list = list(set(node_list))
        # Get a subgraph of a node list
        # edge_list = cur_graph.nonzero()
        #
        # graph_node_from_edge = set()  # Collect all nodes connected by edges
        # for (u, v) in edge_list:
        #     graph_node_from_edge.add(u)
        #     graph_node_from_edge.add(v)
        #
        # node_list = list(set(node_list) & graph_node_from_edge)  # Get the list of nodes in the subgraph, that is, the intersection of the nodes to be modified and the existing edge nodes

        if len(node_list) != 0:
            subgraph, subgraph_edge_index, mapping, _ = k_hop_subgraph(node_list, num_hops=self.config.num_hops, edge_index=cur_graph.nonzero().T, relabel_nodes=True)
            sg = torch.sparse_coo_tensor(subgraph_edge_index, torch.ones_like(subgraph_edge_index[0])).to_dense().to(torch.float32)
            self.graph = sg / (sg.sum(1, keepdims=True) + 1e-6)
            self.subset = subgraph.cpu()

        if not isinstance(self.ewc, EWC):
            self.ewc = EWC(self.model, self.config.ewc_lambda, self.config.ewc_type)
        train_dl, *_ = self.data.get_dataloader(self.norm, self.year)
        self.ewc.register_ewc_params(train_dl, self.loss, self.cur_graph, self.device)

    def influence_node_selection(self, pre_data, cur_data):
        if self.config.detect_strategy == 'original':  # Check the detection strategy specified in the parameters
            # pre_data = pre_data[-288 * 7 - 1:-1, :]  # Select the last week (7 days) of data for both datasets
            # cur_data = cur_data[-288 * 7 - 1:-1, :]
            node_size = pre_data.shape[1]  # Determine the number of nodes in the data
            score = []
            for node in range(node_size):  # Iterate over each node to calculate its KL divergence score
                max_val = max(max(pre_data[:, node]),
                              max(cur_data[:,
                                  node]))  # Find the maximum and minimum values ​​of a node in two time periods
                min_val = min(min(pre_data[:, node]), min(cur_data[:, node]))
                pre_prob, _ = np.histogram(pre_data[:, node], bins=10, range=(min_val, max_val))  # Create a histogram of the data for the node in two time periods, with 10 bins in total, and normalize the histogram to get the probability distribution
                pre_prob = pre_prob * 1.0 / sum(pre_prob)
                cur_prob, _ = np.histogram(cur_data[:, node], bins=10, range=(min_val, max_val))
                cur_prob = cur_prob * 1.0 / sum(cur_prob)
                from scipy.stats import entropy as kldiv
                score.append(kldiv(pre_prob, cur_prob))  # Compute the KL divergence between the two distributions and add it to the list of scores
            return score
        elif self.config.detect_strategy == 'feature':
            self.model.eval()  # Set the model to evaluation mode

            pre_data = self.get_feature(pre_data, self.pre_graph)  # Use the model to extract features from previous and current data, the feature dimension is [num_nodes, batch_size, feature_dim]
            cur_data = self.get_feature(cur_data, self.cur_graph)
            pre_data = (pre_data - pre_data.min(1, keepdims=True)) / (pre_data.max(1, keepdims=True) - pre_data.min(1, keepdims=True))
            cur_data = (cur_data - cur_data.min(1, keepdims=True)) / (cur_data.max(1, keepdims=True) - cur_data.min(1, keepdims=True))
            score = []
            for i in range(pre_data.shape[0]):  # Traverse the nodes in the feature data
                score_ = 0.0
                for j in range(pre_data.shape[2]):  # Traverse each feature dimension
                    if np.isnan(pre_data[i, :, j]).any():
                        continue
                    # pre_data[i, :, j] = (pre_data[i, :, j] - min(pre_data[i, :, j])) / (max(pre_data[i, :, j]) - min(pre_data[i, :, j]))  # Normalize the eigenvalues ​​to the range [0, 1]
                    # cur_data[i, :, j] = (cur_data[i, :, j] - min(cur_data[i, :, j])) / (max(cur_data[i, :, j]) - min(cur_data[i, :, j]))

                    pre_prob, _ = np.histogram(pre_data[i, :, j], bins=10, range=(0, 1))  # Create a histogram of the distribution of the feature over two time periods, with 10 bins, and normalize the histogram to get a probability distribution
                    pre_prob = pre_prob * 1.0 / sum(pre_prob)
                    cur_prob, _ = np.histogram(cur_data[i, :, j], bins=10, range=(0, 1))
                    cur_prob = cur_prob * 1.0 / sum(cur_prob)
                    from scipy.spatial.distance import jensenshannon
                    score__ = jensenshannon(pre_prob, cur_prob)
                    if np.isnan(score__):
                        continue
                    score_ += score__  # Calculate the Jensen-Shannon distance between the two distributions and add to the score
                score.append(score_)  # Add the total score of the node to the score list
            return score  # Returns the indices of the top-k nodes with the highest scores

    def get_feature(self, data, graph):
        data = torch.tensor(data, dtype=torch.float32).cuda()
        if data.ndim == 2:
            data = data.unsqueeze(-1)
        data = self.norm.norm(data) # L, N, D
        dl = torch.utils.data.DataLoader(data.unfold(0, self.config.input_window, 1).permute(0, 3, 1, 2), batch_size=self.config.batch_size)
        feature = []
        with torch.no_grad():
            self.model.eval()
            for batch in dl:
                feature.append(self.model.feature(batch, graph.to(data.dtype)))
        feature = torch.cat(feature, 0) # B, N, D
        if feature.ndim == 4:
            feature = feature.mean(1)
        return feature.transpose(0, 1).cpu().detach().numpy()

    def replay_node_selection(self, influence_node_score, topk):
        if self.config.replay_strategy == 'random':
            return np.random.choice(len(influence_node_score), topk)
        elif self.config.replay_strategy == 'inforeplay':
            return np.argpartition(np.asarray(influence_node_score), topk)[:topk]

class TrafficStream_Model(nn.Module):
    """Some Information about TrafficStream_Model"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = config.backbone
        self.model = Backbone(self.backbone, config)

    def forward(self, x, adj):
        # x = x.transpose(1, 2).flatten(-2)  # B, N, T
        x = self.model(x, adj)
        # x = x.unflatten(-1, (-1, self.config.dim_out)).transpose(1, 2)
        return x

    def feature(self, x, adj):
        x = self.model(x, adj, 'em')
        return x