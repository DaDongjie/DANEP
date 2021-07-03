import numpy as np
import linecache
import networkx as nx
import math
from Utils.utils import *
from sklearn import metrics
from scipy.sparse import *
import tensorflow as tf
from sklearn.preprocessing import normalize
# from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore

class Dataset(object):

    def __init__(self, config):
        # self.graph_file_1 = config['graph_file_1']
        self.graph_file = config['graph_file']
        self.feature_file = config['feature_file']
        self.label_file = config['label_file']

        self.M, self.Z, self.Y = self._load_data()  # W denotes network graph structure Z represents attribute graph structure

        self.X_att, self.att_adj = self.read_att_graph() # self.X_att PPMI矩阵， self.att_adj 属性相似性矩阵

        self.X_net, self.net_adj = self.read_net_graph() # self.X_net PPMI矩阵， self.net_adj 网络的结构

        self.X = np.hstack((self.X_net, self.X_att))
        self.X = zscore(self.X, axis=1) # 标准化
        self.X_num = self.X.shape[1]
        self.num_nodes = self.M.shape[0]
        self.num_feas = self.Z.shape[1]
        self.num_classes = self.Y.shape[1]
        self.num_edges = np.sum(self.M) / 2
        print('nodes {}, edes {}, features {}, classes {}'.format(self.num_nodes, self.num_edges, self.num_feas,
                                                                  self.num_classes))

        self._order = np.arange(self.num_nodes)
        self._index_in_epoch = 0
        self.is_epoch_end = False

    def read_att_graph(self):
        attr = csr_matrix(self.Z).toarray()
        print("calculate cosine similarity..")
        flag = 0
        if (flag == 1):
            X_att_1 = metrics.pairwise.euclidean_distances(attr, attr)
        else:
            X_att_1 = metrics.pairwise.cosine_similarity(attr, attr)
        start = 0
        # num_node = (1/10) * self.M.shape[0]
        num_node = math.sqrt(self.M.shape[0])
        n = math.ceil(num_node)
        for i in X_att_1:
            AA = i
            # print("111:", AA)
            # print("111:", len(AA))
            kth = sorted(AA)[-int(n)]
            # print("1112:", kth)
            AA[AA < kth] = 0
            # print("113:", AA)
            X_att_1[start] = AA
            start = start + 1

        # 创建属性图
        self.G_att = nx.from_numpy_matrix(X_att_1)
        for edge in self.G_att.edges():
            self.G_att[edge[0]][edge[1]]['weight'] = 1
        A = np.asarray(nx.adjacency_matrix(self.G_att, nodelist=None, weight='None').todense())
        print(A.shape)

        self.G_att = self.G_att.to_undirected()
        X_att = nx.to_numpy_array(self.G_att, range(0, self.G_att.number_of_nodes()))
        X_att = self.random_surf(X_att, 5, 0.98)
        X_att = self.ppmi_matrix(X_att)
        print(X_att.shape)
        print("2:", X_att_1.shape)

        return  X_att, X_att_1

    def read_net_graph(self,weighted=False, directed=False):

        lines = linecache.getlines(self.label_file)
        lines = [line.rstrip('\n') for line in lines]

        # ===========load label============
        node_map = {}
        label_map = {}
        Y = []
        cnt = 0
        for idx, line in enumerate(lines):
            line = line.split(' ')
            node_map[line[0]] = idx
            y = []
            for label in line[1:]:
                if label not in label_map:
                    label_map[label] = cnt
                    cnt += 1
                y.append(label_map[label])
            Y.append(y)
        num_classes = len(label_map)
        num_nodes = len(node_map)

        L = np.zeros((num_nodes, num_classes), dtype=np.int32)
        for idx, y in enumerate(Y):
            L[idx][y] = 1

        print('loading graph...')
        # if weighted:
        #     G_1 = nx.read_edgelist(self.graph_file_1, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
        # else:
        #     G_1 = nx.read_edgelist(self.graph_file_1, nodetype=int, create_using=nx.DiGraph())
        #     for edge in G_1.edges():
        #         G_1[edge[0]][edge[1]]['weight'] = 1
        #
        # if not directed:
        #     G_1 = G_1.to_undirected()
        # print("nodes_num_1:",G_1.number_of_nodes())

        if weighted:
            G = nx.read_edgelist(self.graph_file, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
        else:
            G = nx.read_edgelist(self.graph_file, nodetype=int, create_using=nx.DiGraph())
            for edge in G.edges():
                G[edge[0]][edge[1]]['weight'] = 1

        if not directed:
            G = G.to_undirected()
        print("nodes_num:",G.number_of_nodes())

        W = np.zeros((G.number_of_nodes(), G.number_of_nodes()))
        lines = linecache.getlines(self.graph_file)
        lines = [line.rstrip('\n') for line in lines]
        for line in lines:
            line = line.split(' ')
            idx1 = node_map[line[0]]
            idx2 = node_map[line[1]]
            W[idx2, idx1] = 1.0
            W[idx1, idx2] = 1.0

        X_net_1 = W
        # X_net_1 = nx.to_numpy_array(G, range(0, G_1.number_of_nodes()))
        print("X_net_1:", X_net_1.shape)
        # get co-occurrence matrix by random surfing and calculate the PPMI of it
        X_net = self.random_surf(X_net_1, 5, 0.98)
        X_net = self.ppmi_matrix(X_net)
        print(X_net.shape)
        return X_net, X_net_1

    # Normalize the matrix by row so that each row adds up to 1
    def scale_sim_mat(self, mat):
        mat = mat - np.diag(np.diag(mat))  # Diagonal needs to be 0
        for i in range(len(mat)):
            if mat[i].sum() > 0:
                mat[i] = mat[i] / mat[i].sum()
            else:  # Handling those points that only connect to themselves
                mat[i] = 1 / (len(mat) - 1)
                mat[i, i] = 0
        return mat

    # Get the co-occurrence matrix by accumulating the state matrix after every hop, each random walk has a restart probability 1-alpha.
    def random_surf(self, mat, num_hops, alpha):
        num_nodes = len(mat)
        adj_matrix = self.scale_sim_mat(mat)
        p0 = np.eye(num_nodes, dtype='float32')
        p = p0
        a = np.zeros((num_nodes, num_nodes), dtype='float32')
        for i in range(num_hops):
            p = (alpha * np.dot(p, adj_matrix)) + ((1 - alpha) * p0)
            a = a + p
        return a

    # Compute the improved PPMI matrix of the co-occurrence matrix
    def ppmi_matrix(self, mat):
        num_nodes = len(mat)
        mat = self.scale_sim_mat(mat)
        col_sum = np.sum(mat, axis=0).reshape(1, num_nodes)
        col_sum = np.power(col_sum, 0.75)  # smoothing, reduce the effect that low frequency makes high PPMI
        # row_sum all become 1 after scaling, so we don't need to divide it anymore,
        # and multiply num_nodes to make sure the PPMI values are not too small to lose precision
        ppmi = np.log(np.divide(num_nodes * mat, col_sum))
        ppmi[np.isinf(ppmi)] = 0.0
        ppmi[np.isneginf(ppmi)] = 0.0
        ppmi[ppmi < 0.0] = 0.0
        return ppmi

    def _load_data(self):
        lines = linecache.getlines(self.label_file)
        lines = [line.rstrip('\n') for line in lines]

        #===========load label============
        node_map = {}
        label_map = {}
        Y = []
        cnt = 0
        for idx, line in enumerate(lines):
            line = line.split(' ')
            node_map[line[0]] = idx
            y = []
            for label in line[1:]:
                if label not in label_map:
                    label_map[label] = cnt
                    cnt += 1
                y.append(label_map[label])
            Y.append(y)
        num_classes = len(label_map)
        num_nodes = len(node_map)

        L = np.zeros((num_nodes, num_classes), dtype=np.int32)
        for idx, y in enumerate(Y):
            L[idx][y] = 1

        #=========load feature==========
        lines = linecache.getlines(self.feature_file)
        lines = [line.rstrip('\n') for line in lines]

        num_features = len(lines[0].split(' ')) - 1
        Z = np.zeros((num_nodes, num_features), dtype=np.float32)

        for line in lines:
            line = line.split(' ')
            node_id = node_map[line[0]]
            Z[node_id] = np.array([float(x) for x in line[1:]])

        #==========load graph========
        W = np.zeros((num_nodes, num_nodes))
        lines = linecache.getlines(self.graph_file)
        lines = [line.rstrip('\n') for line in lines]
        for line in lines:
            line = line.split(' ')
            idx1 = node_map[line[0]]
            idx2 = node_map[line[1]]
            W[idx2, idx1] = 1.0
            W[idx1, idx2] = 1.0


        return W, Z, L


    def sample(self, batch_size, do_shuffle=True, with_label=True):
        if self.is_epoch_end:
            if do_shuffle:
                np.random.shuffle(self._order)
            else:
                self._order = np.sort(self._order)
            self.is_epoch_end = False
            self._index_in_epoch = 0

        mini_batch = Dotdict()
        end_index = min(self.num_nodes, self._index_in_epoch + batch_size)
        cur_index = self._order[self._index_in_epoch:end_index]
        mini_batch.X = self.X[cur_index] 
        mini_batch.att_adj = self.att_adj[cur_index][:, cur_index]
        mini_batch.net_adj = self.net_adj[cur_index][:, cur_index]

        if with_label:
            mini_batch.Y = self.Y[cur_index]

        if end_index == self.num_nodes:
            end_index = 0
            self.is_epoch_end = True
        self._index_in_epoch = end_index

        return mini_batch

    def sample_by_idx(self, idx):
        mini_batch = Dotdict()
        mini_batch.X = self.X[idx]
        mini_batch.net_adj = self.net_adj[idx][:, idx]
        mini_batch.att_adj = self.att_adj[idx][:, idx]
        mini_batch.Y = self.Y[idx]

        return mini_batch

