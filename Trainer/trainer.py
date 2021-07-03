import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from Utils.utils import *
from Model.SingleAE import SingleAE
import pickle

w_init = lambda:tf.random_normal_initializer(stddev=0.02)


def get_1st_loss(H, adj_mini_batch):  # adj_mini_batch 2708*2708
    # tf.reduce_sum(adj_mini_batch, 1) 2708*1  返回具有给定对角线值的对角张量。  2708*2708
    D = tf.diag(tf.reduce_sum(adj_mini_batch, 1))
    L = D - adj_mini_batch  ## L is laplation-matriX
    return 2 * tf.trace(tf.matmul(tf.matmul(tf.transpose(H), L), H)) # tf.transpose(H) 转置

class Trainer(object):

    def __init__(self, model, config):  #my_graph,
        self.config = config
        self.model = model
        self.att_input_dim = config['att_input_dim']
        self.att_shape = config['att_shape']
        self.drop_prob = config['drop_prob']
        self.beta = config['beta']
        self.alpha = config['alpha']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.num_epochs = config['num_epochs']
        self.model_path = config['model_path']

        self.x = tf.placeholder(tf.float32, [None, self.att_input_dim])
        self.net_adj = tf.placeholder(tf.float32, [None, None])
        self.att_adj = tf.placeholder(tf.float32, [None, None])
        # self.att_adj = tf.placeholder(tf.float32, [None, None])

        self.optimizer, self.loss = self._build_training_graph()
        self.H = self._build_eval_graph()

        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=gpu_config)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()


    def _build_training_graph(self):
       
        att_H, att_recon = self.model.forward_att(self.x, drop_prob=self.drop_prob, reuse=tf.AUTO_REUSE)  # 学到的表示 重构的表示
        print("000:", att_recon.shape)
        print("001:", att_H.shape)
        print("002:", self.net_adj.shape)
        loss_1st_net = get_1st_loss(att_H, self.net_adj) # 学到的表示 网络邻接矩阵
        loss_1st_att = get_1st_loss(att_H, self.att_adj) # 学到的表示 属性邻接矩阵

        #================high-order proximity & semantic proximity=============
        recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.x - att_recon), 1)) # 重构损失

        PR_loss = tf.reduce_mean(loss_1st_att + loss_1st_net) # 邻近度损失

        #==========================================================

        loss = recon_loss * self.beta + PR_loss * self.alpha #+ local_loss * self.alpha 
        
        vars_att = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'att_encoder')


        opt = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, var_list=vars_att)

        return opt, loss

    def _build_eval_graph(self):
        att_H, _ = self.model.forward_att(self.x, drop_prob=0.0, reuse=True)

        return att_H



    def train(self, graph):

        for epoch in range(self.num_epochs):

            idx1 = self.generate_samples(graph)

            index = 0
            cost = 0.0
            cnt = 0
            while True:
                if index > graph.num_nodes:
                    break
                if index + self.batch_size < graph.num_nodes:
                    mini_batch1 = graph.sample_by_idx(idx1[index:index + self.batch_size])

                else:
                    mini_batch1 = graph.sample_by_idx(idx1[index:])

                index += self.batch_size

                loss, _ = self.sess.run([self.loss, self.optimizer],
                                        feed_dict={self.x: mini_batch1.X,
                                                   self.net_adj : mini_batch1.net_adj,
                                                   self.att_adj : mini_batch1.att_adj,
                                                   })

                cost += loss
                cnt += 1

                if graph.is_epoch_end:
                    break
            cost /= cnt

            if epoch % 50 == 0:

                train_emb = None
                train_label = None
                while True:
                    mini_batch = graph.sample(self.batch_size, do_shuffle=False, with_label=True)

                    emb = self.sess.run(self.H,
                                        feed_dict={self.x: mini_batch.X,

                                                   })
                    if train_emb is None:
                        train_emb = emb
                        train_label = mini_batch.Y
                    else:
                        train_emb = np.vstack((train_emb, emb))
                        train_label = np.vstack((train_label, mini_batch.Y))

                    if graph.is_epoch_end:
                        break
                micro_f1, macro_f1 = multiclass_node_classification_eval(train_emb, train_label, 0.7)
                print('Epoch-{}, loss: {:.4f}, Micro_f1 {:.4f}, Macro_fa {:.4f}'.format(epoch, cost, micro_f1, macro_f1))


        self.save_model()


    def infer(self, graph):
        self.sess.run(tf.global_variables_initializer())
        self.restore_model()
        print("Model restored from file: %s" % self.model_path)

        train_emb = None
        train_label = None
        while True:
            mini_batch = graph.sample(self.batch_size, do_shuffle=False, with_label=True)
            emb = self.sess.run(self.H, feed_dict={self.x: mini_batch.X,
                                                   })

            if train_emb is None:
                train_emb = emb
                train_label = mini_batch.Y
            else:
                train_emb = np.vstack((train_emb, emb))
                train_label = np.vstack((train_label, mini_batch.Y))

            if graph.is_epoch_end:
                break

        test_ratio = np.array([0.5, 0.7, 0.9])
        new_method = []
        for tr in test_ratio[-1::-1]:
            print('============train ration-{}=========='.format(1 - tr))
            micro_avg, macro_avg = node_classification_F1(train_emb, train_label, tr)
            micro_avg, macro_avg = node_classification_F1(train_emb, train_label, tr)
            new_method.append('{:.4f}'.format(micro_avg) + ' & ' + '{:.4f}'.format(macro_avg))
        print(' & '.join(new_method))
        
        acc_avg, nmi_avg = node_clustering_ACC(train_emb, train_label)
        # write_embedding(train_emb, './embed/cora.embed')
        return micro_avg, macro_avg, new_method,  acc_avg, nmi_avg
        # return train_emb


    def generate_samples(self, graph):

        order = np.arange(graph.num_nodes)
        np.random.shuffle(order)
        return order

    def save_model(self):
        self.saver.save(self.sess, self.model_path)

    def restore_model(self):
        self.saver.restore(self.sess, self.model_path)
