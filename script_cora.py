import numpy as np
import linecache
from Dataset.dataset import Dataset
from Model.model import Model
from Trainer.trainer import Trainer
from Trainer.pretrainer import PreTrainer
import os
import random
import tensorflow as tf
from Utils.utils import write_embedding, link_prediction


if __name__=='__main__':

    random.seed(9001)

    dataset_config = {'feature_file': './Database/cora/features.txt',
                      'graph_file': './Database/cora/edges.txt',
                      'label_file': './Database/cora/group.txt'}

    graph = Dataset(dataset_config)

    pretrain_config = {
        'att_shape': [512, 256, 128],
        'att_input_dim': graph.X_num,
        'pretrain_params_path': './Log/cora/pretrain_params.pkl'}

    model_config = {
        'att_shape': [512, 256, 128],
        'att_input_dim': graph.X_num,
        'is_init': True,
        'pretrain_params_path': './Log/cora/pretrain_params.pkl'
    }
    Flag = False
    trainer_config = {
        'att_shape': [512, 256, 128],
        'att_input_dim': graph.X_num,
        'drop_prob': 0.2,
        'learning_rate': 1e-5,
        'batch_size': 64,
        'num_epochs': 500,
        'beta': 1,
        'alpha': 20,

        'model_path': './Log/cora/cora_model.pkl',
    }
    if Flag:
        pretrainer = PreTrainer(pretrain_config)
        pretrainer.pretrain(graph.X, 'att')
        Flag = False
    model = Model(model_config)
    trainer = Trainer(model, trainer_config)
    trainer.train(graph)
    micro, macro, new_method, acc, nmi = trainer.infer(graph)



    # New_Method = []
    # Flag = True
    # with open("./result/cora_10_wg.txt", "w") as f:
    #     for beta_i in[1, 10, 50, 100,150, 200,250, 300, 350, 400, 450, 500]:
    #         for alpha_i in [0.001,0.01,0.1,1,10,50,100,200,500]:
    #
    #             tf.reset_default_graph()
    #             trainer_config = {
    #                 'att_shape': [512, 256, 128],
    #                 'att_input_dim': graph.X_num,
    #                 'drop_prob': 0.2,
    #                 'learning_rate': 1e-5,
    #                 'batch_size': 64,
    #                 'num_epochs': 500,
    #                 'beta': beta_i - 1,
    #                 'alpha': alpha_i,
    #
    #                 'model_path': './Log/cora/cora_model.pkl',
    #             }
    #             if Flag:
    #                 pretrainer = PreTrainer(pretrain_config)
    #                 pretrainer.pretrain(graph.X, 'att')
    #                 Flag = False
    #
    #             model = Model(model_config)
    #             trainer = Trainer(model, trainer_config)
    #             trainer.train(graph)
    #             micro, macro, new_method, acc, nmi = trainer.infer(graph)
    #             result_single = 'beta_i={:.4f}'.format(beta_i) + ' & alpha_i={:.4f}'.format(
    #                 alpha_i)  + ' & micro={:.4f}'.format(
    #                 micro) + ' & ' + 'macro={:.4f}'.format(macro) + ' & ' + ' & '.join(new_method) + ' & ' + 'ACC={:.4f}'.format(
    #                 acc) + ' & ' + 'NMI={:.4f}'.format(nmi)
    #             New_Method.append(result_single)
    #
    #             f.write(result_single + '\n')
    #             f.flush()
