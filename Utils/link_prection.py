from __future__ import print_function
import numpy as np
from sklearn.metrics import *
from sklearn.metrics.pairwise import cosine_similarity


def read_embed(inputFileName):
    f = open(inputFileName, 'r')
    lines = f.readlines()
    f.close()
    embed = []
    for line in lines[:]:
        l = line.strip('\n\r').split(',')[1:]
        embed.append(l)
    embed = np.array(embed, dtype=np.float32)
    return embed

def link_prediction(embedding, test_file, result_file):
    embedding_sim = cosine_similarity(embedding)
    y_predict = []
    y_gt = []
    with open(test_file) as fp:
        for line in fp.readlines():
            node1 = int(line.strip('\n\r').split()[0])
            node2 = int(line.strip('\n\r').split()[1])
            label = int(line.strip('\n\r').split()[2])
            y_gt.append(label)
            y_predict.append(embedding_sim[node1, node2])
        roc = roc_auc_score(y_gt, y_predict)
        ap = average_precision_score(y_gt, y_predict)
        if roc < 0.5:
            roc = 1 - roc
        print('ROC:', roc, 'AP:', ap)
        result = 'ROC:' + str(roc) + ',AP:' + str(ap)
        f = open(result_file, 'w')
        f.write(result)
        f.close

def main():
    data='pubmed'
    test_file = './Database/' + data + '.test'
    result_file = './result/' + data + '/link_prection.txt'
    embedFile = './embedding_dane/pubmed.embed'
    embedding = read_embed(embedFile)
    link_prediction(embedding, test_file, result_file)


if __name__ == '__main__':
    main()
