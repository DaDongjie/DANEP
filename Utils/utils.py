import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.metrics import f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
import tensorflow as tf
from sklearn.svm import LinearSVC
import warnings
from sklearn.metrics.pairwise import cosine_similarity


class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__




def small_trick(y_test, y_pred):
    y_pred_new = np.zeros(y_pred.shape, np.bool)
    sort_index = np.flip(np.argsort(y_pred, axis=1), 1)
    for i in range(y_test.shape[0]):
        num = sum(y_test[i])
        for j in range(num):
            y_pred_new[i][sort_index[i][j]] = True
    return y_pred_new


def multi_label_classification(X, Y, ratio):
    X = preprocessing.normalize(X, norm='l2')

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=ratio, random_state=42)

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    # =========train=========
    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg), param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=1)  #
    clf.fit(X_train, y_train)
    print('Best parameters')
    print(clf.best_params_)

    # =========test=========
    y_pred = clf.predict_proba(X_test)
    y_pred = small_trick(y_test, y_pred)

    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")
    
    acc = accuracy_score(y_test, y_pred)

    # fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=2)

    # AUC_s=auc(fpr, tpr, reorder=False)
    AUC_s = 0
    print("acc: %.4f" % (acc))
    print("AUC: %.4f" % (AUC_s))
    print("micro_f1: %.4f" % (micro))
    print("macro_f1: %.4f" % (macro))

    return micro, macro



def check_multi_label_classification(X, Y, ratio):

    X = preprocessing.normalize(X, norm='l2')

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=ratio, random_state=42)

    clf = OneVsRestClassifier(LogisticRegression(solver='liblinear'))
    clf.fit(x_train, y_train)

    y_pred = clf.predict_proba(x_test)
    y_pred = small_trick(y_test, y_pred)

    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)

    return micro, macro




def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)




from sklearn.cluster import KMeans
from sklearn import metrics

def acc_val(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def node_clustering(emb, one_hots):
    label = [np.argmax(one_hot) for one_hot in one_hots]
    ClusterNUm = np.unique(label)


    clf = KMeans(n_clusters=len(ClusterNUm),init="k-means++")
    kmeans = clf.fit(emb)

    cluster_groups = kmeans.labels_
    acc =acc_val(np.array(label),np.array(cluster_groups))
    nmi = metrics.normalized_mutual_info_score(label,cluster_groups)
    return acc,nmi

def node_clustering_ACC(emb, one_hots):
    acc_avg = 0
    nmi_avg = 0
    for i in range(10):
        rnd = np.random.randint(2018)
        acc, nmi  = node_clustering(emb, one_hots)
        acc_avg += acc
        nmi_avg += nmi
    acc_avg /= 10
    nmi_avg /= 10

    print('ACC average value: ' + str(acc_avg))
    print('NMI average value: ' + str(nmi_avg))
    return acc_avg, nmi_avg

def write_embedding(embedding_result, outputFileName):
    f = open(outputFileName, 'w')
    N, dims = embedding_result.shape

    for i in range(N):
        s = ''
        for j in range(dims):
            if j == 0:
                s = str(i) + ',' + str(embedding_result[i, j])
            else:
                s = s + ',' + str(embedding_result[i, j])
        f.writelines(s + '\n')
    f.close()
    
    
def multiclass_node_classification_eval(X, y, ratio, rnd=2018):
    warnings.filterwarnings('ignore')
    y = [np.argmax(one_hot) for one_hot in y]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=rnd)
    print("type:",type(X_train), type(y_train))

    svm = LinearSVC()
    c = 2.0 ** np.arange(-10, 10)
    clf = GridSearchCV(estimator=OneVsRestClassifier(svm), param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=1)
    # clf = LinearSVC()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average='macro')
    micro_f1 = f1_score(y_test, y_pred, average='micro')

    return micro_f1, macro_f1

def node_classification_F1(Embeddings, y, ratio):
    macro_f1_avg = 0
    micro_f1_avg = 0
    for i in range(10):
        rnd = np.random.randint(2018)
        micro_f1, macro_f1  = multiclass_node_classification_eval(Embeddings, y, ratio, rnd)
        micro_f1_avg += micro_f1
        macro_f1_avg += macro_f1
    micro_f1_avg /= 10
    macro_f1_avg /= 10

    print('Micro_f1 average value: ' + str(micro_f1_avg))
    print('Macro_f1 average value: ' + str(macro_f1_avg))
    return micro_f1_avg, macro_f1_avg


def link_prediction(embedding, test_file, result_file):
    embedding_sim = cosine_similarity(embedding)
    y_predict = []
    y_gt = []
    with open(test_file) as fp:
        for line in fp.readlines():
            node1 = int(line.strip('\n\r').split()[0])
            node2 = int(line.strip('\n\r').split()[1])
            label = int(line.strip('\n\r').split()[2])
            print(node1, node2)
            y_gt.append(label)
            y_predict.append(embedding_sim[node1, node2])
        roc = roc_auc_score(y_gt, y_predict)
        ap = average_precision_score(y_gt, y_predict)
        if roc < 0.5:
            roc = 1 - roc
        print('ROC:', roc, 'AP:', ap)
        return roc, ap

