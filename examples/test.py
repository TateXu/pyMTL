import pickle
import numpy as np
from pymtl.linear_regression import MTLRegression as mtl
from sklearn.preprocessing import LabelEncoder


import moabb
import numpy as np
import warnings

from mne.decoding import CSP
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from moabb.datasets import MunichMI, BNCI2014001 
from moabb.paradigms import MotorImagery
from moabb.pipelines.features import TSSF
from copy import deepcopy as dc
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


def score(clf, X, y, scoring):
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    le = LabelEncoder()
    y = le.fit_transform(y)
    acc = cross_val_score(clf, X, y, cv=cv, scoring=scoring)
    return acc.mean()

def load_data(subject, session):
    MI = MotorImagery(fmin=8, fmax=32, events=['left_hand', 'right_hand'])
    dataset = MunichMI()  #  BNCI2014001() # 
    X, y, metadata = MI.get_data(dataset, [subject])
    for nr_ses, name_ses in enumerate(np.unique(metadata.session).tolist()):
        if nr_ses == session:
            ix = metadata.session == name_ses
            return X[ix], y[ix]

file_root = '/home/jxu/File/Code/Git/pyMTL/examples/'

nr_comp = 6
pre_load_data = False
if pre_load_data:
    all_data = []
    all_label = []

    le = LabelEncoder().fit(["left_hand", "right_hand"])
    for nr_subj in range(1, 11):
        X, y = load_data(subject=nr_subj, session=0)
        y = le.transform(y)
        all_data.append(X)
        all_label.append(y)

    source_data = []


    for nr_subj in range(10):
        sf = TSSF(clf_str='SVM', func='clf', n_components=nr_comp, decomp='GED', logvar='Cov')
        source = sf.fit(all_data[nr_subj], all_label[nr_subj]).transform(all_data[nr_subj])
        source_data.append(source)


    source_data = np.asarray(source_data)
    label = np.asarray(all_label)
    with open(file_root + 'MunichMI_{0}.pkl'.format(str(nr_comp)), 'wb') as f:
        pickle.dump([source_data, label], f)
else:
    with open(file_root + 'MunichMI_{0}.pkl'.format(str(nr_comp)), 'rb') as f:
        source_data, label = pickle.load(f)
le = LabelEncoder().fit(["left_hand", "right_hand"])
acc_mat = np.zeros((10, 11))
for nr_subj in range(10):
    
    trained = mtl(max_prior_iter=1000, prior_conv_tol=0.0001, C=1, C_style='ML')
    X_pool = np.delete(dc(source_data), nr_subj, axis=0)
    y_pool = np.delete(dc(label), nr_subj, axis=0)
    trained.fit_multi_task(X_pool, y_pool, verbose=False, n_jobs=1)
    
    X_pool_ = X_pool.reshape(-1, X_pool.shape[-1])
    y_pool_ = y_pool.reshape(-1)

    single_data, single_label = load_data(subject=nr_subj+1, session=0)
    single_label = le.transform(single_label)

    for ind, nr_train_trial in enumerate([120]): # range(120, 120, 10)
        X_train, X_test, y_train, y_test = train_test_split(
            single_data, single_label, test_size=(150-nr_train_trial)/150, random_state=0)

        sf = TSSF(clf_str='SVM', func='clf', n_components=nr_comp, decomp='GED', logvar='Cov')
        fitted_tssf = sf.fit(X_train, y_train)

        source = fitted_tssf.transform(X_train)

        X_all_train = np.concatenate((X_pool_, source), axis=0)
        y_all_train = np.concatenate((y_pool_, y_train), axis=0)
        trained.fit(X_all_train, y_all_train)

        y_predict = np.sign(trained.predict(fitted_tssf.transform(X_test)))
        y_predict = np.int8((y_predict + 1) / 2)
        acc = 1 - np.sum((y_predict - y_test) ** 2) / y_predict.shape[0]
        acc_mat[nr_subj, ind] = acc
        print('S_{0}, #Trial_{1}, Accuracy: {2}'.format(str(nr_subj+1), str(nr_train_trial), str(acc)))
import pdb 
pdb.set_trace()




