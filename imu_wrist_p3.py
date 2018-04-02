import sklearn
from sklearn import ensemble
import numpy as np
from  numpy import genfromtxt
from sklearn import preprocessing
from sklearn import svm
from sklearn.externals import joblib
from sklearn import model_selection
from numpy import random
from imblearn.over_sampling import SMOTE
from sklearn import decomposition
from sklearn import gaussian_process
from sklearn.neural_network import MLPClassifier

def random_forests( X, Y):
    # pos_classes = np.repeat(pos_classes, 5, axis=0).flatten()
    # X = np.concatenate((X, np.take(X, pos_classes, axis=0)))
    # Y = np.concatenate((Y, np.take(Y, pos_classes, axis=0)))
    #scaler = preprocessing.StandardScaler()
    #scaler.fit(X)
    #X = scaler.transform(X)
    clf = ensemble.forest.RandomForestClassifier(n_estimators=500, max_depth=10)
    clf.fit(X, Y)
    return clf

def adaboostin(X, y):
    print("x:", len(X), "y:", len(y), X[0].shape)
    clf = sklearn.ensemble.AdaBoostClassifier(n_estimators=100,)
    # scores = cross_val_score(clf, X[:l1], y[:l1])
    pos_classes = np.where(y == 1.0)
    pos_classes = np.repeat(pos_classes, 5, axis=0).flatten()
    t = np.take(X, pos_classes, axis=0)
    X = np.concatenate((X, np.take(X, pos_classes, axis=0)))
    Y = np.concatenate((y, np.take(y, pos_classes, axis=0)))
    clf.fit(X,Y)
    return clf

def gradboostin(X, y):
    #print("x:", len(X), "y:", len(y), X[0].shape)
    #assert len(X) == len(y)

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    #X, Y = SMOTE(kind='svm', ratio='minority', n_jobs=3).fit_sample(X, y)
    #print("After Smote")
    #print(X.shape,Y.shape)
    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    clf = sklearn.ensemble.GradientBoostingClassifier(n_estimators=500, max_depth=6, random_state=0, verbose=1)
    clf.fit(X, y)
    return clf
    #joblib.dump(clf,"boosting.pkl")
    #y_test_pred = clf.predict(X_test)
    #prec = precision_score(y_test, y_test_pred)
    #recall = recall_score(y_test, y_test_pred)
    #print("belh", prec, recall)

def neural_net( X, Y):
    # print(len(X))
    # pos_classes = np.where(Y == 1.0)
    # pos_classes = np.repeat(pos_classes, 2, axis=0).flatten()
    # X = np.concatenate((X, np.take(X, pos_classes, axis=0)))
    # Y = np.concatenate((Y, np.take(Y, pos_classes, axis=0)))
    # print(len(X))
    # X, Y = SMOTE(kind='borderline1', ratio='minority', n_jobs=3).fit_sample(X, Y)
    '''
    pca = decomposition.PCA(n_components=100)
    pca.fit(X)
    X = pca.transform(X)
    '''
    clf = MLPClassifier(hidden_layer_sizes=500, activation='relu', solver='sgd', alpha=0.0001, learning_rate='adaptive',
                        learning_rate_init=0.01, batch_size=64,max_iter=500)
    clf.fit(X, Y)
    return clf


def GaussianProcess( X, Y):
    X, Y = SMOTE(kind='borderline1', ratio='minority', n_jobs=3).fit_sample(X, Y)
    clf = gaussian_process.GaussianProcessClassifier(max_iter_predict=500, n_jobs=2)
    clf.fit(X, Y)
    return clf


def call_me( X, Y):
    scaler=preprocessing.StandardScaler()
    scaler.fit(X)
    X=scaler.transform(X)
    # X, Y = SMOTE(kind='borderline1',ratio='minority',n_jobs=3).fit_sample(X, Y)
    svm_inst = svm.SVC(C=1.0, kernel='sigmoid',cache_size=1024, probability=False)
    svm_inst.fit(X, Y)
    return svm_inst


def K_CROSS(X):
    random.shuffle(X)
    Y = X[:2, -1]
    X = X[:2, :-1]
    print(X.shape)
    print(Y)
    X = preprocessing.normalize(X)
    X, Y = SMOTE(kind='borderline1', ratio='minority').fit_sample(X, Y)
    print(np.sum(Y))
    print(Y.shape)
    X = X[:int(len(X) / 2)]
    Y = Y[:int(len(Y)) / 2]
    pca = decomposition.PCA(n_components=15)
    pca.fit(X)
    X = pca.transform(X)

    # code to create duplicate positive values
    '''
    pos_classes=np.where(Y==1.0)
    pos_classes=np.repeat(pos_classes,5,axis=0).flatten()
    X=np.concatenate((X,np.take(X,pos_classes,axis=0)))
    Y=np.concatenate((Y,np.take(Y,pos_classes,axis=0)))
    '''

    scoring = ['precision', 'recall']
    svm_inst = svm.SVC(C=1.0, kernel='sigmoid', class_weight='balanced', probability=True)
    svm_inst.fit(X, Y)
    # cv_score=model_selection.cross_validate(svm_inst,X,Y,cv=3,n_jobs=3,scoring=scoring,return_train_score=False)
    # print(cv_score)
    # joblib.dump(svm_inst,"trained_svm.pkl")
    return svm_inst.predict_proba(X[0])