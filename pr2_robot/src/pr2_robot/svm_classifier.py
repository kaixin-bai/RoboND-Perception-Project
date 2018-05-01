#!/usr/bin/env python

"""
Adaptation from train_svm.py
"""

import os
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import cross_validation
from sklearn import metrics
from argparse import ArgumentParser

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{0:.2f}'.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

class SVMClassifier(object):
    """
    Simple Wrapper for Scikit-Learn's svm.SVC Class.
    """
    def __init__(self, model_path='', data=None):
        if model_path:
            if os.path.exists(model_path):
                self.load(model_path)
            else:
                print("Specified model path {} does not exist; Model must be trained and saved!".format(model_path))
        else:
            print("Valid Model Path Must Be Specified For Training and Prediction!")

    def save(self, path, param, model):
        with open(path, 'wb') as f:
            pickle.dump([param, model], f)

    def load(self, path):
        with open(path, 'r') as f:
            param, model = pickle.load(f)
            self._clf = model['clf']
            self._scl = model['scl']
            self._enc = model['enc']
            # see data_collection.py
            self._hsv = param['hsv']
            self._bin = param['bin']
            print self._bin

    @staticmethod
    def format_data(data):
        """
        Expects data formatted as {'name' : [feature_1, feature_2, ...]}.
        Format the features and labels for use with scikit learn
        """
        feature_list = []
        label_list = []

        for name, fs in data.iteritems():
            # ============
            # visualization ...
            #bin = len(fs[0]) / 6

            #fs = np.stack(fs, axis=0)
            #plt.boxplot(fs, notch=True)

            ## annotate ...
            #for i in range(0, bin*6, bin):
            #    plt.axvline(x=i)
            #xs = range(bin/2, bin*6, bin)
            #plt.xticks(xs, ['h','s','v','x','y','z'])
            #plt.title(name)
            #plt.savefig('/tmp/{}.png'.format(name))
            #plt.clf()
            # =============

            for f in fs:
                label_list.append(name)
                feature_list.append(f)
        return np.array(feature_list), np.array(label_list)

    def train(self, X, y):
        """
        Trains the classifier.
        """
        # Load training data
        
        # Scale + Preprocess Data
        scl = StandardScaler().fit(X)
        X = scl.transform(X)
        
        # Encode Labels
        enc = LabelEncoder()
        y = enc.fit_transform(y)

        #Train the classifier
        clf = svm.SVC(kernel='linear')
        clf.fit(X=X, y=y)

        model = {
                'clf' : clf,
                'enc' : enc,
                'scl' : scl
                }

        return model
        
    def predict(self, x):
        single = False
        if np.ndim(x) == 1:
            single = True
            x = np.expand_dims(x, 0)

        x = self._scl.transform(x)
        y = self._clf.predict(x)
        y = self._enc.inverse_transform(y)

        if single:
            return y[0]
        else:
            return y

    def test(self, x, y):
        scl, enc, clf = self._scl, self._enc, self._clf

        x = scl.transform(x)
        y = enc.transform(y)

        # Set up 5-fold cross-validation
        kf = cross_validation.KFold(len(x),
                                    n_folds=5,
                                    shuffle=True,
                                    random_state=1)

        # Perform cross-validation
        scores = cross_validation.cross_val_score(cv=kf,
                                                 estimator=clf,
                                                 X=x, 
                                                 y=y,
                                                 scoring='accuracy'
                                                )
        print('Scores: ' + str(scores))
        print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), 2*scores.std()))

        # Gather predictions
        predictions = cross_validation.cross_val_predict(cv=kf,
                                                  estimator=clf,
                                                  X=x, 
                                                  y=y
                                                 )

        accuracy_score = metrics.accuracy_score(y, predictions)
        print('accuracy score: '+str(accuracy_score))

        confusion_matrix = metrics.confusion_matrix(y, predictions)

        class_names = enc.classes_.tolist()

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(confusion_matrix, classes=enc.classes_,
                              title='Confusion matrix, without normalization')
        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(confusion_matrix, classes=enc.classes_, normalize=True,
                              title='Normalized confusion matrix')
        plt.show()

def main():
    parser = ArgumentParser()
    parser.add_argument('model_path', help='Trained Model File (Pickle)')
    parser.add_argument('data_path', default='', help='Object Features Data File (Pickle)')
    parser.add_argument('--train', action='store_true', help='Force Training if model exists')
    parser.add_argument('--test', action='store_true', help='Test Model')
    args = parser.parse_args()
    print('Got Arguments, {}'.format(args))
    input_model = ('' if args.train else args.model_path)
    svm = SVMClassifier(model_path=input_model)

    if args.data_path and os.path.exists(args.data_path):
        x = pickle.load(open(args.data_path, 'rb'))
        [param, data] = pickle.load(open(args.data_path, 'rb'))
        x, y = svm.format_data(data)
        if args.train:
            model = svm.train(x, y)
            svm.save(args.model_path, param, model)

        svm.load(args.model_path)
        if args.test:
            svm.test(x, y)
    else:
        print('Valid data path must be specified for training/testing!')

if __name__ == '__main__':
    main()
