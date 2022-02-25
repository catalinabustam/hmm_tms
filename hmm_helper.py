import operator
from copy import copy
from scipy.special import softmax
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class HMM_classifier(BaseEstimator):
    def __init__(self, hmm_model=None,n_components=None, n_mix=None):
        self.models = {}
        self.hmm_model = hmm_model
        self.n_components = n_components
        self.n_mix = n_mix
    
    def fit(self, X, Y):
        """
        X: input sequence [[[x1,x2,.., xn]...]]
        Y: output classes [1, 2, 1, ...
        """
        #print("Detect classes:", set(Y))
        #print("Prepare datasets...")
        X_Y = {}
        X_lens = {}

        for c in set(Y):
            X_Y[c] = []
            X_lens[c] = []

        for x, y in zip(X, Y):
            X_Y[y].extend(x)
            X_lens[y].append(len(x))

        for c in set(Y):
            #print("Fit HMM for", c, " class")
            hmm_model = copy(self.hmm_model)
            hmm_model.n_components= self.n_components
            hmm_model.n_mix= self.n_mix 
            
            startprob = np.zeros(self.n_components)
            startprob[0] = 1

            transmat = np.zeros((self.n_components, self.n_components))
            transmat[0,0] = 1
            transmat[-1,-1] = 1


            for i in range(transmat.shape[0]-1):
                if i != transmat.shape[0]:
                    for j in range(i, i+2):
                        transmat[i,j] = 0.5
            hmm_model.startprob_ = np.array(startprob)
            hmm_modeltransmat_ = np.array(transmat)
            
            hmm_model.fit(np.array(X_Y[c]), X_lens[c])
            self.models[c] = hmm_model

    def _predict_scores(self, X):

        """
        X: input sample [[x1,x2,.., xn]
        """
        X_s =[]
        X_len = []
        
        #if X.ndim <3:
        #    X = np.expand_dims(X, axis=0)
        scores = []

        for x in X:
            scores_x = []
            for k, v in self.models.items():
                scores_x.append(v.score(np.array(x)))
            scores.append(scores_x)    
        return scores

    def predict_proba(self, X):
        """
        X: input sample [[x1,x2,.., xn]]
        """
        pred = self._predict_scores(X)

        return np.array(pred)

    def predict(self, X):
        """
        X: input sample [[x1,x2,.., xn]]
        """
        pred = self.predict_proba(X)
        #print(np.argmax(pred, axis=1))

        return np.argmax(pred, axis=1)