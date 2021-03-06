import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def get_time_series(dirpath, subjects, stimulus, scan, parcel,shafer):    

    subject_list = []
    all_data = []
    for subject in subjects:
        file_path = f'{dirpath}/{subject}/{stimulus}/roi_timeseries_0/{scan}/_compcor_ncomponents_5_selector_pc10.linear1.wm1.global0.motion1.quadratic1.gm0.compcor1.csf1/_bandpass_freqs_0.01.0.1/{parcel}/roi_stats.csv'
        series = pd.read_csv(file_path, skiprows=[0,1], header=None, delimiter='\t')
        series = series.T
        series['network'] = shafer['network']
        series_average = series.groupby('network').mean()
        series_average = series_average.drop("Limbic", axis=0)
        df_series = series_average.T.sort_index(axis=1)
        all_data.append(df_series.values)
        subject_list.append(subject)
  
    return all_data, subject_list
    

def get_all_data(scans, stimuli,subjects, dirpath, parcel,schafer): 
    all_c = []
    all_subjects = []

    for scan in scans:

        for stimulus in stimuli:
  
            all_data, subject_list = get_time_series(dirpath, subjects, stimulus, scan, parcel,schafer)
            all_subjects = all_subjects +  subject_list
            all_c = all_c + all_data
          
    return all_c, all_subjects

#from https://stackoverflow.com/questions/50125844/how-to-standard-scale-a-3d-matrix


class StandardScaler3D(StandardScaler):
        
    def fit(self, X, y=None):
        x = np.reshape(X, newshape=(X.shape[0]*X.shape[1], X.shape[2]))
        super().fit(x, y=y)
        return self
    
    def transform(self, X):
        x = np.reshape(X, newshape=(X.shape[0]*X.shape[1], X.shape[2]))
        return np.reshape(super().transform(x), newshape=X.shape)
    

class MinMaxScaler3D(MinMaxScaler):
        
    def fit(self, X, y=None):
        x = np.reshape(X, newshape=(X.shape[0]*X.shape[1], X.shape[2]))
        super().fit(x, y=y)
        return self
    
    def transform(self, X):
        x = np.reshape(X, newshape=(X.shape[0]*X.shape[1], X.shape[2]))
        return np.reshape(super().transform(x), newshape=X.shape)
    

class PCA3D(PCA):

    def fit_transform(self, X, y=None):
        x = np.reshape(X, newshape=(X.shape[0]*X.shape[1], X.shape[2]))
        return np.reshape(super().fit_transform(x, y=y), newshape=(X.shape[0],X.shape[1],-1))
    
    def fit(self, X, y=None):
        x = np.reshape(X, newshape=(X.shape[0]*X.shape[1], X.shape[2]))
        super().fit(x, y=y)
        return self
    
    def transform(self, X):
        x = np.reshape(X, newshape=(X.shape[0]*X.shape[1], X.shape[2]))
        return np.reshape(super().transform(x), newshape=(X.shape[0],X.shape[1],-1))

class LinearDiscriminantAnalysis3D(LinearDiscriminantAnalysis):

    def fit_transform(self, X, y):
        x=X
        return np.reshape(super().fit_transform(x, y=y), newshape=(X.shape[0],X.shape[1],-1))
    
    def fit(self, X, y):
        x = np.reshape(X, newshape=(X.shape[0]*X.shape[1], X.shape[2]))
        y = np.array([list(y)]*X.shape[1]).T.flatten()
        super().fit(x, y=y)
        return self
    
    def transform(self, X):
        x = np.reshape(X, newshape=(X.shape[0]*X.shape[1], X.shape[2]))
        return np.reshape(super().transform(x), newshape=(X.shape[0],X.shape[1],-1))

