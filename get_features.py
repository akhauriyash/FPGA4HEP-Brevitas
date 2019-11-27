from __future__ import print_function
import os, h5py, yaml, torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

seed = 42
np.random.seed(seed)


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

def get_features(yamlConfig, batch_size, test_batch_size):
    h5File = h5py.File('./data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z')
    treeArray = h5File['t_allpar_new'][()]
    print(treeArray.shape)
    print(treeArray.dtype.names)
    features = yamlConfig['Inputs']
    labels = yamlConfig['Labels']
    features_labels_df = pd.DataFrame(treeArray,columns=list(set(features+labels)))
    features_labels_df = features_labels_df.drop_duplicates()
    features_df = features_labels_df[features]
    labels_df = features_labels_df[labels]
    if 'Conv' in yamlConfig['InputType']:
        labels_df = labels_df.drop_duplicates()
    features_val = features_df.values
    labels_val = labels_df.values
    if 'j_index' in features:
        features_val = features_val[:,:-1] # drop the j_index feature
    if 'j_index' in labels:
        labels_val = labels_val[:,:-1] # drop the j_index label
        print(labels_val.shape)
    if yamlConfig['InputType']=='Conv1D':
        features_2dval = np.zeros((len(labels_df), yamlConfig['MaxParticles'], len(features)-1))
        for i in range(0, len(labels_df)):
            features_df_i = features_df[features_df['j_index']==labels_df['j_index'].iloc[i]]
            index_values = features_df_i.index.values
            features_val_i = features_val[np.array(index_values),:]
            nParticles = len(features_val_i)
            if nParticles>yamlConfig['MaxParticles']:
                features_val_i =  features_val_i[0:yamlConfig['MaxParticles'],:]
            else:
                features_val_i = np.concatenate([features_val_i, np.zeros((yamlConfig['MaxParticles']-nParticles, len(features)-1))])
            features_2dval[i, :, :] = features_val_i
        features_val = features_2dval
    elif yamlConfig['InputType']=='Conv2D':
        features_2dval = np.zeros((len(labels_df), yamlConfig['BinsX'], yamlConfig['BinsY'], 1))
        for i in range(0, len(labels_df)):
            features_df_i = features_df[features_df['j_index']==labels_df['j_index'].iloc[i]]
            index_values = features_df_i.index.values
            xbins = np.linspace(yamlConfig['MinX'],yamlConfig['MaxX'],yamlConfig['BinsX']+1)
            ybins = np.linspace(yamlConfig['MinY'],yamlConfig['MaxY'],yamlConfig['BinsY']+1)
            x = features_df_i[features[0]]
            y = features_df_i[features[1]]
            w = features_df_i[features[2]]
            hist, xedges, yedges = np.histogram2d(x, y, weights=w, bins=(xbins,ybins))
            for ix in range(0,yamlConfig['BinsX']):
                for iy in range(0,yamlConfig['BinsY']):
                    features_2dval[i,ix,iy,0] = hist[ix,iy]
        features_val = features_2dval
    X_train_val, X_test, y_train_val, y_test = train_test_split(features_val, labels_val, test_size=0.2, random_state=42)
    if yamlConfig['NormalizeInputs'] and yamlConfig['InputType']!='Conv1D' and yamlConfig['InputType']!='Conv2D':
        scaler = preprocessing.StandardScaler().fit(X_train_val)
        X_train_val = scaler.transform(X_train_val)
        X_test = scaler.transform(X_test)
    if yamlConfig['NormalizeInputs'] and yamlConfig['InputType']!='Conv1D' and yamlConfig['InputType']!='Conv2D' and yamlConfig['KerasLoss']=='squared_hinge':
        scaler = preprocessing.MinMaxScaler().fit(X_train_val)
        X_train_val = scaler.transform(X_train_val)
        X_test = scaler.transform(X_test)
        y_train_val = y_train_val * 2 - 1
        y_test = y_test * 2 - 1
    if yamlConfig['NormalizeInputs'] and yamlConfig['InputType']=='Conv1D':
        reshape_X_train_val = X_train_val.reshape(X_train_val.shape[0]*X_train_val.shape[1],X_train_val.shape[2])
        scaler = preprocessing.StandardScaler().fit(reshape_X_train_val)
        for p in range(X_train_val.shape[1]):
            X_train_val[:,p,:] = scaler.transform(X_train_val[:,p,:])
            X_test[:,p,:] = scaler.transform(X_test[:,p,:])
    if 'j_index' in labels:
        labels = labels[:-1]

    input_shape = X_train_val.shape[1:]
    output_shape = y_train_val.shape[1]

    X_train_val = np.stack(list(chunks(X_train_val, batch_size))[:-1])
    y_train_val = np.stack(list(chunks(y_train_val, batch_size))[:-1])
    
    X_test      = np.stack(list(chunks(X_test,      test_batch_size))[:-1])
    y_test      = np.stack(list(chunks(y_test,      test_batch_size))[:-1])
    
    X_train_val = torch.unsqueeze(torch.tensor(torch.from_numpy(X_train_val), dtype=torch.float), dim=2)
    X_test = torch.unsqueeze(torch.tensor(torch.from_numpy(X_test), dtype=torch.float), dim=2)
    
    y_train_val = torch.unsqueeze(torch.tensor(torch.from_numpy(y_train_val), dtype=torch.float), dim=2)
    y_test = torch.unsqueeze(torch.tensor(torch.from_numpy(y_test), dtype=torch.float), dim=2)
    
    train_loader = list(zip(X_train_val, y_train_val))
    test_loader  = list(zip(X_test, y_test))
    
    return X_train_val, X_test, y_train_val, y_test, labels, train_loader, test_loader, input_shape, output_shape