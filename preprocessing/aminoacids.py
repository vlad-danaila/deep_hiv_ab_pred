import pandas as pd
from sklearn import preprocessing
import numpy as np
from deep_hiv_ab_pred.util.tools import to_torch
from deep_hiv_ab_pred.global_constants import EMBEDDING

def compute_amino_props():
    amino_props = pd.DataFrame.from_dict({
        'A': [1.28, 0.05, 1.00, 0.31, 6.11, 0.42, 0.23],
        'G': [0.00, 0.00, 0.00, 0.00, 6.07, 0.13, 0.15],
        'V': [3.67, 0.14, 3.00, 1.22, 6.02, 0.27, 0.49],
        'L': [2.59, 0.19, 4.00, 1.70, 6.04, 0.39, 0.31],
        'I': [4.19, 0.19, 4.00, 1.80, 6.04, 0.30, 0.45],
        'F': [2.94, 0.29, 5.89, 1.79, 5.67, 0.30, 0.38],
        'Y': [2.94, 0.30, 6.47, 0.96, 5.66, 0.25, 0.41],
        'W': [3.21, 0.41, 8.08, 2.25, 5.94, 0.32, 0.42],
        'T': [3.03, 0.11, 2.60, 0.26, 5.60, 0.21, 0.36],
        'S': [1.31, 0.06, 1.60, -0.04, 5.70, 0.20, 0.28],
        'R': [2.34, 0.29, 6.13, -1.01, 10.74, 0.36, 0.25],
        'K': [1.89, 0.22, 4.77, -0.99, 9.99, 0.32, 0.27],
        'H': [2.99, 0.23, 4.66, 0.13, 7.69, 0.27, 0.30],
        'D': [1.60, 0.11, 2.78, -0.77, 2.95, 0.25, 0.20],
        'E': [1.56, 0.15, 3.78, -0.64, 3.09, 0.42, 0.21],
        'N': [1.60, 0.13, 2.95, -0.60, 6.52, 0.21, 0.22],
        'Q': [1.56, 0.18, 3.95, -0.22, 5.65, 0.36, 0.25],
        'M': [2.35, 0.22, 4.43, 1.23, 5.71, 0.38, 0.32],
        'P': [2.67, 0.00, 2.72, 0.72, 6.80, 0.13, 0.34],
        'C': [1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41]
    }, orient='index')
    amino_props_np = amino_props.values
    amino_props_np = preprocessing.StandardScaler().fit_transform(amino_props_np)
    # mean = amino_props_np.mean(axis = 0)
    amino_props_df = pd.DataFrame(amino_props_np, index=amino_props.index)
    amino_props_df.loc['-'] = [0] * 7
    amino_props_df.loc['X'] = [0] * 7
    return amino_props_df # to get a value: amino_props.loc['C'].values

amino_props = compute_amino_props()
aminoacids = list(amino_props.index)
amino_to_index = { aa: i for (i, aa) in enumerate(aminoacids) }
aminoacids_len = len(aminoacids)

def one_hot():
    one_hot = np.eye(aminoacids_len - 2)
    none_element = np.zeros((2, aminoacids_len - 2))
    result = np.concatenate((one_hot, none_element))
    return result

def amino_props_and_one_hot():
    props_and_one_hot = np.concatenate((amino_props.values, one_hot()), axis = 1)
    return pd.DataFrame(props_and_one_hot, index=amino_props.index)

def get_embeding_matrix():
    if EMBEDDING == 'LEARNED':
        return None
    elif EMBEDDING == 'ONE-HOT':
        return to_torch(one_hot())
    elif EMBEDDING == 'ONE-HOT-AND-PROPS':
        return to_torch(amino_props_and_one_hot().values)
    elif EMBEDDING == 'PROPS-ONLY':
        return to_torch(amino_props.values)
    else:
        raise 'The embedding type must have a valid value.'