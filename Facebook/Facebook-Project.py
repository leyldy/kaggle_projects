
# coding: utf-8

# # Facebook Check-Ins

# In[127]:

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn import neighbors, datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier as KNN


# In[ ]:

#Read data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')





# In[120]:
def prepare_data(df, n_cell_x, n_cell_y):

    size_x = 10. / n_cell_x
    size_y = 10. / n_cell_y
    eps = 0.00001
    xs = np.where(df.x.values < eps, 0, df.x.values - eps)
    ys = np.where(df.y.values < eps, 0, df.y.values - eps)
    pos_x = (xs / size_x).astype(np.int64)
    pos_y = (ys / size_y).astype(np.int64)
    df['grid_cell'] = (pos_y * n_cell_x + pos_x)
    
    
    fw = [500, 1000, 4, 3, 1./22., 2, 10] 
        
    df.x = df.x.values * fw[0]
    df.y = df.y.values * fw[1]
    initial_date = np.datetime64('2014-01-01T01:01', dtype='datetime64[m]') 
    d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm') 
                               for mn in df.time.values)    
    df['hour'] = d_times.hour * fw[2]
    df['weekday'] = d_times.weekday * fw[3]
    df['day'] = (d_times.dayofyear * fw[4]).astype(int)
    df['month'] = d_times.month * fw[5]
    df['year'] = (d_times.year - 2013) * fw[6]
    return df


# In[121]:
def process_one_cell(df_train, df_test, grid_id, threshold, n_cell_x, n_cell_y):
    """   
    Throw in a training dataset and it will split it into local training and testing sets, and 
    do a KNN classification inside one grid cell.
    """   
    
    df_cell_train = df_train.loc[df_train.grid_cell == grid_id]
    place_counts = df_cell_train.place_id.value_counts()
    
    mask = (place_counts[df_cell_train.place_id.values] >= threshold).values 
    df_cell_train = df_cell_train.loc[mask] 
    

    df_cell_test = df_test.loc[df_test.grid_cell == grid_id]
    
    row_ids = df_cell_test.index
    
    
    features = ['x','y','hour','day','weekday','month','year','accuracy']
    
    train_y = df_cell_train['place_id']
    train_x = df_cell_train[features]

    test_x = df_cell_test[features]

    
    knn = KNN(15) 
    knn.fit(train_x, train_y) 
    all_preds = knn.predict_proba(test_x)
    

    preds_per_cell = np.zeros((test_x.shape[0], 3), dtype=int)
    for record in range(len(all_preds)):
        top3_idx = all_preds[record].argsort()[-3:][::-1]
        preds = knn.classes_[top3_idx]
        preds_per_cell[record] = preds
           
    train_acc = knn.score(train_x, train_y) 
    
    return preds_per_cell, row_ids, train_acc




def process_grid(df_train, df_test, threshold, n_cells, n_cell_x, n_cell_y):
    """
    Iterates over all grid cells, return average training and testing accuracies
    """ 
    preds = np.zeros((df_test.shape[0], 3), dtype=int)
    small_train_acc_sum = 0

    
    for grid_id in range(n_cells):
        if grid_id % 100 == 0:
            print('iter: %s' %(grid_id))
            print(small_train_acc_sum / (grid_id - 1))
        

        pred_labels, row_ids, small_train_acc = process_one_cell(df_train, df_test,
                                                                 grid_id, threshold, 
                                                                 n_cell_x, n_cell_y)
        
        small_train_acc_sum += small_train_acc 
        
        preds[row_ids] = pred_labels

    train_acc_avg = small_train_acc_sum/n_cells

    
    print('Generating submission file ...')
    

    df_aux = pd.DataFrame(preds, dtype=str, columns = ['l1', 'l2', 'l3'])  
    

    ds_sub = df_aux.l1.str.cat([df_aux.l2, df_aux.l3], sep=' ')
    

    ds_sub.name = 'place_id'
    ds_sub.to_csv('sub_knn.csv', index=True, header=True, index_label='row_id') 
    
    return train_acc_avg

n_cell_x = 30
n_cell_y = 30
threshold = 3



df_train = prepare_data(train, n_cell_x, n_cell_y) 
df_test = prepare_data(test, n_cell_x, n_cell_y)



process_grid(df_train, df_test, threshold, n_cell_x * n_cell_y, n_cell_x, n_cell_y)






