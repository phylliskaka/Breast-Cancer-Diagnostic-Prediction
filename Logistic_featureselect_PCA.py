import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

import os 
# Read the data using pandas
ROOT_DIR = os.getcwd()
print(ROOT_DIR)
path = os.path.join(ROOT_DIR , 'data.csv')
data = pd.read_csv(path)
print(data.columns)

# remove the columns that cant be used for prediction
y = data.diagnosis 
list = ['Unnamed: 32', 'id', 'diagnosis']
x = data.drop(list, axis = 1)
print(x.head())

#%% 
# understand distribution of label 
plt.figure(figsize=[10,5])
plt.subplot(121)
plt.pie( x = y.value_counts(), labels = y.unique())
plt.title("the percentage of labels")
plt.subplot(122)
plt.bar(x = [0.2,1], height = y.value_counts(), width = 0.6)
plt.title('the number of labels')
plt.show()

#%% 
# Feature selection 
# explore more into data show the distribution of each feature, 
# see which feature is useful for classification 
# show the volin plot for mean, standard error, and worst value in three groups
data_dia  = y
data = x
# standardization of data
data_n_2 = (data - data.mean())/(data.std())
data = pd.concat([y,data_n_2.iloc[:,0:10]], axis=1)
data = pd.melt(data, id_vars='diagnosis', 
                     var_name='features', 
                     value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x='features', y='value', hue='diagnosis',
               data=data,split=True,inner='quart')
plt.xticks(rotation=90)

# check the correlation between features, so we can drop
# off some of them 
# heatmap 
f, ax = plt.subplots(figsize=(18,18))
sns.heatmap(x.corr(), annot=True, linewidths=0.5, fmt='.1f', ax=ax)

#%% 
# features selection 
# according to heatmap, choose one feature from a feature group where
# feature correlation has pearson==1 
# Thus select area mean from [radius_mean, perimeter_mean, area_mean]
# select concavity_mean from [compatness_mean, concavity_mean, concave points_mean]
# select area_se from [radius_se, perimeter_se, area_se]
# select area_worst from [radius_worst, perimeter_worst, area_worst]
# select concavity_worst from [compatness_worst, concavity_worst, concave points_worst]
# select concavity_se from [compatness_se, concavity_se, concave points_se]
# select texture_mean from [texture_mean, texture_worst]
# select area_mean from [area_worst, area_mean]
drop_list1 = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean',
              'radius_se','perimeter_se','radius_worst','perimeter_worst',
              'compactness_worst','concave points_worst','compactness_se','concave points_se',
              'texture_worst','area_worst']
x_1 = x.drop(drop_list1, axis = 1)
x_1.head()

#%% 
# Using PCA to reduce the dimensionality of data 
from sklearn.preprocessing import StandardScaler 
conv = StandardScaler()
std_data = conv.fit_transform(x_1)
# use PCA to reduce dimensionality 
from sklearn.decomposition import PCA 
pca = PCA(n_components=16, svd_solver='full')
transformed_data = pca.fit_transform(std_data)
print(transformed_data.shape)
print(pca.explained_variance_ratio_*100)
print(pca.explained_variance_)

threshold = 0.95
for_test = 0
order = 0 
for index, ratio in enumerate(pca.explained_variance_ratio_):
    if threshold > for_test:
        for_test += ratio 
    else:
        order = index + 1 
        break 

print('the first %d features could represent 95 percents of the viarance' % order)
print(pca.explained_variance_ratio_[:order].sum())
com_col = ['com'+ str(i+1) for i in range(order)]
com_col.append('others')
com_value = [i for i in pca.explained_variance_ratio_[:order]]
com_value.append(1-pca.explained_variance_ratio_[:order].sum())
plt.figure(figsize=[4,4])
plt.pie(x = com_value, labels = com_col)
plt.title('the first 10 components')
plt.show()

#%% 
# using regularization for logistic regression, use gridSearchCV to search
# for the best type of regularization and corrisponding paramenter
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
 
X_train, X_test, y_train, y_test = train_test_split(transformed_data, y, 
                                                    test_size = 0.2)
logistic_reg = LogisticRegression()
para_grid = {
        'penalty': ['l1', 'l2'], 
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000 ]}

CV_log_reg = GridSearchCV(estimator= logistic_reg, param_grid=para_grid, n_jobs=-1)
CV_log_reg.fit(X_train, y_train)
best_para = CV_log_reg.best_params_
print('The best parameters are: ', best_para)

#%% 
# the helper function for ploting confusion matrix and get accuracy 
from sklearn.metrics import confusion_matrix 
def plot_confusion_matrix(label, pred, classes=[0,1], cmap=plt.cm.Blues, title='Confusion Matrix'):
    con_m = confusion_matrix(label,pred)
    plt.imshow(con_m, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    thres = con_m.max()/2
    for j in range(con_m.shape[0]):
        for i in range(con_m.shape[1]):
            plt.text(i,j, con_m[j,i],
                     horizontalalignment = 'center',
                     color='white' if con_m[i,j]>thres else 'black')
    plt.ylabel('true label')
    plt.xlabel('predicted label')
    plt.xticks(classes, classes)
    plt.yticks(classes, classes)
    plt.tight_layout()
    
def print_accuracy(label, y_pred):
    tn, fp, fn, tp = confusion_matrix(label, y_pred).ravel()
    print('Accuracy rate = %.2f' %((tp+tn)/(tn+fp+fn+tp)))

    
#%%
# now using the best parmeters to log the regression model 
logistic_reg = LogisticRegression(C =best_para['C'], penalty=best_para['penalty'])
logistic_reg.fit(X_train, y_train)
y_pred = logistic_reg.predict(X_test)

# result 
plot_confusion_matrix(y_test, y_pred)
plt.show()
print_accuracy(y_test, y_pred)
