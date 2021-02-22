# %%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk

from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge
from common import check_cv_score, grid_search, plot_lc
from common import plot_vc
from common import dtree_params
from common import dtree_extended_params
from common import knn_params
from common import knn_leaf_params
from common import check_cv_score
from common import check_test_score
from common import dtree_sample_split
from common import svm_params 
from common import nn_params 
from common import nn_learning_params 
from common import nn_adam_params 
from common import nn_adam_tuned_params
from common import adaboost_params 
from common import adaboost_estimators
from sklearn import preprocessing

pd.options.display.max_colwidth = 150
pd.options.display.max_rows = 200

data = pd.read_csv('data/iris/iris.data')
x = data.drop('class', axis=1)
y = data['class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=5)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_test.value_counts())
# %% [markdown]
# dt
# nn
# boosting
# svm
# knn
# %% load data
# %% EDA 
data.head()
# %%
data.info()
# %%
data.describe()
# %%
data['class'].value_counts()
# %%
pairplot = sns.pairplot(data, hue='class', markers='*')
# %%
sepal_len = sns.violinplot(y='class', x='sepal_len', data=data, inner='quartile')
# %%
sepal_width = sns.violinplot(y='class', x='sepal_width', data=data, inner='quartile')
# %%
petal_len = sns.violinplot(y='class', x='petal_len', data=data, inner='quartile')
# %%
petal_width = sns.violinplot(y='class', x='petal_width', data=data, inner='quartile')
print(data.head)
print(data.columns)
# %% 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=5)
# %% OLD MERGE
# %% 
# [KNN MODEL]
#
# %% basic grid search on untuned model
knn = KNeighborsClassifier()
print(knn_params)
grid_search(knn, knn_params, x_train, y_train)
plot_vc(knn, x_train, y_train, 'n_neighbors', [i for i in range(1,42)])
# %% grid search leads to below parameters for score of 0.82
tuned_knn = KNeighborsClassifier(n_neighbors=3)
plot_vc(tuned_knn, x_train, y_train, 'p', [i for i in np.linspace(0,5,10)])
p_tuned_knn = KNeighborsClassifier(n_neighbors=3,p=5)
# %% the limit of score seems to be 96.7
check_cv_score(p_tuned_knn, x_train, y_train)
# %% nonetheless I still want to test the ball & kd trees
leaf_params = knn_leaf_params
grid_search(tuned_knn, leaf_params, x_train, y_train)
# %% it seems these models are running into the same score limit..
leaf_knn = KNeighborsClassifier(algorithm='ball_tree',n_neighbors=3,p=5)
plot_vc(leaf_knn, x_train, y_train, 'leaf_size', [i for i in range(1,42)])
# %%
plot_lc(p_tuned_knn, x_train, y_train)
# %% 
# # [DT MODEL]
# #
# %% we start by doing a basic grid search of general params
dtree = DecisionTreeClassifier()
params = dtree_params
print(params)
grid_results = grid_search(dtree, params, x_train, y_train)
grid_results
# %% this model has the best score but it seems to be overfitted
tuned_dtree = DecisionTreeClassifier(criterion='gini',max_depth=29,splitter='random')
plot_lc(tuned_dtree, x_train, y_train)
# %%
plot_vc(dtree, x_train, y_train, 'criterion',['entropy','gini'])
# %%
plot_vc(dtree, x_train, y_train, 'splitter',['random','best'])
# %%
plot_vc(DecisionTreeClassifier(criterion='gini'), x_train, y_train, 'max_depth',[i for i in range(1,42)])
# %% 
tuned_dtree = DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=5)
plot_lc(tuned_dtree, x_train, y_train, ylim=(0.8,1.05))
# %% 
dtree = DecisionTreeClassifier()
params = dtree_extended_params
print(params)
grid_results = grid_search(dtree, params, x_train, y_train)
grid_results
# %% pretty good score here, time to check LC
tuned_gini_dtree = DecisionTreeClassifier(criterion='gini',max_depth=15,min_samples_split=5)
plot_lc(tuned_gini_dtree, x_train, y_train, ylim=(0.72,1.08))
# %% 
plot_vc(tuned_gini_dtree, x_train, y_train, 'min_samples_split',[i for i in range(1,30)])
# %% promising early results with gini & min_samples tweaked..
minsplit_gini_dtree = DecisionTreeClassifier(criterion='gini',max_depth=15,min_samples_split=30)
plot_lc(minsplit_gini_dtree, x_train, y_train, ylim=(0.25,1.05))
# %% 
# [SVM MODEL]
#
# %% we start by doing a grid search
svm = SVC(kernel='poly')
params = svm_params 
print(params)
grid_results = grid_search(svm, params, x_train, y_train)
grid_results
# %%
plot_vc(svm, x_train, y_train, 'C',[i for i in np.linspace(0,2,100)])
# %%
plot_lc(svm, x_train, y_train)
# %%
tuned_svm = SVC(kernel='linear')
plot_lc(tuned_svm, x_train, y_train,ylim=(0.85,1.07))
check_cv_score(tuned_svm, x_train, y_train)
# %% we would still like to investigate other kernels, expecially poly
poly_svm = SVC(kernel='poly')
plot_vc(poly_svm, x_train, y_train, 'degree', [i for i in range(1,5)])
# %% we establish degree 3 for the poly func, but LC looks pretty back..
tuned_poly_svm=SVC(kernel='poly',degree=3)
plot_lc(tuned_poly_svm, x_train, y_train)
plot_vc(poly_svm, x_train, y_train, 'C', [i for i in np.linspace(0,1,50)])
params = { 'C': [i for i in np.linspace(0,1,10)],
       'coef0': [i for i in np.linspace(0,1,10)],
       'gamma': [i for i in np.linspace(0,1,10)]
}
grid_results = grid_search(poly_svm, params, x_train, y_train)
grid_results
# %% we arrive at a fully tuned poly model
tuned_poly_svm=SVC(kernel='poly',degree=3,gamma=1.0/9.0)
plot_lc(tuned_poly_svm, x_train, y_train)
# %% still sucks, going back to linear
tuned_poly_svm=SVC(kernel='linear')
plot_lc(tuned_poly_svm, x_train, y_train)
# %% 
# [NN MODEL]
#
# # %% 
mlp = MLPClassifier()
params = nn_params 
print(params)
# grid_results = grid_search(mlp, params, x_train, y_train)
# grid_results
# %% both relu & tanh models show promise, compare them both
# relu_mlp = MLPClassifier(activation='relu',hidden_layer_sizes=(100,))
# plot_vc(relu_mlp, x_train, y_train, 'alpha', [i for i in np.linspace(0,2,24)])
tanh_mlp = MLPClassifier(activation='tanh',hidden_layer_sizes=(50,50))
# plot_vc(tanh_mlp, x_train, y_train, 'alpha', [i for i in np.linspace(0,2,24)])
# %%
plot_lc(tanh_mlp, x_train, y_train)
# %%
# # %% tanh looks slightly better than relu
# # tuned_relu_mlp = MLPClassifier(activation='relu',hidden_layer_sizes=(100,),alpha=0.087)
# # tuned_tanh_mlp = MLPClassifier(activation='tanh',hidden_layer_sizes=(50,50),alpha=0.087)
# # check_cv_score(tuned_relu_mlp, x_train, y_train)
# # check_cv_score(tuned_tanh_mlp, x_train, y_train)
# # plot_lc(tuned_relu_mlp, x_train, y_train)
# # plot_lc(tuned_tanh_mlp, x_train, y_train)
# # %% let's see if we can improve the performance by tuning init
# # plot_vc(tuned_relu_mlp, x_train, y_train, 'learning_rate_init', [i for i in np.linspace(0,10,100)])
# # plot_vc(tuned_tanh_mlp, x_train, y_train, 'learning_rate_init', [i for i in np.linspace(0,10,100)])
# # %% tuning the init didn't really do much..
# # tuned_init_relu_mlp = MLPClassifier(activation='relu',hidden_layer_sizes=(100,),alpha=0.087,learning_rate_init=0.101)
# # tuned_init_tanh_mlp = MLPClassifier(activation='tanh',hidden_layer_sizes=(50,50),alpha=0,learning_rate_init=0.101)
# # check_cv_score(tuned_init_relu_mlp, x_train, y_train)
# # check_cv_score(tuned_init_tanh_mlp, x_train, y_train)
# # plot_lc(tuned_init_relu_mlp, x_train, y_train)
# # plot_lc(tuned_init_tanh_mlp, x_train, y_train)
# # %%
# # params = nn_learning_params
# # tuned_relu_mlp = MLPClassifier(activation='relu',hidden_layer_sizes=(100,),alpha=0.087)
# tuned_tanh_mlp = MLPClassifier(activation='tanh',hidden_layer_sizes=(50,50),alpha=0.087)
# check_cv_score(tuned_tanh_mlp, x_train, y_train)
# # %%
# # grid_results = grid_search(tuned_relu_mlp, params, x_train, y_train)
# # grid_results
# # %%
# # grid_results = grid_search(tuned_tanh_mlp, params, x_train, y_train)
# # grid_results
# # %%
# # tuned_relu_mlp = MLPClassifier(activation='relu',hidden_layer_sizes=(100,),alpha=0.087,max_iter=300)
# tuned_tanh_mlp = MLPClassifier(activation='tanh',hidden_layer_sizes=(50,50),alpha=0.087,max_iter=300)
# # check_cv_score(tuned_relu_mlp, x_train, y_train)
# check_cv_score(tuned_tanh_mlp, x_train, y_train)
# # %%
# # params = nn_adam_params
# # grid_results = grid_search(tuned_tanh_mlp, params, x_train, y_train)
# # grid_results
# # %%
# params = nn_adam_tuned_params
# grid_results = grid_search(tuned_tanh_mlp, params, x_train, y_train)
# grid_results
# # %%
# tuned_tanh_adam_mlp = MLPClassifier(activation='tanh',hidden_layer_sizes=(50,50),alpha=0.087,max_iter=300,beta_1=0.2,beta_2=0.9,epsilon=7e-8)
# check_cv_score(tuned_tanh_adam_mlp, x_train, y_train)
# # %%
# plot_vc(tuned_tanh_adam_mlp, x_train, y_train, 'epsilon', [1e-8,2e-8,3e-8,4e-8,5e-8,6e-8,7e-8,8e-8])
# # %% unfortunately all of the tweaking doesn't seem to have paid off..
# plot_lc(tuned_tanh_adam_mlp, x_train, y_train)
# # %%
# tuned_tanh_mlp = MLPClassifier(activation='tanh',hidden_layer_sizes=(50,50),alpha=0.087,max_iter=300)
# check_test_score(tuned_tanh_mlp, x_train, y_train, x_test, y_test)
# # %% 
# plot_vc(tuned_tanh_mlp, x_train, y_train, 'max_iter', [100,150,200,250,300,350,400,450,500])
# %% 
# [BOOSTING MODEL]
#
# # %% we start with basic gridsearch:
# adaboost = AdaBoostClassifier()
# params = adaboost_params
# grid_results = grid_search(adaboost, params, x_train, y_train)
# grid_results
# %% we find a decent setup, unfortunately it only gets 80 score
tuned_adaboost = AdaBoostClassifier(learning_rate=0.5,n_estimators=75)
check_cv_score(tuned_adaboost, x_train, y_train)
# %% by itself our DT model achieves 84.4 score
plot_lc(tuned_adaboost, x_train, y_train)
# %% by itself our DT model achieves 84.4 score
plot_vc(tuned_adaboost, x_train, y_train, 'n_estimators', [i for i in range(50,100)])
# %% by itself our DT model achieves 84.4 score
# entropy_dtree = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=36,
#                                        min_samples_leaf=1, min_samples_split=11)
# check_cv_score(entropy_dtree, x_train, y_train)
# # %% lets see wnat happens when we use our tuned DT as the base estimator
# tuned_dt_adaboost = AdaBoostClassifier(base_estimator=entropy_dtree,learning_rate=0.5,n_estimators=75)
# check_cv_score(tuned_dt_adaboost, x_train, y_train)
# # %% oddly enough we only get 83 score here
# plot_vc(tuned_dt_adaboost, x_train, y_train, 'n_estimators', [i for i in range(50,200)])
# # %% unfortunately the validation curve shows no real pattern here..
# # top 3 are 50, 98, & 75 so our original guess was spot on, however even #1 only got score of 84.6
# # one last test with different base estimators
# adaboost = AdaBoostClassifier()
# params = adaboost_estimators
# grid_results = grid_search(adaboost, params, x_train, y_train)
# grid_results
