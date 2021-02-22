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

pd.options.display.max_colwidth = 150
pd.options.display.max_rows = 200

data = pd.read_csv('data/games/Video_games_esrb_rating.csv')
test = pd.read_csv('data/games/test_esrb.csv')

x_train = data.drop(['esrb_rating','title'], axis=1)
y_train = data['esrb_rating']
x_test  = test.drop(['esrb_rating','title'], axis=1)
y_test  = test['esrb_rating']
# %% [markdown]
# dt
# nn
# boosting
# svm
# knn
# %% load data
# %% EDA 
print(data.head)
print(data.columns)
# %%
data['esrb_rating'].value_counts()
test['esrb_rating'].value_counts()
# %%
# %% 
# [KNN MODEL]
#
# %% basic grid search on untuned model
model = KNeighborsClassifier()
print(knn_params)
grid_search(model, knn_params, x_train, y_train)
# %% grid search leads to below parameters for score of 0.82
tuned_model = KNeighborsClassifier(n_neighbors=7,weights='distance',algorithm='brute')
check_cv_score(tuned_model, x_train, y_train)
# %% this lc looks pretty awful tho
plot_lc(tuned_model, x_train, y_train, ylim=(0.65,1.05))
# %% neighbors definitely best where it is 
plot_vc(tuned_model, x_train, y_train, 'n_neighbors', [i for i in range(1,42)])
# %% distance weights seems to be overfitted..
plot_vc(tuned_model, x_train, y_train, 'weights', ['distance','uniform'])
# %% trying n_neighbors with uniform to check
tuned_model = KNeighborsClassifier(weights='uniform',algorithm='brute')
plot_vc(tuned_model, x_train, y_train, 'n_neighbors', [i for i in range(1,42)])
# %% we see big difference, lines look better for n=35, lets check ls
tuned_model = KNeighborsClassifier(n_neighbors=35,weights='uniform',algorithm='brute')
plot_lc(tuned_model, x_train, y_train, ylim=(0.45,1.05))
# %% looks great but could be more accurate..
tuned_model = KNeighborsClassifier(n_neighbors=20,weights='uniform',algorithm='brute')
plot_lc(tuned_model, x_train, y_train, ylim=(0.45,1.05))
# %% accuracy for cross validation doesn't seem to be improving any further
tuned_model = KNeighborsClassifier(n_neighbors=10,weights='uniform',algorithm='brute')
plot_lc(tuned_model, x_train, y_train, ylim=(0.45,1.05))
# %% final model looks pretty good, test score 0.798
final_model = KNeighborsClassifier(n_neighbors=20,weights='uniform',algorithm='brute')
plot_lc(final_model, x_train, y_train, ylim=(0.58,0.9))
check_cv_score(final_model, x_train, y_train)
check_test_score(final_model, x_train, y_train, x_test, y_test)
# %% 
# [DT MODEL]
#
# %% we start by doing a grid search
dtree = DecisionTreeClassifier()
params = dtree_params
print(params)
grid_results = grid_search(dtree, params, x_train, y_train)
grid_results
# %% this model has the best score but it seems to be overfitted
tuned_dtree = DecisionTreeClassifier(criterion='entropy', splitter='random')
plot_lc(tuned_dtree, x_train, y_train)
# %%
plot_vc(dtree, x_train, y_train, 'criterion',['entropy','gini'])
# # %%
plot_vc(dtree, x_train, y_train, 'splitter',['random','best'])
# %%
plot_vc(DecisionTreeClassifier(criterion='entropy'), x_train, y_train, 'max_depth',[i for i in range(1,42)])
# %% we can do some pruning, very interesting lc with flip flop
tuned_dtree = DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=5)
plot_lc(tuned_dtree, x_train, y_train, ylim=(0.66,0.78))
# %% still only 70 so we are going to optimize further..
dtree = DecisionTreeClassifier()
params = dtree_extended_params
print(params)
# grid_results = grid_search(dtree, params, x_train, y_train)
grid_results
# %% promising early results with gini & min_samples tweaked..
gini_dtree = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=24,
                                     min_samples_leaf=1, min_samples_split=6)
plot_lc(gini_dtree, x_train, y_train, ylim=(0.70,0.98))
# %% amazing results after the frist tweaking of the min_samples 
params = dtree_sample_split
print(params)
grid_results = grid_search(dtree, params, x_train, y_train)
grid_results
# %% after further tweaking interesting results with entropy & min_samples
entropy_dtree = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=36,
                                     min_samples_leaf=1, min_samples_split=11)
plot_lc(entropy_dtree, x_train, y_train, ylim=(0.66,1.01))
# %% 2 very good models with gini & entropy, let's plot validation curves to check 
plot_vc(gini_dtree, x_train, y_train, 'min_samples_split',[i for i in range(1,42)])
# %%
plot_vc(entropy_dtree, x_train, y_train, 'min_samples_split',[i for i in range(1,42)])
# %% between the 2 above we can establish that grid search gave us some pretty good results
# there is another parameter to tweak - minimal cost complexity pruning
# %%
plot_vc(gini_dtree, x_train, y_train, 'ccp_alpha',[i for i in np.linspace(0,1,100)])
# %%
# balanced_weights = {}
# for col in x_train.columns:
#     balanced_weights[col] = 1
# print(balanced_weights)
# violent_weights = {}
# for col in x_train.columns:
#     if(('blood'   in col or 'violence' in col) and not 
#        ('fantasy' in col or 'cartoon'  in col or 'mild' in col)):
#         violent_weights[col] = 1
#     else:
#         violent_weights[col] = 0.5
# print(violent_weights)
# adult_weights = {}
# for col in x_train.columns:
#     if('mature' in col or 'sex' in col or 'nudity' in col):
#         adult_weights[col] = 1
#     else:
#         adult_weights[col] = 0.5
# print(adult_weights)
# weights = [balanced_weights, violent_weights, adult_weights]
# %% no clear winner..
check_cv_score(entropy_dtree, x_train, y_train)
# %% going with entropy for final model, score 0.844
check_test_score(entropy_dtree, x_train, y_train, x_test, y_test)
# %% 
# [SVM MODEL]
#
# # %% we start by doing a grid search
svm = SVC()
params = svm_params 
print(params)
grid_results = grid_search(svm, params, x_train, y_train)
grid_results
# %%
tuned_svm = SVC(C=1, kernel='rbf')
# %%
plot_vc(tuned_svm, x_train, y_train, 'gamma',[i for i in np.linspace(0,1,100)])
# %% the default parameters work very well for this model (C=1, kernel='rbf')
plot_lc(tuned_svm, x_train, y_train)
# %% default model achieves whopping 0.908
check_cv_score(tuned_svm, x_train, y_train)
# %% we would still like to investigate other kernels, expecially poly
poly_svm = SVC(C=1, kernel='poly')
plot_vc(poly_svm, x_train, y_train, 'degree', [i for i in range(1,5)])
# %% we  establish the best degree of the poly func, now we tune with that degree
poly_svm = SVC(kernel='poly', degree=1)
plot_vc(poly_svm, x_train, y_train, 'C', [i for i in np.linspace(0,1,50)])
params = { 'C': [i for i in np.linspace(0,1,10)],
       'coef0': [i for i in np.linspace(0,1,10)],
       'gamma': [i for i in np.linspace(0,1,10)]
}
grid_results = grid_search(poly_svm, params, x_train, y_train)
grid_results
# %% we arrive a tuned poly model
tuned_poly_svm = SVC(kernel='poly', degree=1, coef0=0.111)
plot_vc(tuned_poly_svm, x_train, y_train, 'gamma', [i for i in np.linspace(0,1,50)])
# %% unfortunately it's not as good as the original svm model..
plot_lc(tuned_svm, x_train, y_train)
# %%
check_cv_score(tuned_svm, x_train, y_train)
# %%
plot_vc(tuned_svm, x_train, y_train, 'kernel',['linear','poly','rbf','sigmoid','precoputed'])
# %% 
# [NN MODEL]
#
# %% 
mlp = MLPClassifier()
params = nn_params 
print(params)
grid_results = grid_search(mlp, params, x_train, y_train)
grid_results
# %% both relu & tanh models show promise, compare them both
relu_mlp = MLPClassifier(activation='relu',hidden_layer_sizes=(100,))
plot_vc(relu_mlp, x_train, y_train, 'alpha', [i for i in np.linspace(0,2,24)])
tanh_mlp = MLPClassifier(activation='tanh',hidden_layer_sizes=(50,50))
plot_vc(tanh_mlp, x_train, y_train, 'alpha', [i for i in np.linspace(0,2,24)])
# %% tanh looks slightly better than relu
tuned_relu_mlp = MLPClassifier(activation='relu',hidden_layer_sizes=(100,),alpha=0.087)
tuned_tanh_mlp = MLPClassifier(activation='tanh',hidden_layer_sizes=(50,50),alpha=0.087)
check_cv_score(tuned_relu_mlp, x_train, y_train)
check_cv_score(tuned_tanh_mlp, x_train, y_train)
plot_lc(tuned_relu_mlp, x_train, y_train)
plot_lc(tuned_tanh_mlp, x_train, y_train)
# %% let's see if we can improve the performance by tuning init
plot_vc(tuned_relu_mlp, x_train, y_train, 'learning_rate_init', [i for i in np.linspace(0,10,100)])
plot_vc(tuned_tanh_mlp, x_train, y_train, 'learning_rate_init', [i for i in np.linspace(0,10,100)])
# %% tuning the init didn't really do much..
tuned_init_relu_mlp = MLPClassifier(activation='relu',hidden_layer_sizes=(100,),alpha=0.087,learning_rate_init=0.101)
tuned_init_tanh_mlp = MLPClassifier(activation='tanh',hidden_layer_sizes=(50,50),alpha=0,learning_rate_init=0.101)
check_cv_score(tuned_init_relu_mlp, x_train, y_train)
check_cv_score(tuned_init_tanh_mlp, x_train, y_train)
plot_lc(tuned_init_relu_mlp, x_train, y_train)
plot_lc(tuned_init_tanh_mlp, x_train, y_train)
# %%
params = nn_learning_params
tuned_relu_mlp = MLPClassifier(activation='relu',hidden_layer_sizes=(100,),alpha=0.087)
tuned_tanh_mlp = MLPClassifier(activation='tanh',hidden_layer_sizes=(50,50),alpha=0.087)
check_cv_score(tuned_tanh_mlp, x_train, y_train)
# %%
grid_results = grid_search(tuned_relu_mlp, params, x_train, y_train)
grid_results
# %%
grid_results = grid_search(tuned_tanh_mlp, params, x_train, y_train)
grid_results
# %%
# tuned_relu_mlp = MLPClassifier(activation='relu',hidden_layer_sizes=(100,),alpha=0.087,max_iter=300)
tuned_tanh_mlp = MLPClassifier(activation='tanh',hidden_layer_sizes=(50,50),alpha=0.087,max_iter=300)
plot_lc(tuned_tanh_mlp, x_train, y_train)
# check_cv_score(tuned_relu_mlp, x_train, y_train)
# %%
check_cv_score(tuned_tanh_mlp, x_train, y_train)
# %%
params = nn_adam_params
grid_results = grid_search(tuned_tanh_mlp, params, x_train, y_train)
grid_results
# %%
params = nn_adam_tuned_params
grid_results = grid_search(tuned_tanh_mlp, params, x_train, y_train)
grid_results
# %%
tuned_tanh_adam_mlp = MLPClassifier(activation='tanh',hidden_layer_sizes=(50,50),alpha=0.087,max_iter=300,beta_1=0.2,beta_2=0.9,epsilon=7e-8)
check_cv_score(tuned_tanh_adam_mlp, x_train, y_train)
# %%
plot_vc(tuned_tanh_adam_mlp, x_train, y_train, 'epsilon', [1e-8,2e-8,3e-8,4e-8,5e-8,6e-8,7e-8,8e-8])
# %% unfortunately all of the tweaking doesn't seem to have paid off..
plot_lc(tuned_tanh_adam_mlp, x_train, y_train)
# %%
tuned_tanh_mlp = MLPClassifier(activation='tanh',hidden_layer_sizes=(50,50),alpha=0.087,max_iter=300)
check_test_score(tuned_tanh_mlp, x_train, y_train, x_test, y_test)
# %%
plot_vc(tuned_tanh_mlp, x_train, y_train, 'max_iter', [100,150,200,250,300,350,400,450,500])
# %% 
# [BOOSTING MODEL]
#
# %% we start with basic gridsearch:
adaboost = AdaBoostClassifier()
params = adaboost_params
grid_results = grid_search(adaboost, params, x_train, y_train)
grid_results
# %% we find a decent setup, unfortunately it only gets 80 score
tuned_adaboost = AdaBoostClassifier(learning_rate=0.5,n_estimators=75)
plot_lc(tuned_adaboost, x_train, y_train, ylim=(0.67,0.9))
# check_cv_score(tuned_adaboost, x_train, y_train)
# %% by itself our DT model achieves 84.4 score
entropy_dtree = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=36,
                                       min_samples_leaf=1, min_samples_split=11)
check_cv_score(entropy_dtree, x_train, y_train)
# %% lets see wnat happens when we use our tuned DT as the base estimator
tuned_dt_adaboost = AdaBoostClassifier(base_estimator=entropy_dtree,learning_rate=0.5,n_estimators=75)
check_cv_score(tuned_dt_adaboost, x_train, y_train)
# %% oddly enough we only get 83 score here
plot_vc(tuned_dt_adaboost, x_train, y_train, 'n_estimators', [i for i in range(50,200)])
# %% unfortunately the validation curve shows no real pattern here..
# top 3 are 50, 98, & 75 so our original guess was spot on, however even #1 only got score of 84.6
# one last test with different base estimators
adaboost = AdaBoostClassifier()
params = adaboost_estimators
grid_results = grid_search(adaboost, params, x_train, y_train)
grid_results
# %%
check_test_score(tuned_adaboost, x_train, y_train, x_test, y_test)

# %%
