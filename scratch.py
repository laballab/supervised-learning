# models to use: 
# dt
# nn
# boosting
# svm
# knn
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

# basic structure from:
# https://www.kaggle.com/jchen2186/machine-learning-with-iris-dataset
# %%
pd.set_option('display.max_rows', 100)
p = np.linspace(0,20,100)

# %%
plt.plot(p,np.sin(p))
plt.show()
# %%
data = pd.read_csv('data/iris.data')
# %%
# !!! EDA !!! 
data.head()
# %%
data.info()
# %%
data.describe()
# %%
data['class'].value_counts()
# %%
# pairplot = sns.pairplot(data, hue='class', markers='*')
# %%
# sepal_len = sns.violinplot(y='class', x='sepal_len', data=data, inner='quartile')
# %%
# sepal_width = sns.violinplot(y='class', x='sepal_width', data=data, inner='quartile')
# %%
# petal_len = sns.violinplot(y='class', x='petal_len', data=data, inner='quartile')
# %%
# petal_width = sns.violinplot(y='class', x='petal_width', data=data, inner='quartile')
# %%
sk.show_versions()
# %%
x = data.drop('class', axis=1)
y = data['class']
# %%
print(x.shape)
print(y.shape)
test = LogisticRegression()
# %%
test_ks = list(range(1,27))
scores = []
print('k | score')
for k in test_ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x,y)
    y_pred = knn.predict(x)
    score = metrics.accuracy_score(y, y_pred)
    scores.append(score)
    print('%s | %s' % (k,score))

# plt.plot(test_ks, scores)
# plt.xlabel('k')
# plt.ylabel('score')
# plt.title('k-Nearest-Neighbors')
# plt.show()
# %%
def get_score(model, x, y):
    model.fit(x, y)
    y_pred = knn.predict(x)
    return metrics.accuracy_score(y, y_pred)
# %%
logreg = LogisticRegression()
logreg.fit(x, y)
y_pred = logreg.predict(x)
print(metrics.accuracy_score(y, y_pred))
# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=5)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_test.value_counts())
# %%
test_ks = list(range(1,27))
scores = []
print('k | score')
for k in test_ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    score = metrics.accuracy_score(y_test, y_pred)
    scores.append(score)
    print('%s | %s' % (k,score))
# %%
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred))
# %%
dtree = DecisionTreeClassifier()
dtree.fit(x_train,y_train)
y_pred = knn.predict(x_test)
score = metrics.accuracy_score(y_test, y_pred)
print(score)
# %%
clf = GridSearchCV(dtree, {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random']
}, cv=5)
clf.fit(x_train, y_train)
reuslt_df = pd.DataFrame(clf.cv_results_)
reuslt_df[['params','mean_test_score']]
# %%
def grid_search_map(map, x, y):
    total_df = pd.DataFrame(columns=['model','params','mean_test_score'])
    for model in map:
        clf = GridSearchCV(model['model'], 
                           model['params'], cv=5)
        clf.fit(x, y)
        result_df = pd.DataFrame(clf.cv_results_)
        result_df = result_df[['params','mean_test_score']]
        result_df['model'] = model['model']
        total_df = pd.concat([total_df,result_df])

    return total_df.sort_values('mean_test_score',axis=0,ascending=False)
# %%
def grid_search(model, params, x, y):
    clf = GridSearchCV(model, params, cv=5)
    clf.fit(x, y)
    result_df = pd.DataFrame(clf.cv_results_)
    result_df = result_df[['params','mean_test_score']]
    result_df['model'] = model['model']
    return result_df.sort_values('mean_test_score',axis=0,ascending=False)
# %%
testTree = DecisionTreeClassifier()
testknn = KNeighborsClassifier()
model_map = [
    {   'model': testTree, 'params': {
            'criterion': ['gini', 'entropy'],
            'splitter' : ['best', 'random']
        }
    },
    {   'model': testknn, 'params': {
            'n_neighbors': [i for i in range(1,7)],
            'weights' : ['uniform', 'distance']
        }
    }
]
# %%
# grid_search_map(model_map, x_train, y_train)
# %% 

# def plot_lc(model, title, x, y, cv, train_sizes=np.linspace(.1, 1.0, 5), ylim=None):
#     fig, lc_plot = plt.subplots()
#     lc_plot.set_title(title)
#     lc_plot.set_ylim(*ylim) if ylim is not None else None
#     lc_plot.set_xlabel("Training examples")
#     lc_plot.set_ylabel("Score")

#     train_sizes, train_scores, test_scores, fit_times, _ = \
#         learning_curve(model, x, y, cv=cv, n_jobs=4,
#                        train_sizes=train_sizes,
#                        return_times=True)
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     fit_times_mean = np.mean(fit_times, axis=1)
#     fit_times_std = np.std(fit_times, axis=1)
#     print(cross_val_score(model, x, y, scoring='accuracy', cv=cv))

#     # Plot learning curve
#     lc_plot.grid()
#     lc_plot.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                          train_scores_mean + train_scores_std, alpha=0.1,
#                          color="r")
#     lc_plot.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                          test_scores_mean + test_scores_std, alpha=0.1,
#                          color="g")
#     lc_plot.plot(train_sizes, train_scores_mean, 'o-', color="r",
#                  label="Training score")
#     lc_plot.plot(train_sizes, test_scores_mean, 'o-', color="g",
#                  label="Cross-validation score")
#     lc_plot.legend(loc="best")
#     lc_plot

# %%
# def plot_vc(model, x, y, param_key, param_val, cv=5):
#     train_score, test_score = validation_curve(model, x, y, param_key, param_val, cv=cv, scoring="accuracy")
#     # Calculating mean and standard deviation of training score 
#     mean_train_score = np.mean(train_score, axis = 1) 
#     std_train_score = np.std(train_score, axis = 1) 
    
#     # Calculating mean and standard deviation of testing score 
#     mean_test_score = np.mean(test_score, axis = 1) 
#     std_test_score = np.std(test_score, axis = 1) 
    
#     # Plot mean accuracy scores for training and testing scores 
#     plt.plot(param_val, mean_train_score,  
#         label = "Training Score", color = 'b') 
#     plt.plot(param_val, mean_test_score, 
#     label = "Cross Validation Score", color = 'g') 
    
#     # Creating the plot 
#     plt.title("Validation Curve with ") 
#     plt.xlabel(param_key) 
#     plt.ylabel("Accuracy") 
#     plt.tight_layout() 
#     plt.legend(loc = 'best') 
#     plt.show()

# %%
lc_knn = KNeighborsClassifier(n_neighbors=13,weights='distance')
plot_lc(lc_knn, x_train, y_train, cv=10, ylim=(0.7,1.01))
# %%
plot_lc(testTree, x_train, y_train, cv=10, ylim=(0.7,1.01))

# %%
# def params_dict_to_pairs(params):
#     params_list = []
#     for param in params_obj:
#         pair = (param, params_obj[param])
#         params_list.append(pair)
#     return params_list


# %% [markdown]
# test case 1 - knn weight: UNIFORM
# the chart shows 3 & 4 are best score w/o overfitting
weights = ['uniform', 'distance']
model = KNeighborsClassifier(n_neighbors=3)
plot_vc(model, x_train, y_train, 'weights', weights)
# %% [markdown]
# # test case 2 - knn neighbors: 3
# the chart shows uniform is better it's not overfitting
n_neighbors = [i for i in range(1,42)]
model = KNeighborsClassifier()
plot_vc(model, x_train, y_train, 'n_neighbors', n_neighbors)
# %%
knn_model = KNeighborsClassifier(n_neighbors=3)
plot_lc(model, x_train, y_train, cv=10, ylim=(0.7,1.01))
print('knn | %s' % (get_score(knn, x_train, y_train)))
# %%
svn_model =  SVC()
svn_map = [
    {   'model': svn_model, 'params': {
            'C': [i for i in np.linspace(0,1,50)],
            'kernel' : ['linear', 'poly', 'rbf', 'sigmoid']
        }
    }
]
# print(grid_search_map(svn_map, x_train, y_train).to_string())
# %%
svn_linear_model =  SVC(kernel='linear')
# %%
C = [i for i in np.linspace(0.02,1,200)]
plot_vc(svn_model, x_train, y_train, 'C', C)
# %%
plot_lc(svn_linear_model, x_train, y_train, cv=10, ylim=(0.8,1.08))
# %%
svn_poly_model =  SVC(kernel='poly')
# %%
C = [i for i in np.linspace(0.02,1,200)]
plot_vc(svn_model, x_train, y_train, 'C', C)
# %% this model looks GREAT 
plot_lc(svn_poly_model, x_train, y_train, cv=10, ylim=(0.8,1.08))

# %% this model sucks 
svn_model =  SVC(C=0.8,kernel='poly')
plot_lc(svn_model, x_train, y_train, cv=10, ylim=(0.8,1.01))
# %% quite an improvement here
C = [i for i in range(1,42)]
plot_vc(svn_poly_model, x_train, y_train, 'degree', C)
# %% final SVN model
svn_final =  SVC(kernel='poly', degree=4)
plot_lc(svn_final, x_train, y_train, cv=10, ylim=(0.8,1.08))

# %%
