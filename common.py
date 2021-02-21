# %%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn.metrics import accuracy_score
# %% PARAM MAPPINGS
dtree_params = {
    'criterion': ['gini', 'entropy'],
    'splitter' : ['best', 'random'],
    'max_features' : [None, "sqrt", "log2"],
    'max_depth' : [i for i in range(1,42)]
}
dtree_extended_params = {
    'criterion': ['gini', 'entropy'],
    'splitter' : ['best', 'random'],
    'max_features' : [None, "sqrt", "log2"],
    'max_depth' : [i for i in range(1,42)],
    'min_samples_split' : [i for i in range(1,7)],
    'min_samples_leaf' : [i for i in range(1,7)]
}
dtree_sample_split = {
    'criterion': ['gini', 'entropy'],
    'splitter' : ['best', 'random'],
    'max_features' : [None, "sqrt", "log2"],
    'max_depth' : [i for i in range(1,42)],
    'min_samples_split' : [i for i in range(1,42)]
}
svm_params = {
    'C': [i for i in np.linspace(0,1,50)],
    'kernel' : ['linear', 'poly', 'rbf', 'sigmoid']
}
knn_params = {
    'n_neighbors': [i for i in range(1,42)],
    'weights' : ['uniform', 'distance']
}
nn_params = {
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'hidden_layer_sizes': [(100,), (50,50), (30,30,30),(10,10,10,10,10,10,10,10,10)]
}
# %%
def plot_vc(model, x, y, param_key, param_val, cv=5):
    print(cross_val_score(model, x, y, scoring='accuracy', cv=cv))
    train_score, test_score = validation_curve(model, x, y, param_key, param_val, cv=cv, scoring="accuracy", n_jobs=-1)
    mean_train_score = np.mean(train_score, axis = 1) 
    std_train_score = np.std(train_score, axis = 1) 
    mean_test_score = np.mean(test_score, axis = 1) 
    std_test_score = np.std(test_score, axis = 1) 

    params = {}
    params[param_key] = param_val
    print(grid_search(model, params, x, y, cv))

    # Creating the plot 
    plt.plot(param_val, mean_train_score,  
        label = "Training Score", color = 'b') 
    plt.plot(param_val, mean_test_score, 
        label = "Cross Validation Score", color = 'g') 
    plt.title("Validation Curve with %s" % (model)) 
    plt.xlabel(param_key) 
    plt.ylabel("Accuracy") 
    plt.tight_layout() 
    plt.legend(loc = 'best') 
    plt.grid()
    plt.show()
# %%
def plot_lc(model, x, y, cv=5, train_sizes=np.linspace(.1, 1.0, 10), ylim=None):
    print(cross_val_score(model, x, y, scoring='accuracy', cv=cv))
    fig, lc_plot = plt.subplots()
    lc_plot.set_title("Learning curve for %s" % (model))
    lc_plot.set_ylim(*ylim) if ylim is not None else None
    lc_plot.set_xlabel("Training examples")
    lc_plot.set_ylabel("Score")

    train_sizes, train_scores, test_scores = \
        learning_curve(model, x, y, cv=cv, n_jobs=8,
                       train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    lc_plot.grid()
    lc_plot.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    lc_plot.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    lc_plot.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    lc_plot.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    lc_plot.legend(loc="best")
    lc_plot
# %%
def grid_search(model, params, x, y, cv=5):
    clf = GridSearchCV(estimator=model, param_grid=params, cv=cv, n_jobs=-1)
    clf.fit(x, y)
    result_df = pd.DataFrame(clf.cv_results_)
    result_df = result_df[['params','mean_test_score','rank_test_score']]
    return result_df.sort_values('mean_test_score',axis=0,ascending=False)
# %%
def check_test_score(model, x, y, x_test, y_test):
    model.fit(x, y)
    y_pred = model.predict(x_test)
    return accuracy_score(y_test, y_pred)
# %%
def check_cv_score(model, x, y, cv=5):
    fold_scores = cross_val_score(model, x, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print('score: %s\navg: %s' % (fold_scores, sum(fold_scores)/len(fold_scores)))