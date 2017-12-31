import pandas
from sklearn import model_selection
from sklearn.feature_selection import RFE
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe
X = array.iloc[:,:-1]
Y = array.iloc[:,8]
seed = 7
num_trees = 30
param_test1 = {'n_estimators':(1)}
#kfold = model_selection.KFold(n_splits=10, random_state=seed)
#model = GradientBoostingClassifier()
model -= AdaBoostClassifier()
#model = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10),
#param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#model = PassiveAggressiveClassifier(C=1.0, fit_intercept=True, n_iter=5, shuffle=False, verbose=0, loss='hinge', n_jobs=1, random_state=None, warm_start=False)
#results = model_selection.cross_val_score(model, X, Y, cv=kfold)
model.fit(X,Y)
#rfe = RFE(model, 3)
#rfe = rfe.fit(X,Y)
# summarize the selection of the attributes
#print(rfe.support_)
#print(rfe.ranking_)
feature_importance=pandas.Series(model.feature_importances_ ,index=X.columns)
print(feature_importance)
#feature_importance.sort
#print(model.feature_importances_)
#print(model.coef_)
#print(model.estimators_)