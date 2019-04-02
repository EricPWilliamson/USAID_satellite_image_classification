"""
Creates and saves a classification model from a set of pre-defined parameters that were discovered by comparing many
  algorithms, feature configurations, etc.
"""
import pandas as pd
import numpy as np
import pickle
from timeit import default_timer as timer
# machine learning
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import Perceptron
# from sklearn.linear_model import SGDClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# extra sklearn tools:
# from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
# # Disable all the warnings from sklearn:
# import warnings
# warnings.filterwarnings("ignore")

#Path to the best cluster population data:
MERGED_DATA_PATH = './USAID_africa/premerged/merged_csvs/merged_rbt5_with_birch_n250_rbt5_tr120pct.csv'
#Optimal balance tolerance:
BAL_TOL = 2
#Optimal classification algorithm:
OPT_MODEL = RandomForestClassifier(n_estimators=100)
#Output path for pickled classifier:
OUT_PATH = './USAID_africa/Models/classifier_RFC100_Birch250_20180830.pkl'


def balence_classes(df, btol):
    """Even up the support of the classes."""
    #Find the least supported class and muliply by the tolerance coefficient to get max_count:
    ccounts = df['classification'].value_counts()
    max_count = np.min(ccounts.values) * btol
    #Create a new dataframe with balenced support:
    newdf = pd.DataFrame(columns=df.columns.values)
    for x in df.groupby('classification'):
        if x[1].shape[0] > max_count:
            newdf = newdf.append(x[1].sample(max_count).reset_index(drop=True))
        else:
            newdf = newdf.append(x[1].reset_index(drop=True))
    return newdf.reset_index(drop=True)

###Load and prep training data:
master_df = pd.read_csv(MERGED_DATA_PATH)
#Remove the features not involved in the model:
master_df = master_df.drop(['X','Y','pxlX','pxlY'], axis=1)
if BAL_TOL is not None:
    #Drop some samples to balance out the distribution of classes:
    master_df = balence_classes(master_df, BAL_TOL)
X = master_df.drop('classification', axis=1).values
Y = master_df['classification'].values

###Train and validate:
print('Fitting model using {} samples with {} features...'.format(X.shape[0],X.shape[1]))
model = OPT_MODEL
loo = LeaveOneOut()
print('Cross validating over {} splits...'.format(loo.get_n_splits(X)))
train_time = []
test_time = []
y_vals = []
y_preds = []
for train_index,val_index in loo.split(X):
    #Separate training/validation data:
    X_train = X[train_index]
    X_val = X[val_index]
    y_train = Y[train_index]
    y_val = Y[val_index]
    y_vals += list(y_val)
    #Train model:
    t1 = timer()
    model.fit(X_train,y_train)
    train_time += [timer()-t1]
    #Evaluate model:
    t1 = timer()
    y_preds += list(model.predict(X_val))
    test_time += [timer()-t1]
#Display results:
print('Avg training time: {:.2f}s,  Total prediction time for all samples: {:.2f}s'
      .format(np.mean(train_time), np.sum(test_time)))
print(classification_report(y_vals,y_preds))
# precision,recall,fscore,support = precision_recall_fscore_support(y_vals,y_preds)
# D = {'precision':precision,'recall':recall,'fscore':fscore,'support':support}
# score_df = pd.DataFrame(D,index=['farmland','settlement','wilderness'])

###Make a final fit using all the data and pickle it:
model.fit(X,Y)
with open(OUT_PATH,'wb') as pfile:
    pickle.dump(model,pfile,pickle.HIGHEST_PROTOCOL)
print('Final model pickled.')

print('Done')
