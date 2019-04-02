"""
Takes merged USAID and Sentinel-2 data to create a model. The model correlates Sentinel-2 spectral content to the
  classifications from the USAID data.


!!Add the outer fold
"""
import pandas as pd
from satellite_head import *
from timeit import default_timer as timer
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
# extra models:
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
# extra sklearn tools:
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
# Disable all the warnings from sklearn:
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.expand_frame_repr', False)

#Merged data folder:
MERGED_DATA_DIR = './USAID_africa/premerged/merged_csvs/'
# #Path of output file:
# OUTPATH = './USAID_africa/model_results/big_comparison_ssT_ns3_pbt5.csv'
# #Other params:
# N_SPLITS = 3
# BALANCE_CLASSES = False
# BALANCE_TOL = 2
# USE_SS = True
########################################################################################################################

def usaid_eval_model_acc(train_df,model,n_splits):
    """Train and validate the given model over multiple folds."""
    ###Prep train/validation data:
    X_trial = train_df.drop('classification', axis=1)
    Y_trial = train_df['classification']

    ###Test each model for each inner fold:
    kf_inner = KFold(n_splits=n_splits)
    acc = []
    train_time = []
    test_time = []
    idx = X_trial.index
    for train_index, val_index in kf_inner.split(X_trial):
        #Separate training/validation data:
        X_train,X_val = X_trial.loc[idx[train_index]],X_trial.loc[idx[val_index]]
        y_train,y_val = Y_trial[idx[train_index]],Y_trial[idx[val_index]]
        #Train model:
        t1 = timer()
        model.fit(X_train,y_train)
        train_time += [timer()-t1]
        #Evaluate model:
        t1 = timer()
        acc += [model.score(X_val, y_val) * 100]
        test_time += [timer()-t1]
    return np.mean(acc), np.std(acc), np.mean(train_time), np.mean(test_time)

def usaid_eval_model(train_df,model,n_splits):
    """Train and validate the given model over multiple folds."""
    ###Prep train/validation data:
    X_trial = train_df.drop('classification', axis=1)
    Y_trial = train_df['classification']

    ###Test each model for each inner fold:
    kf_inner = KFold(n_splits=n_splits)
    train_time = []
    test_time = []
    y_vals = []
    y_preds = []
    idx = X_trial.index
    for train_index, val_index in kf_inner.split(X_trial):
        #Separate training/validation data:
        X_train,X_val = X_trial.loc[idx[train_index]],X_trial.loc[idx[val_index]]
        y_train,y_val = Y_trial[idx[train_index]],Y_trial[idx[val_index]]
        y_vals += list(y_val)
        #Train model:
        t1 = timer()
        model.fit(X_train,y_train)
        train_time += [timer()-t1]
        #Evaluate model:
        t1 = timer()
        y_preds += list(model.predict(X_val))
        test_time += [timer()-t1]
    print(classification_report(y_vals,y_preds))
    precision,recall,fscore,support = precision_recall_fscore_support(y_vals,y_preds)
    D = {'precision':precision,'recall': recall,'fscore': fscore,'support': support}
    score_df = pd.DataFrame(D,index=train_df.classification.unique())
    return score_df, np.mean(train_time), np.mean(test_time)

def standardize_feats(df):
    feat_cols = df.columns.values[1:]
    feat_mat = df[feat_cols].values
    feat_mat = StandardScaler().fit_transform(feat_mat)
    #Put mat back into df:
    df[feat_cols] = pd.DataFrame(feat_mat)
    return df

def balence_classes(df, btol):
    """Even up the support of the classes."""
    #Find the least supported class and muliply by the tolerance coefficient to get max_count:
    ccounts = df['classification'].value_counts()
    max_count = int(np.floor(np.min(ccounts.values) * btol))
    #Create a new dataframe with balenced support:
    newdf = pd.DataFrame(columns=df.columns.values)
    for x in df.groupby('classification'):
        if x[1].shape[0] > max_count:
            newdf = newdf.append(x[1].sample(max_count).reset_index(drop=True))
        else:
            newdf = newdf.append(x[1].reset_index(drop=True))
    return newdf.reset_index(drop=True)

def eval_multi(outpath, models, n_splits, balance_tol, use_ss, two_class=False):
    ###Load merged data:
    file_list = os.listdir(MERGED_DATA_DIR)
    scores_df = pd.DataFrame(None)
    for f,filename in enumerate(file_list):
        t1 = timer()
        print('\n==={} of {}: {}'.format(f+1, len(file_list), filename))
        master_df = pd.read_csv(MERGED_DATA_DIR+filename)

        ###Prep dataframe:
        #Remove the features not involved in the model:
        master_df = master_df.drop(['X','Y','pxlX','pxlY'], axis=1)
        if use_ss:
            #Standardize the features:
            master_df = standardize_feats(master_df)
        if two_class:
            #Merge farmland and wilderness classes into one:
            master_df.loc[master_df.classification=='farmland', 'classification'] = 'other'
            master_df.loc[master_df.classification=='wilderness','classification'] = 'other'
        if balance_tol is not None:
            #Drop some samples to balance out the distribution of classes:
            master_df = balence_classes(master_df, balance_tol)

        #Within this fold, we will find the following variables!!:
        acc_best = 0
        model_best = 'None'
        features_best = 'None'

        #Test all models:
        print('Beginning model evaluations ({} models, {} samples, {} features, {} CV folds):'.format(len(models), master_df.shape[0], master_df.shape[1]-1, n_splits))
        for model in models:
            model_name = model.__class__.__name__
            if model_name=='RandomForestClassifier':
                model_name = 'RFC_{}'.format(model.n_estimators)
            elif model_name=='ExtraTreesClassifier':
                model_name = 'ETC_{}'.format(model.n_estimators)
            model_name_ex = model_name + (30-len(model_name))*' '

            # ###Simple accuracy measure:
            # avg_acc,acc_stdev,avg_train_time,avg_test_time = usaid_eval_model_acc(master_df,model,n_splits)
            # print("Model: {:} --- Score: {:.3f}% ± {:.3f}  - Training: {:.2f}s - Testing: {:.2f}s".format(model_name,avg_acc,acc_stdev,avg_train_time,avg_test_time))

            ###Detailed scoring by class:
            sdf,avg_train_time,avg_test_time = usaid_eval_model(master_df,model,n_splits)
            print("Model: {:} --- Training: {:.2f}s - Testing: {:.2f}s".format(model_name_ex,avg_train_time,avg_test_time))
            #Put scores into output df:
            score_list = [filename,model_name]
            score_cols = ['Data','Model']
            for cat in master_df.classification.unique():
                score_list += list(sdf.loc[cat].values[:-1])
                score_cols += list(cat + '_' + sdf.columns.values[:-1])
            score_list += [avg_train_time, avg_test_time]
            score_cols += ['Train_time', 'Test_time']
            scores_df = scores_df.append(pd.DataFrame(np.array(score_list).reshape(1,-1), columns=score_cols))

            # if avg_acc > acc_best:
            #     acc_best = avg_acc
            #     acc_best_stdev = acc_stdev
            #     model_best_name = model_name
            #     model_best = model
            #     # features_best = features_desc

        print('Total eval time: {:.1f}s'.format(timer()-t1))
        #Save scores to csv:
        scores_df.to_csv(outpath, index=False)

####################################################################################################################
##############################################  TESTS ##############################################################
####################################################################################################################

###Test all of these models (all from sklearn):
# models = [MLPClassifier(),
#           GaussianProcessClassifier(),
#           AdaBoostClassifier(),
#           QuadraticDiscriminantAnalysis(),
#           LogisticRegression(),
#           SVC(),
#           KNeighborsClassifier(),
#           GaussianNB(),
#           Perceptron(),
#           LinearSVC(),
#           SGDClassifier(),
#           DecisionTreeClassifier(),
#           RandomForestClassifier(n_estimators=100)]
# #Just use reasonably good models:
# models = [MLPClassifier(),
#           AdaBoostClassifier(),
#           QuadraticDiscriminantAnalysis(),
#           LogisticRegression(),
#           SVC(),
#           KNeighborsClassifier(),
#           GaussianNB(),
#           Perceptron(),
#           LinearSVC(),
#           SGDClassifier(),
#           DecisionTreeClassifier(),
#           RandomForestClassifier(n_estimators=100)]
#Only use superior models:
# models = [AdaBoostClassifier(),
#           RandomForestClassifier(n_estimators=500),
#           RandomForestClassifier(n_estimators=1000),
#           ExtraTreesClassifier(n_estimators=500),
#           ExtraTreesClassifier(n_estimators=1000)]

# eval_multi('./USAID_africa/model_results/tight_comparison_ssT_ns30_pbt5.csv',
#            n_splits=30,
#            balance_tol=None,
#            use_ss=True)
#
# eval_multi('./USAID_africa/model_results/tight_comparison_ssF_ns30_pbt5.csv',
#            n_splits=30,
#            balance_tol=None,
#            use_ss=False)
#
# eval_multi('./USAID_africa/model_results/tight_comparison_ssF_ns30_bt2.csv',
#            n_splits=30,
#            balance_tol=2,
#            use_ss=False)
#
# eval_multi('./USAID_africa/model_results/tight_comparison_ssT_ns30_bt2.csv',
#            n_splits=30,
#            balance_tol=2,
#            use_ss=True)

OUTDIR = './USAID_africa/model_results/'
TEST_MODELS = [GaussianNB(),
               AdaBoostClassifier(),
               RandomForestClassifier(n_estimators=100),
               RandomForestClassifier(n_estimators=500),
               RandomForestClassifier(n_estimators=1000),
               RandomForestClassifier(n_estimators=2000),
               ExtraTreesClassifier(n_estimators=100),
               ExtraTreesClassifier(n_estimators=500),
               ExtraTreesClassifier(n_estimators=1000),
               ExtraTreesClassifier(n_estimators=2000)]

TWO_CLASS = True
bt_list = [1,2,3,None]
ssc_list = ['T']
ns_list = [100]
desc = 'twoclasses_corrected'


#Roughly estimate how long these tests will take:
eta = len(TEST_MODELS) * len(ssc_list) * len(bt_list) * sum(ns_list) * 1.8/60
print('ETA: {:.1f}mins'.format(eta))
#Evaluate every combination of settings:
for bt in bt_list:
    for ssc in ssc_list:
        for ns in ns_list:
            #Determine output filename:
            if bt is None:
                output_fn = '{}_ss{}_ns{}_pbt5.csv'.format(desc,ssc,ns)
            else:
                output_fn = '{}_ss{}_ns{}_bt{}pct.csv'.format(desc,ssc,ns,int(bt*100))
            #Check if this filename already exists:
            if os.path.exists(OUTDIR+output_fn):
                print('--{} already exists.'.format(output_fn))
            else:
                print('\n++++Working on '+output_fn+'++++')
                eval_multi(OUTDIR+output_fn,
                           TEST_MODELS,
                           n_splits=ns,
                           balance_tol=bt,
                           use_ss=(ssc=='T'),
                           two_class=TWO_CLASS)

myalert()
print('DONE')
