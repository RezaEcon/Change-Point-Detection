#### Ensemble 1: Logistic0, SVM1, Decision Tree2, RandomForest3, Naive Bayes4
  ## EasyEns6,RUSBoost,BalancedBagging,BalancedRandomForestClassifier
Ensemble1 = 0
#### Ensemble 2: Logistic, SVM, Decision Tree,Random Forest,Naive Bayes
  ## CLF2: 10Logistic, SVM, KNN,Decision Tree,Random Forest,Naive Bayes
  ## EasyEns16,RUSBoost,BalancedBagging,BalancedRandomForestClassifier
Ensemble2 = 11

#### Recession-Expansion Labels VS  Changepoint labels (1)
Changepoint = 0
### window size
window_size = 6
### add main features or not 0!
original = 0
### Sample weight: Penalizing changepoints Way MORE
Penalty = 0
pen1, pen2, pen3=30,50,5
##Rolling window size of each moment TS:
moments_ws1 = 0
moments_ws1d = 3
moments_ws2 = 3
moments_ws2d = 3
moments_ws3 = 0
moments_ws3d = 0
moments_ws4 = 0
moments_ws4d = 0
##How many M1's added to each window:
M1_window = 0
M1_windowd = 2
M2_window = 2
M2_windowd = 2
M3_window = 0
M3_windowd = 0
M4_window = 0
M4_windowd = 0
#### how many lagged lables to pass as feature to 2nd stage
label_length = 1
recurrent = 1
#### Country = Canada0 US1 UK2
Country = 1
## For Canadian dataset: only Country level data:1; Only Regions:2; Whole:0
Region = 0
##### K-Fold Splits
n_splits = 5
##### Feature Selection threshold
perc = .1
Metr = 7

import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
#from sklearn.model_selection import cross_val_score
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from imblearn.ensemble import EasyEnsembleClassifier 
from imblearn.ensemble import RUSBoostClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from scipy.ndimage import shift
#import imblearn
#from imblearn.over_sampling import SMOTE, ADASYN
##############  DATA ############## 

if Country == 0:
    df = pd.read_csv(r"C:\Users\momas\OneDrive\Desktop\Thesis\Thesis\Data\CSV\Canada\balanced_can_md.csv")
    df2 = pd.read_csv(r'C:\Users\momas\OneDrive\Desktop\Thesis\Thesis\Data\CSV\Canada\RecessionDating_Monthly_1981_changepoint.csv')
    df3 = pd.DataFrame(df2, columns= ['target'])
    df4 = pd.read_csv(r'C:\Users\momas\OneDrive\Desktop\Thesis\Thesis\Data\CSV\Canada\Table_can_md_modif.csv')
    df4_Region = pd.DataFrame(df4, columns= ['Dummy_Can'])
    df4_Region = df4_Region.drop(df4_Region.index[[-1, -2, -3, -4, -5, -6]], axis = 0)
    DummyCan = df4_Region.to_numpy()
elif Country == 2:
    df = pd.read_csv(r"C:\Users\momas\OneDrive\Desktop\Thesis\Thesis\Data\CSV\UKMD_April_2022\UKMD_April_2022\balanced_uk_md.csv")
    df2 = pd.read_csv(r'C:\Users\momas\OneDrive\Desktop\Thesis\Thesis\Data\CSV\UKMD_April_2022\UKMD_April_2022\RecessionDating_Monthly_1998_changepoint_UK.csv')
    df3 = pd.DataFrame(df2, columns= ['target'])
elif Country == 1:
    df = pd.read_csv(r"C:\Users\momas\OneDrive\Desktop\Thesis\Thesis\Data\US\2023-02.csv")
    df2 = pd.read_csv(r"C:\Users\momas\OneDrive\Desktop\Thesis\Thesis\Data\US\USREC.csv")
    df2aux = pd.DataFrame.reset_index(pd.DataFrame(df2[1249:], columns= ['USREC']))
    df3 = pd.DataFrame(df2aux, columns= ['USREC'])

#adding labels to original data
df['target'] = df3
y = df['target']
###### Preparing Dataset for training and test
X = df.drop(['target'], axis=1)
X = X.drop(['Date'], axis=1)

#eliminating the features that are discontinued (NaN) in 2020
if Country == 0:
    X = X.drop(X.columns[[-1, -2, -3, -4, -5]], axis = 1)
elif Country == 1:
    ## due to missing values of old periods to make balanced dataset:
    X = X.drop(columns=['ACOGNO','ANDENOx','UMCSENTx','TWEXAFEGSMTHx'], axis = 1)
    X = X.drop(X.index[:43], axis=0)
    y = y.drop(y.index[:42],axis=0)
    ## due to missing values of most recent periods:
    X = X.drop(X.index[-2:],axis=0)
    y = y.drop(y.index[-3:],axis=0)
    ## Filling two missing data for April 2020 for CP3Mx, COMPAPFFx
    X.interpolate(method ='linear', limit_direction ='forward', inplace=True)
elif Country == 2:
    X = X.drop(X.columns[[0]], axis = 1)
    ## due to missing values:
    X = X.drop(['LIBOR_3mth'], axis=1)
    X = X.drop(X.index[-1],axis=0)
    y = y.drop(y.index[-1],axis=0)
    
## Eliminating last row due to assigning the first "window" to 
#the lable of the following period
#first 7 data -> label 8; hence, last 7 data -> No label exists for tomorrow
X = X.drop(X.index[-1],axis=0)

if Region != 0 and Country == 0:
    list_NotCan = []
    list_Can = []
    for i in range(len(X.columns)):
        if DummyCan[i] == 0:
            list_NotCan.append(i)
        if DummyCan[i] == 1:
            list_Can.append(i)
    if Region == 1:
        X = X.drop(X.columns[list_NotCan], axis = 1)
    if Region == 2:
        X = X.drop(X.columns[list_Can], axis = 1)
##transformed y
Sample_size = len(X)-window_size+1
y_aux = y.iloc[window_size:]
X_numpy = X.to_numpy()
y_auxnp = y_aux.to_numpy()
new_y = np.zeros((Sample_size))
new_y = y_auxnp
y_aux2 = y.iloc[window_size-1:-1]
new_yLAGnp = y_aux2.to_numpy()
new_yLAG = np.zeros(Sample_size)
new_yLAG[:] = new_yLAGnp[:]
##### Adding latent factor variables
from sklearn.decomposition import FactorAnalysis
my_fa = FactorAnalysis(n_components=5)
X_transformed = my_fa.fit_transform(X)
#### Adding Moments
M1 = X.rolling(moments_ws1).mean()
M1Diff = X.rolling(moments_ws1d).mean().diff()
M2 = X.rolling(moments_ws2).var()
M2Diff = X.rolling(moments_ws2d).var().diff()
M3 = X.rolling(moments_ws3).skew()
M3Diff = X.rolling(moments_ws3d).skew().diff()
M4 = X.rolling(moments_ws4).kurt()
M4Diff = X.rolling(moments_ws4d).kurt().diff()

###### Some weight initialization values
if Country == 0:
    w = {0:91, 1:9}
elif Country == 1:
    w = {0:89, 1:11}
if Changepoint != 0:
    ##taking difference between state today and last month(period)
    new_u = new_y - new_yLAG
    new_y = abs(new_u)
    w = {0:98.5, 1:1.5}

Sample_weight = np.ones(Sample_size)
X_all_window = np.zeros((Sample_size,(window_size*(len(X.columns)))))
M1_all = np.zeros((Sample_size,(M1_window*(len(X.columns)))))
M1D_all = np.zeros((Sample_size,(M1_windowd*(len(X.columns)))))
M2_all = np.zeros((Sample_size,(M2_window*(len(X.columns)))))
M2D_all = np.zeros((Sample_size,(M2_windowd*(len(X.columns)))))
M3_all = np.zeros((Sample_size,(M3_window*(len(X.columns)))))
M3D_all = np.zeros((Sample_size,(M3_windowd*(len(X.columns)))))
M4_all = np.zeros((Sample_size,(M4_window*(len(X.columns)))))
M4D_all = np.zeros((Sample_size,(M4_windowd*(len(X.columns)))))
M_Totsize = M1_window +M1_windowd + M2_window + M2_windowd + M3_window + M3_windowd + M4_window + M4_windowd
M_all = np.zeros((Sample_size,(M_Totsize*(len(X.columns)))))

if Penalty != 0:
    for i in range(Sample_size):
        if new_yLAG[i] == 0 and new_y[i] == 1:
            Sample_weight[i] = pen1
            if i != Sample_size-1:
                Sample_weight[i+1] = pen2
        elif new_yLAG[i] == 1 and new_y[i] == 1 and Sample_weight[i-1]!=pen1:
            Sample_weight[i] = pen3
            
#function to transform a TS into window frames
def Subsequences(input_ts, window_size):
    #input_t = input_ts.to_numpy()
    output_matrix = np.zeros( (len(input_ts)-window_size+1, window_size))
    for i in range(0,len(input_ts)-window_size+1):
        for j in range(0,window_size):
        #creating Subsequences:   
        #row i of the output matrix is the TS from t=i to t=i+window_size-1
            output_matrix[i,j] = input_ts[i+j]
    return output_matrix
def confusion_metrics(conf_matrix):
# save confusion matrix and slice into four pieces
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    #print('True Positives:', TP)
    #print('True Negatives:', TN)
    #print('False Positives:', FP)
    #print('False Negatives:', FN)
    # calculate accuracy
    conf_accuracy = (float (TP+TN) / float(TP + TN + FP + FN))
    # calculate mis-classification
    conf_misclassification = 1- conf_accuracy
    # calculate the sensitivity (aka Recall)
    #“out of all actual Positives, how many did we predict as Positive”
    if math.isclose(TP + FN, 0.0):
        conf_sensitivity = 0
    else:
        conf_sensitivity = (TP / float(TP + FN))
    # calculate the specificity
    #“out of all actual Negatives, how many did we predict as Negative”
    if math.isclose(TN + FP, 0.0):
        conf_specificity = float(0)
    else:
        conf_specificity = (float(TN) / float(TN + FP))
    # calculate G-mean
    # commonly used as an indicator of  CPD performance
    conf_g_mean = (float(conf_specificity * conf_sensitivity)) ** (0.5)
    # calculate precision
    #“out of all predicted Positive cases, how many were actually Positive”
    if math.isclose(TP + FP, 0.0):
        conf_precision = float(0)
    else:    
        conf_precision = (float(TP) / float(TP + FP))
    # calculate f_1 score
    # a harmonic, or weighted, average of Precision and Sensitivity
    if math.isclose(conf_precision + conf_sensitivity, 0.0):
        conf_f1 = float(0)
        conf_f_half = float(0)
        conf_f3 = float(0)
    else:
        conf_f1 = 2 * (float(conf_precision * conf_sensitivity) / float(conf_precision + conf_sensitivity))
        conf_f_half = 1.25 * ((float(conf_precision * conf_sensitivity)) / float(0.25*conf_precision + conf_sensitivity))
        conf_f3 = 10 * ((float(conf_precision * conf_sensitivity)) / float(9*conf_precision + conf_sensitivity))
    output = [conf_accuracy,conf_misclassification,conf_sensitivity,conf_specificity,conf_g_mean,conf_precision,conf_f1,conf_f3]
    return output
#Transforming X into window frames (now each column becomes #window_size columns)
for i in range(len(X.columns)):
    X_indiv_windowform = Subsequences(X_numpy[:,i],window_size)
    X_all_window[:,range(i*window_size,(i+1)*window_size)] = X_indiv_windowform
    #same for moments:
    M1_all[:,range(i*M1_window,(i+1)*M1_window)] = Subsequences(M1.to_numpy()[window_size-M1_window:,i],M1_window)
    M1D_all[:,range(i*M1_windowd,(i+1)*M1_windowd)] = Subsequences(M1Diff.to_numpy()[window_size-M1_windowd:,i],M1_windowd)
    M2_all[:,range(i*M2_window,(i+1)*M2_window)] = Subsequences(M2.to_numpy()[window_size-M2_window:,i],M2_window)
    M2D_all[:,range(i*M2_windowd,(i+1)*M2_windowd)] = Subsequences(M2Diff.to_numpy()[window_size-M2_windowd:,i],M2_windowd)
    M3_all[:,range(i*M3_window,(i+1)*M3_window)] = Subsequences(M3.to_numpy()[window_size-M3_window:,i],M3_window)
    M3D_all[:,range(i*M3_windowd,(i+1)*M3_windowd)] = Subsequences(M3Diff.to_numpy()[window_size-M3_windowd:,i],M3_windowd)
    M4_all[:,range(i*M4_window,(i+1)*M4_window)] = Subsequences(M4.to_numpy()[window_size-M4_window:,i],M4_window)
    M4D_all[:,range(i*M4_windowd,(i+1)*M4_windowd)] = Subsequences(M4Diff.to_numpy()[window_size-M4_windowd:,i],M4_windowd)
    M_all = np.concatenate((M1_all,M1D_all,M2_all,M2D_all,M3_all,M3D_all,M4_all,M4D_all),axis=1)

####### List of Classifiers to use in the first step #######
model_list = ['logistic Regression', 'SVM', #'KNN',
              'Decision Tree','Random Forest', 'Naive Bayes']
# machine learning model_pipeline 
model_pipeline = []
model_pipeline.append(LogisticRegression(solver='lbfgs',max_iter=50000,class_weight = w))
model_pipeline.append(SVC(probability=True,class_weight='balanced'))
#another way? SVC(probability = True, max_iter = 1000, class_weight = cw)
#model_pipeline.append(KNeighborsClassifier())
model_pipeline.append(DecisionTreeClassifier(class_weight='balanced'))
model_pipeline.append(RandomForestClassifier(max_depth=4,class_weight='balanced'))
model_pipeline.append(GaussianNB())
####### List of Classifiers to use in the first step #######
estimators_step1 = tuple(zip(model_list,model_pipeline))
estimators_step1 = [('lr', LogisticRegression(solver='lbfgs',max_iter=100000,class_weight = w)),
                    ('SVM',SVC(probability=True,class_weight='balanced')),
                    ('dt', DecisionTreeClassifier(class_weight='balanced')),
                    ('rf', RandomForestClassifier(max_depth=4,class_weight='balanced')), ('gnb', GaussianNB())]
if Ensemble1 == 0:
    #stack_method can be chosen: if ‘auto’, it will try to invoke,
    #for each estimator, 'predict_proba', 'decision_function' or 'predict' in that order.
    #otherwise, one of 'predict_proba', 'decision_function' or 'predict'. 
    #If the method is not implemented by the estimator, it will raise an error.
    meta_clf = StackingClassifier(stack_method='predict_proba',estimators=estimators_step1, final_estimator=LogisticRegression(class_weight=w))
elif Ensemble1 == 1:
    meta_clf = StackingClassifier(stack_method='predict_proba',estimators=estimators_step1, final_estimator=SVC(probability=True,class_weight='balanced'))
elif Ensemble1 == 2:
 #   meta_clf = StackingClassifier(estimators=estimators_step1, final_estimator=KNeighborsClassifier())
#elif Ensemble1 == 3:
    meta_clf = StackingClassifier(estimators=estimators_step1, final_estimator=DecisionTreeClassifier(class_weight='balanced'))
elif Ensemble1 == 3:
    meta_clf = StackingClassifier(estimators=estimators_step1, final_estimator=RandomForestClassifier(class_weight='balanced'))
elif Ensemble1 == 4:
    meta_clf = StackingClassifier(estimators=estimators_step1, final_estimator=GaussianNB())
elif Ensemble1 == 6:
    meta_clf = EasyEnsembleClassifier(random_state=42)    
elif Ensemble1 == 7:
    meta_clf = RUSBoostClassifier(random_state=0)
elif Ensemble1 == 8:
    meta_clf = BalancedBaggingClassifier(#sampling_strategy='not majority',
                                         base_estimator= SVC(class_weight='balanced'), 
                                         random_state=42)
elif Ensemble1 == 9:
    meta_clf = BalancedRandomForestClassifier(max_depth=4, random_state=0)
####### Defining evaluation matrices:
num_metrics = 9
evaluation_vec_test = np.zeros((1,num_metrics*(len(X.columns))))
evaluation_vec_train = np.zeros((1,num_metrics*(len(X.columns))))
####### Generating Matrices for evalation of prediction of first stage (of each classifier on each TS)
G_means_vec_metaclassifier_test = np.zeros((1,len(X.columns)))
F3_vec_metaclassifier_test = np.zeros((1,len(X.columns)))
G_means_vec_metaclassifier_train = np.zeros((1,len(X.columns)))
F3_vec_metaclassifier_train = np.zeros((1,len(X.columns)))
auc_vec_metaclassifier_train = np.zeros((1,len(X.columns)))
auc_vec_metaclassifier_test = np.zeros((1,len(X.columns)))
auc_pr_vec_metaclassifier_train = np.zeros((1,len(X.columns)))
auc_pr_vec_metaclassifier_test = np.zeros((1,len(X.columns)))
Sens_test = np.zeros((1,len(X.columns)))
Sens_train = np.zeros((1,len(X.columns)))
y_output = np.zeros((Sample_size,len(X.columns)))
y_prob = np.zeros((Sample_size))
y_final = np.zeros((Sample_size))

############ after transforming each TS into window sizes, feed each transformed TS into 
           # each classifier
kf = KFold(n_splits=n_splits, random_state=None, shuffle=False)
#Selected Features for each fold
SF = []
SFI = []
x = df["Date"]
if Country == 0:
    x = x.drop(x.index[-1],axis=0)
    x = x.drop(x.index[:window_size-1],axis=0)
    t = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in x]
if Country == 1:
    x = x.drop(x.index[-3:],axis=0)
    x = x.drop(x.index[:43+window_size-1],axis=0)
    t = [dt.datetime.strptime(d,'%m/%d/%Y').date() for d in x]
if moments_ws1 != 0 or moments_ws1d != 0 or moments_ws2 != 0 or moments_ws2d != 0 or moments_ws3 != 0 or moments_ws3d != 0 or moments_ws4 != 0 or moments_ws4d != 0:
    X_all_window = np.concatenate((X_all_window,M_all),axis=1)
t_np = np.array(t)
X_all_window_Copy = X_all_window
y_pred_matrix_whole_folds = np.zeros((Sample_size,len(X.columns),n_splits))
average_votes = np.zeros((Sample_size,len(model_list),n_splits))
final_votes = np.zeros((Sample_size,len(model_list),n_splits))
evaluation_measures = ["accuracy","misclassification","sensitivity","specificity","g_mean","precision","f1","f3","auc"]
evaluation_metric_test = np.zeros((len(model_list),num_metrics,n_splits))
evaluation_metric_train = np.zeros((len(model_list),num_metrics,n_splits))
evaluation_matrix_train = np.zeros((len(X.columns),num_metrics,n_splits))
evaluation_matrix_test = np.zeros((len(X.columns),num_metrics,n_splits))
n = math.ceil(perc*(len(X.columns)))
indices = np.zeros((n,n_splits),dtype=int)
if label_length != 0:
    new_yLAGs = np.zeros((Sample_size,label_length))
    for i in range(label_length):
        i+=1
        y_ax = y.iloc[window_size-i:-i]
        new_yLAGsnp = y_ax.to_numpy()
        new_yLAGs[:,i-1] = new_yLAGnp[:]      
    X_all_window = np.concatenate((X_all_window_Copy,new_yLAGs),axis=1)  
for train_index, test_index in kf.split(X_all_window):
    fold_num = 1+round(test_index[0]*n_splits/Sample_size)
    #if fold_num != n_splits:
     #   continue
      #  print('hello'+str(fold_num)) 
    transformer= StandardScaler().fit(X_all_window[train_index,:])   
    X_all_window[train_index,:] = transformer.transform(X_all_window[train_index,:])
    X_all_window[test_index,:] = transformer.transform(X_all_window[test_index,:])
    X_train_auxf, X_test_auxf = X_all_window[train_index,:], X_all_window[test_index,:]
    y_train_auxf, y_test_auxf = new_y[train_index], new_y[test_index]
    Sample_weight_aux = Sample_weight[train_index]
    X_auxf = np.concatenate((X_train_auxf,X_test_auxf),axis=0)
    y_auxf = np.concatenate((y_train_auxf,y_test_auxf),axis=0)
    y_pred_matrix_whole = np.zeros((Sample_size,len(X.columns)))
    for i in range(len(X.columns)):
        new_X = pd.DataFrame(X_auxf[:,list(range((i)*window_size,(i+1)*window_size))+list(range(window_size*len(X.columns)+i*M_Totsize,(window_size*len(X.columns))+(i+1)*M_Totsize))])
        X_train = pd.DataFrame(X_train_auxf[:,list(range((i)*window_size,(i+1)*window_size))+list(range(window_size*len(X.columns)+i*M_Totsize,(window_size*len(X.columns))+(i+1)*M_Totsize))])
        X_test = pd.DataFrame(X_test_auxf[:,list(range((i)*window_size,(i+1)*window_size))+list(range(window_size*len(X.columns)+i*M_Totsize,(window_size*len(X.columns))+(i+1)*M_Totsize))])
        if label_length != 0:
            new_X = pd.DataFrame(X_auxf[:,list(range((i)*window_size,(i+1)*window_size))+list(range(window_size*len(X.columns)+i*M_Totsize,(window_size*len(X.columns))+(i+1)*M_Totsize))+list(range(-label_length,0))])
            X_train = pd.DataFrame(X_train_auxf[:,list(range((i)*window_size,(i+1)*window_size))+list(range(window_size*len(X.columns)+i*M_Totsize,(window_size*len(X.columns))+(i+1)*M_Totsize))+list(range(-label_length,0))])
            X_test = pd.DataFrame(X_test_auxf[:,list(range((i)*window_size,(i+1)*window_size))+list(range(window_size*len(X.columns)+i*M_Totsize,(window_size*len(X.columns))+(i+1)*M_Totsize))+list(range(-label_length,0))])
        y_train = y_train_auxf
        y_test = y_test_auxf
        meta_clf.fit(X_train, y_train,sample_weight=Sample_weight_aux)
        y_pred = meta_clf.predict(X_test)
        y_pred_train = meta_clf.predict(X_train)
        #y_output[range(len(train_index)),i] = y_pred_train
        #y_output[range(len(train_index),Sample_size),i] = y_pred
        #y_pred_matrix_whole[train_index,i+counter*len(X.columns)] = y_pred_train
        #y_pred_matrix_whole[test_index,i+counter*len(X.columns)] = y_pred
        yPR_tr = meta_clf.predict_proba(X_train)[:, 1]
        yPR_tst = meta_clf.predict_proba(X_test)[:, 1]   
        if recurrent !=0:
            aa = np.zeros((Sample_size,1))
            aa [train_index,0] = yPR_tr
            aa [test_index,0] = yPR_tst
            aa = StandardScaler().fit_transform(aa)  
            shift(aa, 1, cval=np.NaN)
            aa_pd = pd.DataFrame(aa)
            aa_pd.interpolate(method ='linear', inplace=True)
            aatr=aa_pd.iloc[train_index].reset_index(drop=True)
            aats=aa_pd.iloc[test_index].reset_index(drop=True)
            X_train2 = pd.concat((X_train,aatr),axis=1)
            X_test2 = pd.concat((X_test,aats),axis=1)
            #X_train_auxf, X_test_auxf = X_all_window2[train_index,:], X_all_window2[test_index,:]
            #X_auxf = np.concatenate((X_train_auxf,X_test_auxf),axis=0)
            #X_train = X_train_auxf
            #X_test = X_test_auxf
            meta_clf.fit(X_train2, y_train,sample_weight=Sample_weight_aux)
            y_pred2 = meta_clf.predict(X_test2)
            y_pred_train2 = meta_clf.predict(X_train2)
            #yPR_tr2 = meta_clf.predict_proba(X_train2)[:, 1]
            #yPR_tst2 = meta_clf.predict_proba(X_test2)[:, 1]
        y_pred_matrix_whole[train_index,i] = y_pred_train2
        y_pred_matrix_whole[test_index,i] = y_pred2            
        evaluation_matrix_train[i,:-1,fold_num-1] = confusion_metrics(confusion_matrix(y_train, y_pred_train2,labels=[0,1]))
        evaluation_matrix_train[i,-1,fold_num-1] = roc_auc_score(y_train, y_pred_train2)
        evaluation_matrix_test[i,:-1,fold_num-1] = confusion_metrics(confusion_matrix(y_test, y_pred2,labels=[0,1]))
        evaluation_matrix_test[i,-1,fold_num-1] = roc_auc_score(y_test, y_pred2)
    y_pred_matrix_whole_folds[:,:,fold_num-1] = y_pred_matrix_whole
    ### Feature Selection:
    sorted_index_array = np.argsort(evaluation_matrix_train[:,Metr,fold_num-1],axis=0)
    indices[:,fold_num-1] = sorted_index_array[-n:]
        #Selected_Features = X.columns[indices[:,mod,fold_num-1]]
        #Selected_Features = Selected_Features.to_list()
        #Selected_Features_df = pd.DataFrame(Selected_Features)
    #for ind in indices[:,mod,fold_num-1]:
    # print(evaluation_matrix_train[0,ind])
    #average_votes[:,mod] = np.mean(y_pred_matrix_whole[:,mod*len(X.columns):(mod+1)*len(X.columns)],axis=1)
    fs_index = indices[:,fold_num-1]
    X_train = y_pred_matrix_whole[np.ix_(train_index,fs_index)]
    X_test = y_pred_matrix_whole[np.ix_(test_index,fs_index)]
    transformer2= StandardScaler().fit(X_train)
    X_train = transformer2.transform(X_train)
    X_test = transformer2.transform(X_test)
    counter = 0
    for model in model_pipeline:    
        model.fit(X_train, y_train,sample_weight=Sample_weight_aux)
        y_pred = meta_clf.predict(X_test)
        y_pred_train = meta_clf.predict(X_train)
        #y_output[range(len(train_index)),i] = y_pred_train
        #y_output[range(len(train_index),Sample_size),i] = y_pred
        #y_pred_matrix_whole[train_index,i+counter*len(X.columns)] = y_pred_train
        #y_pred_matrix_whole[test_index,i+counter*len(X.columns)] = y_pred
        yPR_tr = meta_clf.predict_proba(X_train)[:, 1]
        yPR_tst = meta_clf.predict_proba(X_test)[:, 1]
        final_votes[train_index,counter,fold_num-1] = y_pred_train
        final_votes[test_index,counter,fold_num-1] = y_pred
        evaluation_metric_test[counter,:-1,fold_num-1] = confusion_metrics(confusion_matrix(y_test, final_votes[test_index,counter,fold_num-1],labels=[0,1]))
        evaluation_metric_test[counter,-1,fold_num-1] = roc_auc_score(y_test, final_votes[test_index,counter,fold_num-1])
        evaluation_metric_train[counter,:-1,fold_num-1] = confusion_metrics(confusion_matrix(y_train, final_votes[train_index,counter,fold_num-1],labels=[0,1]))
        evaluation_metric_train[counter,-1,fold_num-1] = roc_auc_score(y_train, final_votes[train_index,counter,fold_num-1])
        result_df = pd.DataFrame({'metric':evaluation_measures,'value over test':evaluation_metric_test[counter,:,fold_num-1],'value over train':evaluation_metric_train[counter,:,fold_num-1]})
        print('clf:'+str(model_list[counter])+" (#test fold:"+str(fold_num)+")")
        print(result_df)
        current_time = dt.datetime.now().strftime("%H.%M.%S")
        with pd.ExcelWriter(r'C:\Users\momas\OneDrive\Desktop\Thesis\Thesis\CODES\Revised Codes\Results\M7\7.xlsx', engine="openpyxl",mode='a',if_sheet_exists='overlay') as writer:
            result_df.to_excel(writer, sheet_name=str(current_time)+' '+str(fold_num),startcol=3*counter,index=False)
        counter +=1
    #Selected_Features_df.to_excel(writer, sheet_name="Features fold"+str(fold_num))
    
        #evaluation_vec_test[counter,num_metrics*(i+1)-2] = roc_auc_score(y_test, final_votes[test_index]) 
        #evaluation_metric_test = confusion_metrics(confusion_matrix(y_test, y_pred,labels=[0,1]))
#    for j in range(num_metrics-2):
 #       evaluation_vec_test[counter,num_metrics*i + j] = evaluation_metric_test[j]
        #evaluation_vec_test[counter,num_metrics*(i+1)-2] = roc_auc_score(y_test, y_pred)
        #evaluation_vec_test[counter,num_metrics*(i+1)-1] = roc_auc_score(y_test, meta_clf.predict_proba(X_test)[:, 1])
        
        #evaluation_metric_train = confusion_metrics(confusion_matrix(y_train, y_pred_train,labels=[0,1]))
        #for j in range(num_metrics-2):
         #   evaluation_vec_train[counter,num_metrics*i + j] = evaluation_metric_train[j]
        #evaluation_vec_train[counter,num_metrics*(i+1)-2] = roc_auc_score(y_train, y_pred_train)
    #    evaluation_vec_train[counter,num_metrics*(i+1)-1] = roc_auc_score(y_train, meta_clf.predict_proba(X_train)[:, 1])

######## Feature Selection based on performance(F0.5) of metaclf:
# Generating Geometric means (and F0.5 and auc_score) for each TS

    #for i in range(len(X.columns)):
     #   G_means_vec_metaclassifier_test[:,i] = evaluation_vec_test[counter,num_metrics*(i)+4]
      #  F3_vec_metaclassifier_test[:,i] = evaluation_vec_test[counter,num_metrics*(i)+7]
       # auc_vec_metaclassifier_test[:,i] = evaluation_vec_test[counter,num_metrics*(i+1)-2]
       # auc_pr_vec_metaclassifier_test[:,i] = evaluation_vec_test[counter,num_metrics*(i+1)-1]
       # Sens_test[:,i] = evaluation_vec_test[counter,num_metrics*(i)+2]
        
       # G_means_vec_metaclassifier_train[:,i] = evaluation_vec_train[counter,num_metrics*(i)+4]
       # F3_vec_metaclassifier_train[:,i] = evaluation_vec_train[counter,num_metrics*(i)+7]
       # auc_vec_metaclassifier_train[:,i] = evaluation_vec_train[counter,num_metrics*(i+1)-2]
       # auc_pr_vec_metaclassifier_train[:,i] = evaluation_vec_train[counter,num_metrics*(i+1)-1]
       # Sens_train[:,i] = evaluation_vec_train[counter,num_metrics*(i)+2]
    
    #X_train, X_test, y_train, y_test = train_test_split(X_all_window,new_y,test_size=test_percent, random_state=42,shuffle=False)
    #meta_clf.fit(X_train, y_train).score(X_test, y_test)
    
    ######## Feature Selection based on performance of the metaclassifier
    #the indices of TS whose performance is in Top 5% in are given to 3rd layer classifier
  #  perc = .25
  #  n = math.ceil(perc*(len(X.columns)))
  #  sorted_index_array = np.argsort(auc_vec_metaclassifier_train)
    
    #export
  #  indices = sorted_index_array[0,-n:]
  #  Selected_Features = X.columns[indices]
  #  Selected_Features = Selected_Features.to_list()
  #  Selected_Features_df = pd.DataFrame(Selected_Features)
  #  for ind in indices:
  #    print(auc_vec_metaclassifier_train[0,ind])
    
  #  X_Selected = np.zeros((Sample_size,n*(window_size+M_Totsize)))
  #  #labels produced from the features selected in the first stage 
  #  X01_Selected = np.zeros((Sample_size,n))
    
  #  for k in range(n):
  #      X_Selected[:,range(k*(window_size+M_Totsize),(k+1)*(window_size+M_Totsize))] = X_auxf[:,list(range(indices[k]*window_size,(1+indices[k])*window_size))+list(range(window_size*len(X.columns)+indices[k]*M_Totsize,window_size*len(X.columns)+(1+indices[k])*M_Totsize))]
  #      X01_Selected[:,k] = y_output[:,indices[k]]
        
  #  if original != 0:
        #X01_Selected = np.concatenate((X01_Selected,X_auxf),axis=1)
  #      X01_Selected = np.concatenate((X01_Selected,X_Selected),axis=1)
    
  #  if label_length != 0:
  #      y_lag = np.zeros((Sample_size,1))
  #      y_aux = y.to_numpy()
  #      y_lag[:,0] = y_aux[window_size-1:-1]
  #      X01_Selected = np.concatenate((X01_Selected,y_lag),axis=1)
    
  #  X_train = X01_Selected[:len(train_index),:]
  #  X_test = X01_Selected[len(train_index):,:]
  #  y_train = y_train_auxf
  #  y_test = y_test_auxf
    
    
    #X_train, X_test, y_train, y_test = train_test_split(X01_Selected, y_auxf,
     #                                                   test_size=test_percent, random_state=42,
      #                                                  shuffle=False)
 #   if Ensemble2 == 0:
 #       meta_clf2 = StackingClassifier(estimators=estimators_step1, final_estimator=LogisticRegression(class_weight=w))
 #   elif Ensemble2 == 1:
 #       meta_clf2 = StackingClassifier(estimators=estimators_step1, final_estimator=SVC(probability=True,class_weight='balanced'))
 #   #elif Ensemble2 == 2:    
     #   meta_clf2 = StackingClassifier(estimators=estimators_step1, final_estimator=KNeighborsClassifier())
 #   elif Ensemble2 == 3:    
 #       meta_clf2 = StackingClassifier(estimators=estimators_step1, final_estimator=DecisionTreeClassifier(class_weight='balanced'))
 #   elif Ensemble2 == 4:    
 #       meta_clf2 = StackingClassifier(estimators=estimators_step1, final_estimator=RandomForestClassifier(class_weight='balanced'))
 #   elif Ensemble2 == 5:    
 #       meta_clf2 = StackingClassifier(estimators=estimators_step1, final_estimator=GaussianNB())
 #   elif Ensemble2 == 10:
 #       meta_clf2 = LogisticRegression(solver='lbfgs',max_iter=50000,class_weight = w)
 #   elif Ensemble2 == 11:
 #       meta_clf2 = SVC(probability=True,class_weight='balanced')
 #   elif Ensemble2 == 12:
 #       meta_clf2 = KNeighborsClassifier()
 #   elif Ensemble2 == 13:
 #       meta_clf2 = DecisionTreeClassifier(class_weight='balanced')
 #   elif Ensemble2 == 14:
 #       meta_clf2 = RandomForestClassifier(max_depth=4,class_weight='balanced')
 #   elif Ensemble2 == 15:
 #       meta_clf2 = GaussianNB()
 #   elif Ensemble2 == 16:
 #       meta_clf2 = EasyEnsembleClassifier(random_state=42)    
 #   elif Ensemble2 == 17:
 #       meta_clf2 = RUSBoostClassifier(random_state=0)
 #   elif Ensemble2 == 18:
 #       meta_clf2 = BalancedBaggingClassifier(#sampling_strategy='not majority',
 #                                            base_estimator= SVC(class_weight='balanced'), 
 #                                            random_state=42)
 #   elif Ensemble2 == 19:
 #       meta_clf2 = BalancedRandomForestClassifier(max_depth=4, random_state=0)
    
 #   meta_clf2.fit(X_train, y_train,sample_weight=Sample_weight_aux)
 #   y_pred = meta_clf2.predict(X_test)
 #   y_pred_train = meta_clf2.predict(X_train)
 #   yPR_tr = meta_clf2.predict_proba(X_train)[:, 1]
 #   yPR_tst = meta_clf2.predict_proba(X_test)[:, 1]    
        
 #   evaluation_final_train = confusion_metrics(confusion_matrix(y_train, y_pred_train,labels=[0,1]))
 #   evaluation_final_test = confusion_metrics(confusion_matrix(y_test, y_pred,labels=[0,1]))
    
    
    #y_output[range(len(y_pred_train)),i-1] = y_pred_train
    #y_output[range(len(y_pred)),i-1] = y_pred
    
    #fpr, tpr, _thresholds = metrics.roc_curve(y_test, y_pred)
    #auc = round(metrics.auc(fpr, tpr),2)
    #evaluation_final_test_auc 
  #  evaluation_final_test.append(roc_auc_score(y_test, y_pred))
  #  if Ensemble2 == 0 or Ensemble2 == 10:
  #      evaluation_final_test.append(roc_auc_score(y_test, yPR_tst))
  #      evaluation_final_test.append(roc_auc_score(y_test, yPR_tst))
  #  elif Ensemble2 == 1 or Ensemble2 == 11:
  #      evaluation_final_test.append(roc_auc_score(y_test, meta_clf2.decision_function(X_test)))
  #  else:
  #      evaluation_final_test.append(roc_auc_score(y_test, yPR_tst))
  #  #evaluation_final_test_auc_prob = roc_auc_score(y_test, meta_clf.predict_proba(X_test)[:, 1])
  #      
  #  evaluation_final_train.append(roc_auc_score(y_train, y_pred_train))
  #  
  ##  if Ensemble2 == 0 or Ensemble2 == 10:
  #      evaluation_final_train.append(roc_auc_score(y_train, meta_clf2.decision_function(X_train)))
  #      evaluation_final_train.append(roc_auc_score(y_train, yPR_tr))
  ##  elif Ensemble2 == 1 or Ensemble2 == 11:
  #      evaluation_final_train.append(roc_auc_score(y_train, yPR_tr))
  #  else:
  #      evaluation_final_train.append(roc_auc_score(y_train, yPR_tr))
  #  #evaluation_train = [evaluation_final_train, evaluation_final_train_auc,evaluation_final_train_auc_prob]
  #  #evaluation_test = [evaluation_final_test, evaluation_final_test_auc,evaluation_final_test_auc_prob]
  #  
    
    
  #  result_df = pd.DataFrame({'metric':evaluation_measures,'value over test':evaluation_final_test,'value over train':evaluation_final_train})
    #fold_num = 1+round(test_index[0]*n_splits/Sample_size)
    
    #Selected Features Indices:
    #SFI.append(indices)
    #Saving Selected Features
    #SF.append(Selected_Features)
    #with pd.ExcelWriter(r'C:\Users\momas\Desktop\Thesis\CODES\10Results_labeladded_Manual.xlsx',
     #                   mode='a') as writer:  
      #result_df.to_excel(writer, sheet_name="fold"+str(fold_num))
      #Selected_Features_df.to_excel(writer, sheet_name="Features fold"+str(fold_num))
    
       
   # print("#fold for test:"+str(fold_num))
    #print(result_df)
    #print("Selected Features are:")
    #print((Selected_Features))
    y_auxfill = np.zeros(Sample_size)
    y_auxfill2 = np.zeros(Sample_size)
    y_auxfill [test_index] = 1
    y_auxfill2 [test_index] = y_test_auxf
    for i in test_index:
       if new_y[i] == 1:
            if i!=test_index[0]:
                y_auxfill2[i-1] = 1
            if i!=test_index[1]:
                y_auxfill2[i-2] = 1
            if i!=test_index[-1]:
                y_auxfill2[i+1] = 1
            if i!=test_index[-2]:
                y_auxfill2[i+2] = 1
    y_auxfill2_idx = np.where(y_auxfill2 == 1)[0]
    
    # corresponding y axis values
    #c = np.concatenate(average_votes_each_window[mod*y_train_size:(mod+1)*y_train_size],average_votes_each_window_test[mod*y_test_size:(mod+1)*y_test_size])
    sns.set_style('whitegrid')
    #sns.despine()
    sns.set_context("talk")
    plt.figure(figsize=(20, 11))
    #ax = plt.figure.add_axes([0, 0, 1, 1])
    plt.xticks(fontsize=17,rotation=50)
    plt.yticks(fontsize=17)
    
    plt.annotate('Test Set', xy=(0.2, 0.9), xytext=((fold_num-.5)/n_splits, .1), xycoords='axes fraction', 
            fontsize=20, ha='center', va='center',
            bbox=dict(boxstyle='round', fc='skyblue'),
            #arrowprops=dict(arrowstyle='-[, widthB=1.0, lengthB=1', lw=1.0)
            )
    for mod in range(len(model_list)):
        #y_final[train_index] , y_final[test_index] = y_pred_train, y_pred
        plt.plot(t, final_votes[:,mod,fold_num-1], label='avr '+str(model_list[mod]))
             #, linestyle='dashed', linewidth = 3,
         #marker='o', markerfacecolor='blue', 
         #        markersize=12)
        #plt.plot(t, final_votes[:,mod], label='Vote'+str(model_list[mod]))
    
    
    plt.ylim(-0.006,1.01)
    #plt.xlim(x[window_size], x[-1])
    pd.index = pd.DatetimeIndex(x, freq='MS')
    plt.fill_between(t, 0, 1, where=new_y, color='darkkhaki', alpha=0.3)
    plt.fill_between(t, 0, .2, where=y_auxfill, color='violet',hatch='x', alpha=0.3)
    #plt.text(test_index[1], 0.1, 'test set', verticalalignment='bottom', horizontalalignment='right',
        #transform=ax.transAxes, 
     #   color='green', fontsize=21)

    #plt.text(test_index[1],
     #        0.1, 'test set',
      #  verticalalignment='bottom', horizontalalignment='right',
        #transform=ax.transAxes,
       # color='green', fontsize=22)
    
    plt.xlabel('Date (Monthly)',fontsize=20)
    # naming the y axis
    plt.ylabel('Average vote',fontsize=20)
    # giving a title to my graphco
    plt.title('Prediction of a Meta-Classifier with a Feature Selection Step(Ens1:'+ str(Ensemble1)+',#test fold:'+str(fold_num)+')',fontsize=22)
    plt.gca().spines["top"].set_alpha(.0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(.0)
    plt.gca().spines["left"].set_alpha(.3)
    # show a legend on the plot
    plt.legend(fontsize=20#, loc = (.48,.84)
               )

    # function to show the plot
    #plt.savefig('./figs/Ensemble_mclfs_'+str(Ensemble1)+str(Ensemble2)+'_fold'+str(fold_num)+'.png', bbox_inches='tight')
    
    plt.show()
    #plt.savefig('./figs/Ensemble_mclfs_'+str(Ensemble1)+str(Ensemble2)+'_fold'+str(fold_num)+'.pgf', bbox_inches='tight')
    
    plt.figure(figsize=(18, 10))
    #ax = plt.figure.add_axes([0, 0, 1, 1])
    plt.xticks(fontsize=17,rotation=50)
    plt.yticks(fontsize=17)
    
    #plt.annotate('Test Set', xy=(0.2, 0.9), xytext=((fold_num-.5)/n_splits, .1), xycoords='axes fraction', 
     #       fontsize=20, ha='center', va='center',
      #      bbox=dict(boxstyle='round', fc='skyblue'),
            #arrowprops=dict(arrowstyle='-[, widthB=1.0, lengthB=1', lw=1.0)
        #    )
    for mod in range(len(model_list)):
        #y_final[train_index] , y_final[test_index] = y_pred_train, y_pred
        plt.plot(t_np[y_auxfill2_idx], final_votes[y_auxfill2_idx,mod,fold_num-1], label=str(model_list[mod]))
             #, linestyle='dashed', linewidth = 3,
         #marker='o', markerfacecolor='blue', 
         #        markersize=12)
        #plt.plot(t, final_votes[:,mod], label='Vote'+str(model_list[mod]))
    plt.ylim(-0.006,1.01)
    #plt.xlim(x[window_size], x[-1])
    #pd.index = pd.DatetimeIndex(x, freq='MS')
    plt.fill_between(t_np[y_auxfill2_idx], 0, 1, where=new_y[y_auxfill2_idx], color='darkkhaki', alpha=0.3)
    #plt.fill_between(t, 0, .2, where=y_auxfill, color='violet',hatch='x', alpha=0.3)
    #plt.text(test_index[1], 0.1, 'test set', verticalalignment='bottom', horizontalalignment='right',
        #transform=ax.transAxes, 
     #   color='green', fontsize=21)

    #plt.text(test_index[1],
     #        0.1, 'test set',
      #  verticalalignment='bottom', horizontalalignment='right',
        #transform=ax.transAxes,
       # color='green', fontsize=22)
    plt.xlabel('Date (Monthly)',fontsize=18)
    plt.xticks(fontsize=14,rotation=60)
    plt.yticks(fontsize=14)
    # naming the y axis
    plt.ylabel('Average vote',fontsize=18)
    # giving a title to my graphco
    plt.title('Prediction of a Meta-Classifier on Periods near the Recession'+' (#test fold:'+str(fold_num)+')',fontsize=22)
    plt.gca().spines["top"].set_alpha(.0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(.0)
    plt.gca().spines["left"].set_alpha(.3)
    # show a legend on the plot
    plt.legend(fontsize=10#, loc = (.48,.84)
               )
    # function to show the plot
    #plt.savefig('./figs/Ensemble_mclfs_'+str(Ensemble1)+str(Ensemble2)+'_fold'+str(fold_num)+'.png', bbox_inches='tight')
    plt.show()
def common_fr(input_f):
    set0 = set(input_f[0])
    for i in range(len(input_f)):
        set0 = set0 & set(input_f[i])
        
    print("Selected Features Common in all K-Folds are:")
    if (set0):
        print(set0)
    else:
        print("No common elements")