
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


df = pd.read_csv('Financial Distress.csv')

#understand data structure
df.info()
df.describe().T
df.head()

#data preprocessing
np.unique(df.isnull().values, return_counts = True) #no missing values

df['Financial Distress'].describe()

#convert target variable into binary for classification 
financial_distress = []
for i, j in enumerate(df['Financial Distress']):
    if j > -0.5:
        j = 0
        financial_distress.append(j)
    else:
        j = 1
        financial_distress.append(j)

df['Financial_distress'] = financial_distress
df = df.drop(['Financial Distress'], axis = 1)
df.info()

#imbalanced dataset
print('Percentage of Healthy observations: ', round(df['Financial_distress'].value_counts()[0]/len(df)*100, 2), '%')
print('Percentage of Financially distressed observations: ', round(df['Financial_distress'].value_counts()[1]/len(df)*100, 2), '%')


#exploratory data analysis
sns.countplot('Financial_distress', data = df)
plt.title('Distribution of Target variable - IMBALANCED!', fontsize = 14)
df_corr = (df.drop(['Company', 'Time', 'Financial_distress'], axis = 1).corr())
df_corr_high = (df.drop(['Company', 'Time', 'Financial_distress'], axis = 1).corr() > 0.8)
sns.heatmap(df_corr) #presense of multicollinearity! Maybe remove these variables to improve model??

X = df.drop(['Company', 'Time', 'Financial_distress', 'x80'], axis = 1)
y = df['Financial_distress']

#we can undersample or oversample (SMOTE) to make target variable more balanced
#Lets do Undersampling first!
distressed = df.loc[df['Financial_distress'] == 1]
healthy_us = df.loc[df['Financial_distress'] == 0].sample(n=200) #randomly undersampled to 200
df_us = pd.concat([distressed, healthy_us]).sample(frac = 1) #sample method shuffles the dataset
df_us.head()

#visualize balance plots
fig, axs = plt.subplots(1, 2, figsize=(10,4))
sns.countplot('Financial_distress', data = df, ax=axs[0])
axs[0].set_title('Imbalanced', fontsize = 14)
sns.countplot('Financial_distress', data = df_us, ax = axs[1])
axs[1].set_title('Balanced', fontsize = 14)

#boxplots
fig, axs = plt.subplots(2,2, figsize = (10,5))
sns.boxplot(x = 'Financial_distress', y = 'x3', data = df, ax = axs[0,0])
axs[0,0].set_title('x3 distribution against Fincial distress')
axs[0,0].set_xlabel('')
plt.subplots_adjust(hspace = 0.4)
sns.boxplot(x = 'Financial_distress', y = 'x13', data = df, ax = axs[0,1])
axs[0,1].set_title('x13 distribution against Fincial distress')
axs[0,1].set_xlabel('')
sns.boxplot(x = 'Financial_distress', y = 'x24', data = df, ax = axs[1,0])
axs[1,0].set_title('x24 distribution against Fincial distress')
sns.boxplot(x = 'Financial_distress', y = 'x50', data = df, ax = axs[1,1])
axs[1,1].set_title('x50 distribution against Fincial distress')

#x80 contains 
np.unique(df.x80) #drop x80 for now as it has 37 classes so due to time constraint better to remove!
X_us = df_us.drop(['Company', 'Time', 'Financial_distress', 'x80'], axis = 1)
y_us = df_us['Financial_distress']

#feature scaling
sc = StandardScaler()
X_us = sc.fit_transform(X_us)

X_train, X_test, y_train, y_test = train_test_split(X_us, y_us, test_size=0.2, random_state=42)
np.unique(y_test, return_counts = True)
40/len(y_test)
28/len(y_test)#test set some what balanced! no need for stratified for now!

y_train = y_train.values
y_test = y_test.values

classifiers = {'SGDClf' : SGDClassifier(loss='log', penalty='l2'),
               'Support Vector Classifier' : SVC(),
               'Random Forest' : RandomForestClassifier(n_estimators=100, max_leaf_nodes=16),
               'Logistic Regression' : LogisticRegression()}

#create and update classifier performance and classifier predictions 
clf_perf = {}
model_pred = {}
for keys, clf in classifiers.items():
    clf.fit(X_train, y_train)
    training_cv = cross_val_score(clf, X_train, y_train, cv = 10)
    clf_perf.update({keys: [training_cv.mean(), training_cv.std()]})
    model_pred.update({keys: clf.predict(X_test)}) 

#reviewing the results, we choose the logistic regression model going forward...
log_us_pred = model_pred['Logistic Regression']
cm_us = confusion_matrix(y_test, log_us_pred)
print(classification_report(y_test, log_us_pred))


#so far undersampling has produced quite low accuracy! lets try SMOTE and then compare results!
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)



for train_index, test_index in sss.split(X, y):
    orig_Xtrain, orig_Xtest = X.iloc[train_index], X.iloc[test_index]
    orig_ytrain, orig_ytest = y.iloc[train_index], y.iloc[test_index]

sc_2 = StandardScaler()
orig_Xtrain = sc_2.fit_transform(orig_Xtrain)
orig_Xtest = sc_2.transform(orig_Xtest)

orig_ytrain = orig_ytrain.values
orig_ytest = orig_ytest.values

#list to append score then average i.e. for cross validation
accuracy_lst = []
precision_lst = []
recall_lst = []
f1_lst = []
auc_lst = []


for train, test in sss.split(orig_Xtrain, orig_ytrain):
    pipeline = make_pipeline(SMOTE(ratio = 'minority', random_state= 42), LogisticRegression())
    model = pipeline.fit(orig_Xtrain[train], orig_ytrain[train])
    predict = model.predict(orig_Xtrain[test])
    
    accuracy_lst.append(accuracy_score(orig_ytrain[test], predict))
    precision_lst.append(precision_score(orig_ytrain[test], predict))
    recall_lst.append(recall_score(orig_ytrain[test], predict))
    f1_lst.append(f1_score(orig_ytrain[test], predict))
    auc_lst.append(roc_auc_score(orig_ytrain[test], predict))
    
print('---' * 45)
print('')
print("accuracy: {}".format(np.mean(accuracy_lst)))
print("precision: {}".format(np.mean(precision_lst)))
print("recall: {}".format(np.mean(recall_lst)))
print("f1: {}".format(np.mean(f1_lst)))
print('---' * 45)

#SMOTE (logistic regression) built now lets test on test data!
smote_predict = model.predict(orig_Xtest)
cm_smote = confusion_matrix(orig_ytest, smote_predict)
print(classification_report(orig_ytest, smote_predict))

recall_score(orig_ytest, smote_predict)
accuracy_score(orig_ytest, smote_predict)
precision_score(orig_ytest, smote_predict)
f1_score(orig_ytest, smote_predict)
round(roc_auc_score(orig_ytest, smote_predict), 3)
round(roc_auc_score(y_test, log_us_pred), 3)
print('---' * 45)
print('Undersampling AUC:', round(roc_auc_score(y_test, log_us_pred), 3))
print('SMOTE AUC:', round(roc_auc_score(orig_ytest, smote_predict), 3))
print('---' * 45)

fpr, tpr, thresholds = roc_curve(orig_ytest, smote_predict)
fpr_us, tpr_us, thresholds_us = roc_curve(y_test, log_us_pred)

def roc_curve_plot(fpr, tpr, label = None):
    plt.plot(fpr, tpr, linewidth = 2, label=label)
    plt.plot([0,1],[0,1], 'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    
roc_curve_plot(fpr, tpr, 'SMOTE')
roc_curve_plot(fpr_us, tpr_us, 'UnderSampling')
plt.legend(loc='bottom right')
