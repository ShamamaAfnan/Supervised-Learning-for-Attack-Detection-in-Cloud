#!/usr/bin/env python
# coding: utf-8

# **Import libraries**

# In[ ]:


import pandas as pd
import numpy as np 
import io
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer,mean_squared_error
from sklearn.metrics import roc_curve, mean_squared_error, homogeneity_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import RocCurveDisplay
from joblib import dump, load
from sklearn.manifold import TSNE


# **Import dataset**

# In[ ]:


from google.colab import files
uploaded = files.upload()
forensics = pd.read_csv(io.BytesIO(uploaded['VMResourceUtilizationSlope.csv']))


# **Exploratory data analysis**

# In[ ]:


forensics.head()


# In[ ]:


forensics.shape


# In[ ]:


forensics.columns


# In[ ]:


forensics['Status'].value_counts()


# In[ ]:


forensics.info()


# In[ ]:


forensics.isnull().sum()


# In[ ]:


forensics = forensics.dropna()


# In[ ]:


forensics.isnull().sum()


# In[ ]:


# find categorical variables

categorical = [var for var in forensics.columns if forensics[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)


# In[ ]:


forensics.describe()


# In[ ]:


# view the categorical variables

forensics[categorical].head()


# In[ ]:


# find numerical variables

numerical = [var for var in forensics.columns if forensics[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)


# In[ ]:


# view summary statistics in numerical variables

forensics[numerical].describe()


# In[ ]:


forensics[numerical].head()


# In[ ]:


# remove features which has only 0

forensics = forensics.loc[:, (forensics != 0).any(axis=0)]


# In[ ]:


forensics.shape


# In[ ]:


# find numerical variables

numerical = [var for var in forensics.columns if forensics[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)


# In[ ]:


forensics['Status']=forensics['Status'].replace(['Normal','Attack'],[0,1])


# In[ ]:


forensics.head()


# In[ ]:


# Shamama


# In[ ]:


forensics['Status'].value_counts()


# In[ ]:


colors =['#069AF3','#FF4500','#3f88c5','#ffba08','#d00000']
fig = go.Figure(go.Pie(labels=['Normal','Attack'],values=forensics.Status.value_counts(),name='Attack'))
fig.update_traces(hole=.4, hoverinfo="label+percent", textfont_size=16,marker={'colors':colors})
fig.update_layout(height=400, width=400, title_text='<b style="color:#000000;">Percentage of Target Variable</b>')
fig.show()


# In[ ]:


df = forensics.drop(['LAST_POLL', 'VMID', 'UUID','dom'], axis=1)


# In[ ]:


# draw boxplots to visualize outliers

plt.figure(figsize=(25,20))

for i, column in enumerate(df.columns[:-1]):
    plt.subplot(6, 4, i+1)
    fig = df.boxplot(column=column)
    fig.set_title('')
    fig.set_ylabel(column)


# In[ ]:


df.corr()


# In[ ]:


plt.figure(figsize=(12,6), dpi =100)
sns.heatmap(round(df.corr(), 2), linewidth=0.5,annot=True,fmt='.1g', cmap="viridis")
plt.title("Feature Correlation Heatmap", fontsize = 12, pad =10, fontweight='bold')


# **Define features & Target**

# In[ ]:


X = df.drop(['Status'], axis=1)


# In[ ]:


y = df['Status']


# **Feature Scaling for Clustering**

# In[ ]:


scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)


# **Split Data to Train Test**

# In[ ]:


#train test split
X_train1, X_test1,y_train, y_test = train_test_split(X, y, test_size=0.3, random_state =1)


# **Scaling Train and test Features for Classification Modeling**

# In[ ]:


X_train = scaler.fit_transform(X_train1)
X_test = scaler.transform(X_test1)
X_train = pd.DataFrame(data=X_train, columns=df.columns[:-1])
X_test = pd.DataFrame(data=X_test, columns=df.columns[:-1])


# In[ ]:


X_train


# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(12,5))
for idx, group in enumerate([('Train', y_train), ('Test', y_test)]):
    data = group[1].value_counts()
    sns.set_theme(style="whitegrid")
    sns.barplot(ax=ax[idx], x=data.index, y=data.values,palette="pastel")
    ax[idx].set_title(f'{group[0]} Label Count')
    ax[idx].set_xlabel(f'{group[0]} Labels')
    ax[idx].set_ylabel('Label Count')

plt.show()


# **Random Oversampling**

# In[ ]:


count_class_0,count_class_1=forensics.Status.value_counts()
df_class_0=forensics[forensics['Status']==0]
df_class_1=forensics[forensics['Status']==1]
df_class_0


# In[ ]:


df_class_1_over=df_class_1.sample(count_class_0, replace=True)
df_over=pd.concat([df_class_0,df_class_1_over],axis=0)


# In[ ]:


print("Random Over Sampling: ")
print(df_over.Status.value_counts())


# In[ ]:


colors =['#069AF3','#FF4500','#3f88c5','#ffba08','#d00000']
fig = go.Figure(go.Pie(labels=['Normal','Attack'],values=df_over.Status.value_counts(),name='Attack'))
fig.update_traces(hole=.4, hoverinfo="label+percent", textfont_size=16,marker={'colors':colors})
fig.update_layout(height=400, width=400, title_text='<b style="color:#000000;">Percentage of Target Variable</b>')
fig.show()


# In[ ]:


df_over=df_over.drop(['LAST_POLL', 'VMID', 'UUID','dom'], axis=1)


# In[ ]:


plt.figure(figsize=(12,6), dpi = 150)
sns.heatmap(round(df.corr(), 2), linewidth=0.5,annot=True,fmt='.1g', cmap="viridis")
plt.title("Feature Correlation Heatmap", fontsize = 12, pad =10, fontweight='bold')


# **Splitting Oversampled Data into Test and Train**

# In[ ]:



X_oversampled = df_over.drop(['Status'], axis=1)


# In[ ]:


y_oversampled = df_over['Status']


# In[ ]:


#train test split
X_train_over1, X_test_over1,y_train_over, y_test_over = train_test_split(X_oversampled, y_oversampled, test_size=0.3, random_state =1)


# In[ ]:


X_train_over = scaler.fit_transform(X_train_over1)
X_test_over = scaler.transform(X_test_over1)
X_train_over = pd.DataFrame(data=X_train_over, columns=df.columns[:-1])
X_test_over = pd.DataFrame(data=X_test_over, columns=df.columns[:-1])


# In[ ]:



import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1,2, figsize=(12,5))
for idx, group in enumerate([('Train', y_train_over), ('Test', y_test_over)]):
    data = group[1].value_counts()
    sns.set_theme(style="whitegrid")
    sns.barplot(ax=ax[idx], x=data.index, y=data.values,palette="pastel")
    ax[idx].set_title(f'{group[0]} Label Count')
    ax[idx].set_xlabel(f'{group[0]} Labels')
    ax[idx].set_ylabel('Label Count')

plt.show()


# In[ ]:


# K-means for EDA 


# In[ ]:


# Elbow method to find the optimal value of K
distortions = []
for i in range(1,7):
    km = KMeans(n_clusters=i,
           init='k-means++',
           n_init=10,
           max_iter=300,
           random_state=0)
    km.fit_predict(X)
    distortions.append(km.inertia_)
plt.plot(range(1,7), distortions, marker ='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.tight_layout()
plt.show()


# In[ ]:


km = KMeans(n_clusters=2,
           init='k-means++',
           n_init=10,
           max_iter=300,
           random_state=1)
y_km = km.fit_predict(scaled_X)


# In[ ]:


y_km


# In[ ]:


homogeneity_score(y, y_km)


# In[ ]:


kmeans = km.fit(scaled_X)
centers = np.array(kmeans.cluster_centers_)
X_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='random').fit_transform(scaled_X)
label = km.fit_predict(scaled_X)

plt.figure(figsize=(10,10))
uniq = np.unique(label)
for i in uniq:
    plt.scatter(X_embedded[label == i , 0] , X_embedded[label == i , 1] , label = i)
plt.scatter(centers[:,0], centers[:,1], marker="x", color='k')
#This is done to find the centroid for each clusters.
plt.legend()
plt.show()


# **Support Vector Machine before balancing**

# In[ ]:


#Support Vector Machine before balancing
svc = SVC(class_weight='balanced')


# In[ ]:


param_grid = {'C':[0.001,0.01,0.1,0.5,1, 10, 100, 1000],'gamma':['scale','auto'], 'kernel': ['linear','rbf']}
grid = GridSearchCV(svc,param_grid)


# In[ ]:


grid.fit(X_train,y_train)


# In[ ]:


grid.best_params_


# In[ ]:


svm = SVC(C=100, gamma= 'scale', kernel = 'linear')


# In[ ]:


svm.fit(X_train, y_train)


# In[ ]:


dump(svm, 'svm.joblib')
loaded_svm = load('svm.joblib')


# In[ ]:


y_pred_svm = loaded_svm.predict(X_test)


# In[ ]:


print(classification_report(y_test, y_pred_svm))


# In[ ]:


plot_confusion_matrix(loaded_svm,X_test,y_test)


# In[ ]:


get_ipython().system('pip install scikit-plot')


# In[ ]:


import scikitplot as skplt


# In[ ]:


svc_disp = RocCurveDisplay.from_estimator(loaded_svm, X_test, y_test)
plt.title('ROC SVM before Balancing')
plt.show()


# In[ ]:


# Evaluation metrics
print('Accuracy : %3f' % accuracy_score(y_test, y_pred_svm))
print('Precision : %3f' % precision_score(y_test, y_pred_svm))
print('Recall : %3f' % recall_score(y_test, y_pred_svm))
print('F1 : %3f' % f1_score(y_test, y_pred_svm))
print('kappa_statistic : %3f' % cohen_kappa_score(y_test, y_pred_svm))


# In[ ]:


#Support Vector Machine with Balancing
svc_over= SVC()


# In[ ]:


param_grid = {'C':[0.001,0.01,0.1,0.5,1],'gamma':['scale','auto']}
grid = GridSearchCV(svc_over,param_grid)


# In[ ]:


grid.fit(X_train_over,y_train_over)


# In[ ]:


grid.best_params_


# In[ ]:


svm_over = SVC(C=1, gamma = 'scale')


# In[ ]:


svm_over.fit(X_train_over,y_train_over)


# In[ ]:


dump(svm_over, 'svm_over.joblib')
loaded_svm_over = load('svm_over.joblib')


# In[ ]:


y_pred_svm_over = loaded_svm_over.predict(X_test_over)


# In[ ]:


print(classification_report(y_test_over,y_pred_svm_over))


# In[ ]:


# Evaluation metrics
print('Accuracy : %3f' % accuracy_score(y_test_over, y_pred_svm_over))
print('Precision : %3f' % precision_score(y_test_over, y_pred_svm_over))
print('Recall : %3f' % recall_score(y_test_over, y_pred_svm_over))
print('F1 : %3f' % f1_score(y_test_over, y_pred_svm_over))
print('kappa_statistic : %3f' % cohen_kappa_score(y_test_over, y_pred_svm_over))


# In[ ]:


plot_confusion_matrix(loaded_svm_over, X_test_over,y_test_over)
plt.title('SVM Over Sampled')


# # Feature Selection

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 500, num = 10)]
# Maximum number of levels in tree
max_depth = [2,3,4,5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]


# In[ ]:


param_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_leaf': min_samples_leaf}
print(param_grid)


# In[ ]:


rf_Model = RandomForestClassifier()


# In[ ]:


from sklearn.model_selection import GridSearchCV
rf_Grid = GridSearchCV(estimator = rf_Model, param_grid = param_grid, cv = 3, verbose=2, n_jobs = 4)


# In[ ]:


rf_Grid.fit(X_train_over,y_train_over)


# In[ ]:


rf_Grid.best_params_


# In[ ]:


forest = RandomForestClassifier(max_depth=4, min_samples_leaf= 1, n_estimators= 391, random_state=1)

forest.fit(X_train,y_train)
importances = forest.feature_importances_
pd.DataFrame({"feature": X.columns, "importances": importances}).sort_values("importances", ascending=False).reset_index(drop=True)


# In[ ]:


#Selected Features
X_sf= df_over[['txpackets_slope', 'txbytes_slope', 'rxbytes_slope', 'rxpackets_slope', 'vdawr_reqs_slope',
           'vdawr_bytes_slope','timesys_slope', 'timeusr_slope',  'timecpu_slope', 
          'cputime_slope', 'memrss_slope']]


# In[ ]:


X_train2, X_test2,y_train_sf, y_test_sf = train_test_split(X_sf, y_oversampled, test_size=0.3, random_state =1)


# In[ ]:


X_train_sf = scaler.fit_transform(X_train2)
X_test_sf = scaler.transform(X_test2)


# # SVM with Selected Features

# In[ ]:


svc = SVC()


# In[ ]:


param_grid = {'C':[0.001,0.01,0.1,0.5,1, 10, 100, 1000],'gamma':['scale','auto'],
             'kernel': ['linear','rbf']}
grid = GridSearchCV(svc,param_grid)


# In[ ]:


grid.fit(X_train_sf,y_train_sf)


# In[ ]:


grid.best_params_


# In[ ]:


svm_sf = SVC(C=100, gamma= 'scale', kernel = 'rbf')


# In[ ]:


svm_sf.fit(X_train_sf, y_train_sf)


# In[ ]:


dump(svm_sf, 'svm_sf.joblib')
loaded_svm_sf = load('svm_sf.joblib')


# In[ ]:


y_pred_svm_sf = loaded_svm_sf.predict(X_test_sf)


# In[ ]:


print(classification_report(y_test_sf, y_pred_svm_sf))


# In[ ]:


# Evaluation metrics
print('Accuracy : %3f' % accuracy_score(y_test_sf, y_pred_svm_sf))
print('Precision : %3f' % precision_score(y_test_sf, y_pred_svm_sf))
print('Recall : %3f' % recall_score(y_test_sf, y_pred_svm_sf))
print('F1 : %3f' % f1_score(y_test_sf, y_pred_svm_sf))
print('kappa_statistic : %3f' % cohen_kappa_score(y_test_sf, y_pred_svm_sf))


# In[ ]:


plot_confusion_matrix(loaded_svm_sf,X_test_sf,y_test_sf)


# # Naive Bayes before Balancing

# In[ ]:


#Naive Bayes without Balancing
bnb=BernoulliNB(class_prior=[0.74,0.26], alpha=1 )
bnb.fit(X_train, y_train)


# In[ ]:


dump(bnb, 'bnb.joblib')
loaded_nb_model = load('bnb.joblib')


# In[ ]:


y_pred_nb = loaded_nb_model.predict(X_test)


# In[ ]:


print('Classification Report')
print(classification_report(y_test, y_pred_nb))


# In[ ]:


plot_confusion_matrix(loaded_nb_model,X_test,y_test)
plt.show()


# In[ ]:


# Evaluation metrics
print('Accuracy : %3f' % accuracy_score(y_test, y_pred_nb))
print('Precision : %3f' % precision_score(y_test, y_pred_nb))
print('Recall : %3f' % recall_score(y_test, y_pred_nb))
print('F1 : %3f' % f1_score(y_test, y_pred_nb))
print('kappa_statistic : %3f' % cohen_kappa_score(y_test, y_pred_nb))


# In[ ]:


skplt.metrics.plot_roc(y_test, loaded_nb_model.predict_proba(X_test), plot_micro=False)
plt.title('ROC Naive bayes Before Balancing')
plt.show()


# # Naive Bayes after Balancing

# In[ ]:



bnb_over=BernoulliNB(class_prior=[0.5,0.5], alpha=1 )
bnb_over.fit(X_train_over, y_train_over)


# In[ ]:


dump(bnb_over, 'bnb_over.joblib')
loaded_nb_over_model = load('bnb_over.joblib')


# In[ ]:


y_pred_nb_over = loaded_nb_over_model.predict(X_test_over)


# In[ ]:


print('Classification Report')
print(classification_report(y_test_over, y_pred_nb_over))


# In[ ]:


plot_confusion_matrix(loaded_nb_over_model,X_test_over,y_test_over)
plt.show()


# In[ ]:


# Evaluation metrics
print('Accuracy : %3f' % accuracy_score(y_test_over, y_pred_nb_over))
print('Precision : %3f' % precision_score(y_test_over, y_pred_nb_over))
print('Recall : %3f' % recall_score(y_test_over, y_pred_nb_over))
print('F1 : %3f' % f1_score(y_test_over, y_pred_nb_over))
print('kappa_statistic : %3f' % cohen_kappa_score(y_test_over, y_pred_nb_over))


# In[ ]:


skplt.metrics.plot_roc(y_test_over, loaded_nb_over_model.predict_proba(X_test_over), plot_micro=False)
plt.title('ROC Naive Bayes Over Sampled')
plt.show()


# # Naive Bayes with Selected Features

# In[ ]:


#Naive Bayes with selected Features
bnb_sf=BernoulliNB(class_prior=[0.5,0.5], alpha=1 )
bnb_sf.fit(X_train_sf, y_train_sf)


# In[ ]:


dump(bnb_sf, 'bnb_sf.joblib')
loaded_nb_model = load('bnb_sf.joblib')


# In[ ]:


y_pred_nb_sf = loaded_nb_model.predict(X_test_sf)


# In[ ]:


print('Classification Report')
print(classification_report(y_test_sf, y_pred_nb_sf))


# In[ ]:


plot_confusion_matrix(loaded_nb_model,X_test_sf,y_test_sf)


# In[ ]:


# Evaluation metrics
print('Accuracy : %3f' % accuracy_score(y_test_sf, y_pred_nb_sf))
print('Precision : %3f' % precision_score(y_test_sf, y_pred_nb_sf))
print('Recall : %3f' % recall_score(y_test_sf, y_pred_nb_sf))
print('F1 : %3f' % f1_score(y_test_sf, y_pred_nb_sf))
print('kappa_statistic : %3f' % cohen_kappa_score(y_test_sf, y_pred_nb_sf))


# In[ ]:


skplt.metrics.plot_roc(y_test_sf, loaded_nb_model.predict_proba(X_test_sf), plot_micro=False)
plt.title('ROC Naive Bayes Selected Features')
plt.show()


# In[ ]:


#Deepali


#  **Decision Tree on Raw Data- Gini Impurity**

# In[ ]:


get_ipython().system('pip install joblib')


# In[ ]:


#Decision Tree Model
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import joblib

dtmodeldump = DecisionTreeClassifier().fit(X_train,y_train)
joblib.dump(dtmodeldump, 'dtmodeldump.joblib')
dtmodel = joblib.load('dtmodeldump.joblib')
y_pred_dt = dtmodel.predict(X_test)


# RUN HERE

# In[ ]:


dtmodel = joblib.load('dtmodeldump.joblib')
y_pred_dt = dtmodel.predict(X_test)


# In[ ]:


names=X_train.columns
condn=["Normal","Attack"]
import graphviz 
dot_data = tree.export_graphviz(dtmodel, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("X_test") 
dot_data = tree.export_graphviz(dtmodel, out_file=None, 
                   feature_names=names,  
                      class_names=condn,
          filled=True, rounded=True,  
                   special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 


# In[ ]:


accuracy = accuracy_score(y_test,y_pred_dt)
c_report = classification_report(y_pred_dt, y_test)
c_matrix = confusion_matrix(y_test, y_pred_dt)


# In[ ]:


print("Classification report:")
print("Accuracy: ", accuracy)
print(c_report)
print("Confusion matrix:")
print(c_matrix)


# In[ ]:


plot_confusion_matrix(dtmodel,X_test,y_test)
skplt.metrics.plot_roc(y_test, dtmodel.predict_proba(X_test), plot_micro=False)
plt.show()


# **Pruning the Decison Tree**

# In[ ]:


path = dtmodel.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
print(ccp_alphas)


# In[ ]:


clfs = []
for ccp_alpha in ccp_alphas:
    clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)


# In[ ]:


clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]
node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
plt.scatter(ccp_alphas,node_counts)
plt.scatter(ccp_alphas,depth)
plt.plot(ccp_alphas,node_counts,label='no of nodes',drawstyle="steps-post")
plt.plot(ccp_alphas,depth,label='depth',drawstyle="steps-post")
plt.legend()
plt.show()


# In[ ]:


train_acc = []
test_acc = []
for c in clfs:
    y_train_pred = c.predict(X_train)
    y_test_pred = c.predict(X_test)
    train_acc.append(accuracy_score(y_train_pred,y_train))
    test_acc.append(accuracy_score(y_test_pred,y_test))

plt.scatter(ccp_alphas,train_acc)
plt.scatter(ccp_alphas,test_acc)
plt.plot(ccp_alphas,train_acc,label='train_accuracy',drawstyle="steps-post")
plt.plot(ccp_alphas,test_acc,label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Accuracy vs alpha')
plt.show()


# In[ ]:


clf_dump = tree.DecisionTreeClassifier(random_state=0,ccp_alpha=0.010)
clf_dump.fit(X_train,y_train)
joblib.dump(clf_dump,'clf_dump.joblib')


# RUN HERE

# In[ ]:


clf_=joblib.load('clf_dump.joblib')
y_train_pred = clf_.predict(X_train)
y_test_pred = clf_.predict(X_test)

print(f'Train score {accuracy_score(y_train_pred,y_train)}')
print(f'Test score {accuracy_score(y_test_pred,y_test)}')

accuracy = accuracy_score(y_test,y_test_pred)
c_report = classification_report(y_test_pred,y_test)
c_matrix = confusion_matrix(y_test, y_test_pred)
print("Classification report:")
print("Accuracy: ", accuracy)
print(c_report)
print("Confusion matrix:")
print(c_matrix)


# In[ ]:


plt.figure(figsize=(20,20))
features = X_train.columns
classes = ['Attack','Normal']
tree.plot_tree(clf_,feature_names=features,class_names=classes,filled=True)
plt.show()


# **Decion Tree on Raw Data- Entropy & Information Gain**

# In[ ]:


#DTree using entropy and information gain
dtmodel_entropydump = DecisionTreeClassifier(criterion="entropy").fit(X_train,y_train)
joblib.dump(dtmodel_entropydump,'dtmodel_entropydump.joblib')


# RUN HERE

# In[ ]:


dtmodel_entropy=joblib.load('dtmodel_entropydump.joblib')
y_pred_dt_entropy = dtmodel_entropy.predict(X_test)
y_pred_dt_entropy = dtmodel_entropy.predict(X_test)


# In[ ]:


names=X_train.columns
condn=["Normal","Attack"]
import graphviz 
dot_data = tree.export_graphviz(dtmodel_entropy, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("X_test") 
dot_data = tree.export_graphviz(dtmodel_entropy, out_file=None, 
                   feature_names=names,  
                      class_names=condn,
          filled=True, rounded=True,  
                   special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 


# In[ ]:


accuracy = accuracy_score(y_test,y_pred_dt_entropy)
c_report = classification_report(y_pred_dt_entropy, y_test)
c_matrix = confusion_matrix(y_test, y_pred_dt_entropy)


# In[ ]:


print("Classification report:")
print("Accuracy: ", accuracy)
print(c_report)
print("Confusion matrix:")
print(c_matrix)


# In[ ]:


plot_confusion_matrix(dtmodel_entropy,X_test,y_test)
skplt.metrics.plot_roc(y_test, dtmodel_entropy.predict_proba(X_test), plot_micro=False)
plt.show()


# **Grid Search on Decision Tree-Gini Index**

# In[ ]:


from sklearn.model_selection import GridSearchCV
params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
grid_search_cvdump = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)
grid_search_cvdump.fit(X_train, y_train)


# In[ ]:


grid_search_cvdump.best_estimator_
joblib.dump(grid_search_cvdump,'grid_search_cvdump.joblib')


# RUN HERE

# In[ ]:


grid_search_cv=joblib.load('grid_search_cvdump.joblib')


# In[ ]:


names=X_train.columns
condn=["Normal","Attack"]
import graphviz 
dot_data = tree.export_graphviz(grid_search_cv.best_estimator_, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("X_test") 
dot_data = tree.export_graphviz(grid_search_cv.best_estimator_, out_file=None, 
                   feature_names=names,  
                      class_names=condn,
          filled=True, rounded=True,  
                   special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 


# In[ ]:


y_pred_gini = grid_search_cv.best_estimator_.predict(X_test)
accuracy = accuracy_score(y_test,y_pred_gini)
c_report = classification_report(y_pred_gini, y_test)
c_matrix = confusion_matrix(y_test, y_pred_gini)
print("Classification report:")
print("Accuracy: ", accuracy)
print(c_report)
print("Confusion matrix:")
print(c_matrix)
plot_confusion_matrix(grid_search_cv,X_test,y_test)


# In[ ]:


params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
grid_search_cv_entropydump = GridSearchCV(DecisionTreeClassifier(criterion="entropy",random_state=42), params, verbose=1, cv=3)
grid_search_cv_entropydump.fit(X_train, y_train)


# **Grid Search On Decision Tree-Entropy**

# In[ ]:


params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
grid_search_cv_entropydump = GridSearchCV(DecisionTreeClassifier(criterion="entropy",random_state=42), params, verbose=1, cv=3)
grid_search_cv_entropydump.fit(X_train, y_train)


# In[ ]:


grid_search_cv_entropydump.best_estimator_
joblib.dump(grid_search_cv_entropydump,'grid_search_cv_entropydump.joblib')


# RUN HERE

# In[ ]:


grid_search_cv_entropy=joblib.load('grid_search_cv_entropydump.joblib')


# In[ ]:


names=X_train.columns
condn=["Normal","Attack"]
import graphviz 
dot_data = tree.export_graphviz(grid_search_cv_entropy.best_estimator_, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("X_test") 
dot_data = tree.export_graphviz(grid_search_cv_entropy.best_estimator_, out_file=None, 
                   feature_names=names,  
                      class_names=condn,
          filled=True, rounded=True,  
                   special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 


# In[ ]:


y_pred_entropy = grid_search_cv_entropy.best_estimator_.predict(X_test)
accuracy = accuracy_score(y_test,y_pred_entropy)
c_report = classification_report(y_pred_entropy, y_test)
c_matrix = confusion_matrix(y_test, y_pred_entropy)
print("Classification report:")
print("Accuracy: ", accuracy)
print(c_report)
print("Confusion matrix:")
print(c_matrix)
plot_confusion_matrix(grid_search_cv_entropy,X_test,y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# **Decision Tree on Oversampled Data- Gini Impurity**

# In[ ]:


#Decision Tree on Oversampled Data

dtmodel_overdump = DecisionTreeClassifier().fit(X_train_over,y_train_over)
joblib.dump(dtmodel_overdump,'dtmodel_overdump.joblib')


# RUN HERE

# In[ ]:


dtmodel_over=joblib.load('dtmodel_overdump.joblib')
y_pred_over_dt = dtmodel.predict(X_test_over)


# In[ ]:


names=X_train.columns
condn=["Normal","Attack"]
import graphviz 
dot_data = tree.export_graphviz(dtmodel_over, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("X_test_over") 
dot_data = tree.export_graphviz(dtmodel_over, out_file=None, 
                   feature_names=names,  
                      class_names=condn,
          filled=True, rounded=True,  
                   special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 


# In[ ]:



accuracy = accuracy_score(y_test_over,y_pred_over_dt)
c_report = classification_report(y_pred_over_dt, y_test_over)
c_matrix = confusion_matrix(y_test_over, y_pred_over_dt)


# In[ ]:


print("Classification report:")
print("Accuracy: ", accuracy)
print(c_report)
print("Confusion matrix:")
print(c_matrix)


# In[ ]:


plot_confusion_matrix(dtmodel_over,X_test_over,y_test_over)


# In[ ]:


skplt.metrics.plot_roc(y_test_over, dtmodel_over.predict_proba(X_test_over), plot_micro=False)
plt.show()


# **Pruned Decision Tree**

# In[ ]:


path = dtmodel_over.cost_complexity_pruning_path(X_train_over, y_train_over)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
print(ccp_alphas)


# In[ ]:


clfs = []
for ccp_alpha in ccp_alphas:
    clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train_over, y_train_over)
    clfs.append(clf)


# In[ ]:


clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]
node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
plt.scatter(ccp_alphas,node_counts)
plt.scatter(ccp_alphas,depth)
plt.plot(ccp_alphas,node_counts,label='no of nodes',drawstyle="steps-post")
plt.plot(ccp_alphas,depth,label='depth',drawstyle="steps-post")
plt.legend()
plt.show()


# In[ ]:


train_acc = []
test_acc = []
for c in clfs:
    y_train_pred = c.predict(X_train_over)
    y_test_pred = c.predict(X_test_over)
    train_acc.append(accuracy_score(y_train_pred,y_train_over))
    test_acc.append(accuracy_score(y_test_pred,y_test_over))

plt.scatter(ccp_alphas,train_acc)
plt.scatter(ccp_alphas,test_acc)
plt.plot(ccp_alphas,train_acc,label='train_accuracy',drawstyle="steps-post")
plt.plot(ccp_alphas,test_acc,label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Accuracy vs alpha')
plt.show()


# In[ ]:


clf_2dump = tree.DecisionTreeClassifier(random_state=0,ccp_alpha=0.0010)
clf_2dump.fit(X_train_over,y_train_over)
joblib.dump(clf_2dump,'clf_2dump.joblib')


# In[ ]:


clf_=joblib.load('clf_2dump.joblib')
y_train_pred = clf_.predict(X_train_over)
y_test_pred = clf_.predict(X_test_over)

print(f'Train score {accuracy_score(y_train_pred,y_train_over)}')
print(f'Test score {accuracy_score(y_test_pred,y_test_over)}')
accuracy = accuracy_score(y_test_over,y_test_pred)
c_report = classification_report(y_test_pred,y_test_over)
c_matrix = confusion_matrix(y_test_over, y_test_pred)
print("Classification report:")
print("Accuracy: ", accuracy)
print(c_report)
print("Confusion matrix:")
print(c_matrix)


# In[ ]:


plt.figure(figsize=(20,20))
features = X_oversampled.columns
classes = ['Attack','Normal']
tree.plot_tree(clf_,feature_names=features,class_names=classes,filled=True)
plt.show()


# **Decision Tree on Oversampled Data- Entropy & Information Gain**

# In[ ]:


dtmodel_over_entropydump = DecisionTreeClassifier(criterion="entropy").fit(X_train_over,y_train_over)
joblib.dump(dtmodel_over_entropydump,'dtmodel_over_entropydump.joblib')


# RUN HERE

# In[ ]:


dtmodel_over_entropy=joblib.load('dtmodel_over_entropydump.joblib')
y_pred_over_dt_entropy = dtmodel.predict(X_test_over)


# In[ ]:


names=X_train.columns
condn=["Normal","Attack"]
import graphviz 
dot_data = tree.export_graphviz(dtmodel_over_entropy, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("X_test_over") 
dot_data = tree.export_graphviz(dtmodel_over_entropy, out_file=None, 
                   feature_names=names,  
                      class_names=condn,
          filled=True, rounded=True,  
                   special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 


# In[ ]:


accuracyen = accuracy_score(y_test_over,y_pred_over_dt_entropy)
c_reporten = classification_report(y_pred_over_dt_entropy, y_test_over)
c_matrixen = confusion_matrix(y_test_over, y_pred_over_dt_entropy)
print("Classification report:")
print("Accuracy: ", accuracyen)
print(c_reporten)
print("Confusion matrix:")
print(c_matrixen)
plot_confusion_matrix(dtmodel_over_entropy,X_test_over,y_test_over)
skplt.metrics.plot_roc(y_test_over, dtmodel_over.predict_proba(X_test_over), plot_micro=False)
plt.show()


# **Grid Search on Oversampled Decision Tree-Gini Impurity**

# In[ ]:


params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
grid_search_cv_overdump = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)
grid_search_cv_overdump.fit(X_train_over, y_train_over)


# In[ ]:


grid_search_cv_overdump.best_estimator_
joblib.dump(grid_search_cv_overdump.best_estimator_,'grid_search_cv_overdump.best_estimator_.joblib')


# RUN HERE

# In[ ]:


grid_search_cv_over_best_estimator_=joblib.load('grid_search_cv_overdump.best_estimator_.joblib')


# In[ ]:



names=X_train.columns
condn=["Normal","Attack"]
import graphviz 
dot_data = tree.export_graphviz(grid_search_cv_over_best_estimator_, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("X_test_over") 
dot_data = tree.export_graphviz(grid_search_cv_over_best_estimator_, out_file=None, 
                   feature_names=names,  
                      class_names=condn,
          filled=True, rounded=True,  
                   special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 


# In[ ]:


y_pred_over = grid_search_cv_over_best_estimator_.predict(X_test_over)
accuracy = accuracy_score(y_test_over,y_pred_over)
c_report = classification_report(y_pred_over, y_test_over)
c_matrix = confusion_matrix(y_test_over, y_pred_over)
print("Classification report:")
print("Accuracy: ", accuracy)
print(c_report)
print("Confusion matrix:")
print(c_matrix)
plot_confusion_matrix(grid_search_cv_over_best_estimator_,X_test_over,y_test_over)


# In[ ]:


skplt.metrics.plot_roc(y_test_over, dtmodel_over.predict_proba(X_test_over), plot_micro=False)
plt.show()


# **Grid Search On oversampled Decision tree-Entropy**

# In[ ]:


params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
grid_search_cv_over_entropydump = GridSearchCV(DecisionTreeClassifier(criterion="entropy",random_state=42), params, verbose=1, cv=3)
grid_search_cv_over_entropydump.fit(X_train_over, y_train_over)


# In[ ]:


grid_search_cv_over_entropydump.best_estimator_
joblib.dump(grid_search_cv_over_entropydump.best_estimator_,'grid_search_cv_over_entropydump.best_estimator_.joblib')


# RUN HERE

# In[ ]:


grid_search_cv_over_entropy_best_estimator_=joblib.load('grid_search_cv_over_entropydump.best_estimator_.joblib')


# In[ ]:


names=X_train.columns
condn=["Normal","Attack"]
import graphviz 
dot_data = tree.export_graphviz(grid_search_cv_over_entropy_best_estimator_, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("X_test_over") 
dot_data = tree.export_graphviz(grid_search_cv_over_entropy_best_estimator_, out_file=None, 
                   feature_names=names,  
                      class_names=condn,
          filled=True, rounded=True,  
                   special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 


# In[ ]:


y_pred_entropyover = grid_search_cv_over_entropy_best_estimator_.predict(X_test_over)
accuracy = accuracy_score(y_test_over,y_pred_entropyover)
c_report = classification_report(y_pred_entropyover, y_test_over)
c_matrix = confusion_matrix(y_test_over, y_pred_entropyover)
print("Classification report:")
print("Accuracy: ", accuracy)
print(c_report)
print("Confusion matrix:")
print(c_matrix)
plot_confusion_matrix(grid_search_cv_over_entropy_best_estimator_,X_test_over,y_test_over)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# Poojitha


# ## KNN on Raw dataset

# In[ ]:


#train test split
X_train5, X_test5,y_train5, y_test5 = train_test_split(X, y, test_size=0.3, random_state =1)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=7)
classifier.fit(X_train5, y_train5)


# In[ ]:


y_pred = classifier.predict(X_test5)


# In[ ]:


from sklearn.metrics import classification_report

accuracy = accuracy_score(y_test5, y_pred)
c_report = classification_report(y_pred, y_test5)
c_matrix = confusion_matrix(y_test5, y_pred)
print("Classification report:")
print("Accuracy: ", accuracy)
print(c_report)
print("Confusion matrix:")
print(c_matrix)
plot_confusion_matrix(classifier,X_test5,y_test5)


# In[ ]:


print('kappa_statistic : %3f' % cohen_kappa_score(y_test5, y_pred))


# In[ ]:


error = []
# Calculating the error rate for K-values between 1 and 30
for i in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train5, y_train5)
    pred_i = knn.predict(X_test5)
    error.append(np.mean(pred_i != y_test5))


# In[ ]:


plt.figure(figsize=(12, 5))
plt.plot(range(1, 30), error, color='red', marker='o',
        markerfacecolor='yellow', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')


# ## With Lasso and Ridge Regression
# 

# In[ ]:


from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from statistics import mean


# In[ ]:


# Dividing the data into training and testing set
X_train6, X_test6, y_train6, y_test6 = train_test_split(X, y, test_size=0.3, random_state =1)


# In[ ]:


# List to maintain the different cross-validation scores
cross_val_scores_ridge = []
 
# List to maintain the different values of alpha
alpha = []
 
# Loop to compute the different values of cross-validation scores
for i in range(1, 9):
    ridgeModel = Ridge(alpha = i * 0.25)
    ridgeModel.fit(X_train6, y_train6)
    scores = cross_val_score(ridgeModel, X, y, cv = 10)
    avg_cross_val_score = mean(scores)*100
    cross_val_scores_ridge.append(avg_cross_val_score)
    alpha.append(i * 0.25)
 
# Loop to print the different values of cross-validation scores
for i in range(0, len(alpha)):
    print(str(alpha[i])+' : '+str(cross_val_scores_ridge[i]))


# In[ ]:


# Building and fitting the Ridge Regression model
ridgeModelChosen = Ridge(alpha = 2)
ridgeModelChosen.fit(X_train6, y_train6)
 
# Evaluating the Ridge Regression model
print(ridgeModelChosen.score(X_test6, y_test6))


# In[ ]:


# List to maintain the cross-validation scores
cross_val_scores_lasso = []
 
# List to maintain the different values of Lambda
Lambda = []
 
# Loop to compute the cross-validation scores
for i in range(1, 9):
    lassoModel = Lasso(alpha = i * 0.25, tol = 0.0925)
    lassoModel.fit(X_train6, y_train6)
    scores = cross_val_score(lassoModel, X, y, cv = 10)
    avg_cross_val_score = mean(scores)*100
    cross_val_scores_lasso.append(avg_cross_val_score)
    Lambda.append(i * 0.25)
 
# Loop to print the different values of cross-validation scores
for i in range(0, len(alpha)):
    print(str(alpha[i])+' : '+str(cross_val_scores_lasso[i]))


# In[ ]:


# Building and fitting the Lasso Regression Model
lassoModelChosen = Lasso(alpha = 2, tol = 0.0925)
lassoModelChosen.fit(X_train6, y_train6)
 
# Evaluating the Lasso Regression model
print(lassoModelChosen.score(X_test6, y_test6))


# In[ ]:


# Building the two lists for visualization
models = ['Ridge Regression', 'Lasso Regression']
scores = [ridgeModelChosen.score(X_test6, y_test6),
         lassoModelChosen.score(X_test6, y_test6)]
 
# Building the dictionary to compare the scores
mapping = {}
mapping['Ridge Regression'] = ridgeModelChosen.score(X_test6, y_test6)
mapping['Lasso Regression'] = lassoModelChosen.score(X_test6, y_test6)
 
# Printing the scores for different models
for key, val in mapping.items():
    print(str(key)+' : '+str(val))


# In[ ]:


y_pred_ridge = ridgeModelChosen.predict(X_test6)
accuracy = accuracy_score(y_test6,np.round(abs(y_pred_ridge))) 
c_report = classification_report(np.round(abs(y_pred_ridge)), y_test6)
c_matrix = confusion_matrix(y_test6, np.round(abs(y_pred_ridge)))
print("Classification report:")
print("Accuracy: ", accuracy)
print(c_report)
print("Confusion matrix:")
print(c_matrix)


# In[ ]:


y_pred_lasso = lassoModelChosen.predict(X_test6)
accuracy = accuracy_score(y_test6,np.round(abs(y_pred_lasso))) 
c_report = classification_report(np.round(abs(y_pred_lasso)), y_test6)
c_matrix = confusion_matrix(y_test6, np.round(abs(y_pred_lasso)))
print("Classification report:")
print("Accuracy: ", accuracy)
print(c_report)
print("Confusion matrix:")
print(c_matrix)


# In[ ]:


print('kappa_statistic (ridge) : %3f' % cohen_kappa_score(y_test6, np.round(abs(y_pred_ridge))))


# In[ ]:


print('kappa_statistic (lasso): %3f' % cohen_kappa_score(y_test6, np.round(abs(y_pred_ridge))))


# ## KNN + GA
# 

# In[ ]:


get_ipython().system('pip install --user pygad')
#import pygad
import random 


# In[ ]:


#defining various steps required for the genetic algorithm
def initilization_of_population(size,n_feat):
    population = []
    for i in range(size):
        chromosome = np.ones(n_feat,dtype=np.bool)
        chromosome[:int(0.3*n_feat)]=False
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population

def fitness_score(population):
    scores = []
    for chromosome in population:
        classifier.fit(X_train.iloc[:,chromosome],y_train)
        predictions = classifier.predict(X_test.iloc[:,chromosome])
        scores.append(accuracy_score(y_test,predictions))
    scores, population = np.array(scores), np.array(population) 
    inds = np.argsort(scores)
    return list(scores[inds][::-1]), list(population[inds,:][::-1])

def selection(pop_after_fit,n_parents):
    population_nextgen = []
    for i in range(n_parents):
        population_nextgen.append(pop_after_fit[i])
    return population_nextgen

def crossover(pop_after_sel):
    population_nextgen=pop_after_sel
    for i in range(len(pop_after_sel)):
        child=pop_after_sel[i]
        child[3:7]=pop_after_sel[(i+1)%len(pop_after_sel)][3:7]
        population_nextgen.append(child)
    return population_nextgen

def mutation(pop_after_cross,mutation_rate):
    population_nextgen = []
    for i in range(0,len(pop_after_cross)):
        chromosome = pop_after_cross[i]
        for j in range(len(chromosome)):
            if random.random() < mutation_rate:
                chromosome[j]= not chromosome[j]
        population_nextgen.append(chromosome)
    #print(population_nextgen)
    return population_nextgen

def generations(size,n_feat,n_parents,mutation_rate,n_gen,X_train,
                                   X_test, y_train, y_test):
    best_chromo= []
    best_score= []
    population_nextgen=initilization_of_population(size,n_feat)
    for i in range(n_gen):
        scores, pop_after_fit = fitness_score(population_nextgen)
        print(scores[:2])
        pop_after_sel = selection(pop_after_fit,n_parents)
        pop_after_cross = crossover(pop_after_sel)
        population_nextgen = mutation(pop_after_cross,mutation_rate)
        best_chromo.append(pop_after_fit[0])
        best_score.append(scores[0])
    return best_chromo,best_score


# In[ ]:


chromo,score=generations(size=200,n_feat=20,n_parents=100,mutation_rate=0.10,
                     n_gen=38,X_train=X_train6,X_test=X_test6,y_train=y_train6,y_test=y_test6)


# In[ ]:


classifier.fit(X_train6.iloc[:,chromo[-1]],y_train6)


# In[ ]:


predictions = classifier.predict(X_test6.iloc[:,chromo[-1]])


# In[ ]:


print("Accuracy score after genetic algorithm is= "+str(accuracy_score(y_test6,predictions)))


# In[ ]:


from sklearn.metrics import classification_report

accuracy = accuracy_score(y_test6, predictions)
c_report = classification_report(predictions, y_test6)
c_matrix = confusion_matrix(y_test6, predictions)
print("Classification report:")
print("Accuracy: ", accuracy)
print(c_report)
print("Confusion matrix:")
print(c_matrix)


# In[ ]:


print('kappa_statistic (lasso): %3f' % cohen_kappa_score(y_test6, predictions))


# In[ ]:





# ## KNN on oversampled data

# In[ ]:


#train test split
X_train_over5, X_test_over5,y_train_over5, y_test_over5 = train_test_split(X_oversampled, y_oversampled, test_size=0.3, random_state =1)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=7)
classifier.fit(X_train_over5, y_train_over5)


# In[ ]:


y_pred_over = classifier.predict(X_test_over5)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test_over5, y_pred_over))
from sklearn.metrics import classification_report

accuracy = accuracy_score(y_test_over5, y_pred_over)
c_report = classification_report(y_pred_over, y_test_over5)
c_matrix = confusion_matrix(y_test_over5, y_pred_over)
print("Classification report:")
print("Accuracy: ", accuracy)
print(c_report)
print("Confusion matrix:")
print(c_matrix)
plot_confusion_matrix(classifier,X_test_over5,y_test_over5)


# In[ ]:


print('kappa_statistic (lasso): %3f' % cohen_kappa_score(y_test_over5, y_pred_over))


# ## KNN on over-sampled date with lasso and ridge regression

# In[ ]:


# Dividing the data into training and testing set
X_train_over6, X_test_over6,y_train_over6, y_test_over6 = train_test_split(X_oversampled, y_oversampled, test_size=0.3, random_state =1)


# In[ ]:


# List to maintain the different cross-validation scores
cross_val_scores_ridge = []
 
# List to maintain the different values of alpha
alpha = []
 
# Loop to compute the different values of cross-validation scores
for i in range(1, 9):
    ridgeModel = Ridge(alpha = i * 0.25)
    ridgeModel.fit(X_train_over6, y_train_over6)
    scores = cross_val_score(ridgeModel, X_oversampled, y_oversampled, cv = 10)
    avg_cross_val_score = mean(scores)*100
    cross_val_scores_ridge.append(avg_cross_val_score)
    alpha.append(i * 0.25)
 
# Loop to print the different values of cross-validation scores
for i in range(0, len(alpha)):
    print(str(alpha[i])+' : '+str(cross_val_scores_ridge[i]))


# In[ ]:


# Building and fitting the Ridge Regression model
ridgeModelChosen = Ridge(alpha = 2)
ridgeModelChosen.fit(X_train_over6, y_train_over6)
 
# Evaluating the Ridge Regression model
print(ridgeModelChosen.score(X_test_over6,y_test_over6))


# In[ ]:


# List to maintain the cross-validation scores
cross_val_scores_lasso = []
 
# List to maintain the different values of Lambda
Lambda = []
 
# Loop to compute the cross-validation scores
for i in range(1, 9):
    lassoModel = Lasso(alpha = i * 0.25, tol = 0.0925)
    lassoModel.fit(X_train_over6, y_train_over6)
    scores = cross_val_score(lassoModel, X_oversampled, y_oversampled, cv = 10)
    avg_cross_val_score = mean(scores)*100
    cross_val_scores_lasso.append(avg_cross_val_score)
    Lambda.append(i * 0.25)
 
# Loop to print the different values of cross-validation scores
for i in range(0, len(alpha)):
    print(str(alpha[i])+' : '+str(cross_val_scores_lasso[i]))


# In[ ]:


# Building and fitting the Lasso Regression Model
lassoModelChosen = Lasso(alpha = 2, tol = 0.0925)
lassoModelChosen.fit(X_train_over6, y_train_over6)
 
# Evaluating the Lasso Regression model
print(lassoModelChosen.score(X_test_over6,y_test_over6))


# In[ ]:


# Building the two lists for visualization
models = ['Ridge Regression', 'Lasso Regression']
scores = [ridgeModelChosen.score(X_test_over6,y_test_over6),
         lassoModelChosen.score(X_test_over6,y_test_over6)]
 
# Building the dictionary to compare the scores
mapping = {}
mapping['Ridge Regression'] = ridgeModelChosen.score(X_test_over6,y_test_over6)
mapping['Lasso Regression'] = lassoModelChosen.score(X_test_over6,y_test_over6)
 
# Printing the scores for different models
for key, val in mapping.items():
    print(str(key)+' : '+str(val))


# In[ ]:


y_pred_ridge_over = ridgeModelChosen.predict(X_test_over6)
accuracy = accuracy_score(y_test_over6,np.round(abs(y_pred_ridge_over)))
c_report = classification_report(np.round(abs(y_pred_ridge_over)), y_test_over6)
c_matrix = confusion_matrix(y_test_over6, np.round(abs(y_pred_ridge_over)))
print("Classification report:")
print("Accuracy: ", accuracy)
print(c_report)
print("Confusion matrix:")
print(c_matrix)


# In[ ]:


y_pred_lasso_over = lassoModelChosen.predict(X_test_over6)
accuracy = accuracy_score(y_test_over6,np.round(abs(y_pred_lasso_over)))
c_report = classification_report(np.round(abs(y_pred_lasso_over)), y_test_over6)
c_matrix = confusion_matrix(y_test_over6, np.round(abs(y_pred_lasso_over)))
print("Classification report:")
print("Accuracy: ", accuracy)
print(c_report)
print("Confusion matrix:")
print(c_matrix)


# In[ ]:


print('kappa_statistic (ridge): %3f' % cohen_kappa_score(y_test_over6, np.round(abs(y_pred_ridge_over))))


# In[ ]:


print('kappa_statistic (lasso): %3f' % cohen_kappa_score(y_test_over6, np.round(abs(y_pred_lasso_over))))


# ##over sampled - knn + ga

# In[ ]:


chromo,score=generations(size=200,n_feat=20,n_parents=100,mutation_rate=0.10,
                     n_gen=38,X_train=X_train_over6,X_test=X_test_over6,y_train=y_train_over6,y_test=y_test_over6)


# In[ ]:


classifier.fit(X_train6.iloc[:,chromo[-1]],y_train6)


# In[ ]:


predictions = classifier.predict(X_test_over6.iloc[:,chromo[-1]])


# In[ ]:


print("Accuracy score after genetic algorithm is= "+str(accuracy_score(y_test_over6,predictions)))


# In[ ]:


from sklearn.metrics import classification_report

accuracy = accuracy_score(y_test_over6, predictions)
c_report = classification_report(predictions, y_test_over6)
c_matrix = confusion_matrix(y_test_over6, predictions)
print("Classification report:")
print("Accuracy: ", accuracy)
print(c_report)
print("Confusion matrix:")
print(c_matrix)


# In[ ]:


print('kappa_statistic (lasso): %3f' % cohen_kappa_score(y_test_over6, predictions))


# In[ ]:


# ## ROC curve
# skplt.metrics.plot_roc(y_test_over6, classifier.predict_proba(X_test_over6), plot_micro=False)
# plt.title('ROC KNN + GA for unbalanced Data')
# plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#Yasaman


# # Xgboost on Imbalance data

# In[ ]:


import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


xgb_cl = xgb.XGBClassifier()

# Init classifier

# Fit
xgb_cl = xgb_cl.fit(X_train, y_train)

# Predict
xgb_preds = xgb_cl.predict(X_test)


# In[ ]:


from xgboost import plot_importance
import matplotlib.pyplot as plt

plot_importance(xgb_cl)
plt.figure(figsize=(10,5))
plt.show()


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, xgb_preds)


# In[ ]:


from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(xgb_cl,X_test,y_test)
plt.show()


# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    "max_depth": [3, 4, 5, 7],
    "learning_rate": [0.1, 0.01, 0.05],
    "gamma": [0, 0.25, 1],
    "reg_lambda": [0, 1, 10],
    "scale_pos_weight": [1, 3, 5],
    "subsample": [0.8],
    "colsample_bytree": [0.5],
}
# Init classifier
xgb_cl = xgb.XGBClassifier(objective="binary:logistic")

# Init Grid Search
grid_cv = GridSearchCV(xgb_cl, param_grid, n_jobs=-1, cv=3, scoring="roc_auc")

# Fit
_ = grid_cv.fit(X_train, y_train)


# In[ ]:


grid_cv.best_params_


# In[ ]:


print('Classification Report')
print(classification_report(y_test, xgb_preds, digits=4))


# #Xgboost on oversampled data
# 

# In[ ]:


# Fit
xgb_cl_over = xgb_cl.fit(X_train_over, y_train_over)

# Predict
xgb_preds = xgb_cl_over.predict(X_test_over)


# In[ ]:


plot_importance(xgb_cl_over)
plt.figure(figsize=(10,5))
plt.show()


# In[ ]:


confusion_matrix(y_test_over, xgb_preds)


# In[ ]:


plot_confusion_matrix(xgb_cl_over,X_test_over,y_test_over)
plt.show()


# In[ ]:


# Fit Grid Search for oversampled data
_ = grid_cv.fit(X_train_over, y_train_over)


# In[ ]:


grid_cv.best_params_


# # Logistic Regression on Imbalanced data

# In[ ]:


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0).fit(X_train, y_train)
preds_LR = clf.predict(X_test)


# In[ ]:


from sklearn.metrics import precision_score
print('Classification Report')
print(classification_report(y_test, preds_LR, digits=4))


# In[ ]:


plot_confusion_matrix(clf,X_test,y_test)
plt.show()


# In[ ]:


param_grid={
    "C":np.logspace(-3,3,7), 
    "penalty":["l1","l2"],
    "solver":['liblinear']
    }# l1 lasso l2 ridge
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,param_grid,cv=10)
logreg_cv.fit(X_train,y_train)

print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)


# # Logistic Regression on oversampled data

# In[ ]:


clf = LogisticRegression(random_state=0).fit(X_train_over, y_train_over)
preds_LR = clf.predict(X_test_over)
print('Classification Report')
print(classification_report(y_test_over, preds_LR, digits=4))
plot_confusion_matrix(clf,X_test_over,y_test_over)
plt.show()


# In[ ]:



logreg_cv.fit(X_train_over,y_train_over)

print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)


# In[ ]:





# In[ ]:





# ## Random Forest on In-Balanced Dataset

# In[ ]:





# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer,mean_squared_error

scaler = StandardScaler()
X_train1, X_test1,y_train, y_test = train_test_split(X, y, test_size=0.3, random_state =1)
X_train = scaler.fit_transform(X_train1)
X_test = scaler.transform(X_test1)



rf_model = RandomForestClassifier().fit(X_train,y_train)
print(rf_model)

dump(rf_model, 'rf_model.joblib')
loaded_rf_model= load('rf_model.joblib')

##testing the model
y_pred_rf = loaded_rf_model.predict(X_test)
y_pred_rf


x = metrics.accuracy_score(y_test, y_pred_rf)
print("RF's Accuracy on raw dataset is: ", x*100)
print(classification_report(y_test, y_pred_rf))

# Evaluation metrics
print('Accuracy : %3f' % accuracy_score(y_test, y_pred_rf))
print('Precision : %3f' % precision_score(y_test, y_pred_rf))
print('Recall : %3f' % recall_score(y_test, y_pred_rf))
print('F1 : %3f' % f1_score(y_test, y_pred_rf))
print('kappa_statistic : %3f' % cohen_kappa_score(y_test, y_pred_rf))

##confusion matrix
plot_confusion_matrix(rf_model,X_test,y_test)
plt.show()

## ROC curve

## ROC curve
skplt.metrics.plot_roc(y_test, rf_model.predict_proba(X_test), plot_micro=False)
plt.title('ROC Random Forest for unbalanced Data')
plt.show()


# ### Grid Search On In-Balanced data

# In[ ]:





# In[ ]:


rfc=RandomForestClassifier(random_state=1)
param_grid = { 
    'n_estimators': [1, 100],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [5,6,7,8],
    'criterion' :['gini', 'entropy']
}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)


# In[ ]:


CV_rfc.best_params_


# In[ ]:


rf_model_grid=RandomForestClassifier(random_state=1, max_features='auto', n_estimators= 100, max_depth=5, criterion='gini')

rf_model_grid.fit(X_train, y_train)

dump(rf_model_grid, 'rf_model_grid.joblib')
loaded_rf_model_grid= load('rf_model_grid.joblib')


pred_rf_grid=loaded_rf_model_grid.predict(X_test)
print("Accuracy for Random Forest with grid_search : ",accuracy_score(y_test, pred_rf_grid))


# In[ ]:


# Evaluation metrics
print('Accuracy : %3f' % accuracy_score(y_test, pred_rf_grid))
print('Precision : %3f' % precision_score(y_test, pred_rf_grid))
print('Recall : %3f' % recall_score(y_test, pred_rf_grid))
print('F1 : %3f' % f1_score(y_test, pred_rf_grid))
print('kappa_statistic : %3f' % cohen_kappa_score(y_test, pred_rf_grid))


# ### Random Forest on OverSampled Data

# In[ ]:





# In[ ]:




## training the model
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


rf_model_over = RandomForestClassifier().fit(X_train_over,y_train_over)
print(rf_model_over)

dump(rf_model_over, 'rf_model_over.joblib')
loaded_rf_model_over= load('rf_model_over.joblib')

##testing the model

y_pred_rf_over = loaded_rf_model_over.predict(X_test_over)
y_pred_rf_over

x = metrics.accuracy_score(y_test_over , y_pred_rf_over)
print("RF's Accuracy is on oversampled data : ", x*100)
print(classification_report(y_test_over , y_pred_rf_over))


# Evaluation metrics
print('Accuracy : %3f' % accuracy_score(y_test_over, y_pred_rf_over))
print('Precision : %3f' % precision_score(y_test_over, y_pred_rf_over))
print('Recall : %3f' % recall_score(y_test_over, y_pred_rf_over))
print('F1 : %3f' % f1_score(y_test_over, y_pred_rf_over))
print('kappa_statistic : %3f' % cohen_kappa_score(y_test_over, y_pred_rf_over))


##confusion matrix
plot_confusion_matrix(rf_model,X_test_over,y_test_over)
plt.show()

## ROC curve
skplt.metrics.plot_roc(y_test_over, rf_model_over.predict_proba(X_test_over), plot_micro=False)
plt.show()


# In[ ]:





# ### Grid Search on OverSampling Data

# In[ ]:


rfc=RandomForestClassifier(random_state=42)
param_grid = { 
    'n_estimators': [1, 100],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [5,6,7,8],
    'criterion' :['gini', 'entropy']
}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train_over,y_train_over)


# In[ ]:


CV_rfc.best_params_


# In[ ]:


rf_model_grid_over=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 100, max_depth=8, criterion='gini')

rf_model_grid_over.fit(X_train_over,y_train_over)

dump(rf_model_grid_over, 'rf_model_grid_over.joblib')
loaded_rf_model_grid_over= load('rf_model_grid_over.joblib')

pred_rf_grid_over=loaded_rf_model_grid_over.predict(X_test_over)
print("Accuracy for Random Forest with grid_search : ",accuracy_score(y_test_over, pred_rf_grid_over))


# In[ ]:


# Evaluation metrics
print('Accuracy : %3f' % accuracy_score(y_test_over, pred_rf_grid_over))
print('Precision : %3f' % precision_score(y_test_over, pred_rf_grid_over))
print('Recall : %3f' % recall_score(y_test_over, pred_rf_grid_over))
print('F1 : %3f' % f1_score(y_test_over, pred_rf_grid_over))
print('kappa_statistic : %3f' % cohen_kappa_score(y_test_over, pred_rf_grid_over))


# In[ ]:




