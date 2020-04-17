#loan paid off status
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn import metrics
from sklearn import feature_selection
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import tree
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.feature_selection import chi2



def modelstats1(Xtrain,Xtest,ytrain,ytest):
    stats=[]
    modelnames=["LR","DecisionTree","KNN","NB"]
    models=list()
    models.append(linear_model.LogisticRegression())
    models.append(tree.DecisionTreeClassifier())
    models.append(neighbors.KNeighborsClassifier())
    models.append(naive_bayes.GaussianNB())
    for name,model in zip(modelnames,models):
        if name=="KNN":
            k=[l for l in range(5,17,2)]
            grid={"n_neighbors":k}
            grid_obj = model_selection.GridSearchCV(estimator=model,param_grid=grid,scoring="f1")
            grid_fit =grid_obj.fit(Xtrain,ytrain)
            model = grid_fit.best_estimator_
            model.fit(Xtrain,ytrain)
            name=name+"("+str(grid_fit.best_params_["n_neighbors"])+")"
            print(grid_fit.best_params_)
        else:
            model.fit(Xtrain,ytrain)
        trainprediction=model.predict(Xtrain)
        testprediction=model.predict(Xtest)
        scores=list()
        scores.append(name+"-train")
        scores.append(metrics.accuracy_score(ytrain,trainprediction))
        scores.append(metrics.precision_score(ytrain,trainprediction))
        scores.append(metrics.recall_score(ytrain,trainprediction))
        scores.append(metrics.roc_auc_score(ytrain,trainprediction))
        stats.append(scores)
        scores=list()
        scores.append(name+"-test")
        scores.append(metrics.accuracy_score(ytest,testprediction))
        scores.append(metrics.precision_score(ytest,testprediction))
        scores.append(metrics.recall_score(ytest,testprediction))
        scores.append(metrics.roc_auc_score(ytest,testprediction))
        stats.append(scores)
    
    colnames=["MODELNAME","ACCURACY","PRECISION","RECALL","AUC"]
    return pd.DataFrame(stats,columns=colnames)



df=pd.read_csv("g:/ML with python/pdata/credit_train.csv")
dforig=pd.read_csv("g:/ML with python/pdata/credit_train.csv")

df
df.info()
df.shape     # (100514, 19)   (row ,column)
df.isnull().sum()
df.isnull().sum()/df.shape[0]

df.apply(lambda x: sum(x.isnull()))
df


df["Years in current job"].isnull().sum()
df_new = df[pd.notnull(df['Years in current job'])]
df_new.apply(lambda x: sum(x.isnull()))

df_new['Credit Score'].fillna(value=df_new['Credit Score'].mean(),inplace=True)     
df1=df_new.round(decimals=0)

df1['Annual Income'].fillna(value=df1['Annual Income'].mean(),inplace=True)     
df2=df1.round(decimals=2)

df2['Months since last delinquent'].fillna(value=df2['Months since last delinquent'].mean(),inplace=True)     
df3=df2.round(decimals=0)

df3.dropna(inplace=True)
df3.apply(lambda x: sum(x.isnull()))
df3.shape

hist = df3['Current Loan Amount'].hist(color='red')
plt.title("Histogram for Current Loan Amount")
plt.show()

sns.distplot(df3['Monthly Debt'], kde=True, color='blue', bins=10)

sns.distplot(df3['Annual Income'], kde=False, color='blue', bins=10)

sns.distplot(df3['Years of Credit History'], kde=True, color='green', bins=10)
#
#class_set= df3['Loan Status']
#pd.unique(class_set)
#
#class_set= df3['Home Ownership']
#pd.unique(class_set)

df3['Home Ownership'] = df3['Home Ownership'].map({'Home Mortgage':0,'Own Home':1,'Rent':2,'HaveMortgage':3})
df3['Term'] = df3['Term'].map({'Short Term':0,'Long Term':1})
df3['Loan Status'] = df3['Loan Status'].map({'Fully Paid':0,'Charged Off':1})
df3.shape


sns.boxplot(y='Current Loan Amount',data=df3)

q=df3['Current Loan Amount'].quantile(0.99)
df4=df3[df3['Current Loan Amount']<q]
df4.shape
print(df4.columns)

df3.drop(["Loan ID","Customer ID"],axis=1,inplace=True)


#X1=df4[['Current Loan Amount',
#       'Credit Score', 'Annual Income',
#        'Monthly Debt', 'Years of Credit History',
#       'Months since last delinquent', 'Current Credit Balance',
#       'Maximum Open Credit']]
#Y=df4['Loan Status']


d1={ label:i  for i,label in enumerate(df3["Years in current job"].unique())} # dictionary 
df3["Years in current job"].replace(d1,inplace=True)

d2={ label:i  for i,label in enumerate(df3["Purpose"].unique())} # dictionary 
df3["Purpose"].replace(d2,inplace=True)





X=df3.drop("Loan Status",axis=1)
y=df3["Loan Status"]
scaler=StandardScaler()
scaled_df=scaler.fit_transform(df3[["Current Loan Amount","Credit Score","Annual Income","Monthly Debt","Years of Credit History","Number of Open Accounts",
              "Current Credit Balance","Maximum Open Credit","Months since last delinquent"]])

#[["Current Loan Amount","Credit Score","Annual Income","Monthly Debt","Years of Credit History","Number of Open Accounts",
#              "Current Credit Balance","Maximum Open Credit"]]

scaled_df = pd.DataFrame(scaled_df, columns=["Current Loan Amount","Credit Score","Annual Income","Monthly Debt","Years of Credit History","Number of Open Accounts",
              "Current Credit Balance","Maximum Open Credit","Months since last delinquent"])

notscaled=df3.drop(["Current Loan Amount","Credit Score","Annual Income","Monthly Debt","Years of Credit History","Number of Open Accounts",
              "Current Credit Balance","Maximum Open Credit","Months since last delinquent"],axis=1)

allcol=notscaled.copy()
for col in scaled_df:
    allcol[col]=scaled_df[col].values

X=allcol.drop("Loan Status",axis=1)
y=allcol["Loan Status"]

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2, random_state=66)
modelstats1(Xtrain,Xtest,ytrain,ytest)

#             MODELNAME  ACCURACY  PRECISION    RECALL       AUC
# 0            LR-train  0.821534   0.991140  0.203021  0.601249
# 1             LR-test  0.822809   0.985092  0.202929  0.601028
# 2  DecisionTree-train  1.000000   1.000000  1.000000  1.000000
# 3   DecisionTree-test  0.757259   0.450159  0.434207  0.641677
# 4        KNN(5)-train  0.843978   0.805381  0.397787  0.685066
# 5         KNN(5)-test  0.798640   0.593659  0.287503  0.615765
# 6            NB-train  0.361262   0.256697  0.980623  0.581848
# 7             NB-test  0.356892   0.253457  0.978738  0.579377



df3.drop("Number of Credit Problems",axis=1,inplace=True)

# No improvement


X=df3.drop(["Loan Status","Bankruptcies"],axis=1)
y=df3["Loan Status"]

chi_sq=chi2(X,y)
chi_sq

p_values=pd.Series(chi_sq[1],index=X.columns)
p_values.sort_values(ascending=False,inplace=True)
p_values[:].plot

df3.drop("Purpose",axis=1,inplace=True)
modelstats1(Xtrain,Xtest,ytrain,ytest)
#             MODELNAME  ACCURACY  PRECISION    RECALL       AUC
# 0            LR-train  0.821534   0.991140  0.203021  0.601249
# 1             LR-test  0.822809   0.985092  0.202929  0.601028
# 2  DecisionTree-train  1.000000   1.000000  1.000000  1.000000
# 3   DecisionTree-test  0.754957   0.444852  0.429719  0.638593
# 4        KNN(5)-train  0.843978   0.805381  0.397787  0.685066
# 5         KNN(5)-test  0.798640   0.593659  0.287503  0.615765
# 6            NB-train  0.361262   0.256697  0.980623  0.581848
# 7             NB-test  0.356892   0.253457  0.978738  0.579377










from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100, bootstrap = True,
                               max_features = 'sqrt')


X=df3.drop("Loan Status",axis=1)
y=df3["Loan Status"]


Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2, random_state=67)
modelstats1(Xtrain,Xtest,ytrain,ytest)

#             MODELNAME  ACCURACY  PRECISION    RECALL       AUC
# 0            LR-train  0.819467   0.925856  0.202453  0.598915
# 1             LR-test  0.814230   0.929336  0.199403  0.597466
# 2  DecisionTree-train  1.000000   1.000000  1.000000  1.000000
# 3   DecisionTree-test  0.755689   0.461761  0.439697  0.644283
# 4        KNN(5)-train  0.809148   0.648162  0.305624  0.629163
# 5         KNN(5)-test  0.737902   0.331625  0.148633  0.530149
# 6            NB-train  0.339276   0.251161  0.998349  0.574862
# 7             NB-test  0.345279   0.257919  0.998851  0.575703
#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(Xtrain,ytrain)

ypred=clf.predict(Xtest)

from sklearn import metrics
print("Recall:",metrics.recall_score(ytest, ypred))


X=df3.drop("Loan Status",axis=1)
y=df3["Loan Status"]

chi_sq=chi2(X,y)
chi_sq

p_values=pd.Series(chi_sq[1],index=X.columns)
p_values.sort_values(ascending=False,inplace=True)

df3.drop("Bankruptcies",axis=1,inplace=True)

df3.drop("Monthly Debt",axis=1,inplace=True)

corr=df3.corr()
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2, random_state=68)
modelstats1(Xtrain,Xtest,ytrain,ytest)



#             MODELNAME  ACCURACY  PRECISION    RECALL       AUC
# 0            LR-train  0.818342   0.926085  0.202566  0.598959
# 1             LR-test  0.818519   0.918919  0.200283  0.597620
# 2  DecisionTree-train  1.000000   1.000000  1.000000  1.000000
# 3   DecisionTree-test  0.758305   0.453626  0.433318  0.642185
# 4        KNN(5)-train  0.809409   0.654584  0.309941  0.631463
# 5         KNN(5)-test  0.737170   0.301426  0.139491  0.523615
# 6            NB-train  0.341617   0.253033  0.998301  0.575574
# 7             NB-test  0.337013   0.250636  0.998115  0.573229























