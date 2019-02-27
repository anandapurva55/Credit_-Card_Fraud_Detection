import sys
import numpy
import pandas  
import matplotlib
import seaborn #for co-relation matrix
import scipy
import sklearn

print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(numpy.__version__))
print('Pandas: {}'.format(pandas.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Seaborn: {}'.format(seaborn.__version__))
print('Scipy: {}'.format(scipy.__version__))
print('Sklearn: {}'.format(sklearn.__version__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('creditcard.csv') #Load the dataset from the csv file using pandas

#v1 to v28 are the results of PCA dimensionality reduction that was used to protect sensitive information in this dataset.
Like location or name of the user.

print(data.shape)
print(data.describe())

#For saving computational power and time, we take 10% of the emtire dataset rather than the whole
data = data.sample(frac = 0.1, random_state = 1)
print(data.shape)
print(data.describe())

#Plot histogram of each parameter
data.hist(figsize=(20,20))
plt.show()

#determine no of fraud cases in dataset
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlier_fraction = len(fraud)/float(len(valid)) #necessary to consider this else we would be considering more fraudelent %
print(outlier_fraction)
print('Fraud Cases: {}'.format(len(fraud)))
print('Valid Cases: {}'.format(len(valid)))

#Correlation matrix :- tell us whether or not to remove objects, tell linear relationships b/w attributes
corrmat = data.corr()
fig = plt.figure(figsize = (14,11))
sns.heatmap(corrmat,vmax=0.8, square=True)

#Get all the columns from the dataframe
columns = data.columns.tolist()

#Filter the columns to remove data we donot want
columns = [c for c in columns if c not in ["Class"]]

#Store the variable we will be predicting on
target = "Class"

X = data[columns]
Y = data[target]

#Print the shapes of X and Y
print(X.shape)
print(Y.shape)

from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#define a random state
state = 1

#define the outlier detection methods
classifiers = {
    "Isolation Forest" : IsolationForest(max_samples = len(X), 
                                        contamination = outlier_fraction,
                                        random_state = state),
    "Local Outlier Factor" : LocalOutlierFactor(n_neighbors = 20, contamination = outlier_fraction) 
    
}


#Fit the model
n_outliers = len(fraud)

for i,(clf_name, clf) in enumerate (classifiers.items()):
    
        #fit the data and tag outliers
        if clf_name == "Local Outlier Factor":
                y_pred = clf.fit_predict(X)
                scores_pred = clf.negative_outlier_factor_
       
        else:
                clf.fit(X)
                scores_pred = clf.decision_function(X)
                y_pred = clf.predict(X)
            
        #Reshape the prediction value to 0 for valid and 1 for fraud
        y_pred[y_pred == 1] = 0
        y_pred[y_pred == -1] = 1
            
        n_errors = (y_pred != Y).sum()
            
        #Run classification metrices
        print('{}: {}'.format(clf_name,n_errors))
        print(accuracy_score(Y, y_pred))
        print(classification_report(Y, y_pred))


