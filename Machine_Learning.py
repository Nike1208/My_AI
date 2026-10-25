import numpy
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix 
from sklearn.neighbors import KNeighborsClassifier
import pandas
import pandas as pd
from pathlib import Path
 

#Variables


#StandardScaler is used to standardize the features by removing the mean and scaling to unit variance.

"""
scale = StandardScaler()
data_file = Path(__file__).parent / "data1.csv"
df = pandas.read_csv(data_file)

X = df[['Weight', 'Volume']]
y = df['CO2']

scaledX = scale.fit_transform(X)

regr = linear_model.LinearRegression()
regr.fit(scaledX, y)

scaled = scale.transform([[2300, 1.3]])

#predcti the value depentent on the CO2 Emission
predictedCO2 = regr.predict([scaled[0]])
print(predictedCO2)

"""

"""
#Decision Tree Classifier

df = pandas.read_csv("data.csv")


d = {'UK':0, 'USA':1,'N':2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES':1, 'NO':0}
df['Go'] = df['Go'].map(d)

features = ['Age', 'Experience', 'Rank', 'Nationality']

X = df[features]
y = df['Go']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

tree.plot_tree(dtree, feature_names=features)

Age1 = int(input("Age: "))
Experience1 = int(input("Experience: "))
Rank1 = int(input("Rank: "))
Nationality1 = int(input("Nationality: "))

print(dtree.predict([[Age1, Experience1, Rank1, Nationality1]]))
"""

"""
#Confusion Matrix

actual = numpy.random.binomial(1, 0.9,size=1000)
predicted = numpy.random.binomial(1, 0.9,size=1000)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])

Accuracy = metrics.accuracy_score(actual, predicted)
#Accuracy measures the overall correctness of the model by calculating the ratio of correctly predicted instances to the total instances. It gives a general idea of how well the model is performing across all classes.

Precision = metrics.precision_score(actual, predicted)
#Precision measures the accuracy of positive predictions by calculating the ratio of true positive instances to the total predicted positive instances. It indicates how many of the predicted positive instances are actually correct.

Sensitivity = metrics.recall_score(actual, predicted)
#Sensitivity, also known as recall, measures the ability of the model to identify positive instances. It is the ratio of true positive instances to the total actual positive instances.

Specificity = metrics.roc_auc_score(actual, predicted)
#Specificity measures the ability of the model to identify negative instances. It is the ratio of true negative instances to the total actual negative instances.

F1_score = metrics.f1_score(actual, predicted)
print("Accuracy: ", Accuracy,"Precision: ", Precision, "Sensitivity: ", Sensitivity, "Specificity: ", Specificity, "F1_score: ", F1_score)

cm_display.plot()
plt.show()

"""

"""

#Hierarchical Clustering

x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]


data = list(zip(x, y))

hierarchical_cluster = AgglomerativeClustering(n_clusters=2, linkage='ward')
labels = hierarchical_cluster.fit_predict(data)
linkage_data = linkage(data, method='ward', metric='euclidean')

dendrogram(linkage_data)
print(labels)

plt.scatter(x, y, c=labels)
plt.show()

"""
"""


#Logistic Regression


#is for modeling the probability of a binary outcome based on one or more predictor variables. It estimates the relationship between the independent variables and the log-odds of the dependent variable, allowing for predictions of class membership. Logistic regression is commonly used in various fields, including healthcare, finance, and social sciences, to analyze and interpret relationships between variables and make informed decisions.


#X represents the size of a tumor in centimeters.
X = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)

#Note: X has to be reshaped into a column from a row for the LogisticRegression() function to work.
#y represents whether or not the tumor is cancerous (0 for "No", 1 for "Yes").
y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

logr = linear_model.LogisticRegression()
logr.fit(X, y)

log_odds = logr.coef_
odds = numpy.exp(log_odds)

Centimeters = float(input("Enter the size of the tumor in centimeters: "))

predicted = logr.predict(numpy.array([Centimeters]).reshape(-1,1))

print(predicted, "The tumor is cancerous" if predicted == 1 else "The tumor is not cancerous"  )
print("The odds of the tumor being cancerous per millimeter are x", odds)

def logit2prob(logr,x):
    log_odds = logr.coef_ * x + logr.intercept_
    odds = numpy.exp(log_odds)
    probability = odds / (1 + odds)
    return(probability)

#printing the probability in percentage of the tumor being cancerous based on the size of the tumor in centimeters.
print(logit2prob(logr, X))


"""
"""

#Grid Search

#is for hyperparameter tuning in machine learning models. It systematically searches through a predefined set of hyperparameters to find the combination that yields the best performance for a given model. By evaluating different hyperparameter configurations, grid search helps optimize model performance and improve predictive accuracy.

iris = datasets.load_iris()

X = iris['data']
y = iris['target']

logit = LogisticRegression(max_iter = 1000)

C = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

scores = []

for choice in C:
    logit.set_params(C=choice)
    logit.fit(X, y)
    scores.append(logit.score(X, y))


print(logit.fit(X, y))
print(logit.score(X, y))
print(scores)
What does the C parameter do in Logistic Regression?
The C parameter in Logistic Regression is a regularization parameter that controls the trade-off between achieving a low training error and a low testing error, which is also known as the bias-variance trade-off. A smaller value of C indicates stronger regularization, which can help prevent overfitting by penalizing large coefficients. Conversely, a larger value of C allows the model to fit the training data more closely, potentially leading to overfitting if the model becomes too complex. In summary, C helps to balance model complexity and generalization performance.
"""

"""

#Categorical Data

#is for handling categorical variables in machine learning models. Categorical data represents discrete values or categories, such as colors, types, or labels. Since many machine learning algorithms require numerical input, categorical data needs to be transformed into a suitable format. Common techniques for handling categorical data include one-hot encoding, label encoding, and ordinal encoding. These methods convert categorical variables into numerical representations that can be used effectively in machine learning models.


cars = pd.read_csv("data1.csv")
ohe_cars = pd.get_dummies(cars[['Car']])

X = pandas.concat([cars[['Volume', 'Weight']], ohe_cars], axis=1)
y = cars['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)

Weight1 = float(input("Weight: "))
Volume1 = float(input("Volume: "))

predictedCO2 = regr.predict([[Volume1, Weight1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]])

colors = pd.DataFrame({'color':['blue', 'red', 'green']})
dummies = pd.get_dummies(colors, drop_first=True)
dummies['color'] = colors['color']

print(dummies)


print(predictedCO2)


"""


"""


#K-Means Clustering

#is for partitioning a dataset into distinct, non-overlapping groups (clusters) based on the similarity of data points. It aims to minimize the variance within each cluster while maximizing the variance between clusters. K-Means is widely used in unsupervised learning tasks, such as customer segmentation, image compression, and anomaly detection.


x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]


data = list(zip(x, y))
intertias = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    intertias.append(kmeans.inertia_)


kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

plt.scatter(x, y, c=kmeans.labels_)
plt.show()

print(data)

"""

"""

#Bootstrap Aggregation (Bagging)

#is for improving the accuracy and stability of machine learning algorithms by combining multiple models trained on different subsets of the training data. It helps to reduce overfitting and increase the robustness of the predictions.


data = datasets.load_wine(as_frame=True)

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state = 22)

dtree = DecisionTreeClassifier(random_state = 22)
dtree.fit(X_train, y_train)

y_pred = dtree.predict(X_test)

print("Train data accuracy: ", accuracy_score(y_true = y_train, y_pred = dtree.predict(X_train)))
print("Test data accuracy: ", accuracy_score(y_true = y_test, y_pred = y_pred))

estimator_range = [2, 4, 6, 8, 10, 12, 14, 16]

models = []
scores = []

for n_estimators in estimator_range:

    clf = BaggingClassifier(n_estimators = n_estimators, random_state = 22)
    clf.fit(X_train, y_train)

    models.append(clf)
    scores.append(accuracy_score(y_true = y_test, y_pred = clf.predict(X_test)))

plt.figure(figsize=(9,6))
plt.plot(estimator_range, scores)

plt.xlabel("n_estimators", fontsize = 18)
plt.ylabel("score", fontsize = 18)
plt.tick_params(labelsize = 16)

oob_model = BaggingClassifier(n_estimators = 12, oob_score = True, random_state = 22)
oob_model.fit(X_train,y_train)

print("OOB score: ", oob_model.oob_score_)

plt.show()

"""


"""

#Cross Validation

X, y = datasets.load_iris(return_X_y=True)

clf = DecisionTreeClassifier(random_state=42)

ss = ShuffleSplit(train_size=0.6, test_size=0.3, n_splits=5)

scores = cross_val_score(clf, X, y, cv = ss)

print("Cross-validation scores: ", scores)
print("Average CV score: ", scores.mean())
print("Standard deviation of CV scores: ", len(scores))



"""



"""
#AUC ROC Curve

#Is for evaluating the performance of binary classification models. It provides a graphical representation of the trade-off between the true positive rate (sensitivity) and the false positive rate (1-specificity) at various threshold settings. The area under the ROC curve (AUC) quantifies the overall ability of the model to discriminate between positive and negative classes, with a higher AUC indicating better model performance.

n = 10000
ratio = .95
n_0 = int((1-ratio) * n)
n_1 = int(ratio * n)

y = np.array([0] * n_0 + [1] * n_1)

y_proba = np.array([1]*n)
y_pred = y_proba > .5


y_proba_2 = np.array(
    np.random.uniform(0, .7, n_0).tolist() +
    np.random.uniform(.3, 1, n_1).tolist()
)
y_pred_2 = y_proba_2 > .5


print(f'accuracy score: {accuracy_score(y, y_pred_2)}')
cf_mat = confusion_matrix(y, y_pred_2)
print('confusion matrix')
print(cf_mat)
print(f'class 0 accuracy: {cf_mat[0][0]/n_0}')
print(f'class 1 accuracy: {cf_mat[1][1]/n_1}')

def plot_roc_curve(true_y, y_prob):

    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plot_roc_curve(y, y_proba_2)
print(f'model 2 AUC score: {roc_auc_score(y, y_proba_2)}')
plt.show()


"""

"""
#AUC ROC Curve Example 2


def plot_roc_curve(true_y, y_prob):

    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

n = 10000
y = np.array([0] * n + [1] * n)

y_prob_1 = np.array(
    np.random.uniform(.25, .5, n//2).tolist() +
    np.random.uniform(.3, .7, n).tolist() +
    np.random.uniform(.5, .75, n//2).tolist()
)
y_prob_2 = np.array(
    np.random.uniform(0, .4, n//2).tolist() +
    np.random.uniform(.3, .7, n).tolist() +
    np.random.uniform(.6, 1, n//2).tolist()
)

print(f'model 1 accuracy score: {accuracy_score(y, y_prob_1>.5)}')
print(f'model 2 accuracy score: {accuracy_score(y, y_prob_2>.5)}')

print(f'model 1 AUC score: {roc_auc_score(y,y_prob_1)}')
print(f'model 2 AUC score: {roc_auc_score(y,y_prob_2)}')

fpr, tpr, thresholds = roc_curve(y, y_prob_2)
plt.plot(fpr,tpr)
plt.show()

"""

"""

#K nearest neighbors

#is for classification and regression. In both cases, the input consists of the k closest training examples in the feature space. The output depends on whether KNN is used for classification or regression:
#In KNN classification, the output is a class membership. An object is classified by a

x = [4, 5, 10, 4, 3, 11, 14, 8, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
classes = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1]

new_x = 8
new_y = 21
new_point = [(new_x,new_y)]



data = list(zip(x, y))
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(data, classes)

prediction = knn.predict(new_point)

plt.scatter(x + [new_x], y + [new_y], c=classes + [prediction[0]])
plt.text(x=new_x-1.7, y=new_y-0.7, s=f"new point, class: {prediction[0]}")
plt.show()


"""


