import os
import inspect
app_path = inspect.getfile(inspect.currentframe())
directory = os.path.realpath(os.path.dirname(app_path))
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import decomposition 
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from numpy import loadtxt  
import operator
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore")


# Load in the raw dataset of training and testing #
train_data = loadtxt(os.path.join(directory, 'train/X_train.txt'), comments="#",  unpack=False)
 
train_label = loadtxt(os.path.join(directory, 'train/y_train.txt'), comments="#", delimiter=",", unpack=False)
 
test_data = loadtxt(os.path.join(directory, 'test/X_test.txt'), comments="#",  unpack=False)
 
test_label = loadtxt(os.path.join(directory, 'test/y_test.txt'), comments="#", delimiter=",", unpack=False)


# Visualize distribution of data
n, bins, patches = plt.hist(train_label, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.xlabel('Value')
plt.ylabel('Number of samples in Human Activity Category')
plt.title('Data illustration') 


######### Hyperparameter selection of depth of tree #####################

Dpt = [40,50,60,70,80,90,100]   # define different depths

depth = []
for i in range(0, len(Dpt)):
    clf = tree.DecisionTreeClassifier(max_depth = Dpt[i])
    # Perform 5-fold cross validation 
    scores = cross_val_score(estimator=clf, X=train_data, y=train_label, cv=5)
    depth.append((scores.mean()))
print("Accuracy of 5-fold Cross Validation :: ", depth)
 

plt.plot(depth) 
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off 
plt.ylabel('Recognition rate of Human Activity')
plt.title('Plot 5-fold cross validation results') 
plt.show()

index, value = max(enumerate(depth), key=operator.itemgetter(1))

###### Human Activity Recognition using Decision Tree Using best depth and testing with unused test data

X_train = train_data
X_test = test_data

clfr = tree.DecisionTreeClassifier(max_depth=Dpt[index])
clfr.fit(X_train, train_label)

## predict decision tree classifier
predictions = clfr.predict(X_test)  # test with unused test data

print("Test Accuracy of Decision Tree Classifier :: ", accuracy_score(test_label, predictions))
 

############## Plot the Confusion matrix
cnf_matrix = confusion_matrix(test_label, predictions)
np.set_printoptions(precision=2)

## create the names of classes for confusion matrix plot
class_names = np.array(['WLKNG', 'WLKNG_UPSTS', 'WLKNG_DWNSTRS', 'SITNG', 'STNDNG', 'LYNG'])

# Plot normalized confusion matrix
plt.figure()
plt.figure(figsize=(40,8))
plot_confusion_matrix.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
