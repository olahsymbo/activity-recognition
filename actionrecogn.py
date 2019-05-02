import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt  
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


###############################################################################
################ Load in the raw dataset of training and testing ##############
###############################################################################

train_data = loadtxt("C:/UCI Act_Rec/train/X_train.txt", comments="#",  unpack=False)
 
train_label = loadtxt("C:/UCI Act_Rec/train/y_train.txt", comments="#", delimiter=",", unpack=False)
 
test_data = loadtxt("C:/UCI Act_Rec/test/X_test.txt", comments="#",  unpack=False)
 
test_label = loadtxt("C:/UCI Act_Rec/test/y_test.txt", comments="#", delimiter=",", unpack=False)


##########################################################################
############## Visualize distribution of data ############################
##########################################################################
n, bins, patches = plt.hist(train_label, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.xlabel('Value')
plt.ylabel('Number of samples in Human Activity Category')
plt.title('Data illustration') 


#########################################################################
######### Hyperparameter selection of depth of tree #####################
#########################################################################

Dpt = [40,50,60,70,80,90,100]   # define different depths

depth = []
for i in range(0, len(Dpt)):
    clf = tree.DecisionTreeClassifier(max_depth = Dpt[i])
    # Perform 5-fold cross validation 
    scores = cross_val_score(estimator=clf, X=train_data, y=train_label, cv=5)  # use 5fold CV to determine best depth on training set
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

##########################################################
##########################################################
###### Human Activity Recognition using Decision Tree ####
### Using best depth and testing with unused test data####
##########################################################
##########################################################

X_train = train_data
X_test = test_data

clfr = tree.DecisionTreeClassifier(max_depth=Dpt[index])
clfr.fit(X_train, train_label)

## predict random forest classifier
predictions = clfr.predict(X_test)  # test with unused test data


print("Test Accuracy of Decision Tree Classifier :: ", accuracy_score(test_label, predictions))
 

 
#######################################################
######################################################
############## Plot the Confusion matrix ##############
#######################################################
######################################################
    
import itertools

## Get the confusion matrix
cnf_matrix = confusion_matrix(test_label, predictions)
np.set_printoptions(precision=2)

## create the names of classes for confusion matrix plot
class_names = np.array(['WLKNG', 'WLKNG_UPSTS', 'WLKNG_DWNSTRS', 'SITNG', 'STNDNG', 'LYNG'])
 
## confusion matrix function for classes
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Plot normalized confusion matrix
plt.figure()
plt.figure(figsize=(40,8))
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()




####################################################################
########### Human Actvity Recognition using ########################
########### PCA dimesionality reduction and Decision Tree ###########
#####################################################################

pca = decomposition.PCA()
pca.fit(train_data)
X_train = pca.transform(train_data)
X_test = pca.transform(test_data)
 
##################################################################### 
##################################################################### 
########### fit decision tree #############################
################################################################### 
clfr = tree.DecisionTreeClassifier(max_depth=Dpt[index])
clfr.fit(X_train, train_label)

## predict decision tree classifier
predictions = clfr.predict(X_test)


print("Test Accuracy of PCA_Decision Tree :: ", accuracy_score(test_label, predictions))
 



###################### Confusion Matrix #####################################
## Get the confusion matrix
cnf_matrix = confusion_matrix(test_label, predictions)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plt.figure(figsize=(40,8))
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


##########################################################
########### Human Actvity Recognition using ##############
########### LASSO Feature Selection and Decision Tree#####
##########################################################
##########################################################

lambdda = 0.02
clf = LassoCV(lambdda)
clf.fit(train_data, train_label)
#print(clf.coef_)
np.count_nonzero(clf.coef_)

sfm = SelectFromModel(clf, prefit = 'True')
X_train = sfm.transform(train_data) 
X_train.shape

X_test = sfm.transform(test_data) 
X_test.shape

print('Selected features = ', X_train.shape[1])

##################################################################### 
##################################################################### 
########### fit decision tree #############################
################################################################### 
clfr = tree.DecisionTreeClassifier(max_depth=Dpt[index])
clfr.fit(X_train, train_label)

## predict decision tree classifier
predictions = clfr.predict(X_test)


print("Test Accuracy of LASSO_Decision Tree :: ", accuracy_score(test_label, predictions))
 

###################### Confusion Matrix #####################################
## Get the confusion matrix
cnf_matrix = confusion_matrix(test_label, predictions)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plt.figure(figsize=(40,8))
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
