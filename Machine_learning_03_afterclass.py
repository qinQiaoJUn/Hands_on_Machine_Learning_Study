from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

mnist = fetch_openml('mnist_784', as_frame=False)
# Here, minst (MINST) is a very popular dataset containing 70,000 handwritten numbers and their labels
# We set as_frame as False to get the data as NumPy arrays

# We should also learn about the dataset we are going to process:
'''
MNIST dataset properties:
# 70,000 images, each with 784 features
# Thus, we have X as a 2D array (each with 784 numbers, and 70,000 in total)
'''

# After fetching the dataset, we get the X and y of this dataset
X, y = mnist.data, mnist.target


# The 784 feature numbers for each image is actually the result of 28*28 length and height
# Thus, we need to convert the 784 numbers into a 28*28 array
# Firstly, we plot one of the data to see what it represents
def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")


# We use the plot_digit method to draw the digit image
first_instance = X[0]
# plot_digit(first_instance)
# plt.show()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn_classifier = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='minkowski', p=2)

'''
TWO IMPORTANT PARAMETERS:
n_neighbors: the number of neighbors, too small would lead to overfitting, and vise versa
weights: the method to calculate weight, uniform (equal weights) or distance (growing weights with nearer distances)
'''

# Train the model
knn_classifier.fit(X_train, y_train)
y_predicted = knn_classifier.predict(X_test)
accuracy = knn_classifier.score(X_test, y_test)
print("The accuracy using KNN is: ", accuracy)








