from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve


mnist = fetch_openml('mnist_784', as_frame=False)
# Here, minst (MINST) is a very popular dataset containing 70,000 handwritten numbers and their labels
# We set as_frame as False to get the data as NumPy arrays

# Acquire a dataset using sklearn-datasets:
'''
sklearn-datasets package contains 3 types of functions to acquire a dataset:
1. fetch_* functions such as fetch_openml() to download real-life datasets
2. load_* functions to load small toy datasets bundled with Scikit-Learn (built-in datasets)
   (so they don’t need to be downloaded over the internet)
3. make_* functions to generate fake datasets, useful for tests
'''

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
some_digit = X[0]
'''
plot_digit(some_digit)
plt.show()
'''

# Start actual classification
'''
First of all, don't forget to set the training set and test set!
We select the first 60,000 X and y as the training set, and the rest as the test set
'''
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Start from a simple task
'''
We could first classify 5's and those not 5's
Thus, we create the target vectors for this simple task:
'''
y_train_5 = (y_train == '5')  # True for all 5s, False for all other digits
y_test_5 = (y_test == '5')

'''
Then we use stochastic gradient descent (SGD) to do the classification
SGD's advantages are:
1. Capable of handling very large datasets efficiently
2. Deals with training instances independently, one at a time, which also makes SGD well suited for online learning
'''
sgd_clf = SGDClassifier(random_state=42)  # Create an SDG classifier
sgd_clf.fit(X_train, y_train_5)  # Use all the training data whose y == '5' to train the classifier

print(sgd_clf.predict([some_digit]))
# Conduct the prediction
# Since some_digit is an 1D array, we need to add another [] outside of it

# Evaluate the performance
'''
Great! We have made the first prediction
However, we don't know the exact performance of this prediction
Therefore, we need to evaluate its performance
Here, we use the cross_val_score() function to evaluate our SGDClassifier model
'''

print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))
'''
The accuracy result is: [0.95035 0.96035 0.9604 ], seems high
However, if we just keep guessing the numbers are "not 5", then for the whole set, we could still get 90% accuracy!
Thus, purely evaluating accuracy may not be the best way
A better way: confusion matrix (CM)
'''

# Confusion matrix
'''
The general idea of a confusion matrix is to count the number of times instances of class A are classified as class B
For example, to know the number of times the classifier confused images of 8s with 0s: 
look at row #8, column #0 of the confusion matrix
'''
y_train_prediction = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
# cross_val_predict() method also performs k-fold cross-validation
# However, it returns predictions instead of scores
cm = confusion_matrix(y_train_5, y_train_prediction)
# Now, cm is the confusion matrix that counts the comparisons between predicted set and real result set
print(cm)

# The output array is as follows:
'''
[[53892   687]
 [ 1891  3530]]
53,892 are correctly classified as non-5s (true negatives)
687 are wrongly classified as 5s (false positives)
1,891 are wrongly classified as non-5s (false negatives)
3,530 are correctly classified as 5s (true positive)

A perfect classifier would only have true positives and true negatives

precision of the classifier: TP/(TP+FP)
recall of the classifier: TP/(TP+FN)
 '''

# We can calculate the precision and recall:
print("The precision score is: ", precision_score(y_train_5, y_train_prediction))
print("The recall score is: ", recall_score(y_train_5, y_train_prediction))

# We also have F1 score that combines these two together:
# F1 score = 2 * (precision * recall / (precision + recall))
# To use F1 score, we simply call the f1_score() function:
print("The F1 score is: ", f1_score(y_train_5, y_train_prediction))

# Noticeable about F1 score:
'''
The F1 score favors classifiers that have similar precision and recall
Because it cares about precision and recall equally
Therefore, if we care more about precision/recall in certain models, F1 score may not work very well
'''

# VERY IMPORTANT:
# Increasing precision reduces recall, and vice versa. This is called the precision/recall trade-off.

# Decision function
'''
For each instance, SGDClassifier computes a score to determine whether it's positive or negative
If this score is higher than the "threshold", the instance will be judged as positive
Otherwise, it will be judged as negative
'''

# QUESTION: How do you decide which threshold to use?
# 1. Get the decision scores for all instances
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")
# 2. With these scores, use precision_recall_curve() function to compute precision and recall for all thresholds
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

# 3. Finally, use Matplotlib to plot precision and recall as functions of the threshold value
plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
plt.vlines(thresholds, 0, 1.0, "k", "dotted", label="threshold")
plt.show()
