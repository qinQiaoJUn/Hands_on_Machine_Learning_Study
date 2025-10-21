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
# plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
# plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
# plt.vlines(thresholds, 0, 1.0, "k", "dotted", label="threshold")
# plt.show()

# We could also draw the graph of precision against recall
# plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")
# beautify the figure: add labels, grid, legend, arrow, and text
# plt.show()


# Suppose that we need 90% precision
'''
1. Yes, we could use the first graph to acquire the approximate locations that satisfy precision >= 90%
However, it may not be that precise
2. We could also calculate the minimum threshold value that satisfies precision >= 90%
NumPy has a argmax() function to realize this
'''
idx_for_90_precision = (precisions >= 0.90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision]
print("The threshold value for 90% precision is: ", threshold_for_90_precision)

# To make predictions, we can run this code instead of using predict():
y_train_prediction_90_precision = (y_scores >= threshold_for_90_precision)  # Generate the boolean array
# We could now verify the precision and recall
print("The precision score for the 90 precision is: ", precision_score(y_train_5, y_train_prediction_90_precision))
# Output: 0.9000345901072293
print("The recall score for the 90 precision is: ", recall_score(y_train_5, y_train_prediction_90_precision))
# Output: 0.4799852425751706


# Another way of evaluating is ROC curve
# This curve shows recall against false positive rate
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
plt.plot([fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")
# beautify the figure: add labels, grid, legend, arrow, and text
plt.show()

# The area under this curve is called AUC
# We could calculate this area to evaluate the model, larger area (closer to 1) represents better performance
print("The AUC score of the curve is: ", roc_auc_score(y_train_5, y_scores))
# Output: 0.9604938554008616


# Until now, we have learned the classification method of SGDClassifier, and how to evaluate its outcome
# Next up, we will switch to another classification method called RandomForestClassifier
forest_classifier = RandomForestClassifier(random_state=42)

# However, RandomForestClassifier doesn't have a decision_function() method
# Luckily, it still has a predict_proba() method that returns class probabilities for each instance
# We still use the cross_val_predict() function to train the forest classifier:
y_probas_forest = cross_val_predict(forest_classifier, X_train, y_train_5, cv=3,
                                    method="predict_proba")
# We can take a look at the first 2 images' prediction results:
print(y_probas_forest[:2])
# Output: [[0.11, 0.89],  89% likely to be '5'
#        [0.99, 0.01]]  Only 1% likely to be '5'

# Pass the trained forest classifier to the precision_recall_curve() function:
y_scores_forest = y_probas_forest[:, 1]
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(y_train_5, y_scores_forest)
# Then draw the graph:
plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2,
         label="Random Forest")
plt.plot(recalls, precisions, "--", linewidth=2, label="SGD")
# beautify the figure: add labels, grid, and legend
# plt.show()


svm_classifier = SVC(random_state=42)
svm_classifier.fit(X_train[:2000], y_train[:2000])
print("Using SVM, we predict the first image: ", svm_classifier.predict([some_digit]))
# Output: Using SVM, we predict the first image:  ['5']

# Decision process
some_digit_scores = svm_classifier.decision_function([some_digit])
print("The decision values are shown in the array: ", some_digit_scores.round(2))
# Output: [[ 3.79,  0.73,  6.06,  8.3 , -0.29,  9.3 ,  1.75,  2.77,  7.21, 4.82]]
# The highest score 9.3 is on the 6th index, which corresponds to '5'
class_id = some_digit_scores.argmax()
print("The class id is: ", class_id)


ovr_classifier = OneVsRestClassifier(SVC(random_state=42))
ovr_classifier.fit(X_train[:2000], y_train[:2000])
print("Forcing to use OvR strategy: ", ovr_classifier.predict([some_digit]))
# Output: Forcing to use OvR strategy:  ['5']

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
print("Using SGD, the predicted result is: ", sgd_clf.predict([some_digit]))
# Output: ['3']
# This is incorrect! Prediction errors do happen!

# To check what exactly leads to the error, we look at the scores that SGDClassifier assignsed to each class:
print("SGD scores for each class: ", sgd_clf.decision_function([some_digit]).round())
# Output: SGD scores for each class: [[-31893., -34420.,  -9531.,   1824., -22320.,  -1386., -26189.,
#         -16148.,  -4604., -12051.]]
# It's not very confident about the result, with only +1824 for '3', and all the others as negative values

# We could also use cross_val_score() function to evaluate the model:
print("The cross validation scores are: ", cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy"))
# Output: The cross validation scores are:  [0.87365, 0.85835, 0.8689 ]

# This is not very high. By scaling the inputs, we can increase the accuracy:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype("float64"))
print("The cross validation scores after scaling are: ", cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))
# Output: The cross validation scores after scaling are: [0.8983, 0.891 , 0.9018]

from sklearn.metrics import ConfusionMatrixDisplay
y_train_prediction2 = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_prediction2)
plt.show()

# Add normalization
ConfusionMatrixDisplay.from_predictions(y_train, y_train_prediction2, normalize="true", values_format=".0%")
plt.show()
