# Chapter 3 Practice 1:
"""
We use Fashion-MNIST as the dataset to reproduce the key points in Chapter 3: Classification
Fashion-MNIST is a MNIST-like dataset containing 70,000 gray images of different clothes:
60,000 of them are examples, and the remaining 10,000 compose the test set

Content:
Each image is 28*28 pixels, the same as the original MNIST dataset
Each pixel has a single pixel value between 0 and 255
The dataset contains 785 columns, with the first column being the label
(The first column is the class label, and the remaining 784 columns stand for pixel values)

Labels:
0 T-shirt/top
1 Trouser
2 Pullover
3 Dress
4 Coat
5 Sandal
6 Shirt
7 Sneaker
8 Bag
9 Ankle boot
"""

# 1. FETCH THE DATASET
from sklearn.datasets import fetch_openml

fashion_mnist = fetch_openml('Fashion-MNIST', as_frame=False)

# 2. LEARNING WHAT DOES IT CONTAIN
X, y = fashion_mnist.data, fashion_mnist.target  # Here X is a matrix, y is a vector, thus in uppercase/lowercase
print("In the Fashion MNIST, X is: ", X)
print("While y is: ", y)
# Output:
#  In the Fashion MNIST, X is:  [[0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]
#  ...
#  [0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]]
#  While y is:  ['9' '0' '0' ... '8' '1' '5']
# That is the same with MNIST!

# 3. SELECT ONE EXAMPLE TO OBSERVE
# We could select one example in the whole dataset first to observe it
import matplotlib.pyplot as plt


def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")


some_digit = X[0]
some_digit_2 = X[1]
some_digit_3 = X[2]
'''
The following code could be commented
'''
# plot_digit(some_digit)
# plt.show()
# print("The corresponding class for this image is: ", y[0])
'''
Ends here
'''

# 4. SET THE TRAINING AND TEST SETS

'''
First of all, don't forget to set the training set and test set!
We select the first 60,000 X and y as the training set, and the rest as the test set
'''
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
# X_train and y_train are set aside for the training set
# While X_test and y_test are for the test set


# 5. START FROM SCRATCH: IS IT AN ANKLE BOOT?
# Create two boolean arrays from the original arrays, let them represent whether each example is an ankle boot or not
y_train_is_nine = (y_train == '9')
y_test_is_nine = (y_test == '9')
# Then we apply SGD to do the classification:
from sklearn.linear_model import SGDClassifier

sgd_classifier = SGDClassifier(random_state=42)  # Create an SDG classifier
sgd_classifier.fit(X_train, y_train_is_nine)  # Use all the training data whose y == '9' to train the classifier
# That is to say, use all the images of ankle boots to train the classifier
print("The prediction for the first example is: ", sgd_classifier.predict([some_digit]))  # Test the first example
print("The prediction for the second example is: ", sgd_classifier.predict([some_digit_2]))  # Test the second example
# Output:
# The prediction for the first example is:  [ True]
# The prediction for the second example is:  [False]
'''
We know that the first example in Fashion-MNIST is an ankle boot
Now let's find out what the second example is:
'''
print("As verification, the second example's label is actually: ", y[1])
# Output: As verification, the second example's label is actually:  0
'''
Therefore, the prediction is correct: the second example is indeed not an ankle boot
However, does it mean that the prediction will be correct globally? Maybe not!
Thus, we have to evaluate its performance
'''

# 6. EVALUATE THE PERFORMANCE
# We still apply the cross_val_score() function in the sklearn module
from sklearn.model_selection import cross_val_score

print(cross_val_score(sgd_classifier, X_train, y_train_is_nine, cv=3, scoring="accuracy"))
# Output: [0.9774  0.97875 0.97835]
'''
Parameter analyses:
sgd_classifier: the model trained, to be evaluated
X_train: the training set (image data)
y_train_is_nine: the result boolean array for the training set (for verifying the prediction results)
cv=3: create 3 "folds", equally (try the best) divide the dataset into 3 halves
      each time select 2 for the training sets, and the remaining 1 for the test set, evaluate for 3 loops in total
scoring="accuracy": evaluate the model's accuracy
'''

'''
However, we cannot judge the quality of model only by precision
We should also consider how many examples are misclassified to other classes
'''
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_prediction = cross_val_predict(sgd_classifier, X_train, y_train_is_nine, cv=3)
# cross_val_predict() method also performs k-fold cross-validation
# However, it returns predictions instead of scores
cm = confusion_matrix(y_train_is_nine, y_train_prediction)
# Now, cm is the confusion matrix that counts the comparisons between predicted set and real result set
print(cm)
# Output: [[53202   798]
#  [  512  5488]]

from sklearn.metrics import precision_score, recall_score

print("The precision is: ", precision_score(y_train_is_nine, y_train_prediction))
# Output: The precision is:  0.8730512249443207
print("The recall is: ", recall_score(y_train_is_nine, y_train_prediction))
# Output: The recall is:  0.9146666666666666

# Solely counting on precision or recall is not a comprehensive way
# Thus, we introduce F1 score:
from sklearn.metrics import f1_score

print("The F1 score is: ", f1_score(y_train_is_nine, y_train_prediction))

# We need to find a "threshold value" to satisfy both precision and recall requirements
# A graph may be more explicit:
from sklearn.metrics import precision_recall_curve

y_scores = cross_val_predict(
    sgd_classifier, X_train, y_train_is_nine, cv=3, method="decision_function"
)

# Get precision, recall, and threshold values
precisions, recalls, thresholds = precision_recall_curve(y_train_is_nine, y_scores)

# Create the figure canvas with its size
plt.figure(figsize=(10, 5))

# Plot precision (blue), recall (green), and the vertical threshold line (dotted)
plt.plot(thresholds, precisions[:-1], "b--", linewidth=2, label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", linewidth=2, label="Recall")
plt.axvline(0, color="k", linestyle=":", linewidth=1.2, label="threshold")

# Find the cross point between precision and recall
import numpy as np

idx = np.argmin(np.abs(precisions[:-1] - recalls[:-1]))
plt.scatter(thresholds[idx], precisions[:-1][idx], color='k', s=30)

# Limit the maximum and minimum values of x-axis, let the 0-point be on the center
x_min, x_max = thresholds.min(), thresholds.max()
x_limit = max(abs(x_min), abs(x_max))
plt.xlim(-x_limit, x_limit)

# Beautify grids, titles, labels, etc.
plt.xlabel("Threshold", fontsize=12)
plt.ylabel("Precision / Recall", fontsize=12)
plt.title("Precision and Recall vs Decision Threshold", fontsize=14)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.grid(True, linestyle="--", alpha=0.6)

# Add a legend
plt.legend(loc="best", fontsize=11)

# Stretch the layout to avoid labels or titles being cut
plt.tight_layout()
plt.show()

# We could also directly draw the curve between precision and recall
y_scores = cross_val_predict(
    sgd_classifier, X_train, y_train_is_nine, cv=3, method="decision_function"
)
# Calculate precision, recall, thresholds
precisions, recalls, thresholds = precision_recall_curve(y_train_is_nine, y_scores)

# Create the figure canvas
plt.figure(figsize=(7, 6))

# Plot the curve of precision vs. recall
plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")

# For threshold value, we could either:
# 1. set a threshold and annotate it in the graph, or
# 2. Calculate the exact threshold value for a certain precision value
# Method 1:
threshold = 3000
idx = np.argmin(np.abs(thresholds - threshold))
# Method 2, find the threshold for at least 90% precision:
idx_for_90_precision = (precisions >= 0.90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision]
print("The threshold value for 90% precision is: ", threshold_for_90_precision)
threshold = threshold_for_90_precision
idx = idx_for_90_precision
# Both methods can correctly plot the graph using this line of code below:
plt.plot(recalls[idx], precisions[idx], "ko", label=f"Point at threshold {threshold:,}")

# Assistance curve
plt.plot([recalls[idx], recalls[idx]], [0, precisions[idx]], "k:", linewidth=1)
plt.plot([0, recalls[idx]], [precisions[idx], precisions[idx]], "k:", linewidth=1)

# Add an arrow annotation that indicates the direction of higher threshold value
plt.annotate(
    "Higher threshold",
    xy=(recalls[idx] + 0.05, precisions[idx] - 0.05),
    xytext=(recalls[idx] + 0.15, precisions[idx] - 0.15),
    arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=8),
    fontsize=11
)

# Axes and types settings
plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.title("Precision vs Recall Curve", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="lower left", fontsize=11)
plt.tight_layout()
plt.show()

from sklearn.metrics import roc_curve

# Get False positive rate (fall-out value), true positive rate (recall), and threshold
# roc_curve() mainly use 2 parameters to calculate:
# 1. The true results (y_train_is_nine), and
# 2. The score calculated by decision function (y_scores)
fpr, tpr, thresholds = roc_curve(y_train_is_nine, y_scores)
idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
plt.plot([fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")
plt.annotate(
    "Higher threshold",
    xy=(fpr_90 + 0.02, tpr_90 + 0.1),
    xytext=(fpr_90 + 0.15, tpr_90 + 0.25),
    arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=8),
    fontsize=11
)
plt.xlabel("False Positive Rate (Fall-Out)", fontsize=12)
plt.ylabel("True Positive Rate (Recall)", fontsize=12)
plt.title("ROC Curve", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="lower right", fontsize=11)

plt.tight_layout()
plt.show()

# Do we have other classification methods? Yes!
# Let's try the RandomForestClassifier, and compare its evaluation result to SGDClassifier!
from sklearn.ensemble import RandomForestClassifier
forest_classifier = RandomForestClassifier(random_state=42)

y_probas_forest = cross_val_predict(forest_classifier, X_train, y_train_is_nine, cv=3,
                                    method="predict_proba")
# We can take a look at the first 2 images' prediction results:
print(y_probas_forest[:2])
# Pass the trained forest classifier to the precision_recall_curve() function:
y_scores_forest = y_probas_forest[:, 1]
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(y_train_is_nine, y_scores_forest)
# Then draw the graph:
plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2,
         label="Random Forest")
plt.plot(recalls, precisions, "--", linewidth=2, label="SGD")
# beautify the figure: add labels, grid, and legend
plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.title("RandomForestClassifier", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="lower right", fontsize=11)
plt.show()

from sklearn.svm import SVC
svm_classifier = SVC(random_state=42)
svm_classifier.fit(X_train[:2000], y_train[:2000])
print("Using SVM, we predict the first image: ", svm_classifier.predict([some_digit]))
print("The true answer is: ", y_train[0])
some_digit_3 = X[1]
print("Using SVM, we predict the second image: ", svm_classifier.predict([some_digit_2]))
print("The true answer is: ", y_train[1])
some_digit_3 = X[2]
print("Using SVM, we predict the third image: ", svm_classifier.predict([some_digit_3]))
print("The true answer is: ", y_train[2])
# Output:
# Using SVM, we predict the first image:  ['9']
# The true answer is:  9
# Using SVM, we predict the second image:  ['0']
# The true answer is:  0
# Using SVM, we predict the third image:  ['3']
# The true answer is:  0

# We have spotted a mistake. However, there could be more mistakes that haven't been discovered
# Thus, we have to analyze all the prediction results, using Confusion Matrix:

'''
The following code for confusion matrix could take longer time to run
thus, it could be commented to run faster, without interrupting other parts of the code
'''
# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train.astype("float64"))
# y_train_prediction_for_CM = cross_val_predict(svm_classifier, X_train_scaled, y_train, cv=3)
# ConfusionMatrixDisplay.from_predictions(y_train, y_train_prediction_for_CM, normalize="true", values_format=".0%")
# plt.show()
'''
Ends here
'''

'''
The following code for confusion matrix could take longer time to run
thus, it could be commented to run faster, without interrupting other parts of the code
'''
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype("float64"))
y_train_prediction_for_CM2 = cross_val_predict(svm_classifier, X_train_scaled, y_train, cv=3)
sample_weight = (y_train_prediction_for_CM2 != y_train)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_prediction_for_CM2,
                                        sample_weight=sample_weight,
                                        normalize="pred", values_format=".0%")
# "normalize" parameter could be set as "true" or "pred" to realize different analyses
plt.show()
'''
Ends here
'''







