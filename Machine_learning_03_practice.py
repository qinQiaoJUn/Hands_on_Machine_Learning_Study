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








