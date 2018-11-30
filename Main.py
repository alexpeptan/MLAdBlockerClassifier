import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC

datafile = 'resources/ad.data'
data = pd.read_csv(datafile, sep=",", header=None, low_memory=False)
print(data.head(20))


# Check whether a given value is a missing value, if yes change it to NaN
def to_num(cell):
    try:
        return np.float(cell)
    except:
        return np.nan


# Apply missing value check to a column / Pandas series
def series_num(series):
    return series.apply(to_num)


# Prepare training data
train_data=data.iloc[:, 0:-1].apply(series_num)
train_data.head(20)

train_data_after_cleanup = train_data.dropna()
train_data_after_cleanup.head(20)


# Prepare training labels
def to_label(str):
    if str == "ad.":
        return 1
    else:
        return 0


train_labels = data.iloc[train_data_after_cleanup.index, -1].apply(to_label)
print(train_labels)
print()


# Training Phase - Perform Support Vector Machines Ad Detection

# import LinearSVC which is a Classifier Object that can perform Support Vector Machines from scikit-learn module

# dual : bool, (default=True)
# Select the algorithm to either solve the dual or primal
# optimization problem. Prefer dual=False when n_samples > n_features.
# We have 1558 features and will use 2200 samples. Will use dual=False

# Adding dual=False will remove the warning:
# Liblinear failed to converge, increase the number of iterations.
# "the number of iterations.", ConvergenceWarning)
# Reference: https://stackoverflow.com/questions/52670012/convergencewarning-liblinear-failed-to-converge-increase-the-number-of-iterati
# This warning did not disappear with increasing max_iter to 1000000

classifier = LinearSVC(max_iter=1000, dual=False)

# Train the Model. Leave out some of the training data for the testing phase
classifier.fit(train_data_after_cleanup[100:2300], train_labels[100:2300])


#  Testing Phase

#  Each image from train_data is a panda dataframe.
#  Its values array will need to be used and reshaped to be seen as a 2D array
def test_model(classifier_param, test_image):
    return classifier_param.predict(test_image.values.reshape(1, -1))


def print_prediction_result(classifier_param, train_data, image_index):
    test_image = train_data.iloc[image_index]
    prediction = test_model(classifier_param, test_image)
    print("Image with index " + str(image_index) + " was rated " + ("Ad" if prediction == 1 else "Non Ad"))


print_prediction_result(classifier, train_data_after_cleanup, 12)
print_prediction_result(classifier, train_data_after_cleanup, 2301)
print_prediction_result(classifier, train_data_after_cleanup, 13)
print_prediction_result(classifier, train_data_after_cleanup, 2302)
