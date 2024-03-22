#Katya Kobari-1201478
#Jehad Hamayel-1200348
import numpy as np
import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
TEST_SIZE = 0.3
K = 3
class NN:
    def __init__(self, trainingFeatures, trainingLabels) -> None:
        self.trainingFeatures = np.array(trainingFeatures)
        self.trainingLabels = np.array(trainingLabels)

    def predict(self, features, k):
        predict_results = []
        for test_feature in features:
            distances_testAndTraining = []
            for train_feature in self.trainingFeatures:
                distance = np.linalg.norm(test_feature - train_feature)
                distances_testAndTraining.append(distance)
            sorted_indices = np.argsort(distances_testAndTraining)
            nearestLabels = []
            for i in sorted_indices[:k]:
                nearestLabels.append(self.trainingLabels[i])
            predicted_label = max(set(nearestLabels), key=nearestLabels.count)
            predict_results.append(predicted_label)

        return predict_results


def load_data(filename):
    """
    Load spam data from a CSV file `filename` and convert into a list of
    features vectors and a list of target labels. Return a tuple (features, labels).

    features vectors should be a list of lists, where each list contains the
    57 features vectors

    labels should be the corresponding list of labels, where each label
    is 1 if spam, and 0 otherwise.
    """
    dataSet_features = []
    class_labels = []
    with open(filename, 'r') as file:
        dataSet = csv.reader(file)
        for row in dataSet:
            dataSet_features.append([float(value) for value in row[:-1]])
            class_labels.append(int(row[-1]))

    return np.array(dataSet_features), np.array(class_labels)


def preprocess(features):
    """
    Normalize each feature by subtracting the mean value and dividing by the standard deviation.
    """
    means_list = np.mean(features, axis=0)
    stds_list = np.std(features, axis=0)
    # Normalize each feature
    normalized_features = []
    for feature_vector in features:
        normalized_feature = []
        for feature, mean, std in zip(feature_vector, means_list, stds_list):
            normalized_value = (feature - mean) / std
            normalized_feature.append(normalized_value)
        normalized_features.append(normalized_feature)
    return normalized_features


def train_mlp_model(features, labels):
    """
    Given a list of features lists and a list of labels, return a
    fitted MLP model trained on the data using sklearn implementation.
    """

    model = MLPClassifier(hidden_layer_sizes=(10, 5), activation='logistic', random_state=42, max_iter=10000)
    model.fit(features, labels)

    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return (accuracy, precision, recall, f1).

    Assume each label is either a 1 (positive) or 0 (negative).
    """
    truePos = trueNeg = falsePos = falseNeg = 0
    for actual, predicted in zip(labels, predictions):
        if actual == 1 and predicted == 1:
            truePos += 1
        elif actual == 0 and predicted == 0:
            trueNeg += 1
        elif actual == 0 and predicted == 1:
            falsePos += 1
        elif actual == 1 and predicted == 0:
            falseNeg += 1

    accuracy = (truePos + trueNeg) / (truePos+falsePos+trueNeg+falseNeg)
    precision = truePos / (truePos + falsePos) if truePos + falsePos != 0 else 0
    recall = truePos / (truePos + falseNeg) if truePos + falseNeg != 0 else 0
    f1_measure = 2 * ((precision * recall) / (precision + recall)) if precision + recall != 0 else 0
    return accuracy, precision, recall, f1_measure

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python template.py spambase.csv")

    # Load data from spreadsheet and split into train and test sets
    features, labels = load_data(sys.argv[1])
    features = preprocess(features)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SIZE)

    # Train a k-NN model and make predictions
    model_nn = NN(X_train, y_train)
    predictions = model_nn.predict(X_test, K)
    accuracy, precision, recall, f1 = evaluate(y_test, predictions)

    # Print results
    print("**** 1-Nearest Neighbor Results ****")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)

    # Train an MLP model and make predictions
    model = train_mlp_model(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy, precision, recall, f1 = evaluate(y_test, predictions)

    # Print results
    print("**** MLP Results ****")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)



if __name__ == "__main__":
    main()
