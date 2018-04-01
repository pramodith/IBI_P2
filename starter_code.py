from helpers import *
from opportunity_dataset import OpportunityDataset
import numpy as np
from scipy.stats import mode
from collections import Counter
import imu_wrist_p3


def create_io_pairs(inputs, labels, window_size, stride):
    # Compute your windowed features here and labels. Right now
    # it just returns the inputs and labels without changing anything.
    X = inputs
    Y = labels
    final_x = []
    final_y = []
    for i in range(0, len(X), stride):
        if i + window_size < len(X):
            inp_x = X[i:i + window_size]
            inp_x = np.mean(inp_x, axis=0)
            out_y = int(np.asscalar(mode(Y[i:i + window_size])[0][0]))
            final_x.append(inp_x)
            final_y.append(out_y)
    # ...
    # ....

    return final_x, final_y


def impute_data(arr):
    # Data imputation code goes here!
    # ...
    # ...
    row_len = len(arr[0])
    delete_indices = []
    for cnt, row in enumerate(arr):
        indices = np.logical_not(np.isnan(row))
        true_cnt = np.count_nonzero(indices == True)
        if true_cnt == len(row):
            continue
        elif true_cnt == 0:
            delete_indices.append(cnt)
        else:
            all_indices = np.arange(len(row))
            row = np.interp(all_indices, all_indices[indices], row[indices])
            arr[cnt] = row
    arr = np.delete(arr, delete_indices)
    arr = arr.reshape(-1, row_len)
    return arr


def test_imputation(dataset):
    # Get the input array on which to perform imputation
    training_data, testing_data = dataset.leave_subject_out(left_out=["S2", "S3", "S4"])
    X_train, Y_train = create_dataset(training_data, dataset.data_map["AccelWristSensors"],
                                      dataset.locomotion_labels["idx"])
    arr = X_train
    out = impute_data(arr)
    baseline = np.load("D:\gatech\sem2\IBI\P2\imputed_data.npy", encoding='latin1')
    return np.sum((out - baseline) ** 2)


def train(X, Y):
    # This is where you train your classifier, right now a dummy
    # classifier which uniformly guesses a label is "trained"
    # ....
    # ....
    clf = imu_wrist_p3.call_me(X, Y)
    model = {"clf": clf}
    # ....
    # ....
    return model


def test(X, model):
    # This is where you compute predictions using your trained classifier
    # ...
    Y = model["clf"].predict(X)
    return Y


def cv_train_test(dataset, sensors, labels, window, stride):
    """
    Template code for performing leave on subject out cross-validation
    """
    subjects = dataset.subject_data.keys()
    Y_pred_total, Y_test_total = [], []

    # Leave one subject out cross validation
    for subj in subjects:
        training_data, testing_data = dataset.leave_subject_out(left_out=subj)

        X_train, Y_train = create_dataset(training_data, sensors, labels["idx"])
        X_test, Y_test = create_dataset(testing_data, sensors, labels["idx"])
        Y_test = np.asarray(Y_test)
        # Impute missing inputs data
        X_train = impute_data(X_train)
        X_test = impute_data(X_test)
        # Compute features and labels for train and test set
        X_train, Y_train = create_io_pairs(X_train, Y_train, window, stride)
        # print(Counter(Y_train))
        X_test, Y_test = create_io_pairs(X_test, Y_test, window, stride)
        # print(Counter(Y_test))
        Y_test = np.asarray(Y_test)
        model = train(X_train, Y_train)

        # Make predictions on the test set here
        Y_pred = test(X_test, model)

        # Append prediction and current labels to cv dataset
        Y_pred_total.append(Y_pred.reshape((Y_pred.size, 1)))
        Y_test_total.append(Y_test.reshape((Y_test.size, 1)))

    # Perform evaluations
    eval_preds(Y_pred_total, Y_test_total, labels["classes"])


if __name__ == "__main__":
    # Example inputs to cv_train_test function, you would use
    # these inputs for  problem 2
    window_sizes = [5]
    strides = [2]
    dataset = OpportunityDataset()
    sensors = dataset.data_map["ImuWristSensors"]

    # print(test_imputation(dataset))
    # Locomotion labels
    for i in range(0, len(window_sizes)):
        for j in range(0, len(strides)):
            cv_train_test(dataset, sensors, dataset.locomotion_labels, window_sizes[i], strides[j])

# Activity labels
# cv_train_test(dataset, sensors, dataset.activity_labels)
