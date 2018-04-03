from helpers import *
from opportunity_dataset import OpportunityDataset
import numpy as np
from scipy.stats import mode
from part4_classifier import Part4Classifier
from sklearn.externals import joblib


# import imu_wrist_p3

def create_io_pairs(inputs, labels, window_size, stride):
    """
    [STUDENT] Computes your windowed features here and labels
    :param inputs:
    :param labels:
    :param window_size:
    :param stride:
    :return:
    """
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
    return final_x, final_y


def impute_data(arr):
    """
    [STUDENT] Data imputation code goes here!
    :param arr:
    :return:
    """
    arr = arr.T
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
    arr = arr.T
    return arr


def test_imputation(dataset):
    # Get the input array on which to perform imputation
    training_data, testing_data = dataset.leave_subject_out(left_out=["S2", "S3", "S4"])
    X_train, Y_train = create_dataset(training_data, dataset.data_map["AccelWristSensors"],
                                      dataset.locomotion_labels["idx"])
    arr = X_train
    out = impute_data(arr)
    baseline = np.load("imputed_data.npy", encoding='latin1')
    return np.sum((out - baseline) ** 2)


def train(X, Y):
    """
    [STUDENT] This is where you train your classifier, right now a dummy classifier which uniformly guesses a label is "trained"
    :param X:
    :param Y:
    :return:
    """
    # classifier = imu_wrist_p3.call_me(X, Y)
    classifier = Part4Classifier()
    classifier.train(X, Y)
    return {"clf": classifier}


def test(X, model):
    """
    [STUDENT] This is where you compute predictions using your trained classifier
    :param X:
    :param model:
    :return:
    """
    return model["clf"].predict(X)


def cv_train_test(dataset, sensors, labels, window, stride, model_file_name):
    """
    Template code for performing leave on subject out cross-validation
    """
    subjects = dataset.subject_data.keys()
    Y_pred_total, Y_test_total = [], []

    print("Subjects: ", subjects)
    # Leave one subject out cross validation
    for subj in subjects:
        print("Subject: ", subj)
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
        joblib.dump(model["clf"], model_file_name + "sub:{}-model.pkl".format(subj))

        # Make predictions on the test set here
        Y_pred = test(X_test, model)

        # Append prediction and current labels to cv dataset
        Y_pred_total.append(Y_pred.reshape((Y_pred.size, 1)))
        Y_test_total.append(Y_test.reshape((Y_test.size, 1)))

    # Perform evaluations
    eval_preds(Y_pred_total, Y_test_total, labels["classes"])


def custom_run():
    """
    Run
    :return:
    """
    # Example inputs to cv_train_test function, you would use
    # these inputs for  problem 2
    window_sizes = [5, 10, 15]
    strides = [2, 3, 5]
    print("Window Sizes: {}, Strides: {}\n".format(window_sizes, strides))
    dataset = OpportunityDataset()
    sensors = dataset.data_map["FullBodySensors"]
    print("Computing Imputation error...")
    imputation_diff = test_imputation(dataset)
    print("Imputation error: {} \n".format(imputation_diff))
    # Locomotion labels
    for window_size in window_sizes:
        for stride in strides:
            print("Window Size: {}, stride:{} \n".format(window_size, stride))
            cv_train_test(dataset, sensors, dataset.locomotion_labels, window_size, stride,
                          model_file_name="FullBodySensors-W{}-S{}".format(window_size, stride))


if __name__ == "__main__":
    custom_run()

# Activity labels
# cv_train_test(dataset, sensors, dataset.activity_labels)
