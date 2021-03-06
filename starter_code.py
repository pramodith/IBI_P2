from helpers import *
from opportunity_dataset import OpportunityDataset
import numpy as np
from scipy.stats import mode
from part4_classifier import Part4Classifier
from sklearn.externals import joblib
from scipy.fftpack import fftfreq, rfft
from sklearn.decomposition import PCA


def create_io_pairs(inputs, labels, window_size, stride, feature_extraction_method, post_processor):
    """
    [STUDENT] Computes your windowed features here and labels
    :param inputs:
    :param labels:
    :param window_size:
    :param stride:
    :param feature_extraction_method: lambda/function to extract features from x
    :param post_processor: lambda/function post_processor to apply to each feature
    :return:
    """
    print("Creating io pairs...")
    if isinstance(feature_extraction_method, str) and feature_extraction_method == "FFT_REL_POW":
        io_pairs = create_io_pairs_fft(inputs, labels, window_size, stride)
    else:
        X = inputs
        Y = labels
        final_x = []
        final_y = []
        for i in range(0, len(X), stride):
            if i + window_size < len(X):
                inp_x = X[i:i + window_size]
                inp_x = feature_extraction_method(inp_x)  # np.mean(inp_x, axis=0)
                out_y = int(np.asscalar(mode(Y[i:i + window_size])[0][0]))
                final_x.append(inp_x)
                final_y.append(out_y)
        io_pairs = final_x, final_y
    print("Done")
    if post_processor:
        print("Post Processing...")
        return post_processor(io_pairs[0]), io_pairs[1]


def create_io_pairs_fft(inputs, labels, window_size, stride):
    # Compute your windowed features here and labels. Right now
    # it just returns the inputs and labels without changing anything.
    cnt = 0
    X = inputs
    Y = labels
    final_x = []
    final_y = []
    print("FFT. ", len(X))
    for i in range(0, len(X), stride):
        if i % 100 == 0:
            print("FFT i:", i)
        if i + window_size < len(X):
            inp_x = X[i:i + window_size]
            # inp_x = np.mean(inp_x,axis=0)
            # inp_x = np.std(inp_x,axis=0)
            inp_x = np.abs(rfft(inp_x, axis=0))
            if np.count_nonzero(np.sum(inp_x, axis=0) == 0) > 0:
                continue
            inp_x /= np.sum(inp_x, axis=0)
            final_x.append([])
            tr_inp_x = inp_x.T
            for r in tr_inp_x:
                coeff = fftfreq(r.shape[0])
                hist, bins = np.histogram(coeff, weights=r, bins=5)
                final_x[cnt].extend(hist)
            cnt += 1
            out_y = int(np.asscalar(mode(Y[i:i + window_size])[0][0]))
            final_y.append(out_y)

    return np.asarray(final_x), final_y


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


def cv_train_test(dataset, sensors, labels, window, stride, model_file_name, feature_extraction_method,
                  post_processor):
    """
    Template code for performing leave on subject out cross-validation

    :param dataset:
    :param sensors:
    :param labels:
    :param window:
    :param stride:
    :param model_file_name:
    :param feature_extraction_method:
    :param post_processor:
    :return:
    """

    subjects = dataset.subject_data.keys()
    Y_pred_total, Y_test_total = [], []

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
        X_train, Y_train = create_io_pairs(X_train, Y_train, window, stride,
                                           feature_extraction_method=feature_extraction_method,
                                           post_processor=post_processor)
        # print(Counter(Y_train))
        X_test, Y_test = create_io_pairs(X_test, Y_test, window, stride,
                                         feature_extraction_method=feature_extraction_method,
                                         post_processor=post_processor)
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


def custom_run(dataset, is_locomotion, sensor_data_field, window_size, stride, post_processor, fem_method="MEAN"):
    """
    Runs one instance of training/test

    :param dataset:
    :param is_locomotion:
    :param sensor_data_field:
    :param window_size:
    :param stride:
    :param post_processor:
    :param fem_method:
    :return:
    """
    SENSORS = ["AccelWristSensors", "ImuWristSensors", "FullBodySensors"]
    assert sensor_data_field in SENSORS
    print("Sensor: ", sensor_data_field)

    fem_methods = {
        "MEAN": lambda x: np.mean(x, axis=0),
        "RMS": lambda x: np.linalg.norm(x, axis=0) / np.sqrt(window_size),
        "STD_DEV": lambda x: np.std(x, axis=0),
        "MEAN_ABS_DEV": lambda x: NotImplementedError,
        "FFT_REL_POW": "FFT_REL_POW"
    }
    assert fem_method in fem_methods.keys()
    print("Feature Extraction Method: ", fem_method)

    # Example inputs to cv_train_test function, you would use these inputs for  problem 2
    print("Locomotion labels" if is_locomotion else "Activity labels")

    sensors = dataset.data_map[sensor_data_field]
    labels = dataset.locomotion_labels if is_locomotion else dataset.activity_labels

    print("Training...")
    print("Window Size: {}, stride:{} ".format(window_size, stride))

    cv_train_test(dataset, sensors, labels, window_size, stride,
                  model_file_name="{}-W{}-S{}".format(sensor_data_field, window_size, stride),
                  feature_extraction_method=fem_methods[fem_method], post_processor=post_processor)


def window_stride_generator(gen_pairs=True, window_sizes=[5, 10, 15], strides=[2, 3, 5]):
    pairs = zip(window_sizes, strides)
    if gen_pairs:
        for pair in pairs:
            yield pair[0], pair[1]
    else:
        for window_size in window_sizes:
            for stride in strides:
                yield window_size, stride


def pca_post_processor(X, components=20, iterated_power=50):
    """
    :param X:
    :param components:
    :return:
    """
    pca = PCA(n_components=components, whiten=True, iterated_power=iterated_power, copy=False)
    print("PCA:", pca)
    return pca.fit_transform(np.array(X))


def main_method():
    print("Loading Dataset...")
    dataset = OpportunityDataset()
    print("Done...")
    print("Computing Imputation error...")
    imputation_diff = test_imputation(dataset)
    assert imputation_diff == 0, "Imputation error: {}".format(imputation_diff)
    print("Imputation error: 0")

    for w, s in window_stride_generator(window_sizes=[5], strides=[2]):
        custom_run(dataset=dataset, is_locomotion=False, sensor_data_field="FullBodySensors", window_size=w, stride=s,
                   post_processor=pca_post_processor, fem_method="FFT_REL_POW")


def test_pca():
    np.random.seed(1)
    X = np.array([[-1, 2, -2], [-2, 8, -4], [-3, 18, -6], [1, 2, 2], [2, 8, 4], [3, 18, 9]])
    print("pca result: ", pca_post_processor(X, components=2, iterated_power=4))
    print("pca result: ", pca_post_processor(X, components=1, iterated_power=4))

    X = np.array([[-1, 2], [-2, 8], [-3, 18], [1, 2], [2, 8], [3, 18]])
    print("pca result: ", pca_post_processor(X, components=2, iterated_power=4))
    print("pca result: ", pca_post_processor(X, components=1, iterated_power=4))


if __name__ == "__main__":
    main_method()
    # test_pca()
