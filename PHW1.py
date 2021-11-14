import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def combination(dataset, numerical_feature_list, categorical_feature_list):
    # Scalers
    scaler_standard = preprocessing.StandardScaler()
    scaler_minmax = preprocessing.MinMaxScaler()
    scaler_robust = preprocessing.RobustScaler()
    scaler_maxabs = preprocessing.MaxAbsScaler()
    scaler_normalize = preprocessing.Normalizer()
    scalers_list = [scaler_standard, scaler_minmax, scaler_robust, scaler_maxabs, scaler_normalize]
    scalers_name = ["standard", "minmax", "robust", "maxabs", "normalize"]

    # Encoders
    encoder_label = preprocessing.LabelEncoder()
    encoder_ordinal = preprocessing.OrdinalEncoder()
    encoders_list = [encoder_label, encoder_ordinal]
    encoders_name = ["label", "ordinal"]

    # Models
    model_tree_entropy = DecisionTreeClassifier(criterion='entropy')
    model_tree_gini = DecisionTreeClassifier()
    model_logistic = LogisticRegression()
    model_svm = SVC()
    models_list = [model_tree_entropy, model_tree_gini, model_logistic, model_svm]
    models_name = ['decisiontreeentropy', 'decisiontreegini', 'logistic', 'svm']

    # Params for decision tree
    tree_param_grid = {'max_depth': [2, 4, 6, 8, 10, 12],
                       'min_samples_split': [2, 3, 4]}

    # Params for logistic regression
    logistic_param_grid = {'solver': ['newton-cg', 'lbfgs', 'liblinear'],
                           'penalty': ['l2'], 'C': [100, 10, 1.0, 0.1, 0.01]}

    # Params for svm
    svm_param_grid = {'C': [0.1, 1, 10, 100, 1000],
                      'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                      'kernel': ['rbf']}

    params = [tree_param_grid, tree_param_grid, logistic_param_grid, svm_param_grid]

    i = 0
    j = 0
    dataset_list = []
    result_dict = {}

    # scalers x encoders = 10 combination
    # When categorical features to encode exist
    if len(categorical_feature_list) != 0:
        for encoder in encoders_list:
            # Copy the dataset to encode
            dataset_copy = dataset.copy()

            # Encoding
            if len(categorical_feature_list) > 1:
                if type(encoder) == preprocessing.LabelEncoder:
                    dataset_copy[categorical_feature_list] = dataset.apply(encoder.fit_transform)
                else:
                    dataset_copy[categorical_feature_list] = encoder.fit_transform(
                        dataset[categorical_feature_list])
            elif len(categorical_feature_list) == 1:
                if type(encoder) == preprocessing.LabelEncoder:
                    dataset_copy[categorical_feature_list[0]] = dataset.apply(encoder.fit_transform)
                else:
                    dataset_copy[categorical_feature_list[0]] = encoder.fit_transform(
                        dataset[categorical_feature_list])

            encoded_dataset = dataset_copy

            # When numerical features to scale exist
            if len(numerical_feature_list) != 0:
                for scaler in scalers_list:
                    # Input the copy of the encoded dataset in the list
                    dataset_list.append(encoded_dataset)

                    # Scaling
                    if len(numerical_feature_list) > 1:
                        dataset_list[j][numerical_feature_list] = scaler.fit_transform(dataset[numerical_feature_list])
                    elif len(categorical_feature_list) == 1:
                        dataset_list[j][numerical_feature_list[0]] = scaler.fit_transform(
                            dataset[numerical_feature_list])

                    for model, modelName, param_grid in zip(models_list, models_name, params):
                        for k in [3, 5, 10]:
                            # Save in the result dictionary
                            dataset_name = scalers_name[j] + "_" + encoders_name[i] + '_' + modelName
                            result_dict[dataset_name] = dataset_list[j]

                            # Test the model
                            model_test = GridSearchCV(model, param_grid, cv=k, n_jobs=4, verbose=1)
                            model_test.fit(X_train, y_train)

                            # Save the results in the following variables
                            best_estimator = model_test.best_estimator_
                            best_parameters = model_test.best_params_
                            best_score = model_test.best_score_

                            # Show the results of evaluation
                            print("The results of evaluation:", dataset_name)
                            print('k = ' + str(k) + ', best estimator :', best_estimator)
                            print('k = ' + str(k) + ', best parameters:', best_parameters)
                            print('k = ' + str(k) + ', best score     :', best_score)
                            print()

                    j = j + 1

                i = i + 1

            # When numerical features to scale not exist
            else:
                for model, modelName, param_grid in zip(models_list, models_name, params):
                    for k in [3, 5, 10]:
                        # Save in the result dictionary
                        dataset_name = encoders_name[i] + '_' + modelName
                        result_dict[dataset_name] = dataset_list[i]

                        # Test the model
                        model_test = GridSearchCV(model, param_grid, cv=k, n_jobs=4, verbose=1)
                        model_test.fit(X_train, y_train)

                        # Save the results in the following variables
                        best_estimator = model_test.best_estimator_
                        best_parameters = model_test.best_params_
                        best_score = model_test.best_score_

                        # Show the results of evaluation
                        print("The results of evaluation:", dataset_name)
                        print('k = ' + str(k) + ', best estimator :', best_estimator)
                        print('k = ' + str(k) + ', best parameters:', best_parameters)
                        print('k = ' + str(k) + ', best score     :', best_score)
                        print()

                i = i + 1

    # When categorical features to encode not exist
    else:
        # When numerical features to scale exist
        if len(numerical_feature_list) != 0:
            for scaler in scalers_list:
                # Input the copy of the dataset in the list
                dataset_list.append(dataset.copy())

                # Scaling
                if len(numerical_feature_list) > 1:
                    dataset_list[i][numerical_feature_list] = scaler.fit_transform(dataset[numerical_feature_list])
                elif len(categorical_feature_list) == 1:
                    dataset_list[i][numerical_feature_list[0]] = scaler.fit_transform(dataset[numerical_feature_list])

                for model, modelName, param_grid in zip(models_list, models_name, params):
                    for k in [3, 5, 10]:
                        # Save in the result dictionary
                        dataset_name = scalers_name[i] + "_" + modelName
                        result_dict[dataset_name] = dataset_list[i]

                        # Test the model
                        model_test = GridSearchCV(model, param_grid, cv=k, n_jobs=4, verbose=1)
                        model_test.fit(X_train, y_train)

                        # Save the results in the following variables
                        best_estimator = model_test.best_estimator_
                        best_parameters = model_test.best_params_
                        best_score = model_test.best_score_

                        # Show the results of evaluation
                        print("The results of evaluation:", dataset_name)
                        print('k = ' + str(k) + ', best estimator :', best_estimator)
                        print('k = ' + str(k) + ', best parameters:', best_parameters)
                        print('k = ' + str(k) + ', best score     :', best_score)
                        print()

                i = i + 1

    return result_dict


# Get the data
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', header=None)
data.columns = ['ID', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                'Normal Nucleoli', 'Mitoses', 'Target']

# Check the data
print('Number of instances = %d' % (data.shape[0]))
print('Number of attributes = %d' % (data.shape[1]))
print('\nData before preprocessed')
print(data.head())

# Replace the dirty data
data = data.replace('?', np.NaN)
print('Number of missing values:')
print(data.isna().sum())
print()

# Fill the na data with median
data = data.fillna(data.median())
data = data.drop(['ID'], axis=1)
print('Number of missing values after preprocessed:')
print(data.isna().sum())
print('\nData after preprocessed')
print(data.head())

# Separate the data
X_train = data.drop(columns='Target')
y_train = data.Target

# Start examination
result = combination(data, data.columns, [])
