# Manual

# def combination(dataset, numerical_feature_list, categorical_feature_list)

Scale and enocde the dataset with the combination of Standard, MinMax, Robust, MaxAbs, Normalizer scalers and Label, Ordinal encoders.  
Classify every combination of dataset with DecisionTreeClassifier(), LogisticRegression(), and SVC().  

##	Parameter  
dataset: Dataframe to be scaled and encoded  
numerical_feature_list: List containing the numerical feature names  
categorical_feature_list: List containing the categorical feature names

##	How to operate
1.	Make the dataset you want to proceed clean.
2.	Make the name lists of numerical and categorical feature.
3.	Call the function with the variables you made on step 1 and 2 as parameters.

##	Examples
    df = pd.read_csv('something.csv')
    df.fillna(df.mean(), inplace=True)
    combination(df, ['Numerical1', 'Numerical2', 'Numerical3'], ['Categorical1', 'Categorical2', 'Categorical3'])

##	Return
	Dictionary that contains all the combination of the scaled and encoded dataset.  
	Key: Name that by which types it is scaled and encoded Ex) "standard_label"  
	Value: Corresponding dataset as dataframe
