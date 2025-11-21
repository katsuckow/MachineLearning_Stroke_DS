import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import sys
assert sys.version_info >= (3, 7)
import sklearn #"scikit-learn"
assert sklearn.__version__ >= "1.2"
from pandas.plotting import scatter_matrix
assert pd.__version__ >="1.5.0"
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn import pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score

#load dataset
data = np.loadtxt('healthcare-dataset-stroke-data.csv', delimiter=',', dtype=str, encoding='utf-8')

data1 = data.copy()

# remove outlier "Other" in gender
data = data[data[:, 1] != "Other"]

# create security copy of data1








# column names - removing ID column
column_names = data[0, 1:-1].tolist()

# data remove columns in row zero
data = data[1:,:]



# create 3 new columns at the end of the dataset for the groups of different glucose levels
# these will be later one coded

# low sugar: glucose below 85
# normal sugar: for glucose between 85 and 120
# high sugar: for glucose above 120

lowsugar = np.zeros(5109)
lowsugar = np.array(lowsugar).reshape(-1, 1)
data = np.concatenate([data, lowsugar], axis=1)

data[data[:, 8].astype(float) <= 85, -1] = '1'
print(data[1:5,:])

nosugar = np.zeros(5109)
nosugar = np.array(nosugar).reshape(-1, 1)
data = np.concatenate([data, nosugar], axis=1)

data[(data[:, 8].astype(float) > 85)&(data[:, 8].astype(float)<= 120), -1] = '1'
print(data[1:5,:])

highsugar = np.zeros(5109)
highsugar = np.array(highsugar).reshape(-1, 1)
data = np.concatenate([data, highsugar], axis=1)

data[(data[:, 8].astype(float) > 120), -1] = '1'
print(data[1:5,:], data.shape)




# potentially remove children - see discussion in data analysis
# commented out because removing children did not improve results
# data = data[data[:, 2].astype(float) > 20, :]

print(np.unique(data[:,-1], return_counts = True))

# separating data: prep for matrix X and target column y
# prep for training
X = data[:, [1,2,3,4,5,6,7,8,9,10,12,13,14]]
print("New X:",X[1:5,],X.shape)
y = data[:, 11]




# print(column_names)
# print("Column names:", column_names)
print("\nLabels in target column\n", np.unique(y, return_counts=True))

print(X.shape,y.shape)
# train-test split Verfahren
# die Daten werden in ein Traininsset gesplittet
# wobei 20 % der Daten für das Testen und 80% auf das Training verteilt wird
# da die Werte in der Zielspalte sehr ungleich verteilt sind (deutlich mehr Einsen für Stroke Patienten als für nicht-stroke Patienten), wird über die Zielspalte stratifiziert
# also gleiches Verhältnis der Verteilung der Stroke Patienten in der Trainings als auch in der Testmenge
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


print("X_train",X_train.shape,X_test.shape,y_train.shape,y_test.shape)
print("Data1 Shape, data.shape",data1.shape, data.shape)



print(np.unique(y_test, return_counts = True), np.unique(y_train, return_counts = True))







# Look at the values for checking
print(X_train, X_test, y_train, y_test)

# We decided to use the Encoder before the Imputer because I wanna use the KNNImputer for the missing bmi & smoking values
# - gender: female 1, male 0
# - ever_married: yes 1, no 0
# - work_type: 'children' 0, 'Govt_job' 1, 'Private' 2, 'Self-employed' 3, 'Never_worked' 4
# - residence_type: urban 1, rural 0
# - smoking_status: never smoked 0,  formerly smoked 1, smokes 2, -> unknown NA -> imputer

# - one hot encoder for the last three columns 0,1 [10,11,12]


# Define the order for ordinal encoding
category_orders = [
    ["Male", "Female"],
    ["No", "Yes"],
    ['children', 'Govt_job', 'Private', 'Self-employed', 'Never_worked'], # ordered by estimated stress level
    ["Rural", "Urban"],
    ["never smoked", "formerly smoked", "smokes"],
]


# column transformer for categorical columns
transformer_category_columns = ColumnTransformer(

transformers=[
    (
        "ordinal_encoder",
        OrdinalEncoder(
            categories=category_orders,
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
        ),
        [0, 4, 5, 6, 9],  # Indices of categorical columns to encode
    ),
    (
        "onehot_encoder",
        OneHotEncoder(
            categories="auto",
            handle_unknown="ignore",  # recommended to avoid errors at inference  # optional: returns dense arrays
        ),
        [10, 11, 12], # one hot encoder for additional 3 columns -- three different glucose levels
    ),
],
remainder="passthrough",
)

# complete data preparation pipeline
data_preparation_pipeline = Pipeline(
    steps=[
        ("transformer_category_columns", transformer_category_columns),
        ("str_to_float", FunctionTransformer(lambda X: np.where(X == 'N/A', 'nan', X).astype(float))),
        ("scaler", MinMaxScaler()),
        ("imputer", KNNImputer(n_neighbors=10)),
    ]
)


# prepare the training and test data
X_train_prepared = data_preparation_pipeline.fit_transform(X_train)
X_test_prepared = data_preparation_pipeline.transform(X_test)


##########################
# RandomForestClassification mit Voreil, dass man viele Trees gleichzeitig auusprobieren kann
# Daten lassen sich besser ausbalanzieren
###############

X, y = make_classification(n_samples=2000, n_features=13)
clf = RandomForestClassifier(max_depth=20, random_state=42,min_samples_leaf=20,class_weight='balanced')
clf.fit(X_train_prepared, y_train)

importances = clf.feature_importances_
print("Feature importances:")
print(importances)
print(data1[0,:])


###################################
# Decision tree bringt auhc mit Gridsearch keine besonders guten Ergebnisse
######################################

# tree_para = {'criterion':['entropy'],'splitter':['best'],'max_depth':[16,18,20,25],"min_samples_leaf":[5,6,7]}
# clf = GridSearchCV(DecisionTreeClassifier(), tree_para, cv=5)

# tree = DecisionTreeClassifier(max_depth=20,min_samples_leaf=4) # to handle the imbalanced dataset
# tree.fit(X_train_prepared, y_train)
print("\nTree:\ntraining score" , clf.score(X_train_prepared, y_train))
y_train_pred = clf.predict(X_train_prepared)

tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train,y_train_pred).ravel().tolist()
print("TN:", tn_train,"FP:", fp_train,"FN:", fn_train,"TP:", tp_train)

print("Tree: \ntest score" , clf.score(X_test_prepared, y_test))
y_test_pred = clf.predict(X_test_prepared)
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel().tolist()
print("TN:", tn,"FP:", fp,"FN:", fn,"TP:",tp)
# print("recall: ", recall_score(y_test, y_test_pred))

print("confusion matrix: \n", confusion_matrix(y_test_pred, y_test))


#######################################
#
# Feature-importance extraction + permutation importance
#
########################################

input_cols = [
    "gender","age","hypertension","heart_disease","ever_married",
    "work_type","Residence_type","avg_glucose_level","bmi","smoking_status",
    "lowsugar","nosugar","highsugar"
]

ct = data_preparation_pipeline.named_steps["transformer_category_columns"]
try:
    # sklearn >=1.0 supports get_feature_names_out on ColumnTransformer
    feature_names = ct.get_feature_names_out(input_cols)
except Exception:
    # fallback: create generic names if not available
    feature_names = [f"f{i}" for i in range(X_train_prepared.shape[1])]

# tree-based importances
importances = clf.feature_importances_
order = np.argsort(importances)[::-1]
print("Feature importances (RandomForest):")
for idx in order:
    print(f"{feature_names[idx]}: {importances[idx]:.4f}")
