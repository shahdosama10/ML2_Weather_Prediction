# %%
# =================================================================================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.naive_bayes import GaussianNB
# =================================================================================================
# %%
# =================================================================================================
# Preprocessing
# =================================================================================================


# load 'weather_forecast_data.csv' dataset
df = pd.read_csv('weather_forecast_data.csv')

# =================================================================================================
# %%
# =================================================================================================
# get copy from the original to preprocess
# =================================================================================================

df_pre = df.copy()

# =================================================================================================
# %%
# =================================================================================================
# check missing values
# =================================================================================================

# to know the number of the rows
print(f"total records:",len(df), "\n")


# to get the number of missing values in each column
print("missing records in each column:","\n")
print(df_pre.isnull().sum())

print("-"*80)

print("Records with null values: ","\n")
print(df_pre[df_pre.isnull().any(axis=1)])

print("-"*80)
# according to the output there are missing values
# =================================================================================================
# %%
# =================================================================================================
# Handle missing values with dropping them
# =================================================================================================

df_dropped_nulls= df_pre.dropna()
print(f"total records without nulls:",len(df_dropped_nulls), "\n")

df_dropped_nulls.head()
print("-"*80)
# =================================================================================================
# %%
# =================================================================================================
#  Handle missing values with replacing them with Avg
# =================================================================================================


# get the numerical features only because we can't get mean for categorical feature
df_numerical_features_only=df_pre.select_dtypes(include="number")

# replace the null values with the average of the numerical features
df_numerical_filled_avg = df_numerical_features_only.fillna(df_numerical_features_only.mean())

# concatenate the numerical features with the target column "Rain" and create a new dataframe "df_filledAvg"
df_filled_avg=pd.concat([df_numerical_filled_avg,df_pre["Rain"]], axis=1)
print("DataFrame after replacing null values with the average:")
print(df_filled_avg)

print("-"*80)
# =================================================================================================
# %%
# =================================================================================================
# determine targets & features
# =================================================================================================

targets_columns=["Rain"]

df_targets_filled_avg = df_filled_avg[targets_columns]
df_features_filled_avg = df_filled_avg.drop(columns=targets_columns)

df_targets_dropped_nulls = df_dropped_nulls[targets_columns]
df_features_dropped_nulls = df_dropped_nulls.drop(columns=targets_columns)

print("Avg data:")
print(df_features_filled_avg.head())
print(df_targets_filled_avg.head())
print("-"*80)

print("Dropped nulls data:")
print(df_features_dropped_nulls.head())
print(df_targets_dropped_nulls.head())
print("-"*80)

# =================================================================================================
# %%
# =================================================================================================
# splitting data into train , test
# =================================================================================================

# make the 80% from the data training set and 20% from the data testing set
# random state to ensure that the split return the same data each run

df_features_train_avg, df_features_test_avg, df_targets_train_avg, df_targets_test_avg = train_test_split(df_features_filled_avg, df_targets_filled_avg, test_size=0.2, random_state=42) 
df_features_train_dropped, df_features_test_dropped, df_targets_train_dropped, df_targets_test_dropped = train_test_split(df_features_dropped_nulls, df_targets_dropped_nulls, test_size=0.2, random_state=42) 

print(len(df_features_train_avg))
print(len(df_features_test_avg))
print(len(df_targets_train_avg))
print(len(df_targets_test_avg))
print("-"*80)

print(len(df_features_train_dropped))
print(len(df_features_test_dropped))
print(len(df_targets_train_dropped))
print(len(df_targets_test_dropped))
print("-"*80)

# =================================================================================================
# %%
# =================================================================================================
# encode the targets
# =================================================================================================

# final targets will be worked on 

label_encoder = LabelEncoder()
df_targets_train_avg = label_encoder.fit_transform(df_targets_train_avg.values.ravel())
df_targets_test_avg = label_encoder.fit_transform(df_targets_test_avg.values.ravel())

df_targets_train_dropped = label_encoder.fit_transform(df_targets_train_dropped.values.ravel())
df_targets_test_dropped = label_encoder.fit_transform(df_targets_test_dropped.values.ravel())


# =================================================================================================
# %%
# =================================================================================================
# check scaling of data
# =================================================================================================

print("filled avg data:")
print(df_filled_avg.describe().T)
print("-"*80)

print("dropped nulls data:")
print(df_dropped_nulls.describe().T)
print("-"*80)

# according to the output from min, max the numeric features dosn't have the same scale
# =================================================================================================
# %%
# =================================================================================================
# features are scaled
# =================================================================================================

scaler = MinMaxScaler()

# the scaler return ndarray

df_features_train_avg = scaler.fit_transform(df_features_train_avg)
df_features_test_avg = scaler.fit_transform(df_features_test_avg)


df_features_train_dropped = scaler.fit_transform(df_features_train_dropped)
df_features_test_dropped = scaler.fit_transform(df_features_test_dropped)


# convert the ndarray to DataFrame
# final features will be worked on

df_features_train_avg = pd.DataFrame(df_features_train_avg, columns=df_features_filled_avg.columns)
df_features_test_avg = pd.DataFrame(df_features_test_avg, columns=df_features_filled_avg.columns)


df_features_train_dropped = pd.DataFrame(df_features_train_dropped, columns=df_features_dropped_nulls.columns)
df_features_test_dropped = pd.DataFrame(df_features_test_dropped, columns=df_features_dropped_nulls.columns)


print("Avg Features:")
print(df_features_train_avg.describe().T)
print(df_features_test_avg.describe().T)
print("-"*80)

print("Dropped Nulls Features:")
print(df_features_train_dropped.describe().T)
print(df_features_test_dropped.describe().T)
print("-"*80)

# =================================================================================================
# %%
# =================================================================================================
# Implement Decision Tree, k-Nearest Neighbors (kNN) and naïve Bayes
# Evaluate accuracy, precision, and recall 
# =================================================================================================


def evaluateModels(target, predictions):
    # get the percentage 
    accuracy = accuracy_score(target, predictions) * 100
    precision = precision_score(target, predictions) * 100
    recall = recall_score(target, predictions) * 100

    print(f"Accuracy: {accuracy:.2f}%", f"Precision: {precision:.2f}%", f"Recall: {recall:.2f}%")
    return accuracy, precision, recall

# =================================================================================================
# %%
# =================================================================================================
# KNN with scikit-learn
# =================================================================================================


# KNN with scikit-learn using 5 Neighbors and brute force
knnModel = KNeighborsClassifier(n_neighbors=5, algorithm='brute')

# using technique of replacing the nulls values with the mean
knnModel.fit(df_features_train_avg, df_targets_train_avg)

knnPredictions = knnModel.predict(df_features_test_avg)
print("KNN using technique of replacing the nulls values with the mean ")
knn_accuracy_avg, knn_precision_avg, knn_recall_avg = evaluateModels(df_targets_test_avg, knnPredictions)

print("-"*80)

# using technique of dropping the nulls
print("KNN using technique of dropping the nulls")

knnModel.fit(df_features_train_dropped, df_targets_train_dropped)
knnPredictions = knnModel.predict(df_features_test_dropped)
knn_accuracy_dropped, knn_precision_dropped, knn_recall_dropped = evaluateModels(df_targets_test_dropped, knnPredictions)
print("-"*80)

# =================================================================================================
# %%
# =================================================================================================
# KNN from scratch
# =================================================================================================


# get the distances between two points
def eculidean_distance(p, q):
    distance = 0
    for i in range(len(q)):
       distance += ( (p[i] - q[i] ) ** 2 )

    return np.sqrt(distance)

# =================================================================================================
# %%
# =================================================================================================
# find the neighbors of a point
# =================================================================================================


# find the neighbors of a point (x_test)
# loop over the x_train to find the neighbors
def find_neighbours(x_train, x_test, y_train):
    n = len(x_train)
    distances = np.zeros(n)
    
    for i in range(n):
        distances[i] = eculidean_distance(x_train[i], x_test)


    # convert distances and y_train to data frame to can concatenate
    distances = pd.DataFrame(distances, columns=['Distance'])
    y_train = pd.DataFrame(y_train, columns=['Target'])
    neighbours = pd.concat([distances,y_train], axis=1)

    # sort the neighbors according to the distances
    neighbours = neighbours.sort_values(by='Distance', ascending=True).reset_index(drop=True)

    return neighbours

# ================================================================================================
# %%
# =================================================================================================
# get y predict for a one x test
# =================================================================================================


# take the neighbors and k 
# Return the value with the highest count
def get_y_predict(neighbours, k):
    # get first k rows
    top_k = neighbours.head(k)

    # count the number of 0s and 1s
    label_counts = top_k['Target'].value_counts()

    # return the value with the highest count
    return label_counts.idxmax()

# =================================================================================================
# %%
# =================================================================================================
# get y predict for the test data
# =================================================================================================

# return y predictions for the whole test set
def predict(x_train, x_test, y_train, k):
    y_predictions = np.zeros(len(x_test))
    x_test = x_test.to_numpy()

    # loop over the x_test
    for i in range(len(x_test)):

        # get the neighnours
        neighbours = find_neighbours(x_train.to_numpy(), x_test[i], y_train)

        # get the y prediction and update the list of predictions
        y_predictions[i] = get_y_predict(neighbours, k)

    return y_predictions

# =================================================================================================
# %%
# =================================================================================================
# different k values for KNN algorithm
# =================================================================================================

# KNN from Scratch with k = 3
knn_scratch_predictions = predict(df_features_train_avg, df_features_test_avg, df_targets_train_avg, 3)
print("KNN FROM SCRATCH WITH K = 3")
knn3_scratch_accuracy, knn3_scratch_precision, knn3_scratch_recall = evaluateModels(df_targets_test_avg, knn_scratch_predictions)
print("-"*80)

# KNN from Scratch with k = 5
knn_scratch_predictions = predict(df_features_train_avg, df_features_test_avg, df_targets_train_avg, 5)
print("KNN FROM SCRATCH WITH K = 5")
knn5_scratch_accuracy, knn5_scratch_precision, knn5_scratch_recall = evaluateModels(df_targets_test_avg, knn_scratch_predictions)
print("-"*80)

# KNN from Scratch with k = 7
knn_scratch_predictions = predict(df_features_train_avg, df_features_test_avg, df_targets_train_avg, 7)
print("KNN FROM SCRATCH WITH K = 7")
knn7_scratch_accuracy, knn7_scratch_precision, knn7_scratch_recall = evaluateModels(df_targets_test_avg, knn_scratch_predictions)
print("-"*80)

# KNN from Scratch with k = 9
knn_scratch_predictions = predict(df_features_train_avg, df_features_test_avg, df_targets_train_avg, 9)
print("KNN FROM SCRATCH WITH K = 9")
knn9_scratch_accuracy, knn9_scratch_precision, knn9_scratch_recall = evaluateModels(df_targets_test_avg, knn_scratch_predictions)
print("-"*80)

# KNN from Scratch with k = 11
knn_scratch_predictions = predict(df_features_train_avg, df_features_test_avg, df_targets_train_avg, 11)
print("KNN FROM SCRATCH WITH K = 11")
knn11_scratch_accuracy, knn11_scratch_precision, knn11_scratch_recall = evaluateModels(df_targets_test_avg, knn_scratch_predictions)
print("-"*80)

# =================================================================================================
# %%
# =================================================================================================
# Implement Decision Tree
# =================================================================================================

dt_model_avg = DecisionTreeClassifier(random_state=42)

dt_model_avg.fit(df_features_train_avg, df_targets_train_avg)  #usign averge filled

dt_preds_avg = dt_model_avg.predict(df_features_test_avg)

print("Decision Tree usign averge filled")
dt_accuracy_avg , dt_precision_avg , dt_recall_avg = evaluateModels(df_targets_test_avg , dt_preds_avg)

print("-"*80)

dt_model_dropped = DecisionTreeClassifier(random_state=42)


dt_model_dropped.fit(df_features_train_dropped, df_targets_train_dropped) # using dropped nulls

dt_preds_dropped = dt_model_dropped.predict(df_features_test_dropped)

print("Decision Tree usign dropped nulls")
dt_accuracy_dropped , dt_precision_dropped , dt_recall_dropped = evaluateModels(df_targets_test_dropped , dt_preds_dropped)
print("-"*80)

# =================================================================================================
# %%
# =================================================================================================
# Decision Tree Explanation Report
# plot of the decision tree
# =================================================================================================

#for Avgerage-Filled
plt.figure(figsize=(20, 10))
plot_tree(
    dt_model_avg, 
    feature_names=df_features_filled_avg.columns, 
    class_names=["No Rain", "Rain"], 
    filled=True, 
    rounded=True
)
plt.title("Decision Tree")
plt.show()
    

# =================================================================================================
# %%
# =================================================================================================
# Naive Bayes Algorithm
# =================================================================================================

# initialize the classifer
model = GaussianNB()


# Model the features with avg replacement
# fit the model
model.fit(df_features_train_avg, df_targets_train_avg)

# predict the target values
y_pred_avg = model.predict(df_features_test_avg)

# evaluate the target values
print("Avg data:")

nb_accuracy_avg , nb_precision_avg , nb_recall_avg = evaluateModels(df_targets_test_avg ,y_pred_avg )
print("-"*80)

# =================================================================================================
# %%
# =================================================================================================
# Model the features with dropped nulls
# =================================================================================================

# fit the model
model.fit(df_features_train_dropped , df_targets_train_dropped)

# predict the target values
y_pred_dropped = model.predict(df_features_test_dropped)

# evaluate the target values
print("Dropped Nulls data:")

nb_accuracy_dropped , nb_precision_dropped , nb_recall_dropped = evaluateModels(df_targets_test_dropped ,y_pred_dropped )
print("-"*80)


# =================================================================================================
# %%
# =================================================================================================
# Compare between Decision Tree, k-Nearest Neighbors (kNN) and naïve Bayes
# =================================================================================================

# Compare the performance of your implementations by evaluating accuracy, precision, and recall metrics. 

data = {
    "Model": [
        "KNN(K=5) ",
        "Decision Tree ",
        "Naive Bayes "
    ],
    "Accuracy (%)": [knn_accuracy_avg, dt_accuracy_avg, nb_accuracy_avg],
    "Precision (%)": [knn_precision_avg, dt_precision_avg, nb_precision_avg],
    "Recall (%)": [knn_recall_avg, dt_recall_avg, nb_recall_avg]
}


comparison_table = pd.DataFrame(data)

print("\nComparison of Models (Replacing Nulls with Mean)\n")
print(comparison_table)
print("-"*80)

# =================================================================================================
# %%
# =================================================================================================
# compare the performance of your custom kNN implementation with the pre-built kNN algorithms in scikit-learn
# =================================================================================================



data = {
    "Model": [
        "KNN (Scratch Implementation)",
        "KNN (Library Implementation)"
    ],
    "Accuracy (%)": [knn5_scratch_accuracy, knn_accuracy_avg],
    "Precision (%)": [knn5_scratch_precision, knn_precision_avg],
    "Recall (%)": [knn5_scratch_recall, knn_recall_avg]
}

comparison_table = pd.DataFrame(data)

print("\nComparison of KNN Implementations\n")
print(comparison_table)
print("-"*80)


# =================================================================================================
# %%
# =================================================================================================
# evaluating the performance of scikit learn implementations of 3 algorithms with respect to the different handling missing
# =================================================================================================

data = {
    "Model": [
        "KNN(K=5) ",
        "Decision Tree ",
        "Naive Bayes "
    ],
    "Accuracy (%)": [knn_accuracy_avg, dt_accuracy_avg, nb_accuracy_avg],
    "Precision (%)": [knn_precision_avg, dt_precision_avg, nb_precision_avg],
    "Recall (%)": [knn_recall_avg, dt_recall_avg, nb_recall_avg]
}


comparison_table = pd.DataFrame(data)

print("\nComparison of Models (Replacing Nulls with Mean)\n")
print(comparison_table)
print("-"*80)


data = {
    "Model": [
        "KNN(K=5) ",
        "Decision Tree ",
        "Naive Bayes "
    ],
    "Accuracy (%)": [knn_accuracy_dropped, dt_accuracy_dropped, nb_accuracy_dropped],
    "Precision (%)": [knn_precision_dropped, dt_precision_dropped, nb_precision_dropped],
    "Recall (%)": [knn_recall_dropped, dt_recall_dropped, nb_recall_dropped]
}


comparison_table = pd.DataFrame(data)

print("\nComparison of Models (Dropping Nulls)\n")
print(comparison_table)

print("-"*80)

# =================================================================================================
# %%
# =================================================================================================
# evaluating the performance of your implementations of the k-Nearest Neighbors (kNN) from scratch with different k values at least 5 values. 
# Compare these results with the performance of the corresponding algorithms implemented using scikit-learn. 
# =================================================================================================

data = {
    "Model": [
        "KNN (Scratch, K=3)",
        "KNN (Scratch, K=5)",
        "KNN (Scratch, K=7)",
        "KNN (Scratch, K=9)",
        "KNN (Scratch, K=11)",
        "KNN (scikit-learn, K=5)",
        "Decision Tree",
        "Naive Bayes"
      
    ],
    "Accuracy (%)": [
        knn3_scratch_accuracy, knn5_scratch_accuracy,
        knn7_scratch_accuracy, knn9_scratch_accuracy,
        knn11_scratch_accuracy,
        knn_accuracy_avg, dt_accuracy_avg, nb_accuracy_avg
       
    ],
    "Precision (%)": [
        knn3_scratch_precision, knn5_scratch_precision,
        knn7_scratch_precision, knn9_scratch_precision,
        knn11_scratch_precision,
        knn_precision_avg, dt_precision_avg, nb_precision_avg
      
    ],
    "Recall (%)": [
        knn3_scratch_recall, knn5_scratch_recall,
        knn7_scratch_recall, knn9_scratch_recall,
        knn11_scratch_recall,
        knn_recall_avg, dt_recall_avg, nb_recall_avg
        
    ]
}

comparison_table = pd.DataFrame(data)

print("\nComparison of Models (Including KNN from Scratch with Various K Values)\n")
print(comparison_table)
print("-"*80)


# =================================================================================================