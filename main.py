import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

train_data = pd.read_csv("train_v2.csv", low_memory=False)
test_data = pd.read_csv("test_v2.csv", low_memory=False)
# *Load the training and testing data

# print("Training data:")
# print(train_data.info())
# print(train_data.head())

# print("\nTesting data:")
# print(test_data.info())
# print(test_data.head())

# print("\nTarget variable in training data:")
# print(train_data["loss"].head())
# *basic explantion of data and the few key points in each

_ = train_data.isnull().sum()
__ = test_data.isnull().sum()

print("Missing values in training data:")
print(_)
# *finding how many items are null to be sorted later

rows_with_null = train_data[train_data.isnull().any(axis=1)]

if not rows_with_null.empty:
    print("Rows with null values:")
    print(rows_with_null)
else:
    print("No rows with null values found in the DataFrame.")
# *pinpointing NULL values to be fixed or changed later when the data is present

columns_with_null = _[_ > 0]

if not columns_with_null.empty:
    print("Columns with null values:")
    print(columns_with_null)
else:
    print("No null values found in the DataFrame.")
# *also printing columns to double check and cross reference if
# *all values were found

print("\nMissing values in testing data:")
print(__)

rows_with_null = test_data[test_data.isnull().any(axis=1)]

if not rows_with_null.empty:
    print("Rows with null values:")
    print(rows_with_null)
else:
    print("No rows with null values found in the DataFrame.")

columns_with_null = __[__ > 0]

if not columns_with_null.empty:
    print("Columns with null values:")
    print(columns_with_null)
else:
    print("No null values found in the DataFrame.")
# *double checking test_data as well

train_data.dropna(inplace=True)
test_data.dropna(inplace=True)
# *remove rows with missing values

plt.figure(figsize=(10, 6))
sns.boxplot(data=train_data.drop(columns=["id", "loss"]))
plt.title("Boxplot of Numerical Features (excluding 'id' and 'loss')")
plt.xticks(rotation=45)
plt.show()
# *check for outliers

print("Summary statistics of numerical features:")
print(train_data.describe())
# *summary statistics of numerical features

correlation_matrix = train_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False, fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=train_data, x="loss", bins=20, kde=True)
plt.title("Distribution of Target Variable 'loss'")
plt.xlabel("Loss")
plt.ylabel("Frequency")
plt.show()
# *visualize distributions

X = train_data.drop(columns=["id", "loss"])
y = train_data["loss"]
# *prepare data for modeling

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# *split data into training and validation sets

model = LogisticRegression(
    max_iter=1000
)  #!severe amount of iterations because of large dataset
model.fit(X_train, y_train)
# *train logistic regression model (ML from here on out)

y_pred = model.predict(X_val)
# *predict on validation

accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)
# *evaluate model

param_grid = {"C": [0.001, 0.01, 0.1, 1, 10]}
# *define hyperparameters for grid search

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
# *grid search

best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)
# *get best hyperparameters

best_model = LogisticRegression(**best_params)
best_model.fit(X_train, y_train)
# *retrain model with best hyperparameters

y_pred = best_model.predict(X_val)
# *predict on validation set

accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy (after hyperparameter tuning):", accuracy)
# *evaluate model again to show any change

X_test = test_data.drop(columns=["id"])
# *prepare data for prediction

test_predictions = best_model.predict(X_test)
# *make predictions on testing data

submission_df = pd.DataFrame({"id": test_data["id"], "loss": test_predictions})
submission_df.to_csv("submission.csv", index=False)
# *submit predictions as a seperate CSV
