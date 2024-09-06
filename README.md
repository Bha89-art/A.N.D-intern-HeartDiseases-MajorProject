# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
# Load the Dataset
data = pd.read_csv('heart.csv')
# Step 2: Inspecting the data to know its structure

print(data.head())
print(data.info())
print(data.describe())
# Step 3: Separate features (X) and target (y)
# Here, 'HeartDisease' is the label I'm trying to predict
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']
# Step 4: Identify categorical and numerical columns
# This helps in deciding the preprocessing steps for each type
categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
numerical_columns = [col for col in X.columns if col not in categorical_columns]
# Step 5: Preprocessing the numerical data
# Filling in missing values with the mean and scaling the data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
# Step 6: Preprocessing the categorical data
# Handling missing values by filling with the most frequent value and encoding the categories
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
# Step 7: Combine the preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])
# Step 8: Creating a pipeline that combines preprocessing with model training
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
# Step 9: Train-test split to evaluate the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Hyperparameter tuning using GridSearchCV
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20]
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
# Step 11: Model Evaluation

y_pred = best_model.predict(X_test)
y_pred_prob = best_model.predict_proba(X_test)[:, 1]

# Print the classification report and accuracy
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_prob))
# Step 12: Confusion Matrix
# Visualizing the confusion matrix to see how well the model is classifying
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# Step 13: ROC Curve
# Plotting the ROC curve to visualize the model's performance
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
# Step 15: Feature Importance from RandomForest
# Visualizing the importance of each feature based on the trained RandomForest model
importances = best_model.named_steps['classifier'].feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title('Feature Importances')
plt.bar(range(X_test_transformed.shape[1]), importances[indices], align='center')
plt.xticks(range(X_test_transformed.shape[1]), best_model.named_steps['preprocessor'].get_feature_names_out()[indices], rotation=90)
plt.xlim([-1, X_test_transformed.shape[1]])
plt.show()
# Step 16: Partial Dependence Plots
# These plots help in understanding how features impact the prediction
fig, ax = plt.subplots(figsize=(12, 8))
display = PartialDependenceDisplay.from_estimator(
    best_model,
    X_train,
    features=numerical_columns,
    ax=ax
)
plt.suptitle('Partial Dependence Plots')
plt.subplots_adjust(top=0.9)
plt.show()
