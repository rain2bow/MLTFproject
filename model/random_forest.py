import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load the datasets
train_data = pd.read_csv('uncleaned_imbalanced_train_dataset.csv')
test_data = pd.read_csv('uncleaned_test_dataset.csv')

# Display the first few rows of the datasets
print("Training Data:")
print(train_data.head())
print("Test Data:")
print(test_data.head())

# Define feature columns and target variable
feature_cols = ['step', 'amount', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
target_col = 'isFraud'

# Split the training and test data into features (X) and target (y)
X_train = train_data[feature_cols]
y_train = train_data[target_col]
X_test = test_data[feature_cols]
y_test = test_data[target_col]

# Display the first few rows of the feature sets
print("X_train:")
print(X_train.head())
print("y_train:")
print(y_train.head())

# Initialize the Random Forest classifier
rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto')

# Train the model
rf.fit(X_train, y_train)

# Display feature importances of the model
importances = rf.feature_importances_
print("Feature Importances:")
for feature, importance in zip(feature_cols, importances):
    print(f"{feature}: {importance}")

# Make predictions on the test data using the model
y_pred = rf.predict(X_test)

# Calculate the probability scores
y_pred_proba = rf.predict_proba(X_test)[:, 1]

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Display the evaluation results
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
print("ROC AUC Score:")
print(roc_auc)

# Plot ROC AUC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
