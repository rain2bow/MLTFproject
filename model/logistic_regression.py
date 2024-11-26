import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load the datasets
train_data = pd.read_csv('uncleaned_balanced_train_dataset.csv')
test_data = pd.read_csv('uncleaned_test_dataset.csv')

# Define feature columns and target variable
feature_cols = ['step', 'amount', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
target_col = 'isFraud'

# Split the training and test data into features (X) and target (y)
X_train = train_data[feature_cols]
y_train = train_data[target_col]
X_test = test_data[feature_cols]
y_test = test_data[target_col]

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display the first few rows of the scaled feature sets
print("X_train_scaled:")
print(X_train_scaled[:5])
print("X_test_scaled:")
print(X_test_scaled[:5])

# Initialize the logistic regression model with predefined parameters
logreg = LogisticRegression(C=1, penalty='l2', solver='liblinear')

# Train the model
logreg.fit(X_train_scaled, y_train)

# Make predictions on the test data using the model
y_pred = logreg.predict(X_test_scaled)

# Calculate the probability scores
y_pred_proba = logreg.predict_proba(X_test_scaled)[:, 1]

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
# plt.figure()
# plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.show()
tnr = 1 - fpr
fnr = 1 - tpr

# 绘制 TNR-FNR 曲线
plt.figure()
plt.plot(fnr, tnr, color='blue', lw=2, label=f'TNR-FNR curve (area = {roc_auc_score(y_test, y_pred_proba):.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # 对角线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Negative Rate (FNR)')
plt.ylabel('True Negative Rate (TNR)')
plt.title('TNR-FNR Curve')
plt.legend(loc="lower left")
plt.savefig("./pics/lr.png")
plt.show()
