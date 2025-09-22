import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import joblib

# Load the enhanced synthetic data
df = pd.read_csv('student_data_enhanced.csv')

# 1. Define features (X) and target (y)
features = ['attendance', 'marks', 'attempts', 'fees_due']
target = 'did_dropout'

X = df[features]
y = df[target]

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# stratify=y ensures the same proportion of dropout cases in both sets

# 3. Calculate the class imbalance ratio
class_imbalance_ratio = np.sum(y_train == 0) / np.sum(y_train == 1)
print(f"Class imbalance ratio (0/1): {class_imbalance_ratio:.2f}")

# 4. Train the LightGBM model with tuned parameters
print("\nTraining the LightGBM model with tuned hyperparameters on enhanced data...")
model = lgb.LGBMClassifier(
    objective='binary',
    metric='auc',
    random_state=42,
    is_unbalance=True, # Addresses class imbalance
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=100
)
model.fit(X_train, y_train)

# 5. Predict on the test set
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

# 6. Evaluate the model's performance
print("\n--- Model Evaluation (After Tuning & Enhanced Data) ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Save the trained model
joblib.dump(model, 'dropout_prediction_model_final.pkl')
print("\nFinal tuned model saved as 'dropout_prediction_model_final.pkl'")