import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from xgboost import XGBClassifier

#--- Load data from .csv files ---
train_df=pd.read_csv('./train_enhanced.csv')
test_df=pd.read_csv('./test_enhanced.csv')

X_train=train_df.drop(['label','filename'], axis=1)
y_train=train_df['label']
groups_train=train_df['filename'].values

X_test=test_df.drop(['label','filename'], axis=1)
y_test=test_df['label']
# Handle potential NaNs from silent chunks
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

#--- Scaling & Encoding ---
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

le=LabelEncoder()
y_train_enc=le.fit_transform(y_train)
y_test_enc=le.transform(y_test)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    random_state=42,
    n_jobs=1
)
#xgb: slow learning rate 0.05 to avoid overfitting
xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05, 
    max_depth=4,
    subsample=0.7,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=1,
    eval_metric='mlogloss'
)
#SVM
svm = SVC(
    C=10,
    kernel='rbf',
    probability=True, #required for soft voting
    random_state=42
)
#ensemble
ensemble = VotingClassifier(
    estimators=[('rf',rf),('xgb',xgb),('svm',svm)],
    voting='soft'
)
#tuning the weights
param_grid = {
    'weights': [
        [1, 1, 1],  # Equal democracy
        [2, 1, 1],  # Trust Random Forest more
        [1, 2, 1],  # Trust XGBoost more
        [1, 1, 2],  # Trust SVM more 
        [2, 2, 1],  # Trust Trees more than SVM
        [1, 2, 2]   # Trust Math (XGB+SVM) more than RF
    ]
}
cv_splitter=GroupShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
print("\nStarting Hyperparameter Tuning...")
search=GridSearchCV(
    estimator=ensemble,
    param_grid=param_grid,
    cv=cv_splitter,
    scoring='accuracy',
    n_jobs=2,
    verbose=1
)
search.fit(X_train_scaled, y_train_enc, groups=groups_train)

print(f"\nBest Weights Found: {search.best_params_['weights']}")
print(f"Best Validation Accuracy: {search.best_score_:.2%}")

best_model = search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
acc = accuracy_score(y_test_enc, y_pred)

print(f"\nFINAL TEST ACCURACY: {acc:.2%}")
print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

# Save Confusion Matrix for Report
cm = confusion_matrix(y_test_enc, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Final Ensemble Matrix (Acc: {acc:.2%})')
plt.savefig('final_ensemble_matrix.png')