import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob) if y_prob is not None else None
    }

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = joblib.load('data_processed/train_test_data.pkl')

    # Base models for voting
    models = [
        ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
        ('CatBoost', CatBoostClassifier(verbose=0, random_state=42)),
        ('LightGBM', LGBMClassifier(random_state=42))
    ]

    voting = VotingClassifier(estimators=models, voting='soft')
    v_results = train_and_evaluate(voting, X_train, X_test, y_train, y_test)
    pd.DataFrame([v_results], index=['Ensemble']).to_csv('results/ensemble_results.csv')
    print('Ensemble results saved to results/ensemble_results.csv')

    # Stacking
    base_estimators = [
        ('RF', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('SVM', SVC(probability=True, random_state=42)),
        ('KNN', KNeighborsClassifier(n_neighbors=5)),
        ('XGB', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
        ('Cat', CatBoostClassifier(verbose=0, random_state=42)),
        ('LGB', LGBMClassifier(random_state=42))
    ]
    stack = StackingClassifier(estimators=base_estimators, final_estimator=LogisticRegression())
    s_results = train_and_evaluate(stack, X_train, X_test, y_train, y_test)
    pd.DataFrame([s_results], index=['Stacking']).to_csv('results/stacking_results.csv')
    print('Stacking results saved to results/stacking_results.csv')
