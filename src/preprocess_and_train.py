import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path):
    df = pd.read_csv(path)
    if 'session_id' in df.columns:
        df.drop(columns=['session_id'], inplace=True)

    numerical_features = ['network_packet_size', 'login_attempts', 'session_duration', 'ip_reputation_score', 'failed_logins']
    categorical_features = ['protocol_type', 'encryption_used', 'browser_type']
    target = 'attack_detected'

    # Impute missing values
    df[numerical_features] = df[numerical_features].fillna(df[numerical_features].median())
    df[categorical_features] = df[categorical_features].fillna('Unknown')

    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize
    scaler = StandardScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    path = 'dataset/cybersecurity_intrusion_data.csv'
    X_train, X_test, y_train, y_test = load_and_preprocess(path)

    # Save processed data
    joblib.dump((X_train, X_test, y_train, y_test), 'data_processed/train_test_data.pkl')
    print('Preprocessing complete. Data saved to data_processed/train_test_data.pkl')
