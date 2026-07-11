from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
from src.crypto_signals.data import make_synthetic_crypto_data
from src.crypto_signals.model import SignalModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

frame = make_synthetic_crypto_data(n_rows=500)
dataset = SignalModel.build_training_dataset([frame])
print('dataset shape', dataset.shape)
print('target nunique', dataset['target'].nunique())
print('target unique', dataset['target'].unique())
print('target value counts')
print(dataset['target'].value_counts())
model = SignalModel(model_path='models/crypto_signal_v1.joblib')
X = dataset.drop(columns=['target'])
y = dataset['target']
print('X shape', X.shape)
print('feature_columns count', len(model.feature_columns))
print('first feature cols', model.feature_columns[:10])
print('X columns first 10', list(X.columns[:10]))

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print('encoded classes', le.classes_)
print('encoded unique', sorted(set(y_encoded)))
print('encoded counts')
print({label: list(y_encoded).count(label) for label in set(y_encoded)})

if len(y_encoded) > 0:
    X_train, X_val, y_train, y_val = train_test_split(X[model.feature_columns], y_encoded, test_size=0.15, shuffle=False)
    print('train shapes', X_train.shape, X_val.shape)
    print('y_train unique', sorted(set(y_train)), 'y_val unique', sorted(set(y_val)))
else:
    print('y_encoded empty')
