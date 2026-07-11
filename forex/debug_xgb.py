from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
from src.crypto_signals.data import make_synthetic_crypto_data
from src.crypto_signals.model import SignalModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

frame = make_synthetic_crypto_data(n_rows=500)
dataset = SignalModel.build_training_dataset([frame])
X = dataset.drop(columns=['target'])
y = dataset['target']
le = LabelEncoder()
y_encoded = le.fit_transform(y)
feature_columns = SignalModel(model_path='models/crypto_signal_v1.joblib').feature_columns
X_train, X_val, y_train, y_val = train_test_split(X[feature_columns], y_encoded, test_size=0.15, shuffle=False)
print('X_train type', type(X_train), X_train.shape)
print('y_train type', type(y_train), y_train.shape)
print('y_train unique', sorted(set(y_train)))
print('classes', le.classes_)
print('dtype y_train', y_train.dtype)
print('first y_train', y_train[:10])

clf = XGBClassifier(objective='multi:softprob', use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1)
try:
    clf.fit(X_train, y_train)
    print('fit succeeded without num_class')
except Exception as e:
    print('fit failed without num_class:', type(e).__name__, e)

clf2 = XGBClassifier(objective='multi:softprob', use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1, num_class=3)
try:
    clf2.fit(X_train, y_train)
    print('fit succeeded with num_class=3')
except Exception as e:
    print('fit failed with num_class=3:', type(e).__name__, e)
