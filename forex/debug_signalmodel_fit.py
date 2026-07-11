from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
from src.crypto_signals.data import make_synthetic_crypto_data
from src.crypto_signals.model import SignalModel

frame = make_synthetic_crypto_data(n_rows=500)
dataset = SignalModel.build_training_dataset([frame])
model = SignalModel(model_path='models/crypto_signal_v1.joblib')
X = dataset.drop(columns=['target'])
y = dataset['target']
print('X shape', X.shape)
print('y shape', y.shape)
print('y unique', y.unique())
try:
    metrics = model.fit(X, y)
    print('metrics', metrics)
except Exception as e:
    print('ERROR', type(e).__name__, e)
    import traceback
    traceback.print_exc()
