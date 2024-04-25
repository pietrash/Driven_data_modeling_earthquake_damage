from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

from data.data_preparation import get_train_data

# Load data
X, y, _ = get_train_data(encoded_x=True)

# Adjust y
y -= 1
y = y.values.ravel()

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

tpot = TPOTClassifier(generations=50, population_size=5, verbosity=10, n_jobs=-1, random_state=0,
                      periodic_checkpoint_folder='tpot_checkpoints', max_eval_time_mins=45)
tpot.fit(X_train, y_train)
tpot.export('tpot_export.py', 'tpot_data_output')
