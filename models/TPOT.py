from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier


def perform_search(X, y):
    # Adjust y
    y -= 1
    y = y.values.ravel()

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    tpot = TPOTClassifier(generations=1000, population_size=5, verbosity=10, n_jobs=-1, random_state=0,
                          periodic_checkpoint_folder='tpot_checkpoints', max_eval_time_mins=45)
    tpot.fit(X_train, y_train)
    tpot.export('tpot_export.py')
