# import all models
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, make_scorer
from sklearn.model_selection import RandomizedSearchCV
# write a function to train the model


def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 5, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,              # number of combinations to try
        cv=5,                   # 5-fold cross-validation
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    return random_search


# write a function to get error or best fit model at return
def evaluate_classifier(model, X_test, y_test, model_name):
    print(f"Running model: {model_name}")

    preds = model.predict(X_test)

    print(f"\n{model_name} Performance:")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))
