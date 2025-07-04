from utils import load_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

def train_models():
    df = load_data()
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{name} - MSE: {mse:.2f}, R2: {r2:.2f}")





def tune_models():
    df = load_data()
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    models_params = {
        'Decision Tree': (
            DecisionTreeRegressor(),
            {'max_depth': [2, 4, 6],
             'min_samples_split': [2, 5, 10],
             'min_samples_leaf': [1, 2, 4]}
        ),
        'Random Forest': (
            RandomForestRegressor(),
            {'n_estimators': [50, 100],
             'max_depth': [None, 10, 20],
             'min_samples_split': [2, 5]}
        )
    }

    for name, (model, params) in models_params.items():
        grid = GridSearchCV(model, params, cv=3, scoring='r2')
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{name} Tuning - MSE: {mse:.2f}, R2: {r2:.2f}")
        print(f"Best Params: {grid.best_params_}")

if __name__ == "__main__":
    tune_models()