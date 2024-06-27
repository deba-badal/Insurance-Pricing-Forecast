import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Step 1: Load Data
data = pd.read_csv('C://Users//debas//Downloads//insurance.csv')

# Step 2: Data Preprocessing
# Assume the dataset has columns 'age', 'bmi', 'children', 'sex', 'smoker', 'region', 'expenses'
X = data.drop('expenses', axis=1)
y = data['expenses']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the preprocessing steps for numerical and categorical features
numeric_features = ['age', 'bmi', 'children']
categorical_features = ['sex', 'smoker', 'region']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Step 3: Model Training
# You can try both Linear Regression and XGBoost

# Linear Regression Model
linear_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

linear_model.fit(X_train, y_train)

# XGBoost Model
xgboost_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(objective='reg:squarederror'))
])

xgboost_model.fit(X_train, y_train)

# Step 4: Evaluation
# Evaluate Linear Regression Model
y_pred_linear = linear_model.predict(X_test)
print("Linear Regression Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred_linear))
print("MSE:", mean_squared_error(y_test, y_pred_linear))
print("R2 Score:", r2_score(y_test, y_pred_linear))

# Evaluate XGBoost Model
y_pred_xgboost = xgboost_model.predict(X_test)
print("\nXGBoost Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred_xgboost))
print("MSE:", mean_squared_error(y_test, y_pred_xgboost))
print("R2 Score:", r2_score(y_test, y_pred_xgboost))
