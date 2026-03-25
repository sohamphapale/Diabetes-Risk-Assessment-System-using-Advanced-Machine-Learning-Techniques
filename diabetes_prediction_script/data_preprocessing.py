import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# load dataset ✅


def load_diabetes_health_data(db_path: str):
    return pd.read_csv("./data/diabetes-data.csv")


def remove_redundant_columns(df):
    return df.drop(['Pregnancies', 'SkinThickness'], axis=1)

# replace na values ✅


def replace_na_values(df: pd.DataFrame):
    # replace 0 (0 is also nan value) with nan
    cols = ['Glucose', 'BloodPressure', 'Insulin', 'BMI']
    df[cols] = df[cols].replace(0, np.nan)
    # replace na values with
    # 1.  mean() -> Normal distribution
    # 2.  median() for -> Skewed / Outliers present
    df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean())
    df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
    df['Insulin'] = df['Insulin'].fillna(df['Insulin'].median())
    df['BMI'] = df['BMI'].fillna(df['BMI'].median())
    return df


# create bins you give df and it return df with bins
def transform_numerical_to_categorical(df: pd.DataFrame):
    # 1: Glucose
    df['Glucose'] = pd.cut(
        df['Glucose'],
        bins=[0, 70, 99, 125, float("inf")],
        labels=['Low-glucose', "Normal", "Pre-diabetes", "Diabetes"]
    )
    # 2: BloodPressure
    df['BloodPressure'] = pd.cut(
        df['BloodPressure'],
        bins=[0, 40, 60, 80, 90, float("inf")],
        labels=['Very-Low', "Low", "Normal", "Elevated", "High"]
    )
    # 3: Insulin
    df['Insulin'] = pd.cut(
        df['Insulin'],
        bins=[0, 50, 150, 300,  float("inf")],
        labels=['Low', "Normal", "High", "Very High"]
    )
    # 4: Insulin
    df['BMI'] = pd.cut(
        df['BMI'],
        bins=[0, 18.5, 25, 30, float("inf")],
        labels=['Underweight', "Normal", "Overweight", "Obese"]
    )
    # 5: DiabetesPedigreeFunction
    df['DiabetesPedigreeFunction'] = pd.cut(
        df['DiabetesPedigreeFunction'],
        bins=[0, .3, .8,   float("inf")],
        labels=['Low', "Medium", "High"]
    )
    # 6: Age
    df['Age'] = pd.cut(
        df['Age'],
        bins=[0, 25, 35, 50, 65, float("inf")],
        labels=["Very-Young", "Young", "Middle", "Senior", "Old"]
    )
    return df


# applay onehot incoding
def encode_categorical_features(df: pd.DataFrame):
    cat_cols = ['Glucose', 'BloodPressure', 'Insulin', 'BMI',
                'DiabetesPedigreeFunction', 'Age']
    return pd.get_dummies(df, columns=cat_cols)


# prepare features
def prepare_feature(df: pd.DataFrame):
    X = df.drop(['Outcome'], axis=1)
    y = df['Outcome']
    return X, y


# train test split dataset
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=0.2, random_state=42)
