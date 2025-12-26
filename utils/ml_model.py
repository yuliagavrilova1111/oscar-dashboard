"""ML модель для прогнозирования побед на Оскаре"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import streamlit as st
from typing import Tuple, Dict, Any

def prepare_ml_data(df: pd.DataFrame) -> pd.DataFrame:
    """Подготовка данных для ML модели"""
    ml_df = df.copy()
    sorted_df = df.sort_values(by=['name', 'year_ceremony'])
    sorted_df['nominee_prev_noms'] = sorted_df.groupby('name').cumcount()
    sorted_df['nominee_prev_wins'] = sorted_df.groupby('name')['winner'].cumsum() - sorted_df['winner']
    sorted_df['nominee_prev_win_rate'] = (100*sorted_df['nominee_prev_wins'] / sorted_df['nominee_prev_noms']).round(2)
    ml_df = sorted_df.copy()
    ml_df['years_since_last_nomination'] = ml_df.groupby('name')['year_ceremony'].diff()
    ml_df['years_since_last_nomination'] = ml_df['years_since_last_nomination'].fillna(0)
    ml_df['film_noms_this_year'] = ml_df.groupby(['year_ceremony', 'film'])['film'].transform('count')
    ml_df['category_competitiveness'] = ml_df.groupby(['year_ceremony', 'category'])['category'].transform('count')
    ml_df['cat_prev_noms'] = ml_df.groupby('category').cumcount()
    ml_df['cat_prev_wins'] = ml_df.groupby('category')['winner'].cumsum() - ml_df['winner']
    ml_df['cat_prev_win_rate'] = (100*ml_df['cat_prev_wins']/ml_df['cat_prev_noms']).round(2)
    ml_df['nominee_prev_noms'] = ml_df['nominee_prev_noms'].fillna(0)
    ml_df['nominee_prev_wins'] = ml_df['nominee_prev_wins'].fillna(0)
    ml_df['nominee_prev_win_rate'] = ml_df['nominee_prev_win_rate'].fillna(0)

    ml_df = ml_df[(ml_df['film'].notna()) & (ml_df['name'].notna())]
    ml_df['cat_prev_win_rate'] = ml_df['cat_prev_win_rate'].fillna(0)

    from sklearn.preprocessing import LabelEncoder

    category_mean = ml_df.groupby('category')['winner'].transform('mean')
    ml_df['category_mean'] = category_mean
    le = LabelEncoder()
    ml_df['le_decade'] = le.fit_transform(ml_df['decade'])
    era_dummies = pd.get_dummies(ml_df['era'])
    ml_df = pd.concat([ml_df, era_dummies], axis=1)
    ml_df['had_previous_noms'] = (ml_df['nominee_prev_noms']>0).astype(int)
    ml_df['had_previous_wins'] = (ml_df['nominee_prev_wins']>0).astype(int)
    ml_df['lot_of_noms'] = (ml_df['film_noms_this_year']>7).astype(int)
    ml_df['film_experience'] = (ml_df['lot_of_noms'] & ml_df['had_previous_wins']).astype(int)
    ml_df['few_years_since_last_nom'] = (ml_df['years_since_last_nomination']<5)
    ml_df['film_power'] = pd.cut(ml_df['film_noms_this_year'], bins=[-1, 0, 5, 9, 14], labels=['weak', 'med', 'strong', 'super'])
    ml_df['film_and_cat'] = ml_df['film_noms_this_year'] * ml_df['category_competitiveness']
    ml_df['cat_total'] = ml_df['category_competitiveness'] * ml_df['is_major_cat']
    ml_df['comeback'] = (ml_df['years_since_last_nomination']>5).astype(int)
    film_power_dummies = pd.get_dummies(ml_df['film_power'])
    ml_df = pd.concat([ml_df, film_power_dummies], axis=1)

    return ml_df


def get_features() -> list:
    """Список используемых признаков"""
    features = [
        'nominee_prev_noms', 'nominee_prev_win_rate',
        'years_since_last_nomination', 'film_noms_this_year',
        'category_competitiveness', 'Золотой век',
        'Новый Голливуд', 'Эпоха блокбастеров', 'Современный период',
        'had_previous_noms', 'had_previous_wins', 'lot_of_noms',
        'film_experience', 'film_and_cat', 'cat_total',
        'comeback', 'med', 'strong', 'super'
    ]
    return features

def train_random_forest(
    ml_df: pd.DataFrame,
    test_size: float = 0.3,
    use_smote: bool = True,
    use_gridsearch: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    """Обучение модели Random Forest"""

    features = get_features()
    X = ml_df[features]
    y = ml_df['winner']
    
    # Преобразование булевых колонок
    bool_cols = X.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        X[col] = X[col].astype(int)
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Балансировка классов
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        st.info('SMOTE применен.')
    else:
        X_train_bal, y_train_bal = X_train, y_train
    
    # Обучение модели
    if use_gridsearch:
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 3, 5],
            'class_weight': ['balanced']
        }
        rf = RandomForestClassifier(random_state=42)
        gridRF = GridSearchCV(rf, rf_params, cv=5, scoring='roc_auc')
        gridRF.fit(X_train_bal, y_train_bal)
        best_rf = gridRF.best_estimator_
        best_params = gridRF.best_params_
        best_score = gridRF.best_score_
    else:
        best_rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
        best_rf.fit(X_train_bal, y_train_bal)
        best_params = best_rf.get_params()
        best_score = None
    
    # Предсказания
    predictions = best_rf.predict(X_test)
    pred_proba = best_rf.predict_proba(X_test)[:, 1]
    
    # Метрики
    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, pred_proba)
    report = classification_report(y_test, predictions, output_dict=True)
    cm = confusion_matrix(y_test, predictions)
    
    # Важность признаков
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': best_rf.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    results = {
        'model': best_rf,
        'features': features,
        'X_test': X_test,
        'y_test': y_test,
        'predictions': predictions,
        'pred_proba': pred_proba,
        'ml_df': ml_df,
        'feature_importance': feature_importance,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'classification_report': report,
        'confusion_matrix': cm,
        'best_params': best_params,
        'best_score': best_score
    }
    
    return best_rf, results