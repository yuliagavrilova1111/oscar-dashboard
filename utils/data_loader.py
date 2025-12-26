"""Модуль загрузки и фильтрации данных"""
import pandas as pd
import streamlit as st
from typing import Optional, Tuple

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/oscars_for_dashboard.csv')
        return df
    except FileNotFoundError:
        st.error('Файл не найден')
        return pd.DataFrame()
    except Exception as e:
        st.error('Ошибка загрузки данных')
        return pd.DataFrame()

def apply_filters(
    df: pd.DataFrame,
    year_range: Optional[Tuple[int, int]] = None,
    category: Optional[str] = None,
    film: Optional[str] = None,
    actor: Optional[str] = None,
    director: Optional[str] = None,
    studio: Optional[str] = None
) -> pd.DataFrame:
    """Применение фильтров к данным"""
    df_filtered = df.copy()
    
    if year_range:
        df_filtered = df_filtered[
            (df_filtered['year_ceremony'] >= year_range[0]) & 
            (df_filtered['year_ceremony'] <= year_range[1])
        ]
    
    if category and category != 'Все категории':
        df_filtered = df_filtered[df_filtered['category'] == category]
    
    if film and film.strip():
        df_filtered = df_filtered[
            df_filtered['film'].str.contains(film, case=False, na=False)
        ]
    
    if actor and actor.strip():
        df_filtered = df_filtered[
            (df_filtered['actor_actress'] == True) & 
            (df_filtered['name'].str.contains(actor, case=False, na=False))
        ]
    
    if director and director.strip():
        df_filtered = df_filtered[
            (df_filtered['is_director'] == True) & 
            (df_filtered['name'].str.contains(director, case=False, na=False))
        ]
    
    if studio and studio != 'Все студии':
        df_filtered = df_filtered[df_filtered['name'] == studio]
    
    return df_filtered


def get_basic_metrics(df: pd.DataFrame) -> dict:
    """Получение метрик"""
    count_noms = len(df)
    count_wins = df['winner'].sum()
    percent_wins = count_wins / count_noms * 100 if count_noms > 0 else 0
    unique_movies = df['film'].nunique()
    unique_categories = df['category'].nunique()
    
    return {
        'count_noms': count_noms,
        'count_wins': count_wins,
        'percent_wins': percent_wins,
        'unique_movies': unique_movies,
        'unique_categories': unique_categories
    }
