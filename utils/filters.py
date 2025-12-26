"""Модуль для создания фильтров в сайдбаре"""
import streamlit as st
import pandas as pd
from typing import Dict, Any

def create_sidebar_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """Создание всех фильтров в сайдбаре"""

    st.sidebar.title('Фильтры')
    min_year = df['year_ceremony'].min()
    max_year = df['year_ceremony'].max()
    year_range = st.sidebar.slider(
        label='Годы',
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        help='Выберите диапазон лет для анализа'
    )

    category_options = ['Все категории'] + sorted(df['category'].unique().tolist())
    categories = st.sidebar.selectbox(
        label='Категории',
        options=category_options,
        help='Выберите категорию'
    )

    films = st.sidebar.text_input(
        label='Фильмы',
        placeholder='Название',
        help='Введите название фильма'
    )

    actors = st.sidebar.text_input(
        label='Актеры',
        placeholder='Имя',
        help='Введите имя актера/актрисы'
    )

    directors = st.sidebar.text_input(
        label='Режиссёры',
        placeholder='Имя',
        help='Введите имя режиссёра'
    )

    studios = st.sidebar.selectbox(
        label='Студии',
        options=['Все студии'] + sorted(df[df['is_studio']]['name'].unique().tolist()),
        help='Выберите название студии'
    )

    if st.sidebar.button('Сбросить фильтры'):
        st.rerun()
    return {
        'year_range': year_range,
        'categories': categories,
        'films': films,
        'actors': actors,
        'directors': directors,
        'studios': studios
    }