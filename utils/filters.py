"""Модуль для создания фильтров в сайдбаре"""
import streamlit as st
import pandas as pd
from typing import Dict, Any

def create_sidebar_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """Создание всех фильтров в сайдбаре"""
    
    st.sidebar.title('Фильтры')

    min_year = int(df['year_ceremony'].min())
    max_year = int(df['year_ceremony'].max())
    
    # Инициализация состояния сброса
    if 'reset_filters' not in st.session_state:
        st.session_state.reset_filters = False
    
    # Если нужно сбросить фильтры
    if st.session_state.reset_filters:
        # Удаляем все ключи фильтров
        for key in ['year_range', 'category', 'film', 'actor', 'director', 'studio']:
            if key in st.session_state:
                del st.session_state[key]
        # Сбрасываем флаг
        st.session_state.reset_filters = False
        # Используем st.rerun()
        st.rerun()
    
    # Инициализация значений по умолчанию
    # Делаем это ПОСЛЕ возможного сброса
    if 'year_range' not in st.session_state:
        st.session_state.year_range = (min_year, max_year)
    
    if 'category' not in st.session_state:
        st.session_state.category = 'Все категории'
    
    if 'film' not in st.session_state:
        st.session_state.film = ''
    
    if 'actor' not in st.session_state:
        st.session_state.actor = ''
    
    if 'director' not in st.session_state:
        st.session_state.director = ''
    
    if 'studio' not in st.session_state:
        st.session_state.studio = 'Все студии'

    # Создание виджетов с использованием ключей
    year_range = st.sidebar.slider(
        label='Годы',
        min_value=min_year,
        max_value=max_year,
        help='Выберите диапазон лет для анализа',
        key='year_range'
    )

    category_options = ['Все категории'] + sorted(df['category'].unique().tolist())
    
    categories = st.sidebar.selectbox(
        label='Категории',
        options=category_options,
        help='Выберите категорию',
        key='category'
    )

    films = st.sidebar.text_input(
        label='Фильмы',
        placeholder='Название',
        help='Введите название фильма',
        key='film'
    )

    actors = st.sidebar.text_input(
        label='Актеры',
        placeholder='Имя',
        help='Введите имя актера/актрисы',
        key='actor'
    )

    directors = st.sidebar.text_input(
        label='Режиссёры',
        placeholder='Имя',
        help='Введите имя режиссёра',
        key='director'
    )

    studios_options = ['Все студии'] + sorted(df[df['is_studio']]['name'].unique().tolist())
    
    studios = st.sidebar.selectbox(
        label='Студии',
        options=studios_options,
        help='Выберите название студии',
        key='studio'
    )

    if st.sidebar.button('Сбросить фильтры', type='secondary', use_container_width=True):
        st.session_state.reset_filters = True
        st.rerun()

    return {
        'year_range': st.session_state.year_range,
        'categories': st.session_state.category,
        'films': st.session_state.film,
        'actors': st.session_state.actor,
        'directors': st.session_state.director,
        'studios': st.session_state.studio
    }