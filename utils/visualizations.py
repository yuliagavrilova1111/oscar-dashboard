"""Модуль для создания графиков"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
from typing import Optional, List, Dict, Any

# ВКЛАДКА 1: Основные тренды

def plot_nominations_by_year_with_extremes(df: pd.DataFrame) -> go.Figure:
    """График номинаций по годам с выделением экстремумов"""
    noms_by_year = df.groupby('year_ceremony').size().reset_index(name='noms')
    top5_min = noms_by_year.nsmallest(5, 'noms')
    top5_max = noms_by_year.nlargest(5, 'noms')
    
    fig = px.line(noms_by_year, x='year_ceremony', y='noms', 
                  labels={'year_ceremony': 'Год', 'noms': 'Количество номинаций'},
                  title='Количество номинаций по годам')
    
    fig.add_trace(go.Scatter(
        x=top5_min['year_ceremony'], y=top5_min['noms'],
        mode='markers', name='Минимум (топ-5)',
        marker=dict(size=8, color='red', symbol='circle')
    ))
    
    fig.add_trace(go.Scatter(
        x=top5_max['year_ceremony'], y=top5_max['noms'],
        mode='markers', name='Максимум (топ-5)',
        marker=dict(size=8, color='green', symbol='circle')
    ))
    
    fig.update_layout(showlegend=True)
    return fig

def plot_wins_by_year_with_extremes(df: pd.DataFrame) -> go.Figure:
    """График побед по годам с выделением экстремумов"""
    wins_by_year = df.groupby('year_ceremony')['winner'].sum().reset_index(name='wins')
    top5_min = wins_by_year.nsmallest(5, 'wins')
    top5_max = wins_by_year.nlargest(5, 'wins')
    
    fig = px.line(wins_by_year, x='year_ceremony', y='wins',
                  labels={'year_ceremony': 'Год', 'wins': 'Количество побед'},
                  title='Количество побед по годам')
    
    fig.add_trace(go.Scatter(
        x=top5_min['year_ceremony'], y=top5_min['wins'],
        mode='markers', name='Минимум (топ-5)',
        marker=dict(size=8, color='red', symbol='circle')
    ))
    
    fig.add_trace(go.Scatter(
        x=top5_max['year_ceremony'], y=top5_max['wins'],
        mode='markers', name='Максимум (топ-5)',
        marker=dict(size=8, color='green', symbol='circle')
    ))
    
    fig.update_layout(showlegend=True)
    return fig

def plot_win_percentage_by_year_with_extremes(df: pd.DataFrame) -> go.Figure:
    """График процента побед по годам с выделением экстремумов"""
    noms_by_year = df.groupby('year_ceremony').size().reset_index(name='noms')
    wins_by_year = df.groupby('year_ceremony')['winner'].sum().reset_index(name='wins')
    
    percent_df = pd.merge(noms_by_year, wins_by_year, on='year_ceremony')
    percent_df['win_percentage'] = percent_df['wins'] / percent_df['noms'] * 100
    top5_min = percent_df.nsmallest(5, 'win_percentage')
    top5_max = percent_df.nlargest(5, 'win_percentage')
    
    fig = px.line(percent_df, x='year_ceremony', y='win_percentage',
                  labels={'year_ceremony': 'Год', 'win_percentage': 'Процент побед (%)'},
                  title='Процент побед по годам')
    
    fig.add_trace(go.Scatter(
        x=top5_min['year_ceremony'], y=top5_min['win_percentage'],
        mode='markers', name='Минимум (топ-5)',
        marker=dict(size=8, color='red', symbol='circle')
    ))
    
    fig.add_trace(go.Scatter(
        x=top5_max['year_ceremony'], y=top5_max['win_percentage'],
        mode='markers', name='Максимум (топ-5)',
        marker=dict(size=8, color='green', symbol='circle')
    ))
    
    fig.update_layout(showlegend=True)
    return fig

# ВКЛАДКА 2: Сравнение категорий

def plot_specific_category_nominations(df: pd.DataFrame, category: str) -> go.Figure:
    """Номинации по годам для конкретной категории"""
    category_data = df[df['category'] == category]
    cat_noms = category_data.groupby('year_ceremony').size().reset_index(name='noms')
    
    fig = px.line(cat_noms, x='year_ceremony', y='noms', 
                  labels={'year_ceremony': 'Год', 'noms': 'Количество номинаций'},
                  title=f'Количество номинаций по годам в категории "{category}"')
    return fig

def plot_specific_category_wins(df: pd.DataFrame, category: str) -> go.Figure:
    """Победы по годам для конкретной категории"""
    category_data = df[df['category'] == category]
    cat_wins = category_data.groupby('year_ceremony')['winner'].sum().reset_index(name='wins')
    
    fig = px.line(cat_wins, x='year_ceremony', y='wins',
                  labels={'year_ceremony': 'Год', 'wins': 'Количество побед'},
                  title=f'Количество побед по годам в категории "{category}"')
    return fig

def plot_category_heatmap(df: pd.DataFrame, selected_categories: List[str]) -> go.Figure:
    """Тепловая карта по категориям"""
    cat_data = df[df['category'].isin(selected_categories)]
    heatmap_data = cat_data.groupby(['year_ceremony', 'category']).size().unstack(fill_value=0)
    
    fig = px.imshow(heatmap_data, 
                    labels=dict(x='Год', y='Категория', color='Количество'),
                    aspect='auto')
    return fig

def plot_category_stability(df: pd.DataFrame) -> go.Figure:
    """Наличие категорий по десятилетиям"""
    stability_table = df.groupby(['decade', 'category']).size().unstack(fill_value=0)
    binary_table = (stability_table > 0).astype(int)
    
    fig = px.imshow(binary_table,
                    labels=dict(x='Категория', y='Десятилетие'),
                    color_continuous_scale=[[0, 'gray'], [1, 'blue']])
    
    fig.update_coloraxes(
        colorbar=dict(
            title='Наличие категории',
            tickvals=[0.1, 0.9],
            ticktext=['Отсутствует', 'Присутствует'],
            orientation='v',
            len=1.3
        )
    )
    
    return fig

def plot_competitive_categories(df: pd.DataFrame) -> go.Figure:
    """Самые конкурентные категории"""
    cat_difficulty = df.groupby('category').agg(
        nominations=('winner', 'count'),
        wins=('winner', 'sum')
    ).reset_index().sort_values(by='nominations', ascending=False)
    
    cat_difficulty['win_rate'] = (100 * cat_difficulty['wins'] / cat_difficulty['nominations']).round(2)
    most_competitive = cat_difficulty[cat_difficulty['nominations'] > 10].sort_values(by='win_rate').head(15)
    
    fig = px.bar(most_competitive, x='win_rate', y='category',
                 labels={'category': 'Категория', 'win_rate': 'Процент побед (%)'},
                 color='win_rate', color_continuous_scale='Viridis')
    
    return fig

def plot_category_types_pie(df: pd.DataFrame) -> go.Figure:
    """Распределение по типам категорий (круговая диаграмма)"""
    types_count = df.groupby('category_type')['category'].nunique().reset_index(name='count')
    fig = px.pie(types_count, values='count', names='category_type',
                 title='Распределение категорий по типам')
    return fig

def plot_category_types_heatmap(df: pd.DataFrame) -> go.Figure:
    """Доля номинаций по типам категорий"""
    count_by_decades = df.groupby('decade').size()
    nominations_by_type = df.groupby(['decade', 'category_type']).size()
    percent_table = (nominations_by_type / count_by_decades * 100).unstack(fill_value=0)
    
    fig = px.imshow(percent_table, 
                    labels=dict(x='Тип категории', y='Десятилетие'), 
                    aspect='auto',
                    title='Доля номинаций по типам категорий (%)')
    
    fig.update_coloraxes(
        colorbar=dict(title='Процент номинаций')
    )
    
    return fig

# ВКЛАДКА 3: Анализ по периодам

def plot_period_comparison(df: pd.DataFrame, period_type: str) -> go.Figure:
    """Сравнение по периодам (десятилетиям или эпохам)"""
    if period_type == 'Десятилетиям':
        period_df = df.groupby('decade').agg(
            noms=('winner', 'count'),
            wins=('winner', 'sum')
        ).reset_index()
        period_col = 'decade'
        title_suffix = 'десятилетиям'
    else:
        period_df = df.groupby('era').agg(
            noms=('winner', 'count'),
            wins=('winner', 'sum')
        ).reset_index()
        period_col = 'era'
        title_suffix = 'историческим периодам'
    
    period_df = period_df.rename(columns={'noms': 'Номинации', 'wins': 'Победы'})
    
    fig = px.bar(period_df, x=period_col, y=['Номинации', 'Победы'], barmode='group',
                 title=f'Номинации и победы по {title_suffix}',
                 labels={period_col: period_col, 'value': 'Количество', 'variable': 'Тип'})
    
    return fig

# ВКЛАДКА 4: Актеры и режиссеры

def plot_top_actors_by_wins(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Топ актеров по победам"""
    actors_df = df[df['actor_actress'] == True]
    actors_stats = actors_df.groupby('name').agg(
        noms=('winner', 'count'),
        wins=('winner', 'sum')
    ).reset_index()
    
    top_df = actors_stats.sort_values(by='wins', ascending=False).head(top_n)
    
    fig = px.bar(top_df, x='wins', y='name',
                 labels={'name': 'Актер/Актриса', 'wins': 'Победы'},
                 color='wins', color_continuous_scale='Teal',
                 title=f'Топ-{top_n} актеров по количеству побед на Оскаре')
    
    return fig

def plot_actors_without_wins(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Самые номинируемые актеры без побед"""
    actors_df = df[df['actor_actress'] == True]
    actors_stats = actors_df.groupby('name').agg(
        noms=('winner', 'count'),
        wins=('winner', 'sum')
    ).reset_index()
    
    eternal_df = actors_stats[actors_stats['wins'] == 0].sort_values(by='noms', ascending=False).head(top_n)
    
    fig = px.bar(eternal_df, x='noms', y='name',
                 labels={'name': 'Актер/Актриса', 'noms': 'Номинации'},
                 color='noms', color_continuous_scale='Teal',
                 title=f'Самые номинируемые актеры без побед (топ-{top_n})')
    
    return fig

def plot_gender_analysis(df: pd.DataFrame) -> go.Figure:
    """Распределение актерских номинаций по полу"""
    actors_df = df[df['actor_actress'] == True]
    gender_df = actors_df.groupby(['decade', 'gender']).size().reset_index(name='count')
    
    fig = px.bar(gender_df, x='decade', y='count', color='gender', barmode='group',
                 color_discrete_map={'male': 'steelblue', 'female': 'lightpink'},
                 labels={'decade': 'Десятилетие', 'count': 'Количество номинаций', 'gender': 'Пол'},
                 title='Распределение актерских номинаций по полу и десятилетиям')
    
    return fig

def plot_top_directors_by_wins(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Топ режиссеров по победам"""
    directors_df = df[df['is_director'] == True]
    directors_stats = directors_df.groupby('name').agg(
        noms=('winner', 'count'),
        wins=('winner', 'sum')
    ).reset_index()
    
    top_df = directors_stats.sort_values(by='wins', ascending=False).head(top_n)
    
    fig = px.bar(top_df, x='wins', y='name',
                 labels={'name': 'Режиссер', 'wins': 'Количество побед'},
                 color='wins', color_continuous_scale='Cividis',
                 title=f'Топ-{top_n} режиссеров по количеству побед на Оскаре')
    
    return fig

def plot_directors_without_wins(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Топ режиссеров по номинациям без побед"""
    directors_df = df[df['is_director'] == True]
    directors_stats = directors_df.groupby('name').agg(
        noms=('winner', 'count'),
        wins=('winner', 'sum')
    ).reset_index()
    
    top_df = directors_stats[directors_stats['wins'] == 0].sort_values(by='noms', ascending=False).head(top_n)
    
    fig = px.bar(top_df, x='noms', y='name',
                 labels={'name': 'Режиссер', 'noms': 'Количество номинаций'},
                 color='noms', color_continuous_scale='Cividis',
                 title=f'Топ-{top_n} режиссеров по количеству номинаций без побед')
    
    return fig

# ВКЛАДКА 5: Студии

def plot_studio_treemap_golden_age(df: pd.DataFrame) -> go.Figure:
    """Treemap для студий Золотого века Голливуда"""
    studios_df = df[df['is_studio'] == True]
    early_era = studios_df[studios_df['year_ceremony'] < 1950]
    
    studio_stats = early_era.groupby('name').agg(
        noms=('winner', 'count'),
        wins=('winner', 'sum')
    ).reset_index().sort_values(by='wins', ascending=False)
    
    studio_stats['percent_of_wins'] = studio_stats['wins'] / studio_stats['noms'] * 100
    
    fig = px.treemap(
        studio_stats,
        path=['name'],
        values='wins',
        title='Распределение побед среди студий Золотого века Голливуда',
        color_continuous_scale='Plasma',
        hover_data={
            'name': True,
            'noms': True,
            'wins': True,
            'percent_of_wins': ':.1f'
        }
    )
    
    fig.update_traces(
        textinfo='label+value+percent parent',
        textfont_size=14
    )
    
    return fig

def plot_studio_treemap_modern(df: pd.DataFrame) -> go.Figure:
    """Treemap для студий современного периода"""
    studios_df = df[df['is_studio'] == True]
    modern_era = studios_df[studios_df['year_ceremony'] >= 1950]
    
    studio_stats = modern_era.groupby('name').agg(
        noms=('winner', 'count'),
        wins=('winner', 'sum')
    ).reset_index().sort_values(by='wins', ascending=False)
    
    studio_stats['percent_of_wins'] = studio_stats['wins'] / studio_stats['noms'] * 100
    
    fig = px.treemap(
        studio_stats,
        path=['name'],
        values='wins',
        title='Распределение побед среди студий современного периода',
        color_continuous_scale='Plasma',
        hover_data={
            'name': True,
            'noms': True,
            'wins': True,
            'percent_of_wins': ':.1f'
        }
    )
    
    fig.update_traces(
        textinfo='label+value+percent parent',
        textfont_size=14
    )
    
    return fig

# ВКЛАДКА 6: Прогнозный анализ

def plot_confusion_matrix(cm, labels=['НЕТ', 'ДА']) -> go.Figure:
    """Матрица ошибок"""
    fig = px.imshow(
        cm, text_auto=True,
        labels=dict(x='Предсказанные', y='Реальные', color='Количество'),
        x=labels, y=labels,
        aspect='auto',
        color_continuous_scale='Blues',
        title='Матрица ошибок'
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig

def plot_feature_importance(importance_df: pd.DataFrame) -> go.Figure:
    """Важность признаков"""
    fig = px.bar(
        importance_df,
        x='importance',
        y='feature',
        color='importance',
        color_continuous_scale='Viridis',
        labels={'feature': 'Признак', 'importance': 'Важность'},
        title='Важность признаков в модели Random Forest'
    )
    return fig

# УНИВЕРСАЛЬНЫЕ ФУНКЦИИ

def create_simple_line_chart(df: pd.DataFrame, x_col: str, y_col: str, 
                            title: str = None, x_label: str = None, 
                            y_label: str = None) -> go.Figure:
    """Создание простого линейного графика"""
    if title is None:
        title = f'{y_col} по {x_col}'
    if x_label is None:
        x_label = x_col
    if y_label is None:
        y_label = y_col
    
    fig = px.line(df, x=x_col, y=y_col, 
                  labels={x_col: x_label, y_col: y_label},
                  title=title)
    return fig

def create_simple_bar_chart(df: pd.DataFrame, x_col: str, y_col: str,
                           title: str = None, color_col: str = None) -> go.Figure:
    """Создание простой столбчатой диаграммы"""
    if title is None:
        title = f'{y_col} по {x_col}'
    
    if color_col:
        fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=title)
    else:
        fig = px.bar(df, x=x_col, y=y_col, title=title)
    
    return fig

def create_heatmap(df: pd.DataFrame, title: str = None, 
                  color_scale: str = 'Viridis') -> go.Figure:
    """Создание тепловой карты"""
    if title is None:
        title = 'Тепловая карта'
    
    fig = px.imshow(df, title=title, color_continuous_scale=color_scale)
    return fig