import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title='Oscar Awards Dashboard',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded'
)

@st.cache_data
def load_data():
    df = pd.read_csv('data/oscars_for_dashboard.csv')
    return df
df = load_data()

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

st.title("Анализ премии Оскар")

df_filtered = df.copy()
if year_range:
    df_filtered = df_filtered[(df_filtered['year_ceremony']>=year_range[0]) & (df_filtered['year_ceremony']<=year_range[1])]
if categories and categories != 'Все категории':
    df_filtered = df_filtered[df_filtered['category']==categories]
if films and films.strip():
    df_filtered = df_filtered[df_filtered['film'].str.contains(films, case=False, na=False)]
if actors and actors.strip():
    df_filtered = df_filtered[(df_filtered['actor_actress']==True) & (df_filtered['name'].str.contains(actors, case=False, na=False))]
if directors and directors.strip():
    df_filtered = df_filtered[(df_filtered['is_director']==True) & (df_filtered['name'].str.contains(directors, case=False, na=False))]
if studios and studios != 'Все студии':
    df_filtered = df_filtered[df_filtered['name']==studios]

if st.button('Экспортировать отфильтрованные данные'):
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label='Скачать CSV',
        data=csv,
        file_name='oscar_filtered.csv',
        mime='text/csv'
    )

if len(df_filtered) == 0:
    st.error('По выбранным фильтрам данных не найдено. Пожалуйста, измените критерии поиска.')
    st.stop()

st.success(f'Найдено записей: {len(df_filtered)}')

with st.expander('Data'):
    st.write(f'**Всего строк: {len(df_filtered)}**')
    st.write('**X**')
    X = df_filtered.drop('winner', axis=1)
    X
    st.write('**y**')
    y = df_filtered['winner']
    y

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    'Основные тренды',
    'Сравнение категорий',
    'Анализ по периодам',
    'Анализ актеров и режиссеров',
    'Анализ студий',
    'Прогнозный анализ'
])
with tab1:
    st.subheader('Ключевые показатели')
    count_noms = len(df_filtered)
    count_wins = df_filtered['winner'].sum()
    percent_wins = count_wins / count_noms * 100 if count_noms > 0 else 0
    unique_movies = df_filtered['film'].nunique()
    unique_categories = df_filtered['category'].nunique()
    metric1, metric2, metric3, metric4, metric5 = st.columns(5)
    with metric1:
        st.metric(
            label='Номинаций',
            value=count_noms,
            help='Всего номинаций в выбранном периоде'
            )
    
    with metric2:
        st.metric(
            label='Побед',
            value=count_wins,
            help='Всего побед в выбранном периоде'
        )
    
    with metric3:
        st.metric(
            label='Процент побед',
            value=f'{percent_wins:.1f}%',
            help='Процент побед от общего числа номинаций'
        )
    
    with metric4:
        st.metric(
            label='Фильмов',
            value=unique_movies,
            help='Уникальных фильмов в выбранном периоде'
        )
    
    with metric5:
        st.metric(
            label='Категорий',
            value=unique_categories,
            help='Уникальных категорий в выбранном периоде'
        )
    
    st.markdown('---')
    st.subheader('Динамика по годам')

    col1, col2, col3 = st.columns(3)
    with col1:
        noms_by_year = df_filtered.groupby('year_ceremony').size().reset_index(name='noms')
        top5_min_noms_by_year = noms_by_year.nsmallest(5, 'noms')
        top5_max_noms_by_year = noms_by_year.nlargest(5, 'noms')
        fig1 = px.line(noms_by_year, x='year_ceremony', y='noms', labels={'year_ceremony': 'Год', 'noms': 'Количество номинаций'},
                       title='Количество номинаций по годам')
        fig1.add_traces(go.Scatter(
            x=top5_min_noms_by_year['year_ceremony'],
            y=top5_min_noms_by_year['noms'],
            mode='markers',
            name='Минимум (топ-5)',
            marker=dict(size=5, color='red')
        ))
        fig1.add_traces(go.Scatter(
            x=top5_max_noms_by_year['year_ceremony'],
            y=top5_max_noms_by_year['noms'],
            mode='markers',
            name='Максимум (топ-5)',
            marker=dict(size=5, color='green')
        ))
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        wins_by_year = df_filtered.groupby('year_ceremony')['winner'].sum().reset_index(name='wins')
        top5_min_wins = wins_by_year.nsmallest(5, 'wins')
        top5_max_wins = wins_by_year.nlargest(5, 'wins')
        fig2 = px.line(wins_by_year, x='year_ceremony', y='wins', labels={'year_ceremony': 'Год', 'wins': 'Количество побед'},
                       title='Количество побед по годам')
        fig2.add_traces(go.Scatter(
            x=top5_min_wins['year_ceremony'],
            y=top5_min_wins['wins'],
            mode='markers',
            name='Минимум (топ-5)',
            marker=dict(size=5, color='red')
        ))
        fig2.add_traces(go.Scatter(
            x=top5_max_wins['year_ceremony'],
            y=top5_max_wins['wins'],
            mode='markers',
            name='Максимум (топ-5)',
            marker=dict(size=5, color='green')
        ))
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        percent_of_wins_by_year = pd.merge(noms_by_year, wins_by_year, on='year_ceremony')
        percent_of_wins_by_year['win_percentage'] = percent_of_wins_by_year['wins'] / percent_of_wins_by_year['noms'] * 100
        top5_min_percent = percent_of_wins_by_year.nsmallest(5, 'win_percentage')
        top5_max_percent = percent_of_wins_by_year.nlargest(5, 'win_percentage')
        fig3 = px.line(percent_of_wins_by_year, x='year_ceremony', y='win_percentage',
                       labels={'year_ceremony': 'Год', 'win_percentage': 'Процент побед (%)'},
                       title='Процент побед по годам')
        fig3.add_traces(go.Scatter(
            x=top5_min_percent['year_ceremony'],
            y=top5_min_percent['win_percentage'],
            mode='markers',
            name='Минимум (топ-5)',
            marker=dict(size=5, color='red')
        ))
        fig3.add_traces(go.Scatter(
            x=top5_max_percent['year_ceremony'],
            y=top5_max_percent['win_percentage'],
            mode='markers',
            name='Максимум (топ-5)',
            marker=dict(size=5, color='green')
        ))
        st.plotly_chart(fig3, use_container_width=True)
    

with tab2:
    st.markdown('### Сравнение категорий')
    if categories != 'Все категории':
        st.markdown(f'#### Анализ категории **{categories}**')
        category_data = df_filtered[df_filtered['category']==categories]
        col4, col5 = st.columns(2)
        with col4:
            cat_noms = category_data.groupby('year_ceremony').size().reset_index(name='noms')
            fig4 = px.line(cat_noms, x='year_ceremony', y='noms', labels={'year_ceremony': 'Год', 'noms': 'Количество номинаций'},
                       title=f'Количество номинаций по годам в категории "{categories}"')
            st.plotly_chart(fig4, use_container_width=True)
        with col5:
            cat_wins = category_data.groupby('year_ceremony')['winner'].sum().reset_index(name='wins')
            fig5 = px.line(cat_wins, x='year_ceremony', y='wins', labels={'year_ceremony': 'Год', 'wins': 'Количество побед'},
                       title=f'Количество побед по годам в категории "{categories}"')
            st.plotly_chart(fig5, use_container_width=True)

    st.markdown('#### Тепловая карта номинаций')
    top_categories = df_filtered['category'].value_counts().sort_values(ascending=False).head(10).index.tolist()
    compare_categories = st.multiselect(
        label='Выберите категории для сравнения:',
        options=top_categories,
        default=top_categories[:3]
    )
    if compare_categories:
        cat_comparison_data = df_filtered[df_filtered['category'].isin(compare_categories)]
        heatmap_cat_data = cat_comparison_data.groupby(['year_ceremony', 'category']).size().unstack(fill_value=0)
        fig6 = px.imshow(heatmap_cat_data, labels=dict(x='Год', y='Категория', color='Количество'), aspect='auto')
        st.plotly_chart(fig6, use_container_width=True)
    st.markdown('---')
    
    st.markdown('#### Наличие категорий по десятилетиям')
    stability_table = df_filtered.groupby(['decade', 'category']).size().unstack(fill_value=0)
    binary_stability_table = (stability_table>0).astype(int)
    fig7 = px.imshow(
        binary_stability_table,
        labels=dict(x='Категория', y='Десятилетие'),
        color_continuous_scale=[[0, 'gray'], [1, 'blue']]
    )
    fig7.update_coloraxes(
        colorbar=dict(
            title='Наличие категории',
            tickvals=[0.1, 0.9],
            ticktext=['Отсутсвует', 'Присутствует'],
            orientation='v',
            len=1.3
        )
    )
    st.plotly_chart(fig7, use_container_width=True)
    st.markdown('---')

    st.markdown('#### Самые конкурентные категории')
    category_difficulty = df_filtered.groupby('category').agg(
        nominations=('winner', 'count'),
        wins=('winner', 'sum')
    ).reset_index().sort_values(by='nominations', ascending=False)
    category_difficulty['win_rate'] = (100*category_difficulty['wins']/category_difficulty['nominations']).round(2)
    most_competitive = category_difficulty[category_difficulty['nominations']>10].sort_values(by='win_rate').head(15)
    fig8 = px.bar(most_competitive, x='win_rate', y='category',
                             labels={'category': 'Категория', 'win_rate': 'Процент побед (%)'},
                             color='win_rate', color_continuous_scale='Viridis')
    st.plotly_chart(fig8, use_container_width=True)
    st.markdown('---')
    
    st.markdown('#### Распределение по типам')
    categories_types_count = df_filtered.groupby('category_type')['category'].nunique().reset_index(name='count')
    fig9 = px.pie(categories_types_count, values='count', names='category_type')
    st.plotly_chart(fig9, use_container_width=True)
    st.markdown('---')

    st.markdown('#### Доля номинаций Оскара по типам категорий')
    count_by_decades = df_filtered.groupby('decade').size()
    nominations_by_type = df_filtered.groupby(['decade', 'category_type']).size()
    percent_table = (nominations_by_type / count_by_decades * 100).unstack(fill_value=0)
    fig10 = px.imshow(percent_table, labels=dict(x='Тип категории', y='Десятилетие'), aspect='auto')
    fig10.update_coloraxes(
        colorbar=dict(title='Процент номинаций')
    )
    st.plotly_chart(fig10, use_container_width=True)

with tab3:
    st.markdown('### Анализ по периодам')
    period_type = st.radio(
        'Анализировать по:',
        ['Десятилетиям', 'Историческим периодам']
    )
    if period_type == 'Десятилетиям':
        decade_df = df_filtered.groupby('decade').agg(
            noms=('winner', 'count'),
            wins=('winner', 'sum')
        ).reset_index()
        decade_df = decade_df.rename(columns={'noms': 'Номинации', 'wins': 'Победы'})
        fig11 = px.bar(decade_df, x='decade', y=['Номинации', 'Победы'], barmode='group',
                    title='Номинации и победы по десятилетиям / историческим периодам',
                    labels={'decade': 'Десятилетие', 'value': 'Количество', 'variable': 'Тип'},
                    )
        st.plotly_chart(fig11)
    else:
        eras_df = df_filtered.groupby('era').agg(
            noms=('winner', 'count'),
            wins=('winner', 'sum')
        ).reset_index()
        eras_df = eras_df.rename(columns={'noms': 'Номинации', 'wins': 'Победы'})
        fig12 = px.bar(eras_df, x='era', y=['Номинации', 'Победы'], barmode='group',
                      title='Номинации и победы по историческим периодам',
                      labels={'era': 'Исторический период', 'value': 'Количество', 'variable': 'Тип'})
        st.plotly_chart(fig12)

with tab4:
    st.markdown('### Анализ актеров и режиссеров')
    choice = st.radio(
        label='Анализировать:',
        options=['Актеры', 'Режиссеры']
    )
    if choice == 'Актеры':
        actors_df = df_filtered[df_filtered['actor_actress']==True]
        actors_noms_wins = actors_df.groupby('name').agg(
            noms=('winner', 'count'),
            wins=('winner', 'sum')
        ).reset_index()
        top_actors_by_wins = actors_noms_wins.sort_values(by='wins', ascending=False).head(10)
        eternal_actors = actors_noms_wins[actors_noms_wins['wins']==0].sort_values(by='noms', ascending=False).head(10)
        col1, col2 = st.columns(2)
        with col1:
            fig13 = px.bar(top_actors_by_wins, x='wins', y='name', labels={'name': 'Актер/Актриса', 'wins': 'Победы'},
                        color='wins', color_continuous_scale='Teal', title='Топ-10 актеров по количеству побед на Оскаре')
            st.plotly_chart(fig13)
        with col2:
            fig14 = px.bar(eternal_actors, x='noms', y='name',
                           labels={'name': 'Актер/Актриса', 'noms': 'Номинации'},
                           color='noms', color_continuous_scale='Teal',
                           title='Самые номинируемые актеры без побед')
            st.plotly_chart(fig14)

        gender_analysis = actors_df.groupby(['decade', 'gender'], observed=True).size().reset_index(name='count')
        st.markdown('---')
        st.markdown('#### Распределение актерских номинаций по полу и десятилетиям')
        fig15 = px.bar(gender_analysis, x='decade', y='count', color='gender', barmode='group',
                       color_discrete_map={'male': 'steelblue', 'female': 'lightpink'},
                       labels={'decade': 'Десятилетие', 'count': 'Количество номинаций', 'gender': 'Пол'})
        st.plotly_chart(fig15)
        
    else:
        directors_df = df_filtered[df_filtered['is_director']==True]
        directors = directors_df.groupby('name').agg(
            noms=('winner', 'count'),
            wins=('winner', 'sum')
        ).reset_index()
        directors_wins = directors.sort_values(by='wins', ascending=False).head(10)
        directors_noms = directors[directors['wins']==0].sort_values(by='noms', ascending=False).head(10)
        col1, col2 = st.columns(2)
        with col1:
            fig16 = px.bar(directors_wins, x='wins', y='name', color='wins', color_continuous_scale='Cividis',
                   labels={'name': 'Режиссер', 'wins': 'Количество побед'},
                   title='Топ-10 режиссеров по количеству побед на Оскаре')
            st.plotly_chart(fig16)
        with col2:
            fig17 = px.bar(directors_noms, x='noms', y='name', color='noms', color_continuous_scale='Cividis',
                   labels={'name': 'Режиссер', 'noms': 'Количество номинаций'},
                   title='Топ-10 режиссеров по количеству номинаций без побед')
            st.plotly_chart(fig17)

with tab5:
    st.markdown('### Анализ студий')
    choice = st.radio(
        'Анализировать студии:',
        ['Золотого века Голливуда', 'Современного периода'])
    
    studios = df[df['is_studio']==True]
    if choice == 'Золотого века Голливуда':
        early_era = studios[studios['year_ceremony'] < 1950]
        early_studios_wins = early_era.groupby('name').agg(
            noms=('winner', 'count'),
            wins=('winner', 'sum')
        ).reset_index().sort_values(by='wins', ascending=False)
        early_studios_wins['percent_of_wins'] = early_studios_wins['wins'] / early_studios_wins['noms'] * 100
        fig18 = px.treemap(
            early_studios_wins,
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

        fig18.update_traces(
            textinfo='label+value+percent parent',
            textfont_size=14
        )

        st.plotly_chart(fig18)
    else:
        modern_era = studios[studios['year_ceremony'] >= 1950]
        modern_era_wins = modern_era.groupby('name').agg(
            noms=('winner', 'count'),
            wins=('winner', 'sum')
        ).reset_index().sort_values(by='wins', ascending=False)
        modern_era_wins['percent_of_wins'] = modern_era_wins['wins'] / modern_era_wins['noms'] * 100
        fig19 = px.treemap(
            modern_era_wins,
            path=['name'],
            values='wins',
            title='Распределение побед среди студий современного периода',
            color_continuous_scale='Plasma',
            hover_data={
                'noms': True,
                'wins': True,
                'percent_of_wins': ':.1f'
            }
        )

        fig19.update_traces(
            textinfo='label+value+percent parent',
            textfont_size=14
        )

        st.plotly_chart(fig19)

with tab6:
    st.header('Прогнозный анализ побед на Оскаре')
    st.subheader('1. Подготовка данных для модели')
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
    features = [
        'nominee_prev_noms', 'nominee_prev_win_rate',
        'years_since_last_nomination', 'film_noms_this_year',
        'category_competitiveness', 'Золотой век',
        'Новый Голливуд', 'Эпоха блокбастеров', 'Современный период',
        'had_previous_noms', 'had_previous_wins', 'lot_of_noms',
        'film_experience', 'film_and_cat', 'cat_total',
        'comeback', 'med', 'strong', 'super'
    ]
    bool_cols = ml_df[features].select_dtypes(include=['bool']).columns
    for i in bool_cols:
        ml_df[i] = ml_df[i].astype(int)
    
    st.write("Данные подготовлены")
    st.write(f"**Всего записей:** {len(ml_df)}")
    st.write(f"**Используется фич:** {len(features)}")

    st.subheader('2. Настройка модели')
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider(
            label='Размер тестовой выборки (%)',
            min_value=10,
            max_value=40,
            value=30,
            help='Доля данных для тестирования'
        )
    
    with col2:
        use_smote = st.checkbox(
            'Использовать SMOTE',
            value=True,
            help='Балансировка классов (рекомендуется при дисбалансе)'
        )
    
    with col3:
        use_gridsearch = st.checkbox(
            'Оптимизация гиперпараметров',
            value=True,
            help='GridSearchCV для поиска лучших параметров'
        )
    
    if st.button('Обучить модель Random Forest', type='primary'):
        with st.spinner('Обучение модели...'):
            try:
                X = ml_df[features]
                y = ml_df['winner']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, 
                                                                    random_state=42, stratify=y)
                if use_smote:
                    smote = SMOTE(random_state=42)
                    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
                    st.info('SMOTE применен.')
                else:
                    X_train_bal, y_train_bal = X_train, y_train
                
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
                    st.success(f'Лучшие параметры: {gridRF.best_params_}')
                    st.success(f'Лучший ROC-AUC: {gridRF.best_score_}')
                else:
                    best_rf = RandomForestClassifier(
                        n_estimators=200,
                        max_depth=20,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        class_weight='balanced',
                        random_state=42)
                    best_rf.fit(X_train_bal, y_train_bal)
                
                predictions = best_rf.predict(X_test)
                pred_proba = best_rf.predict_proba(X_test)[:, 1]

                # Сохранение в session_state
                st.session_state['model'] = best_rf
                st.session_state['features'] = features
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['predictions'] = predictions
                st.session_state['pred_proba'] = pred_proba
                st.session_state['ml_df'] = ml_df
                st.session_state['feature_importance'] = pd.DataFrame({
                    'feature': features,
                    'importance': best_rf.feature_importances_
                }).sort_values(by='importance', ascending=False)

                st.success('Модель успешно обучена.')
            except Exception as e:
                st.error(f'Ошибка при обучении модели: {str(e)}')
                st.info("Проверьте наличие всех необходимых столбцов в данных")
    
    if 'model' in st.session_state:
        st.subheader('3. Оценка модели')
        model = st.session_state['model']
        predictions = st.session_state['predictions']
        y_test = st.session_state['y_test']
        pred_proba = st.session_state['pred_proba']

        accuracy = accuracy_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, pred_proba)
        report = classification_report(y_test, predictions, output_dict=True)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric('Точность (Accuracy)', f'{accuracy:.2%}')
        
        with col2:
            st.metric('ROC-AUC', f'{roc_auc:.3f}')
        
        with col3:
            precision = report['True']['precision']
            st.metric('Precision (победы)', f'{precision:.2%}')
        
        with col4:
            recall = report['True']['recall']
            st.metric('Recall (победы)', f'{recall:.2%}')
        
        with col5:
            f1 = report['True']['f1-score']
            st.metric('f1-score (победы)', f'{f1:.2%}')
        
        st.markdown("##### Матрица ошибок")
        cm = confusion_matrix(y_test, predictions)
        fig_cm = px.imshow(
            cm,text_auto=True,
            labels=dict(x='Предсказанные', y='Реальные', color='Количество'),
            x = ['НЕТ', 'ДА'],
            y = ['НЕТ', 'ДА'],
            aspect='auto',
            color_continuous_scale='Blues'
            )
        fig_cm.update_layout(
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_cm)
    
        st.subheader('5. Важность признаков')
        importance_df = st.session_state['feature_importance']
        fig_importance = px.bar(
            importance_df,
            x='importance',
            y='feature',
            color='importance',
            color_continuous_scale='Viridis',
            labels={'feature': 'Признак', 'importance': 'Важность'}
        )
        st.plotly_chart(fig_importance)