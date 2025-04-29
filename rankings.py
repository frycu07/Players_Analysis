import streamlit as st
import pandas as pd
import seaborn as sns

def show_rankings(df, selected_teams, selected_positions, selected_minutes, filter_mode):
    """
    Wyświetla rankingi dla wybranych statystyk z uwzględnieniem filtrów
    """
    if df.empty:
        st.warning("Brak danych do wyświetlenia rankingów.")
        return

    # Filtrowanie danych według wybranych kryteriów
    filtered_df = filter_data(df, selected_teams, selected_positions, selected_minutes, filter_mode)

    if filtered_df.empty:
        st.warning("Brak zawodników spełniających kryteria! Zmień filtry.")
        return

    # Statystyki do pominięcia w rankingach
    excluded_stats = ['Weight', 'Age', 'Height', 'Market value', 'Minutes played']

    # Pobieranie wszystkich dostępnych statystyk numerycznych
    numeric_columns = filtered_df.select_dtypes(include=['number']).columns
    available_stats = [col for col in numeric_columns if col not in excluded_stats]

    # Wybór statystyki do rankingu
    selected_stat = st.selectbox('Wybierz statystykę do rankingu:', available_stats)

    # Liczba zawodników do wyświetlenia
    top_n = st.slider('Liczba zawodników w rankingu:', min_value=5, max_value=50, value=10)

    # Sortowanie malejąco/rosnąco
    sort_order = st.radio('Kolejność sortowania:', ['Malejąco', 'Rosnąco'])
    ascending = sort_order == 'Rosnąco'

    # Tworzenie rankingu
    ranking_df = filtered_df[['Player', 'Team within selected timeframe', 'Position', selected_stat]].copy()
    # Usuwamy wiersze z wartościami NA dla wybranej statystyki
    ranking_df = ranking_df.dropna(subset=[selected_stat])
    ranking_df = ranking_df.sort_values(by=selected_stat, ascending=ascending)
    ranking_df = ranking_df.head(top_n).reset_index(drop=True)
    ranking_df.index = ranking_df.index + 1  # Numeracja od 1

    # Obliczanie percentyli dla wybranej statystyki
    max_value = ranking_df[selected_stat].max()  # Używamy max z ranking_df, który ma już usunięte NA
    if max_value > 0:
        percentiles = (ranking_df[selected_stat] / max_value * 100).round(1)
    else:
        percentiles = pd.Series([0] * len(ranking_df))

    # Wyświetlanie rankingu w formie tabeli z paskami postępu
    st.subheader(f'Ranking - {selected_stat}')

    # Tworzenie tabeli HTML z paskami postępu
    col_widths = {'rank': '50px', 'player': '200px', 'team': '150px', 'position': '100px', 
                  'bar': '300px', 'value': '100px'}

    table_html = '<table style="width:100%; border-collapse: collapse;">'
    table_html += '<tr>'
    table_html += f'<th style="width:{col_widths["rank"]}; text-align: center; border: 1px solid black;">#</th>'
    table_html += f'<th style="width:{col_widths["player"]}; text-align: left; border: 1px solid black;">Zawodnik</th>'
    table_html += f'<th style="width:{col_widths["team"]}; text-align: left; border: 1px solid black;">Drużyna</th>'
    table_html += f'<th style="width:{col_widths["position"]}; text-align: left; border: 1px solid black;">Pozycja</th>'
    table_html += f'<th style="width:{col_widths["bar"]}; text-align: left; border: 1px solid black;">Wartość</th>'
    table_html += f'<th style="width:{col_widths["value"]}; text-align: right; border: 1px solid black;">Liczba</th>'
    table_html += '</tr>'

    for idx, row in ranking_df.iterrows():
        # idx jest 1-based, ale percentiles jest 0-based, więc używamy idx-1 jako indeksu
        percentile = percentiles.iloc[idx-1]
        color = sns.color_palette('RdYlGn', as_cmap=True)(percentile / 100)
        rgb_color = f'rgb({int(color[0] * 255)}, {int(color[1] * 255)}, {int(color[2] * 255)})'

        bar_html = f'<div style="width:{col_widths["bar"]}; background:lightgrey; border:1px solid black; height:20px; position:relative;">'
        bar_html += f'<div style="width:{percentile}%; background:{rgb_color}; height:100%;"></div></div>'

        table_html += '<tr>'
        table_html += f'<td style="text-align: center; border: 1px solid black;">{idx}</td>'  # idx jest już poprawnym numerem rankingu (1-based)
        table_html += f'<td style="border: 1px solid black;">{row["Player"]}</td>'
        table_html += f'<td style="border: 1px solid black;">{row["Team within selected timeframe"]}</td>'
        table_html += f'<td style="border: 1px solid black;">{row["Position"]}</td>'
        table_html += f'<td style="border: 1px solid black;">{bar_html}</td>'
        table_html += f'<td style="text-align: right; border: 1px solid black;">{row[selected_stat]:.2f}</td>'
        table_html += '</tr>'

    table_html += '</table>'
    st.markdown(table_html, unsafe_allow_html=True)

def filter_data(df, teams, positions, minutes, filter_mode):
    """
    Filtruje dane według wybranych kryteriów
    """
    if df.empty:
        return df

    filtered = df.copy()

    try:
        if filter_mode == 'Tylko dla filtrowanych zawodników':
            if teams != 'Wszystkie':
                filtered = filtered[filtered['Team within selected timeframe'] == teams]
            if positions:  # Jeśli wybrano jakieś pozycje
                filtered = filtered[filtered['Position'].apply(
                    lambda x: any(pos in [p.strip() for p in str(x).split(',')] for pos in positions)
                )]
            filtered = filtered[filtered['Minutes played'] >= minutes]

        if filtered.empty:
            st.warning("Filtry zwróciły pusty zbiór danych")
            return df  # Zwracamy oryginalne dane zamiast pustego zbioru

    except Exception as e:
        st.error(f"Błąd podczas filtrowania: {str(e)}")
        return df

    return filtered 
