import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px


def compare_teams(df):
    """
    Funkcja wyświetlająca porównanie statystyk między drużynami.

    Pozwala na wybór drużyn i statystyk do porównania, oraz wyświetla
    ranking wszystkich drużyn dla wybranej statystyki.
    """
    st.title('Porównanie Drużyn')

    if df.empty:
        st.warning("Brak danych do analizy.")
        return

    # Pobierz listę wszystkich drużyn
    teams = df['Team within selected timeframe'].dropna().unique()

    # Wybór drużyny głównej do analizy
    selected_team = st.selectbox('Wybierz główną drużynę do analizy:', options=list(teams))

    # Filtruj dane dla wybranej drużyny
    team_data = df[df['Team within selected timeframe'] == selected_team].copy()

    if team_data.empty:
        st.warning(f"Brak danych dla drużyny {selected_team}.")
        return

    # Dodaj filtr dla minimalnej liczby minut rozegranych
    min_minutes_played = st.slider(
        "Minimalna liczba minut rozegranych przez zawodnika:",
        min_value=0,
        max_value=int(team_data['Minutes played'].max()),
        value=0
    )

    # Filtruj zawodników według minimalnej liczby minut
    filtered_team_data = team_data[team_data['Minutes played'] >= min_minutes_played].copy()

    if filtered_team_data.empty:
        st.warning(f"Brak zawodników z co najmniej {min_minutes_played} minut.")
        return

    # Używamy przefiltrowanych danych do dalszych obliczeń
    team_data = filtered_team_data

    # Lista statystyk do analizy (statystyki per 90 minut)
    per_90_stats = [col for col in df.columns if 'per 90' in col]

    # Dodaj inne ważne statystyki, które nie mają "per 90" w nazwie
    other_important_stats = [
        'Goals', 'Assists', 'xG', 'xA', 'Shots', 'Passes', 
        'Accurate passes, %', 'Successful defensive actions', 
        'Defensive duels won, %', 'Aerial duels won, %'
    ]

    # Wszystkie statystyki do wyboru
    all_stats = per_90_stats + [stat for stat in other_important_stats if stat not in per_90_stats]

    # Wybór statystyk do analizy z opcją wyboru wszystkich
    st.sidebar.header('Wybierz statystyki')
    select_all = st.sidebar.checkbox('Wybierz wszystkie', value=True)

    if select_all:
        stats_to_analyze = all_stats
    else:
        stats_to_analyze = st.sidebar.multiselect(
            "Wybierz statystyki do analizy:", 
            options=all_stats,
            default=all_stats[:min(5, len(all_stats))]
        )

    if not stats_to_analyze:
        st.warning("Wybierz przynajmniej jedną statystykę do analizy.")
        return

    # Oblicz statystyki drużyno we per 90 minut
    team_stats_per_90 = {}
    for stat in stats_to_analyze:
        if 'per 90' in stat:
            # Dla statystyk już wyrażonych per  90, obliczamy średnią ważoną minutami
            weighted_stat = (team_data[stat] * team_data['Minutes played']).sum() / team_data['Minutes played'].sum()
            team_stats_per_90[stat] = weighted_stat
        else:
            # Dla statystyk, które nie są wyrażone per 90, przeliczamy je
            base_stat = stat.replace(' per 90', '')
            if base_stat in team_data.columns:
                total_stat = team_data[base_stat].sum()
                per_90_value = (total_stat / team_data['Minutes played'].sum()) * 90
                team_stats_per_90[f"{base_stat} per 90"] = per_90_value

    # Wybór drużyn do porównania
    teams_to_compare = st.multiselect(
        "Wybierz drużyny do porównania:", 
        options=[t for t in teams if t != selected_team],
        default=[]
    )

    if not teams_to_compare:
        st.warning("Wybierz przynajmniej jedną drużynę do porównania.")
        return

    # Wybór statystyki do porównania
    stat_to_compare = st.selectbox(
        "Wybierz statystykę do porównania:", 
        options=stats_to_analyze
    )

    # Oblicz wybraną statystykę dla wszystkich wybranych drużyn
    comparison_data = {}
    # Use the correct key format based on whether the stat already has "per 90" in its name
    stat_key = stat_to_compare if 'per 90' in stat_to_compare else f"{stat_to_compare} per 90"
    if stat_key in team_stats_per_90:
        comparison_data[selected_team] = team_stats_per_90[stat_key]

    for team in teams_to_compare:
        team_df = df[df['Team within selected timeframe'] == team].copy()
        if not team_df.empty:
            team_minutes = team_df['Minutes played'].sum()
            if 'per 90' in stat_to_compare:
                weighted_stat = (team_df[stat_to_compare] * team_df['Minutes played']).sum() / team_df['Minutes played'].sum()
                comparison_data[team] = weighted_stat
            else:
                base_stat = stat_to_compare.replace(' per 90', '')
                if base_stat in team_df.columns:
                    total_stat = team_df[base_stat].sum()
                    per_90_value = (total_stat / team_minutes) * 90
                    comparison_data[team] = per_90_value

    # Oblicz statystyki dla wszystkich drużyn (nie tylko wybranych do porównania)
    all_teams_data = {}
    # Use the correct key format based on whether the stat already has "per 90" in its name
    stat_key = stat_to_compare if 'per 90' in stat_to_compare else f"{stat_to_compare} per 90"
    if stat_key in team_stats_per_90:
        all_teams_data[selected_team] = team_stats_per_90[stat_key]

    for team in teams:
        if team != selected_team:  # Już mamy dane dla wybranej drużyny
            team_df = df[df['Team within selected timeframe'] == team].copy()
            if not team_df.empty:
                team_minutes = team_df['Minutes played'].sum()
                if 'per 90' in stat_to_compare:
                    weighted_stat = (team_df[stat_to_compare] * team_df['Minutes played']).sum() / team_df['Minutes played'].sum()
                    all_teams_data[team] = weighted_stat
                else:
                    base_stat = stat_to_compare.replace(' per 90', '')
                    if base_stat in team_df.columns:
                        total_stat = team_df[base_stat].sum()
                        per_90_value = (total_stat / team_minutes) * 90
                        all_teams_data[team] = per_90_value

    # Stwórz DataFrame do porównania (tylko wybrane drużyny)
    comparison_df = pd.DataFrame({
        'Drużyna': list(comparison_data.keys()),
        'Wartość': list(comparison_data.values())
    })

    # Stwórz DataFrame dla wszystkich drużyn
    all_teams_df = pd.DataFrame({
        'Drużyna': list(all_teams_data.keys()),
        'Wartość': list(all_teams_data.values())
    })

    # Sortuj według wartości
    comparison_df = comparison_df.sort_values('Wartość', ascending=False)
    all_teams_df = all_teams_df.sort_values('Wartość', ascending=False)

    # Dodaj indeks rangowy (1-based) po sortowaniu
    ranks = list(range(1, len(all_teams_df) + 1))

    # Przygotuj kolory dla drużyn
    colors = ['rgba(246, 78, 139, 0.6)' if team == selected_team else 'rgba(58, 71, 80, 0.6)' 
              for team in all_teams_df['Drużyna']]

    # Stwórz nowy DataFrame z wszystkimi kolumnami na raz, aby uniknąć fragmentacji
    all_teams_df = pd.DataFrame({
        'Drużyna': all_teams_df['Drużyna'].values,
        'Wartość': all_teams_df['Wartość'].values,
        'Ranga': ranks,
        'Kolor': colors
    })

    # Oblicz percentyl dla wybranej drużyny
    selected_team_value = all_teams_data[selected_team]
    better_teams = sum(1 for value in all_teams_data.values() if value > selected_team_value)
    worse_teams = sum(1 for value in all_teams_data.values() if value < selected_team_value)
    equal_teams = sum(1 for value in all_teams_data.values() if value == selected_team_value) - 1  # Odejmujemy 1, żeby nie liczyć wybranej drużyny dwa razy
    total_teams = len(all_teams_data)

    # Sprawdź, czy mamy dane dla co najmniej jednej drużyny
    if total_teams > 0:
        # Oblicz percentyl (im wyższy, tym lepiej)
        # Używamy formuły, która uwzględnia drużyny o równej wartości
        percentile = 100 - (better_teams / total_teams * 100)
    else:
        percentile = 0
        st.warning("Brak danych dla innych drużyn, nie można obliczyć percentyla.")

    # Wyświetl informacje o percentylu
    st.subheader(f"Pozycja drużyny {selected_team} w lidze dla statystyki {stat_to_compare}")

    # Dodaj wykres wskaźnikowy (gauge) dla percentyla
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percentile,
        title={'text': f"Percentyl drużyny {selected_team}"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "rgba(246, 78, 139, 0.6)"},
            'steps': [
                {'range': [0, 25], 'color': "lightgray"},
                {'range': [25, 50], 'color': "gray"},
                {'range': [50, 75], 'color': "lightblue"},
                {'range': [75, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': percentile
            }
        }
    ))

    fig_gauge.update_layout(
        height=300,
        width=600
    )

    st.plotly_chart(fig_gauge)

    # Wyświetl dodatkowe informacje tekstowe
    st.write(f"Percentyl: {percentile:.1f}% (im wyższy, tym lepiej)")
    st.write(f"Liczba drużyn lepszych: {better_teams}")
    st.write(f"Liczba drużyn gorszych: {worse_teams}")
    st.write(f"Liczba drużyn z taką samą wartością: {equal_teams}")
    st.write(f"Łączna liczba drużyn: {total_teams}")

    # Dodaj wykres rankingowy wszystkich drużyn
    st.subheader(f"Ranking wszystkich drużyn dla statystyki {stat_to_compare}")

    if len(all_teams_df) > 0:

        # Stwórz wykres słupkowy z rankingiem
        fig_ranking = px.bar(
            all_teams_df,
            x='Drużyna',
            y='Wartość',
            title=f"Ranking drużyn dla statystyki {stat_to_compare}",
            labels={'Wartość': f'{stat_to_compare}', 'Drużyna': ''},
            text=all_teams_df['Wartość'].apply(lambda x: f"{x:.2f}")
        )

        # Ustaw kolory słupków
        fig_ranking.update_traces(marker_color=all_teams_df['Kolor'])

        # Sprawdź, czy wybrana drużyna jest w DataFrame
        if selected_team in all_teams_df['Drużyna'].values:
            # Dodaj adnotację z rangą dla wybranej drużyny
            selected_team_rank = all_teams_df[all_teams_df['Drużyna'] == selected_team]['Ranga'].values[0]
            selected_team_index = all_teams_df[all_teams_df['Drużyna'] == selected_team].index[0]

            fig_ranking.add_annotation(
                x=selected_team_index,
                y=all_teams_df.loc[selected_team_index, 'Wartość'],
                text=f"Ranga: {selected_team_rank}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40
            )

        fig_ranking.update_layout(
            height=500,
            width=800,
            xaxis_tickangle=-45
        )

        st.plotly_chart(fig_ranking)

        # Wyświetl dane w tabeli w stylu podobnym do analizy zawodników
        st.subheader(f"Tabela rankingowa drużyn dla statystyki {stat_to_compare}")

        # Definicje szerokości kolumn i inicjalizacja tabeli HTML
        col_widths = {'team': '200px', 'bar': '300px', 'value': '250px'}
        table_html = '<table style="width:100%; border-collapse: collapse;">'
        table_html += '<tr><th style="width:{}; text-align: left; border: 1px solid black;">Drużyna</th>'.format(
            col_widths['team'])
        table_html += '<th style="width:{}; text-align: left; border: 1px solid black;">Wykres</th>'.format(
            col_widths['bar'])
        table_html += '<th style="width:{}; text-align: left; border: 1px solid black;">Wartość</th></tr>'.format(
            col_widths['value'])

        # Iteruj przez wszystkie drużyny w rankingu
        for index, row in all_teams_df.iterrows():
            team_name = row['Drużyna']
            team_value = row['Wartość']
            team_rank = row['Ranga']
            total_teams = len(all_teams_df)

            # Oblicz percentyl dla drużyny (im wyższy rank, tym niższy percentyl)
            team_percentile = 100 - ((team_rank - 1) / total_teams * 100)

            # Wybierz kolor na podstawie percentyla
            import seaborn as sns
            color = sns.color_palette('RdYlGn', as_cmap=True)(team_percentile / 100)
            rgb_color = f'rgb({int(color[0] * 255)}, {int(color[1] * 255)}, {int(color[2] * 255)})'

            # Dodaj podświetlenie dla wybranej drużyny
            team_style = 'font-weight: bold; background-color: rgba(246, 78, 139, 0.2);' if team_name == selected_team else ''

            # Stwórz pasek reprezentujący percentyl
            bar_html = f'<div style="width:{col_widths["bar"]}; background:lightgrey; border:1px solid black; height:20px; position:relative;">'
            bar_html += f'<div style="width:{team_percentile}%; background:{rgb_color}; height:100%;"></div></div>'

            # Wyświetl wartość, ranking i percentyl
            value_display = f'{team_value:.2f} ({team_rank}/{total_teams}, {team_percentile:.1f}th percentile)'

            # Dodaj wiersz do tabeli
            table_html += f'<tr style="{team_style}"><td style="width:{col_widths["team"]}; border: 1px solid black;">{team_name}</td>'
            table_html += f'<td style="width:{col_widths["bar"]}; border: 1px solid black;">{bar_html}</td>'
            table_html += f'<td style="width:{col_widths["value"]}; border: 1px solid black;">{value_display}</td></tr>'

        # Zamknij tabelę i wyświetl ją
        table_html += '</table>'
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.warning("Brak danych dla innych drużyn, nie można wyświetlić rankingu.")

    # Wybór rodzaju wykresu
    chart_type = st.radio('Wybierz typ wykresu:', ['Wykres słupkowy', 'Radar Chart'])

    if chart_type == 'Wykres słupkowy':
        # Wykres porównawczy
        fig = px.bar(
            comparison_df, 
            x='Drużyna', 
            y='Wartość',
            title=f"Porównanie {stat_to_compare} między drużynami",
            labels={'Wartość': f'{stat_to_compare}', 'Drużyna': ''},
            text=comparison_df['Wartość'].apply(lambda x: f"{x:.2f}")
        )

        # Podświetl wybraną drużynę
        fig.update_traces(
            marker_color=[
                'rgba(58, 71, 80, 0.6)' if x != selected_team else 'rgba(246, 78, 139, 0.6)' 
                for x in comparison_df['Drużyna']
            ]
        )

        fig.update_layout(
            height=500,
            width=800
        )

        st.plotly_chart(fig)

        # Definicje szerokości kolumn i inicjalizacja tabeli HTML
        col_widths = {'team': '200px', 'bar': '300px', 'value': '250px'}
        table_html = '<table style="width:100%; border-collapse: collapse;">'
        table_html += '<tr><th style="width:{}; text-align: left; border: 1px solid black;">Drużyna</th>'.format(
            col_widths['team'])
        table_html += '<th style="width:{}; text-align: left; border: 1px solid black;">Wykres</th>'.format(
            col_widths['bar'])
        table_html += '<th style="width:{}; text-align: left; border: 1px solid black;">Wartość</th></tr>'.format(
            col_widths['value'])

        # Oblicz percentyle dla wszystkich drużyn
        max_value = max(comparison_data.values())
        min_value = min(comparison_data.values())
        value_range = max_value - min_value if max_value != min_value else 1

        # Znajdź ranking dla każdej drużyny w porównaniu
        sorted_teams = sorted(comparison_data.items(), key=lambda x: x[1], reverse=True)
        team_ranks = {team: rank+1 for rank, (team, _) in enumerate(sorted_teams)}
        total_compared_teams = len(comparison_data)

        for team, value in comparison_data.items():
            # Normalizuj wartość do percentyla (0-100)
            if value_range > 0:
                team_percentile = ((value - min_value) / value_range) * 100
            else:
                team_percentile = 50  # Jeśli wszystkie wartości są takie same

            # Znajdź ranking drużyny w całej lidze
            team_league_rank = all_teams_df[all_teams_df['Drużyna'] == team]['Ranga'].values[0] if team in all_teams_df['Drużyna'].values else "N/A"
            total_league_teams = len(all_teams_df)

            # Wybierz kolor na podstawie percentyla
            color = sns.color_palette('RdYlGn', as_cmap=True)(team_percentile / 100)
            rgb_color = f'rgb({int(color[0] * 255)}, {int(color[1] * 255)}, {int(color[2] * 255)})'

            # Dodaj podświetlenie dla wybranej drużyny
            team_style = 'font-weight: bold; background-color: rgba(246, 78, 139, 0.2);' if team == selected_team else ''

            # Stwórz pasek reprezentujący percentyl
            bar_html = f'<div style="width:{col_widths["bar"]}; background:lightgrey; border:1px solid black; height:20px; position:relative;">'
            bar_html += f'<div style="width:{team_percentile}%; background:{rgb_color}; height:100%;"></div></div>'

            # Wyświetl wartość z rankingiem i percentylem
            value_display = f'{value:.2f} ({team_ranks[team]}/{total_compared_teams}, {team_percentile:.1f}% w porównaniu)'
            if team_league_rank != "N/A":
                value_display += f' [{team_league_rank}/{total_league_teams} w lidze]'

            # Dodaj wiersz do tabeli
            table_html += f'<tr style="{team_style}"><td style="width:{col_widths["team"]}; border: 1px solid black;">{team}</td>'
            table_html += f'<td style="width:{col_widths["bar"]}; border: 1px solid black;">{bar_html}</td>'
            table_html += f'<td style="width:{col_widths["value"]}; border: 1px solid black;">{value_display}</td></tr>'

        # Zamknij tabelę i wyświetl ją
        table_html += '</table>'
        st.markdown(table_html, unsafe_allow_html=True)

    elif chart_type == 'Radar Chart':
        # Przygotowanie danych do wykresu radarowego
        teams_list = list(comparison_data.keys())
        stats_list = stats_to_analyze

        # Wybieramy maksymalnie 25 statystyk dla czytelności wykresu
        if len(stats_list) > 25:
            st.warning('Wybrano zbyt wiele statystyk. Dla wykresu radarowego pokazanych zostanie pierwsze 25.')
            stats_list = stats_list[:25]

        # Tworzenie wykresu radarowego
        fig = go.Figure()

        # Oblicz statystyki dla wszystkich wybranych drużyn i wszystkich wybranych statystyk
        all_team_stats = {}
        for team in teams_list:
            all_team_stats[team] = {}
            team_df = df[df['Team within selected timeframe'] == team].copy()
            if not team_df.empty:
                team_minutes = team_df['Minutes played'].sum()
                for stat in stats_list:
                    if 'per 90' in stat:
                        weighted_stat = (team_df[stat] * team_df['Minutes played']).sum() / team_df['Minutes played'].sum()
                        all_team_stats[team][stat] = weighted_stat
                    else:
                        base_stat = stat.replace(' per 90', '')
                        if base_stat in team_df.columns:
                            total_stat = team_df[base_stat].sum()
                            per_90_value = (total_stat / team_minutes) * 90
                            all_team_stats[team][stat] = per_90_value

        # Znajdź maksymalne wartości dla każdej statystyki
        max_stats = {}
        for stat in stats_list:
            max_stats[stat] = max([all_team_stats[team].get(stat, 0) for team in teams_list])

        # Oblicz rankingi dla każdej statystyki
        stat_rankings = {}
        for stat in stats_list:
            # Zbierz wartości dla wszystkich drużyn dla tej statystyki
            stat_values = {team: all_team_stats[team].get(stat, 0) for team in teams_list}
            # Sortuj drużyny według wartości (malejąco)
            sorted_teams = sorted(stat_values.items(), key=lambda x: x[1], reverse=True)
            # Przypisz rankingi
            stat_rankings[stat] = {team: rank+1 for rank, (team, _) in enumerate(sorted_teams)}

        # Dodaj ślady dla każdej drużyny
        colors = ['rgb(99, 110, 250)', 'rgb(239, 85, 59)', 'rgb(0, 204, 150)', 'rgb(171, 99, 250)', 'rgb(255, 161, 90)']

        for i, team in enumerate(teams_list):
            radar_values = []
            hover_texts = []

            for stat in stats_list:
                if stat in all_team_stats[team]:
                    # Normalizuj wartość do percentyla (0-100)
                    if max_stats[stat] > 0:
                        percentile = (all_team_stats[team][stat] / max_stats[stat]) * 100
                    else:
                        percentile = 0

                    actual_value = all_team_stats[team][stat]
                    team_rank = stat_rankings[stat][team]
                    total_teams = len(teams_list)

                    radar_values.append(percentile)
                    hover_texts.append(f"Wartość: {actual_value:.2f}<br>Ranking: {team_rank}/{total_teams}<br>Percentyl: {percentile:.1f}%")
                else:
                    radar_values.append(0)
                    hover_texts.append("Brak danych")

            color_idx = i % len(colors)
            fig.add_trace(go.Scatterpolar(
                r=radar_values,
                theta=stats_list,
                fill='toself',
                name=team,
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>",
                fillcolor=f'rgba({colors[color_idx][4:-1]}, 0.2)',
                line=dict(color=colors[color_idx])
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            height=600,
            width=800
        )

        st.plotly_chart(fig)

        # Dodaj tabelę z wartościami dla wykresu radarowego
        st.subheader("Tabela wartości dla wykresu radarowego")

        # Przygotuj dane do tabeli
        table_data = []
        for stat in stats_list:
            row = {'Statystyka': stat}
            for team in teams_list:
                if stat in all_team_stats[team]:
                    value = all_team_stats[team][stat]
                    rank = stat_rankings[stat][team]
                    total = len(teams_list)
                    percentile = (all_team_stats[team][stat] / max_stats[stat]) * 100 if max_stats[stat] > 0 else 0
                    row[team] = f"{value:.2f} ({rank}/{total}, {percentile:.1f}%)"
                else:
                    row[team] = "N/A"
            table_data.append(row)

        # Debug: Sprawdź, czy dane tabeli są tworzone
        st.write("Debug: Dane tabeli zostały utworzone")
        st.write(f"Debug: Liczba wierszy w tabeli: {len(table_data)}")
        st.write(f"Debug: Przykładowy wiersz: {table_data[0] if table_data else 'Brak danych'}")

        # Utwórz DataFrame i wyświetl tabelę
        table_df = pd.DataFrame(table_data)

        # Debug: Sprawdź, czy DataFrame został utworzony
        st.write("Debug: DataFrame został utworzony")
        st.write(f"Debug: Kształt DataFrame: {table_df.shape}")

        # Wyświetl tabelę
        st.write("Debug: Wyświetlanie tabeli...")
        st.table(table_df)
        st.write("Debug: Tabela powinna być wyświetlona powyżej")

def analyze_team_stats(df):
    """
    Funkcja wyświetlająca analizę statystyk drużyn na podstawie danych o zawodnikach.

    Pokazuje statystyki per 90 minut dla każdej drużyny, obliczone na podstawie
    statystyk indywidualnych zawodników.
    """
    st.title('Analiza Drużyny')

    if df.empty:
        st.warning("Brak danych do analizy.")
        return

    # Pobierz listę wszystkich drużyn
    teams = df['Team within selected timeframe'].dropna().unique()

    # Wybór drużyny do analizy
    selected_team = st.selectbox('Wybierz drużynę do analizy:', options=list(teams))

    # Filtruj dane dla wybranej drużyny
    team_data = df[df['Team within selected timeframe'] == selected_team].copy()

    if team_data.empty:
        st.warning(f"Brak danych dla drużyny {selected_team}.")
        return

    # Dodaj filtr dla minimalnej liczby minut rozegranych
    min_minutes_played = st.slider(
        "Minimalna liczba minut rozegranych przez zawodnika:",
        min_value=0,
        max_value=int(team_data['Minutes played'].max()),
        value=0
    )

    # Filtruj zawodników według minimalnej liczby minut
    filtered_team_data = team_data[team_data['Minutes played'] >= min_minutes_played].copy()

    if filtered_team_data.empty:
        st.warning(f"Brak zawodników z co najmniej {min_minutes_played} minut.")
        return

    # Używamy przefiltrowanych danych do dalszych obliczeń
    team_data = filtered_team_data

    # Oblicz sumę minut rozegranych przez wszystkich zawodników drużyny
    total_minutes = team_data['Minutes played'].sum()

    # Wyświetl podstawowe informacje o drużynie
    st.subheader(f"Statystyki drużyny: {selected_team}")
    st.write(f"Łączna liczba minut rozegranych przez zawodników: {total_minutes:.0f}")
    st.write(f"Liczba zawodników w drużynie: {len(team_data)}")

    # Lista statystyk do analizy (statystyki per 90 minut)
    per_90_stats = [col for col in df.columns if 'per 90' in col]

    # Dodaj inne ważne statystyki, które nie mają "per 90" w nazwie
    other_important_stats = [
        'Goals', 'Assists', 'xG', 'xA', 'Shots', 'Passes', 
        'Accurate passes, %', 'Successful defensive actions', 
        'Defensive duels won, %', 'Aerial duels won, %'
    ]

    # Wszystkie statystyki do wyboru
    all_stats = per_90_stats + [stat for stat in other_important_stats if stat not in per_90_stats]

    # Wybór statystyk do analizy z opcją wyboru wszystkich
    st.sidebar.header('Wybierz statystyki')
    select_all = st.sidebar.checkbox('Wybierz wszystkie', value=True)

    if select_all:
        stats_to_analyze = all_stats
    else:
        stats_to_analyze = st.sidebar.multiselect(
            "Wybierz statystyki do analizy:", 
            options=all_stats,
            default=all_stats[:min(5, len(all_stats))]
        )

    if not stats_to_analyze:
        st.warning("Wybierz przynajmniej jedną statystykę do analizy.")
        return

    # Dodaj sekcję debugowania
    st.subheader("Debugowanie - Obliczanie statystyk drużynowych")
    debug_expander = st.expander("Pokaż szczegóły debugowania", expanded=False)

    with debug_expander:
        st.write("### Krok 1: Obliczanie statystyk drużynowych per 90 minut")
        st.write(f"Liczba statystyk do analizy: {len(stats_to_analyze)}")
        st.write(f"Przykładowe statystyki: {stats_to_analyze[:5] if len(stats_to_analyze) > 5 else stats_to_analyze}")

    # Oblicz statystyki drużynowe per 90 minut
    team_stats_per_90 = {}
    for stat in stats_to_analyze:
        if 'per 90' in stat:
            # Dla statystyk już wyrażonych per 90, obliczamy średnią ważoną minutami
            weighted_stat = (team_data[stat] * team_data['Minutes played']).sum() / team_data['Minutes played'].sum()
            team_stats_per_90[stat] = weighted_stat

            with debug_expander:
                st.write(f"Statystyka '{stat}' (już per 90): {weighted_stat:.4f}")
        else:
            # Dla statystyk, które nie są wyrażone per 90, przeliczamy je
            base_stat = stat.replace(' per 90', '')
            if base_stat in team_data.columns:
                # Filtrujemy dane, aby usunąć NA dla danej statystyki
                stat_data = team_data.dropna(subset=[base_stat])
                if not stat_data.empty:
                    total_stat = stat_data[base_stat].sum()
                    stat_minutes = stat_data['Minutes played'].sum()
                    per_90_value = (total_stat / stat_minutes) * 90
                    stat_key = f"{base_stat} per 90"
                    team_stats_per_90[stat_key] = per_90_value

                    with debug_expander:
                        st.write(f"Statystyka '{base_stat}' przeliczona na per 90:")
                        st.write(f"  - Suma statystyki: {total_stat:.4f}")
                        st.write(f"  - Suma minut (po usunięciu NA): {stat_minutes:.4f}")
                        st.write(f"  - Wartość per 90: {per_90_value:.4f}")
                        st.write(f"  - Klucz w słowniku: '{stat_key}'")
                else:
                    with debug_expander:
                        st.write(f"Statystyka '{base_stat}': Wszystkie wartości to NA, nie można obliczyć")
            else:
                with debug_expander:
                    st.write(f"Statystyka '{base_stat}' nie istnieje w danych")

    with debug_expander:
        st.write(f"### Wynik: Obliczono {len(team_stats_per_90)} statystyk drużynowych per 90 minut")
        st.write("Przykładowe wartości:")
        sample_stats = list(team_stats_per_90.items())[:5] if len(team_stats_per_90) > 5 else list(team_stats_per_90.items())
        for stat, value in sample_stats:
            st.write(f"  - {stat}: {value:.4f}")

    # Oblicz statystyki dla wszystkich drużyn dla wybranych statystyk
    with debug_expander:
        st.write("### Krok 2: Obliczanie statystyk dla wszystkich drużyn")
        st.write(f"Liczba drużyn w lidze: {len(teams)}")

    all_teams_data = {}

    for stat in stats_to_analyze:
        with debug_expander:
            st.write(f"\n#### Obliczanie statystyki '{stat}' dla wszystkich drużyn")

        # Ensure consistent key format for all_teams_data
        # Always store with the same format as it will be accessed later
        stat_key_for_all_teams = stat if 'per 90' in stat else f"{stat} per 90"
        all_teams_data[stat_key_for_all_teams] = {}
        # Use the correct key format based on whether the stat already has "per 90" in its name
        stat_key = stat if 'per 90' in stat else f"{stat} per 90"

        with debug_expander:
            st.write(f"Klucz używany do dostępu do team_stats_per_90: '{stat_key}'")

        if stat_key in team_stats_per_90:
            all_teams_data[stat_key_for_all_teams][selected_team] = team_stats_per_90[stat_key]

            with debug_expander:
                st.write(f"Wartość dla drużyny {selected_team}: {team_stats_per_90[stat_key]:.4f}")
        else:
            with debug_expander:
                st.write(f"⚠️ Klucz '{stat_key}' nie istnieje w team_stats_per_90! Obliczam wartość bezpośrednio.")

            # Oblicz wartość bezpośrednio dla wybranej drużyny, podobnie jak dla innych drużyn
            if 'per 90' in stat:
                # Filtrujemy dane, aby usunąć NA dla danej statystyki
                stat_data = team_data.dropna(subset=[stat])
                if not stat_data.empty:
                    weighted_stat = (stat_data[stat] * stat_data['Minutes played']).sum() / stat_data['Minutes played'].sum()
                    all_teams_data[stat_key_for_all_teams][selected_team] = weighted_stat

                    with debug_expander:
                        st.write(f"Obliczona wartość dla drużyny {selected_team}: {weighted_stat:.4f}")
                else:
                    teams_without_data = 0  # Resetujemy licznik, bo nie mamy danych dla wybranej drużyny
                    with debug_expander:
                        st.write(f"  - Drużyna {selected_team}: Wszystkie wartości to NA dla '{stat}'")
            else:
                base_stat = stat.replace(' per 90', '')
                if base_stat in team_data.columns:
                    # Filtrujemy dane, aby usunąć NA dla danej statystyki
                    stat_data = team_data.dropna(subset=[base_stat])
                    if not stat_data.empty:
                        total_stat = stat_data[base_stat].sum()
                        stat_minutes = stat_data['Minutes played'].sum()
                        per_90_value = (total_stat / stat_minutes) * 90
                        all_teams_data[stat_key_for_all_teams][selected_team] = per_90_value

                        with debug_expander:
                            st.write(f"Obliczona wartość dla drużyny {selected_team}: {per_90_value:.4f}")
                    else:
                        teams_without_data = 0  # Resetujemy licznik, bo nie mamy danych dla wybranej drużyny
                        with debug_expander:
                            st.write(f"  - Drużyna {selected_team}: Wszystkie wartości to NA dla '{base_stat}'")
                else:
                    teams_without_data = 0  # Resetujemy licznik, bo nie mamy danych dla wybranej drużyny
                    with debug_expander:
                        st.write(f"  - Drużyna {selected_team}: Statystyka '{base_stat}' nie istnieje w danych")

        teams_with_data = 1  # Już mamy wybraną drużynę
        teams_without_data = 0

        for team in teams:
            if team != selected_team:  # Już mamy dane dla wybranej drużyny
                team_df = df[df['Team within selected timeframe'] == team].copy()
                if not team_df.empty:
                    team_minutes = team_df['Minutes played'].sum()
                    if 'per 90' in stat:
                        # Filtrujemy dane, aby usunąć NA dla danej statystyki
                        stat_data = team_df.dropna(subset=[stat])
                        if not stat_data.empty:
                            weighted_stat = (stat_data[stat] * stat_data['Minutes played']).sum() / stat_data['Minutes played'].sum()
                            all_teams_data[stat_key_for_all_teams][team] = weighted_stat
                            teams_with_data += 1
                        else:
                            teams_without_data += 1
                            with debug_expander:
                                st.write(f"  - Drużyna {team}: Wszystkie wartości to NA dla '{stat}'")
                    else:
                        base_stat = stat.replace(' per 90', '')
                        if base_stat in team_df.columns:
                            # Filtrujemy dane, aby usunąć NA dla danej statystyki
                            stat_data = team_df.dropna(subset=[base_stat])
                            if not stat_data.empty:
                                total_stat = stat_data[base_stat].sum()
                                stat_minutes = stat_data['Minutes played'].sum()
                                per_90_value = (total_stat / stat_minutes) * 90
                                all_teams_data[stat_key_for_all_teams][team] = per_90_value
                                teams_with_data += 1
                            else:
                                teams_without_data += 1
                                with debug_expander:
                                    st.write(f"  - Drużyna {team}: Wszystkie wartości to NA dla '{base_stat}'")
                        else:
                            teams_without_data += 1
                            with debug_expander:
                                st.write(f"  - Drużyna {team}: Statystyka '{base_stat}' nie istnieje w danych")

        with debug_expander:
            st.write(f"Podsumowanie dla statystyki '{stat}':")
            st.write(f"  - Drużyny z danymi: {teams_with_data}")
            st.write(f"  - Drużyny bez danych: {teams_without_data}")
            st.write(f"  - Łącznie drużyn: {teams_with_data + teams_without_data}")

            # Pokaż przykładowe wartości
            if all_teams_data[stat_key_for_all_teams]:
                st.write("Przykładowe wartości:")
                sample_teams = list(all_teams_data[stat_key_for_all_teams].items())[:3] if len(all_teams_data[stat_key_for_all_teams]) > 3 else list(all_teams_data[stat_key_for_all_teams].items())
                for team, value in sample_teams:
                    st.write(f"  - {team}: {value:.4f}")
            else:
                st.write("⚠️ Brak danych dla tej statystyki dla wszystkich drużyn!")

    # Oblicz percentyle dla wybranych statystyk
    with debug_expander:
        st.write("### Krok 3: Obliczanie percentyli dla statystyk")

    percentiles = {}
    for stat in stats_to_analyze:
        with debug_expander:
            st.write(f"\n#### Obliczanie percentyla dla statystyki '{stat}'")

        # Use the correct key format for all_teams_data
        stat_key_for_all_teams = stat if 'per 90' in stat else f"{stat} per 90"

        if stat_key_for_all_teams in all_teams_data:
            stat_values = all_teams_data[stat_key_for_all_teams]

            with debug_expander:
                st.write(f"Liczba drużyn z danymi dla tej statystyki: {len(stat_values)}")

            if selected_team in stat_values:
                selected_team_value = stat_values[selected_team]
                better_teams = sum(1 for value in stat_values.values() if value > selected_team_value)
                equal_teams = sum(1 for value in stat_values.values() if value == selected_team_value) - 1  # Odejmujemy 1, żeby nie liczyć wybranej drużyny dwa razy
                worse_teams = sum(1 for value in stat_values.values() if value < selected_team_value)
                total_teams = len(stat_values)

                with debug_expander:
                    st.write(f"Wartość dla drużyny {selected_team}: {selected_team_value:.4f}")
                    st.write(f"Liczba drużyn lepszych: {better_teams}")
                    st.write(f"Liczba drużyn równych: {equal_teams}")
                    st.write(f"Liczba drużyn gorszych: {worse_teams}")
                    st.write(f"Suma kontrolna: {better_teams + equal_teams + worse_teams + 1} (powinna być równa {total_teams})")

                if total_teams > 0:
                    percentile = 100 - (better_teams / total_teams * 100)
                    percentiles[stat] = percentile

                    with debug_expander:
                        st.write(f"Obliczony percentyl: {percentile:.2f}%")
                        st.write(f"Formuła: 100 - ({better_teams} / {total_teams} * 100)")
                else:
                    percentiles[stat] = 0

                    with debug_expander:
                        st.write("⚠️ Brak drużyn z danymi, percentyl ustawiony na 0")
            else:
                with debug_expander:
                    st.write(f"⚠️ Drużyna {selected_team} nie ma wartości dla tej statystyki!")
        else:
            with debug_expander:
                st.write(f"⚠️ Brak danych dla statystyki '{stat_key_for_all_teams}' w all_teams_data!")

    with debug_expander:
        st.write("\n### Podsumowanie percentyli:")
        st.write(f"Obliczono percentyle dla {len(percentiles)} statystyk")
        if percentiles:
            st.write("Przykładowe percentyle:")
            sample_percentiles = list(percentiles.items())[:5] if len(percentiles) > 5 else list(percentiles.items())
            for stat, perc in sample_percentiles:
                st.write(f"  - {stat}: {perc:.2f}%")

    # Wyświetl statystyki drużynowe w formie tabeli z percentylami i pozycją w lidze
    st.subheader("Statystyki drużynowe per 90 minut")

    # Oblicz pozycję w lidze dla każdej statystyki
    with debug_expander:
        st.write("### Krok 4: Obliczanie pozycji w lidze dla statystyk")
        st.write(f"Liczba statystyk w team_stats_per_90: {len(team_stats_per_90)}")

    positions_in_league = {}
    for stat_key in team_stats_per_90.keys():
        with debug_expander:
            st.write(f"\n#### Obliczanie pozycji dla statystyki '{stat_key}'")

        # Use the correct key format for all_teams_data
        # The key in all_teams_data should always have "per 90" suffix
        stat_key_for_all_teams = stat_key if 'per 90' in stat_key else f"{stat_key} per 90"

        with debug_expander:
            st.write(f"Klucz w team_stats_per_90: '{stat_key}'")
            st.write(f"Odpowiadający klucz w all_teams_data: '{stat_key_for_all_teams}'")

        if stat_key_for_all_teams in all_teams_data:
            stat_values = all_teams_data[stat_key_for_all_teams]

            with debug_expander:
                st.write(f"Liczba drużyn z danymi dla tej statystyki: {len(stat_values)}")

            if selected_team in stat_values:
                # Sortuj drużyny według statystyki (od najlepszej do najgorszej)
                sorted_teams = sorted(stat_values.items(), key=lambda x: x[1], reverse=True)

                # Znajdź pozycję wybranej drużyny
                position = next(i+1 for i, (team, _) in enumerate(sorted_teams) if team == selected_team)
                positions_in_league[stat_key] = position

                with debug_expander:
                    st.write(f"Wartość dla drużyny {selected_team}: {stat_values[selected_team]:.4f}")
                    st.write(f"Pozycja w lidze: {position} z {len(sorted_teams)}")

                    # Pokaż kilka drużyn przed i po wybranej drużynie w rankingu
                    st.write("Fragment rankingu:")
                    start_idx = max(0, position - 3)
                    end_idx = min(len(sorted_teams), position + 2)
                    for i in range(start_idx, end_idx):
                        team, value = sorted_teams[i]
                        marker = "👉 " if team == selected_team else "   "
                        st.write(f"  {marker}{i+1}. {team}: {value:.4f}")
            else:
                with debug_expander:
                    st.write(f"⚠️ Drużyna {selected_team} nie ma wartości dla tej statystyki!")
        else:
            with debug_expander:
                st.write(f"⚠️ Brak danych dla statystyki '{stat_key_for_all_teams}' w all_teams_data!")

    with debug_expander:
        st.write("\n### Podsumowanie pozycji w lidze:")
        st.write(f"Obliczono pozycje dla {len(positions_in_league)} statystyk")
        if positions_in_league:
            st.write("Przykładowe pozycje:")
            sample_positions = list(positions_in_league.items())[:5] if len(positions_in_league) > 5 else list(positions_in_league.items())
            for stat, pos in sample_positions:
                st.write(f"  - {stat}: {pos}")

    with debug_expander:
        st.write("### Krok 5: Tworzenie tabeli z wynikami")
        st.write("Przygotowanie danych do tabeli:")
        st.write(f"Liczba statystyk w team_stats_per_90: {len(team_stats_per_90)}")
        st.write(f"Liczba percentyli: {len(percentiles)}")
        st.write(f"Liczba pozycji w lidze: {len(positions_in_league)}")

    # Przygotuj dane do tabeli z debugowaniem
    statystyki = list(team_stats_per_90.keys())
    wartosci = list(team_stats_per_90.values())
    percentyle = []
    pozycje = []

    with debug_expander:
        st.write("\n#### Mapowanie kluczy między słownikami:")

    for stat in team_stats_per_90.keys():
        # Dla percentyli - use the original stat name as it's used in percentiles dictionary
        # This is because we store percentiles with the original stat name from stats_to_analyze
        original_stat = stat.replace(" per 90", "") if " per 90" in stat else stat
        percentile_value = percentiles.get(original_stat, 0)
        percentyle.append(percentile_value)

        with debug_expander:
            st.write(f"Statystyka: '{stat}'")
            st.write(f"  - Klucz dla percentyli: '{original_stat}'")
            st.write(f"  - Wartość percentyla: {percentile_value:.2f}%")

        # Dla pozycji w lidze
        position_key = stat
        position_value = positions_in_league.get(position_key, "N/A")
        pozycje.append(position_value)

        with debug_expander:
            st.write(f"  - Klucz dla pozycji: '{position_key}'")
            st.write(f"  - Wartość pozycji: {position_value}")

    # Formatuj wartości liczbowe przed utworzeniem DataFrame
    wartosci_formatted = [f"{x:.2f}" for x in wartosci]
    percentyle_formatted = [f"{x:.1f}%" if x > 0 else "N/A" for x in percentyle]

    with debug_expander:
        st.write("\n#### Wartości po formatowaniu:")
        st.write(f"Wartości: {wartosci_formatted[:5]}...")
        st.write(f"Percentyle: {percentyle_formatted[:5]}...")

    # Utwórz DataFrame z już sformatowanymi wartościami
    stats_df = pd.DataFrame({
        'Statystyka': statystyki,
        'Wartość per 90 minut': wartosci_formatted,
        'Percentyl': percentyle_formatted,
        'Pozycja w lidze': pozycje
    })

    with debug_expander:
        st.write("\n#### DataFrame po formatowaniu:")
        st.dataframe(stats_df)
        st.write("\n#### Gotowa tabela zostanie wyświetlona poniżej")

    st.table(stats_df)

    # Wizualizacja statystyk drużynowych
    st.subheader("Wizualizacja statystyk drużynowych")

    # Wybór rodzaju wykresu
    chart_type = st.radio('Wybierz typ wykresu:', ['Wykres słupkowy', 'Radar Chart'])

    if chart_type == 'Wykres słupkowy':
        # Definicje szerokości kolumn i inicjalizacja tabeli HTML
        col_widths = {'stat': '200px', 'bar': '300px', 'value': '250px'}
        table_html = '<table style="width:100%; border-collapse: collapse;">'
        table_html += '<tr><th style="width:{}; text-align: left; border: 1px solid black;">Statystyka</th>'.format(
            col_widths['stat'])
        table_html += '<th style="width:{}; text-align: left; border: 1px solid black;">Wykres</th>'.format(
            col_widths['bar'])
        table_html += '<th style="width:{}; text-align: left; border: 1px solid black;">Wartość</th></tr>'.format(
            col_widths['value'])

        for stat in stats_to_analyze:
            # Use the correct key format based on whether the stat already has "per 90" in its name
            stat_key = stat if 'per 90' in stat else f"{stat} per 90"
            if stat in percentiles and stat_key in team_stats_per_90:
                stat_percentile = percentiles[stat]
                stat_value = team_stats_per_90[stat_key]
                position = positions_in_league.get(stat_key, len(teams))
                total_teams = len(teams)

                color = sns.color_palette('RdYlGn', as_cmap=True)(stat_percentile / 100)
                rgb_color = f'rgb({int(color[0] * 255)}, {int(color[1] * 255)}, {int(color[2] * 255)})'

                bar_html = f'<div style="width:{col_widths["bar"]}; background:lightgrey; border:1px solid black; height:20px; position:relative;">'
                bar_html += f'<div style="width:{stat_percentile}%; background:{rgb_color}; height:100%;"></div></div>'

                value_display = f'{stat_value:.2f} ({position}/{total_teams}, {stat_percentile:.1f}th percentile)'

                table_html += f'<tr><td style="width:{col_widths["stat"]}; border: 1px solid black;">{stat}</td>'
                table_html += f'<td style="width:{col_widths["bar"]}; border: 1px solid black;">{bar_html}</td>'
                table_html += f'<td style="width:{col_widths["value"]}; border: 1px solid black;">{value_display}</td></tr>'

        table_html += '</table>'
        st.markdown(table_html, unsafe_allow_html=True)

    elif chart_type == 'Radar Chart':
        # Wybieramy maksymalnie 25 statystyk dla czytelności wykresu
        if len(stats_to_analyze) > 25:
            st.warning('Wybrano zbyt wiele statystyk. Dla wykresu radarowego pokazanych zostanie pierwsze 25.')
            radar_stats = stats_to_analyze[:25]
        else:
            radar_stats = stats_to_analyze

        # Przygotowanie danych do wykresu radarowego
        radar_values = []
        hover_texts = []
        for stat in radar_stats:
            # Use the correct key format based on whether the stat already has "per 90" in its name
            stat_key = stat if 'per 90' in stat else f"{stat} per 90"
            if stat in percentiles and stat_key in team_stats_per_90:
                percentile = percentiles[stat]
                actual_value = team_stats_per_90[stat_key]
                radar_values.append(percentile)
                hover_texts.append(f"Wartość: {actual_value:.2f}<br>Percentyl: {percentile:.1f}%")
            else:
                radar_values.append(0)
                hover_texts.append("Brak danych")

        # Tworzenie wykresu radarowego
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=radar_values,
            theta=radar_stats,
            fill='toself',
            name=selected_team,
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            fillcolor='rgba(99, 110, 250, 0.2)',
            line=dict(color='rgb(99, 110, 250)')
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            height=600,
            width=800
        )

        st.plotly_chart(fig)

    # Moduł pokazujący pozycję w lidze dla wszystkich statystyk
    st.subheader("Pozycja w lidze dla wszystkich statystyk")

    # Przygotuj dane do wizualizacji
    position_data = []
    for stat_key in team_stats_per_90.keys():
        # Find the corresponding stat in all_teams_data
        original_stat = stat_key.replace(" per 90", "") if " per 90" in stat_key else stat_key
        if original_stat in all_teams_data and stat_key in positions_in_league:
            stat_values = all_teams_data[original_stat]
            if selected_team in stat_values:
                total_teams = len(stat_values)
                position = positions_in_league[stat_key]
                better_teams = position - 1
                worse_teams = total_teams - position
                percentile = percentiles.get(original_stat, 0)

                position_data.append({
                    'Statystyka': stat_key,
                    'Pozycja': position,
                    'Liczba drużyn lepszych': better_teams,
                    'Liczba drużyn gorszych': worse_teams,
                    'Percentyl': percentile,
                    'Łączna liczba drużyn': total_teams
                })

    if position_data:
        position_df = pd.DataFrame(position_data)

        # Sortuj według percentyla (od najwyższego do najniższego)
        position_df = position_df.sort_values('Percentyl', ascending=False)

        # Wykres słupkowy z pozycją w lidze i percentylem
        for i, row in position_df.iterrows():
            stat = row['Statystyka']
            position = row['Pozycja']
            better = row['Liczba drużyn lepszych']
            worse = row['Liczba drużyn gorszych']
            percentile = row['Percentyl']
            total = row['Łączna liczba drużyn']

            # Stwórz wykres słupkowy pokazujący pozycję w lidze
            fig_position = go.Figure()

            # Dodaj słupek dla drużyn lepszych
            if better > 0:
                fig_position.add_trace(go.Bar(
                    x=['Drużyny lepsze'],
                    y=[better],
                    name='Drużyny lepsze',
                    marker_color='rgba(246, 78, 139, 0.6)',
                    text=[better],
                    textposition='auto'
                ))

            # Dodaj słupek dla wybranej drużyny
            fig_position.add_trace(go.Bar(
                x=['Twoja drużyna'],
                y=[1],
                name='Twoja drużyna',
                marker_color='rgba(58, 71, 80, 0.6)',
                text=['Ty'],
                textposition='auto'
            ))

            # Dodaj słupek dla drużyn gorszych
            if worse > 0:
                fig_position.add_trace(go.Bar(
                    x=['Drużyny gorsze'],
                    y=[worse],
                    name='Drużyny gorsze',
                    marker_color='rgba(6, 147, 227, 0.6)',
                    text=[worse],
                    textposition='auto'
                ))

            fig_position.update_layout(
                title=f"{stat} - Pozycja: {position}/{total} (Percentyl: {percentile:.1f}%)",
                xaxis_title="",
                yaxis_title="Liczba drużyn",
                height=300,
                width=600,
                barmode='group'
            )

            st.plotly_chart(fig_position)
    else:
        st.warning("Brak danych do wyświetlenia pozycji w lidze.")

    # Wybór statystyki do szczegółowej analizy percentyla
    st.subheader("Szczegółowa analiza percentyla")

    # Wybór statystyki do szczegółowej analizy
    if stats_to_analyze:
        selected_stat_for_percentile = st.selectbox(
            "Wybierz statystykę do szczegółowej analizy percentyla:", 
            options=stats_to_analyze
        )

        # Find the corresponding stat in all_teams_data
        original_stat = selected_stat_for_percentile.replace(" per 90", "") if " per 90" in selected_stat_for_percentile else selected_stat_for_percentile
        if original_stat in all_teams_data:
            stat_values = all_teams_data[original_stat]
            if selected_team in stat_values:
                selected_team_value = stat_values[selected_team]
                better_teams = sum(1 for value in stat_values.values() if value > selected_team_value)
                worse_teams = sum(1 for value in stat_values.values() if value < selected_team_value)
                equal_teams = sum(1 for value in stat_values.values() if value == selected_team_value) - 1  # Odejmujemy 1, żeby nie liczyć wybranej drużyny dwa razy
                total_teams = len(stat_values)

                if total_teams > 0:
                    # Oblicz percentyl (im wyższy, tym lepiej)
                    percentile = 100 - (better_teams / total_teams * 100)

                    # Dodaj wykres wskaźnikowy (gauge) dla percentyla
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=percentile,
                        title={'text': f"Percentyl drużyny {selected_team} dla {selected_stat_for_percentile}"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "rgba(246, 78, 139, 0.6)"},
                            'steps': [
                                {'range': [0, 25], 'color': "lightgray"},
                                {'range': [25, 50], 'color': "gray"},
                                {'range': [50, 75], 'color': "lightblue"},
                                {'range': [75, 100], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': percentile
                            }
                        }
                    ))

                    fig_gauge.update_layout(
                        height=300,
                        width=600
                    )

                    st.plotly_chart(fig_gauge)

                    # Wyświetl dodatkowe informacje tekstowe
                    st.write(f"Percentyl: {percentile:.1f}% (im wyższy, tym lepiej)")
                    st.write(f"Liczba drużyn lepszych: {better_teams}")
                    st.write(f"Liczba drużyn gorszych: {worse_teams}")
                    st.write(f"Liczba drużyn z taką samą wartością: {equal_teams}")
                    st.write(f"Łączna liczba drużyn: {total_teams}")

                    # Stwórz DataFrame dla wszystkich drużyn
                    all_teams_df = pd.DataFrame({
                        'Drużyna': list(stat_values.keys()),
                        'Wartość': list(stat_values.values())
                    })

                    # Sortuj według wartości
                    all_teams_df = all_teams_df.sort_values('Wartość', ascending=False)

                    # Dodaj indeks rangowy (1-based) po sortowaniu
                    ranks = list(range(1, len(all_teams_df) + 1))

                    # Przygotuj kolory dla drużyn
                    colors = ['rgba(246, 78, 139, 0.6)' if team == selected_team else 'rgba(58, 71, 80, 0.6)' 
                              for team in all_teams_df['Drużyna']]

                    # Stwórz nowy DataFrame z wszystkimi kolumnami na raz, aby uniknąć fragmentacji
                    all_teams_df = pd.DataFrame({
                        'Drużyna': all_teams_df['Drużyna'].values,
                        'Wartość': all_teams_df['Wartość'].values,
                        'Ranga': ranks,
                        'Kolor': colors
                    })

                    # Stwórz wykres słupkowy z rankingiem
                    fig_ranking = px.bar(
                        all_teams_df,
                        x='Drużyna',
                        y='Wartość',
                        title=f"Ranking drużyn dla statystyki {selected_stat_for_percentile}",
                        labels={'Wartość': f'{selected_stat_for_percentile}', 'Drużyna': ''},
                        text=all_teams_df['Wartość'].apply(lambda x: f"{x:.2f}")
                    )

                    # Ustaw kolory słupków
                    fig_ranking.update_traces(marker_color=all_teams_df['Kolor'])

                    # Sprawdź, czy wybrana drużyna jest w DataFrame
                    if selected_team in all_teams_df['Drużyna'].values:
                        # Dodaj adnotację z rangą dla wybranej drużyny
                        selected_team_rank = all_teams_df[all_teams_df['Drużyna'] == selected_team]['Ranga'].values[0]
                        selected_team_index = all_teams_df[all_teams_df['Drużyna'] == selected_team].index[0]

                        fig_ranking.add_annotation(
                            x=selected_team_index,
                            y=all_teams_df.loc[selected_team_index, 'Wartość'],
                            text=f"Ranga: {selected_team_rank}",
                            showarrow=True,
                            arrowhead=1,
                            ax=0,
                            ay=-40
                        )

                    fig_ranking.update_layout(
                        height=500,
                        width=800,
                        xaxis_tickangle=-45
                    )

                    st.plotly_chart(fig_ranking)

    # Moduł pokazujący topowe i najsłabsze cechy drużyny
    st.subheader("Topowe i najsłabsze cechy drużyny")

    if position_data:
        # Używamy wcześniej przygotowanych danych z position_df
        # Sortuj według percentyla (od najwyższego do najniższego)
        top_features = position_df.sort_values('Percentyl', ascending=False).head(5)
        worst_features = position_df.sort_values('Percentyl', ascending=True).head(5)

        # Wyświetl topowe cechy
        st.write("### Topowe cechy drużyny")

        # Stwórz wykres słupkowy dla topowych cech
        fig_top = px.bar(
            top_features,
            x='Statystyka',
            y='Percentyl',
            title=f"Najlepsze cechy drużyny {selected_team}",
            labels={'Percentyl': 'Percentyl (%)', 'Statystyka': ''},
            text=top_features['Percentyl'].apply(lambda x: f"{x:.1f}%"),
            color='Percentyl',
            color_continuous_scale='Viridis'
        )

        fig_top.update_layout(
            xaxis_tickangle=-45,
            height=400,
            width=700
        )

        st.plotly_chart(fig_top)

        # Tabela z topowymi cechami
        top_table = top_features[['Statystyka', 'Pozycja', 'Łączna liczba drużyn', 'Percentyl']]
        top_table['Percentyl'] = top_table['Percentyl'].apply(lambda x: f"{x:.1f}%")
        st.table(top_table)

        # Wyświetl najsłabsze cechy
        st.write("### Najsłabsze cechy drużyny")

        # Stwórz wykres słupkowy dla najsłabszych cech
        fig_worst = px.bar(
            worst_features,
            x='Statystyka',
            y='Percentyl',
            title=f"Najsłabsze cechy drużyny {selected_team}",
            labels={'Percentyl': 'Percentyl (%)', 'Statystyka': ''},
            text=worst_features['Percentyl'].apply(lambda x: f"{x:.1f}%"),
            color='Percentyl',
            color_continuous_scale='Viridis'
        )

        fig_worst.update_layout(
            xaxis_tickangle=-45,
            height=400,
            width=700
        )

        st.plotly_chart(fig_worst)

        # Tabela z najsłabszymi cechami
        worst_table = worst_features[['Statystyka', 'Pozycja', 'Łączna liczba drużyn', 'Percentyl']]
        worst_table['Percentyl'] = worst_table['Percentyl'].apply(lambda x: f"{x:.1f}%")
        st.table(worst_table)
    else:
        st.warning("Brak danych do wyświetlenia topowych i najsłabszych cech.")
