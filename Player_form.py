import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import numpy as np
import re

def parse_match_data(match_text):
    """
    Parse match data from text in format "Team 1 - Team 2 Result"
    where Result is in format "Team1Goals : Team2Goals"

    Returns:
    - home_team: Name of the home team (Team 1)
    - away_team: Name of the away team (Team 2)
    - home_goals: Number of goals scored by the home team
    - away_goals: Number of goals scored by the away team
    - result_type: 'W' (win), 'L' (loss), or 'D' (draw) from perspective of home team
    """
    try:
        # Extract teams and result using regex
        match_pattern = r'(.+?)\s*-\s*(.+?)\s+(\d+)\s*:\s*(\d+)'
        match = re.search(match_pattern, match_text)

        if match:
            home_team = match.group(1).strip()
            away_team = match.group(2).strip()
            home_goals = int(match.group(3))
            away_goals = int(match.group(4))

            # Determine result type
            if home_goals > away_goals:
                result_type = 'W'  # Home team won
            elif home_goals < away_goals:
                result_type = 'L'  # Home team lost
            else:
                result_type = 'D'  # Draw

            return home_team, away_team, home_goals, away_goals, result_type
        else:
            return None, None, None, None, None
    except Exception:
        return None, None, None, None, None

def show_player_form():
    """
    Display and process player form data from a separate file.
    This mode uses a separate file upload and doesn't use data from other modes.
    """
    st.title('Analiza Formularzy Zawodników')

    # Add file uploader specific to this mode
    uploaded_file = st.file_uploader("Wybierz plik Excel z danymi formularzy zawodników", 
                                     type=['xlsx', 'xls'], 
                                     key="player_form_uploader")

    if uploaded_file is None:
        st.warning("Proszę wybrać plik z danymi formularzy zawodników.")
        return

    try:
        # Load data from the uploaded file
        df = pd.read_excel(uploaded_file).fillna(0)

        # Wczytaj nazwy kolumn z pliku Kolumny2.xlsx
        try:
            # Flag to track if column names have been successfully replaced
            columns_replaced = False

            # Try to read the Excel file with default parameters
            try:
                kolumny_df = pd.read_excel("Kolumny2.xlsx")

                # Jeśli plik Kolumny2.xlsx ma kolumny, użyj ich do nadpisania nazw kolumn w df
                if not kolumny_df.empty and len(kolumny_df.columns) > 0:
                    # Sprawdź czy liczba kolumn się zgadza
                    if len(df.columns) == len(kolumny_df.columns):
                        # Get the first row of kolumny_df if it contains the actual column names
                        if kolumny_df.shape[0] > 0:
                            # Check if the first row contains the actual column names
                            first_row = kolumny_df.iloc[0].tolist()

                            # If the first row contains strings and not all are NaN, use it as column names
                            if all(isinstance(x, str) for x in first_row if pd.notna(x)) and any(pd.notna(x) for x in first_row):
                                df.columns = first_row
                                columns_replaced = True
                            else:
                                # Use the column headers if they're not unnamed
                                column_headers = kolumny_df.columns.tolist()
                                if not all('Unnamed' in str(header) for header in column_headers):
                                    df.columns = column_headers
                                    columns_replaced = True
                        else:
                            # Use the column names from kolumny_df if they're not unnamed
                            column_headers = kolumny_df.columns.tolist()
                            if not all('Unnamed' in str(header) for header in column_headers):
                                df.columns = column_headers
                                columns_replaced = True
            except Exception:
                pass

            # If columns haven't been replaced yet, try with header=None
            if not columns_replaced:
                try:
                    kolumny_df_no_header = pd.read_excel("Kolumny2.xlsx", header=None)

                    if kolumny_df_no_header.shape[0] > 0 and kolumny_df_no_header.shape[1] == len(df.columns):
                        first_row_no_header = kolumny_df_no_header.iloc[0].tolist()

                        # Check if first row contains valid column names
                        if all(isinstance(x, str) for x in first_row_no_header if pd.notna(x)) and any(pd.notna(x) for x in first_row_no_header):
                            df.columns = first_row_no_header
                            columns_replaced = True
                except Exception:
                    pass

            # If columns haven't been replaced yet, try with skiprows=1
            if not columns_replaced:
                try:
                    kolumny_df_skiprows = pd.read_excel("Kolumny2.xlsx", skiprows=1)

                    if len(kolumny_df_skiprows.columns) == len(df.columns):
                        # Use the column headers (which should be the first row of data)
                        column_headers_skiprows = kolumny_df_skiprows.columns.tolist()
                        if not all('Unnamed' in str(header) for header in column_headers_skiprows):
                            df.columns = column_headers_skiprows
                            columns_replaced = True
                except Exception:
                    pass

            # If columns haven't been replaced yet, try with xlrd
            if not columns_replaced:
                try:
                    import xlrd
                    # Open the workbook
                    workbook = xlrd.open_workbook("Kolumny2.xlsx")
                    # Get the first sheet
                    sheet = workbook.sheet_by_index(0)
                    # Read the first row
                    if sheet.nrows > 0:
                        first_row_xlrd = sheet.row_values(0)

                        if len(first_row_xlrd) == len(df.columns):
                            # Check if names are valid
                            if all(isinstance(x, str) for x in first_row_xlrd if x) and any(x for x in first_row_xlrd):
                                df.columns = first_row_xlrd
                                columns_replaced = True
                except Exception:
                    pass

            if not columns_replaced:
                st.warning("Nie udało się nadpisać nazw kolumn. Używam oryginalnych nazw kolumn.")
        except Exception as e:
            st.warning(f"Nie udało się wczytać pliku Kolumny2.xlsx: {str(e)}. Nazwy kolumn nie zostały nadpisane.")


        # Identify columns for date, match, and result
        date_column = None
        match_column = None
        result_column = None

        # Check for date column
        date_column_candidates = ['Data', 'Date', 'Datum', 'DataMeczu', 'MatchDate']
        for col in date_column_candidates:
            if col in df.columns:
                date_column = col
                break

        # Check for match column
        match_column_candidates = ['Mecz', 'Match', 'Przeciwnik', 'Opponent', 'Rywal', 'Spotkanie']
        for col in match_column_candidates:
            if col in df.columns:
                match_column = col
                break

        # Check for result column
        result_column_candidates = ['Wynik', 'Result', 'Score', 'Rezultat']
        for col in result_column_candidates:
            if col in df.columns:
                result_column = col
                break

        # Allow user to select columns if not automatically identified
        st.subheader('Identyfikacja kolumn')

        if date_column:
            st.write(f"Automatycznie zidentyfikowano kolumnę daty: '{date_column}'")
        else:
            date_column = st.selectbox(
                'Wybierz kolumnę zawierającą datę meczu:',
                options=df.columns.tolist(),
                index=0 if len(df.columns) > 0 else None
            )

        if match_column:
            st.write(f"Automatycznie zidentyfikowano kolumnę meczu: '{match_column}'")
        else:
            match_column = st.selectbox(
                'Wybierz kolumnę zawierającą informację o meczu/przeciwniku:',
                options=df.columns.tolist(),
                index=0 if len(df.columns) > 0 else None
            )

        if result_column:
            st.write(f"Automatycznie zidentyfikowano kolumnę wyniku: '{result_column}'")
        else:
            result_column = st.selectbox(
                'Wybierz kolumnę zawierającą wynik meczu:',
                options=df.columns.tolist(),
                index=0 if len(df.columns) > 0 else None
            )

        # Check if all required columns are identified
        if not all([date_column, match_column]):
            st.warning("Nie wszystkie wymagane kolumny zostały zidentyfikowane. Wykres nie może być wygenerowany.")
            return

        # Parse match data to extract teams and results
        st.subheader('Analiza danych meczowych')

        # Create new columns for parsed match data
        if match_column in df.columns:
            # Extract unique team names from the match column
            all_teams = set()

            # Process each match entry
            home_teams = []
            away_teams = []
            home_goals = []
            away_goals = []
            result_types = []

            for match_text in df[match_column]:
                home_team, away_team, h_goals, a_goals, result_type = parse_match_data(str(match_text))

                home_teams.append(home_team)
                away_teams.append(away_team)
                home_goals.append(h_goals)
                away_goals.append(a_goals)
                result_types.append(result_type)

                # Add teams to the set of all teams
                if home_team:
                    all_teams.add(home_team)
                if away_team:
                    all_teams.add(away_team)

            # Add the parsed data as new columns
            df['Home_Team'] = home_teams
            df['Away_Team'] = away_teams
            df['Home_Goals'] = home_goals
            df['Away_Goals'] = away_goals
            df['Result_Type'] = result_types

            # Let user select which team the player belongs to
            player_team = st.selectbox(
                'Wybierz drużynę zawodnika:',
                options=sorted(list(all_teams)),
                index=0 if len(all_teams) > 0 else None
            )

            # Add a column to indicate if the player's team is playing at home or away
            if player_team:
                df['Is_Home'] = df['Home_Team'] == player_team
                df['Is_Away'] = df['Away_Team'] == player_team

                # Add a column for player's team goals and opponent goals
                df['Team_Goals'] = df.apply(
                    lambda row: row['Home_Goals'] if row['Is_Home'] else row['Away_Goals'] if row['Is_Away'] else None, 
                    axis=1
                )
                df['Opponent_Goals'] = df.apply(
                    lambda row: row['Away_Goals'] if row['Is_Home'] else row['Home_Goals'] if row['Is_Away'] else None, 
                    axis=1
                )

                # Add a column for match result from player's team perspective
                df['Player_Result'] = df.apply(
                    lambda row: row['Result_Type'] if row['Is_Home'] else 
                                ('W' if row['Result_Type'] == 'L' else 
                                 'L' if row['Result_Type'] == 'W' else 'D') if row['Is_Away'] else None,
                    axis=1
                )

            else:
                st.warning("Nie można określić drużyny zawodnika. Wybierz drużynę z listy.")
                return

        # Ensure date column is in datetime format
        try:
            df[date_column] = pd.to_datetime(df[date_column])
        except:
            st.warning(f"Nie można przekonwertować kolumny '{date_column}' na format daty. Wykres może nie działać poprawnie.")

        # Create a section for the line chart
        st.subheader('Wykres liniowy statystyk w czasie')

        # Get all numeric columns for statistics selection
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove any columns that are not statistics (like ID, etc.)
        exclude_columns = ['ID', 'id', 'Nr', 'nr']
        stat_columns = [col for col in numeric_columns if col not in exclude_columns]

        # Filter for selecting statistics (max 5)
        selected_stats = st.multiselect(
            'Wybierz statystyki do wyświetlenia (max 5):',
            options=stat_columns,
            default=stat_columns[:min(5, len(stat_columns))]
        )

        # Limit to max 5 statistics
        if len(selected_stats) > 5:
            st.warning("Można wybrać maksymalnie 5 statystyk. Wyświetlam pierwsze 5 wybranych.")
            selected_stats = selected_stats[:5]

        # Filter for time range
        min_date = df[date_column].min().date()
        max_date = df[date_column].max().date()
        date_range = st.date_input(
            "Wybierz zakres dat:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        # Handle single date selection
        if isinstance(date_range, datetime.date):
            start_date, end_date = date_range, date_range
        else:
            start_date, end_date = date_range

        # Filter dataframe by date range
        df_filtered = df[(df[date_column].dt.date >= start_date) & (df[date_column].dt.date <= end_date)]

        # Filter for match results (win/loss/draw) from player's perspective
        # Use the Player_Result column instead of the original result column
        if 'Player_Result' in df.columns:
            result_types = ['W', 'L', 'D']  # Win, Loss, Draw
            result_labels = {
                'W': 'Wygrane',
                'L': 'Przegrane',
                'D': 'Remisy'
            }
            selected_results = st.multiselect(
                'Filtruj wyniki meczów:',
                options=[result_labels[rt] for rt in result_types],
                default=[result_labels[rt] for rt in result_types]
            )

            # Convert selected labels back to codes
            selected_codes = [rt for rt in result_types if result_labels[rt] in selected_results]

            if selected_codes:
                df_filtered = df_filtered[df_filtered['Player_Result'].isin(selected_codes)]
        elif result_column:
            # Fallback to original result column if Player_Result is not available
            result_types = df[result_column].unique().tolist()
            selected_results = st.multiselect(
                'Filtruj wyniki meczów:',
                options=result_types,
                default=result_types
            )
            if selected_results:
                df_filtered = df_filtered[df_filtered[result_column].isin(selected_results)]

        # Filter for home/away matches using the parsed data
        if 'Is_Home' in df.columns and 'Is_Away' in df.columns:
            match_locations = ['Dom', 'Wyjazd']
            selected_locations = st.multiselect(
                'Filtruj mecze (dom/wyjazd):',
                options=match_locations,
                default=match_locations
            )

            # Apply the filter
            if selected_locations:
                if 'Dom' in selected_locations and 'Wyjazd' in selected_locations:
                    # Both selected, no filtering needed
                    pass
                elif 'Dom' in selected_locations:
                    # Only home matches
                    df_filtered = df_filtered[df_filtered['Is_Home']]
                elif 'Wyjazd' in selected_locations:
                    # Only away matches
                    df_filtered = df_filtered[df_filtered['Is_Away']]
        else:
            # Fallback to original location column if parsed data is not available
            location_column = None
            location_column_candidates = ['Miejsce', 'Location', 'Venue', 'Stadium', 'HomeAway']
            for col in location_column_candidates:
                if col in df.columns:
                    location_column = col
                    break

            # Filter for home/away matches
            if location_column:
                match_locations = df[location_column].unique().tolist()
                selected_locations = st.multiselect(
                    'Filtruj mecze (dom/wyjazd):',
                    options=match_locations,
                    default=match_locations
                )
                if selected_locations:
                    df_filtered = df_filtered[df_filtered[location_column].isin(selected_locations)]

        # Create the line chart if there are selected statistics
        if selected_stats and not df_filtered.empty:
            # Create a single figure for all selected statistics
            import plotly.graph_objects as go
            fig = go.Figure()

            # Add a trace for each selected statistic
            for stat in selected_stats:
                # Add match information as hover text
                hover_data = []
                for i, row in df_filtered.iterrows():
                    # Use parsed match data if available
                    if all(col in row for col in ['Home_Team', 'Away_Team', 'Team_Goals', 'Opponent_Goals', 'Player_Result']):
                        # Determine if it's a home or away match
                        location = "Dom" if row['Is_Home'] else "Wyjazd" if row['Is_Away'] else "Nieznane"

                        # Format the result with team names and scores
                        if row['Is_Home']:
                            opponent = row['Away_Team']
                            match_display = f"{row['Home_Team']} - {row['Away_Team']} {int(row['Home_Goals'])}:{int(row['Away_Goals'])}"
                        elif row['Is_Away']:
                            opponent = row['Home_Team']
                            match_display = f"{row['Home_Team']} - {row['Away_Team']} {int(row['Home_Goals'])}:{int(row['Away_Goals'])}"
                        else:
                            opponent = "Nieznane"
                            match_display = row[match_column]

                        # Format the result type
                        result_label = {
                            'W': 'Wygrana',
                            'L': 'Przegrana',
                            'D': 'Remis'
                        }.get(row['Player_Result'], 'Nieznany')

                        match_info = (
                            f"Statystyka: {stat}<br>"
                            f"Mecz: {match_display}<br>"
                            f"Przeciwnik: {opponent}<br>"
                            f"Wynik: {int(row['Team_Goals'])}:{int(row['Opponent_Goals'])} ({result_label})<br>"
                            f"Miejsce: {location}"
                        )
                    else:
                        # Fallback to original data
                        match_info = f"Statystyka: {stat}<br>Mecz: {row[match_column]}"
                        if result_column in row:
                            match_info += f", Wynik: {row[result_column]}"
                        if location_column and location_column in row:
                            match_info += f", Miejsce: {row[location_column]}"

                    hover_data.append(match_info)

                # Add a trace for this statistic
                fig.add_trace(go.Scatter(
                    x=df_filtered[date_column],
                    y=df_filtered[stat],
                    mode='lines+markers',
                    name=stat,
                    text=hover_data,
                    hovertemplate=f'<b>{date_column}:</b> %{{x}}<br><b>Wartość:</b> %{{y}}<br>%{{text}}<extra></extra>'
                ))

            # Customize the layout for the combined chart
            fig.update_layout(
                title='Statystyki w czasie',
                xaxis_title=date_column,
                yaxis_title='Wartość',
                hovermode='closest',
                legend=dict(
                    title='Statystyki',
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                )
            )

            # Display the chart with all statistics
            st.plotly_chart(fig)
        elif not selected_stats:
            st.warning("Wybierz przynajmniej jedną statystykę do wyświetlenia.")
        elif df_filtered.empty:
            st.warning("Brak danych spełniających wybrane kryteria filtrowania.")

    except Exception as e:
        st.error(f"Błąd podczas przetwarzania danych: {str(e)}")
