def normalize_data(filtered_df, selected_stats):
    """Normalizuje wybrane statystyki i oblicza percentyle."""
    if filtered_df.empty:
        return filtered_df

    # Tworzymy kopię DataFrame
    normalized_df = filtered_df.copy()

    # Przygotowujemy słowniki do przechowywania danych dla nowych kolumn
    rank_data = {}
    percentile_data = {}

    # Dla każdej wybranej statystyki
    for stat in selected_stats:
        # Tworzymy ranking tylko dla zawodników z wartością > 0
        non_zero_mask = filtered_df[stat] > 0

        # Inicjalizujemy kolumny zerami
        rank_data[f'{stat}_rank'] = [0] * len(filtered_df)
        percentile_data[f'{stat}_percentile'] = [0] * len(filtered_df)

        if non_zero_mask.any():
            # Obliczamy ranking (1 dla najwyższej wartości)
            ranks = filtered_df[non_zero_mask][stat].rank(ascending=False, method='min')

            # Obliczamy percentyl jako (pozycja / liczba zawodników z wartością > 0) * 100
            total_non_zero = non_zero_mask.sum()
            percentiles = (ranks / total_non_zero) * 100

            # Zapisujemy wartości do słowników
            for i, (idx, is_non_zero) in enumerate(zip(filtered_df.index, non_zero_mask)):
                if is_non_zero:
                    rank_idx = ranks.index.get_loc(idx)
                    rank_data[f'{stat}_rank'][i] = ranks.iloc[rank_idx]
                    percentile_data[f'{stat}_percentile'][i] = percentiles.iloc[rank_idx]

    # Tworzymy nowe DataFrame z obliczonymi wartościami
    import pandas as pd
    rank_df = pd.DataFrame(rank_data, index=filtered_df.index)
    percentile_df = pd.DataFrame(percentile_data, index=filtered_df.index)

    # Łączymy wszystkie DataFrame za pomocą pd.concat
    result_df = pd.concat([normalized_df, rank_df, percentile_df], axis=1)

    return result_df
