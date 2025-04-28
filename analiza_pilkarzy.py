def normalize_data(filtered_df, selected_stats):
    """Normalizuje wybrane statystyki i oblicza percentyle."""
    if filtered_df.empty:
        return filtered_df
    
    # Tworzymy kopię DataFrame
    normalized_df = filtered_df.copy()
    
    # Dla każdej wybranej statystyki
    for stat in selected_stats:
        # Tworzymy ranking tylko dla zawodników z wartością > 0
        non_zero_mask = filtered_df[stat] > 0
        if non_zero_mask.any():
            # Obliczamy ranking (1 dla najwyższej wartości)
            normalized_df.loc[non_zero_mask, f'{stat}_rank'] = filtered_df[non_zero_mask][stat].rank(ascending=False, method='min')
            
            # Obliczamy percentyl jako (pozycja / liczba zawodników z wartością > 0) * 100
            total_non_zero = non_zero_mask.sum()
            normalized_df.loc[non_zero_mask, f'{stat}_percentile'] = (normalized_df.loc[non_zero_mask, f'{stat}_rank'] / total_non_zero) * 100
            
            # Dla zawodników z wartością 0 ustawiamy rank i percentyl na 0
            normalized_df.loc[~non_zero_mask, f'{stat}_rank'] = 0
            normalized_df.loc[~non_zero_mask, f'{stat}_percentile'] = 0
        else:
            # Jeśli wszyscy mają 0, ustawiamy ranking i percentyl na 0
            normalized_df[f'{stat}_rank'] = 0
            normalized_df[f'{stat}_percentile'] = 0
    
    return normalized_df 