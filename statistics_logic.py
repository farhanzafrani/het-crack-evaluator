import pandas as pd
import numpy as np
from scipy.stats import norm

def filter_data(df, specs):
    """
    Replicates f_app_eval_data.m logic for filtering.
    """
    filtered_df = df.copy()
    
    # Material Filtering (Single)
    if specs.get('material'):
        filtered_df = filtered_df[filtered_df['material'] == specs['material']]
        
    # Multi-selection filters
    for key in ['supplier', 'clearance', 'timeStampMeas']:
        if specs.get(key) and 'Keine Auswahl' not in specs[key]:
            filtered_df = filtered_df[filtered_df[key].isin(specs[key])]
            
    # Thickness Range
    if 'minthick' in specs:
        filtered_df = filtered_df[filtered_df['thick'] >= specs['minthick']]
    if 'maxthick' in specs:
        filtered_df = filtered_df[filtered_df['thick'] <= specs['maxthick']]
        
    # Min Specimens Per Experiment
    if 'min_n_spec' in specs:
        # Each row is an experiment with a list of measurements
        filtered_df = filtered_df[filtered_df['measdata_HET'].apply(len) >= specs['min_n_spec']]
        
    # Experiment list selection (by LabProt)
    if specs.get('selected_labs'):
        filtered_df = filtered_df[filtered_df['LabProt'].isin(specs['selected_labs'])]
        
    return filtered_df

def calculate_quantiles(df, evaluation_type, q_level):
    """
    Calculates direct and Gaussian quantiles for each experiment and globally.
    """
    if df.empty:
        return {}
    
    data_col = 'measdata_HET' if evaluation_type == 'Hole Expansion Coefficient' else 'measdata_Strain'
    
    # 1. Per Experiment Quantiles
    results = []
    all_measurements = []
    
    for _, row in df.iterrows():
        measurements = np.array([m for m in row[data_col] if not np.isnan(m)])
        if len(measurements) == 0:
            continue
            
        all_measurements.extend(measurements)
        
        # Direct
        q_direct = np.quantile(measurements, q_level)
        
        # Gaussian
        mu, sigma = norm.fit(measurements)
        q_gauss = norm.ppf(q_level, mu, sigma)
        
        results.append({
            'LabProt': row['LabProt'],
            'q_direct': q_direct,
            'q_gauss': q_gauss,
            'n': len(measurements)
        })
        
    # 2. Global Quantiles
    all_measurements = np.array(all_measurements)
    global_stats = {}
    if len(all_measurements) > 0:
        mu_g, sigma_g = norm.fit(all_measurements)
        global_stats = {
            'q_direct': np.quantile(all_measurements, q_level),
            'q_gauss': norm.ppf(q_level, mu_g, sigma_g),
            'mu': mu_g,
            'sigma': sigma_g,
            'total_n': len(all_measurements)
        }
        
    return {
        'per_experiment': results,
        'global': global_stats
    }

def convert_threshold(value, from_type):
    """
    Converts threshold value between HEC (%) and Major Strain.
    HEC -> Strain: ln(1 + HEC/100)
    Strain -> HEC: 100 * (exp(Strain) - 1)
    """
    if from_type == 'Hole Expansion Coefficient':
        return 100 * (np.exp(value) - 1)
    else:
        return np.log(1 + value/100)
