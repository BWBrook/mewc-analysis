import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import haversine_distances
from common import load_config

def ensure_directory(path):
    """Ensure that the directory exists; create it if it does not."""
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {path}")

def load_data(config):
    """Load observation and site data from CSV files."""
    # Paths to data files
    obs_table_path = Path(config['output_table']).with_suffix('.csv')  # Add .csv extension
    site_table_path = Path(config['site_table'])
    
    # Read in the observation table
    if not obs_table_path.exists():
        raise FileNotFoundError(f"Observation table not found at {obs_table_path}")
    obs_df = pd.read_csv(obs_table_path)
    
    # Read in the site attributes table
    if not site_table_path.exists():
        raise FileNotFoundError(f"Site attributes table not found at {site_table_path}")
    site_df = pd.read_csv(site_table_path)
    
    # Convert timestamp to datetime
    obs_df['timestamp'] = pd.to_datetime(obs_df['timestamp'], format='%d/%m/%Y %H:%M:%S')
    
    # Extract date from timestamp
    obs_df['date'] = obs_df['timestamp'].dt.date
    
    return obs_df, site_df

def create_species_site_occupancy(obs_df, prefix, results_path):
    """Create and save a species x site occupancy table."""
    # Get presence of species at each site
    presence_df = obs_df[['camera_site', 'class_name']].drop_duplicates()
    presence_df['presence'] = 1
    
    # Pivot to create occupancy table
    occupancy_table = presence_df.pivot(index='class_name', columns='camera_site', values='presence').fillna(0).astype(int)
    
    # Define output path
    output_file = Path(results_path) / f"{prefix}_species_site_occupancy.csv"
    
    # Save to CSV
    occupancy_table.to_csv(output_file)
    print(f"Species x Site Occupancy table saved to {output_file}")
    return occupancy_table

def create_species_site_abundance_count(obs_df, prefix, results_path):
    """Create and save a species x site abundance table based on count."""
    # Sum counts per species per site
    abundance_df = obs_df.groupby(['class_name', 'camera_site'])['count'].sum().reset_index()
    
    # Pivot to create abundance table
    abundance_table = abundance_df.pivot(index='class_name', columns='camera_site', values='count').fillna(0).astype(int)
    
    # Define output path
    output_file = Path(results_path) / f"{prefix}_species_site_abundance_count.csv"
    
    # Save to CSV
    abundance_table.to_csv(output_file)
    print(f"Species x Site Abundance (Count) table saved to {output_file}")
    return abundance_table

def create_species_site_abundance_detection_rate(obs_df, site_df, prefix, results_path):
    """Create and save a species x site abundance table based on detection rate."""
    # Get number of unique days each species was detected at each site
    species_site_days = obs_df.groupby(['class_name', 'camera_site'])['date'].nunique().reset_index()
    species_site_days.rename(columns={'date': 'days_detected'}, inplace=True)
    
    # Merge with site operational days
    species_site_days = species_site_days.merge(site_df[['camera_site', 'op_days']], on='camera_site', how='left')
    
    # Handle potential division by zero or missing op_days
    species_site_days['op_days'].replace(0, np.nan, inplace=True)
    
    # Calculate detection rate
    species_site_days['detection_rate'] = species_site_days['days_detected'] / species_site_days['op_days']
    
    # Replace NaN detection rates with 0 (if op_days was 0 or missing)
    species_site_days['detection_rate'] = species_site_days['detection_rate'].fillna(0)
    
    # Pivot to create detection rate table
    detection_rate_table = species_site_days.pivot(index='class_name', columns='camera_site', values='detection_rate').fillna(0)
    
    # Define output path
    output_file = Path(results_path) / f"{prefix}_species_site_detection_rate.csv"
    
    # Save detection rate table to CSV
    detection_rate_table.to_csv(output_file)
    print(f"Species x Site Detection Rate table saved to {output_file}")
    return detection_rate_table

def create_site_distance_matrix(site_df, prefix, results_path):
    """Create and save a site x site distance matrix in kilometers."""
    # Extract site coordinates
    coords_df = site_df[['camera_site', 'lat', 'lon']].set_index('camera_site')
    
    # Convert latitudes and longitudes to radians
    coords_df['lat_rad'] = np.radians(coords_df['lat'])
    coords_df['lon_rad'] = np.radians(coords_df['lon'])
    
    # Create array of coordinates in radians
    coords = coords_df[['lat_rad', 'lon_rad']].values
    
    # Calculate pairwise haversine distances (in radians)
    pairwise_dist_rad = haversine_distances(coords)
    
    # Convert distances from radians to kilometers
    earth_radius_km = 6371  # Earth's radius in kilometers
    pairwise_dist_km = pairwise_dist_rad * earth_radius_km
    
    # Create distance matrix DataFrame
    distance_matrix = pd.DataFrame(pairwise_dist_km, index=coords_df.index, columns=coords_df.index)
    
    # Define output path
    output_file = Path(results_path) / f"{prefix}_site_distance_matrix_km.csv"
    
    # Save distance matrix to CSV
    distance_matrix.to_csv(output_file)
    print(f"Site x Site Distance Matrix saved to {output_file}")
    return distance_matrix

def compute_diversity_metrics(abundance_df):
    """Compute diversity metrics for each site."""
    # Pivot abundance data to have sites as rows and species as columns
    abundance_pivot = abundance_df.pivot(index='camera_site', columns='class_name', values='count').fillna(0)
    
    # Define function to compute diversity metrics
    def compute_metrics(counts):
        counts = counts[counts > 0]
        S_obs = len(counts)
        total_count = counts.sum()
        proportions = counts / total_count
        F1 = sum(counts == 1)
        F2 = sum(counts == 2)
        
        # Chao1 estimator
        if F2 > 0:
            Chao1 = S_obs + (F1 ** 2) / (2 * F2)
        else:
            Chao1 = S_obs + (F1 * (F1 - 1)) / (2 * (F2 + 1))  # Adjusted formula to avoid division by zero
        
        # Simpson's Diversity Index
        D = sum(proportions ** 2)
        Simpson_diversity = 1 - D
    
        # Shannon entropy
        H = -sum(proportions * np.log(proportions))
    
        # Pielou's Evenness
        if S_obs > 1:
            Evenness = H / np.log(S_obs)
        else:
            Evenness = np.nan  # Evenness is undefined when only one species is present
    
        return pd.Series({
            'observed_richness': S_obs,
            'Chao1': Chao1,
            'Simpson_diversity': Simpson_diversity,
            'Shannon_entropy': H,
            'Pielou_evenness': Evenness
        })
    
    # Apply the function to each site
    diversity_metrics = abundance_pivot.apply(compute_metrics, axis=1)
    return diversity_metrics

def save_diversity_metrics(diversity_metrics, prefix, results_path):
    """Save diversity metrics to a CSV file."""
    # Define output path
    output_file = Path(results_path) / f"{prefix}_site_diversity_metrics.csv"
    
    # Save to CSV
    diversity_metrics.to_csv(output_file)
    print(f"Diversity Metrics table saved to {output_file}")

def main():
    """Main function to execute all tasks."""
    # Load configuration
    config = load_config()
    
    # Extract configuration parameters
    prefix = config.get('prefix', 'output')  # Default to 'output' if not specified
    results_path = config.get('results_path', 'results')  # Default to 'results' if not specified
    
    # Ensure the results directory exists
    ensure_directory(results_path)
    
    # Load data
    obs_df, site_df = load_data(config)
    
    # Create and save Species x Site Occupancy Table
    occupancy_table = create_species_site_occupancy(obs_df, prefix, results_path)
    
    # Create and save Species x Site Abundance Table (Count)
    abundance_count_table = create_species_site_abundance_count(obs_df, prefix, results_path)
    
    # Create and save Species x Site Abundance Table (Detection Rate)
    abundance_detection_rate_table = create_species_site_abundance_detection_rate(obs_df, site_df, prefix, results_path)
    
    # Create and save Site x Site Distance Matrix
    distance_matrix = create_site_distance_matrix(site_df, prefix, results_path)
    
    # Compute Diversity Metrics
    diversity_metrics = compute_diversity_metrics(abundance_count_table)
    
    # Save Diversity Metrics
    save_diversity_metrics(diversity_metrics, prefix, results_path)
    
    # Optional: Print summaries for verification
    print("\n--- Summary of Outputs ---")
    print(f"Occupancy Table shape: {occupancy_table.shape}")
    print(f"Abundance Count Table shape: {abundance_count_table.shape}")
    print(f"Detection Rate Table shape: {abundance_detection_rate_table.shape}")
    print(f"Distance Matrix shape: {distance_matrix.shape}")
    print(f"Diversity Metrics shape: {diversity_metrics.shape}")
    print("--- End of Summary ---\n")

if __name__ == "__main__":
    main()

# This should instead be part of a new mewc-analysis GitHub repository