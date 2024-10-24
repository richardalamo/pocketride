import pandas as pd
import matplotlib.pyplot as plt
import os
import urllib.request
from sklearn.base import BaseEstimator, TransformerMixin

tripdata_files = {
    'jan_2022': 'fhvhv_tripdata_2022-01.parquet',
    'feb_2022': 'fhvhv_tripdata_2022-02.parquet',
    'mar_2022': 'fhvhv_tripdata_2022-03.parquet',
    'apr_2022': 'fhvhv_tripdata_2022-04.parquet',
    'may_2022': 'fhvhv_tripdata_2022-05.parquet',
    'jun_2022': 'fhvhv_tripdata_2022-06.parquet',
    'jul_2022': 'fhvhv_tripdata_2022-07.parquet',
    'aug_2022': 'fhvhv_tripdata_2022-08.parquet',
    'sep_2022': 'fhvhv_tripdata_2022-09.parquet',
    'oct_2022': 'fhvhv_tripdata_2022-10.parquet',
    'nov_2022': 'fhvhv_tripdata_2022-11.parquet',
    'dec_2022': 'fhvhv_tripdata_2022-12.parquet',
    'jan_2023': 'fhvhv_tripdata_2023-01.parquet',
    'feb_2023': 'fhvhv_tripdata_2023-02.parquet',
    'mar_2023': 'fhvhv_tripdata_2023-03.parquet',
    'apr_2023': 'fhvhv_tripdata_2023-04.parquet',
    'may_2023': 'fhvhv_tripdata_2023-05.parquet',
    'jun_2023': 'fhvhv_tripdata_2023-06.parquet',
    'jul_2023': 'fhvhv_tripdata_2023-07.parquet',
    'aug_2023': 'fhvhv_tripdata_2023-08.parquet',
    'sep_2023': 'fhvhv_tripdata_2023-09.parquet',
    'oct_2023': 'fhvhv_tripdata_2023-10.parquet',
    'nov_2023': 'fhvhv_tripdata_2023-11.parquet',
    'dec_2023': 'fhvhv_tripdata_2023-12.parquet',
    'jan_2024': 'fhvhv_tripdata_2024-01.parquet',
    'feb_2024': 'fhvhv_tripdata_2024-02.parquet',
    'mar_2024': 'fhvhv_tripdata_2024-03.parquet',
    'apr_2024': 'fhvhv_tripdata_2024-04.parquet',
    'may_2024': 'fhvhv_tripdata_2024-05.parquet',
    'jun_2024': 'fhvhv_tripdata_2024-06.parquet',
    'jul_2024': 'fhvhv_tripdata_2024-07.parquet'
}

def get_DataFrame(file_name, local_path='data/'):
    """
    Tries to load a Parquet file from a local path.
    If the file is not found, downloads it from TLC's cloudfront URL and saves it locally.
    
    Parameters:
    - file_name (str): Name of the file to load/download (e.g., 'yellow_tripdata_2024-07.parquet').
    - local_path (str): Path to the local directory where the file is stored or will be saved. Default is 'data/'.
    
    Returns:
    - pd.DataFrame: DataFrame containing the data from the Parquet file.
    """
    # Ensure the local path exists
    os.makedirs(local_path, exist_ok=True)

    # Construct file paths
    local_file = os.path.join(local_path, file_name)
    tlc_url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{file_name}"

    try:
        # Try reading the file from the local path
        print(f"Trying to load {file_name} from {local_file}")
        df = pd.read_parquet(local_file)
        return df

    except FileNotFoundError:
        print(f"{file_name} not found locally. Attempting to download from {tlc_url}")

        # Download the file from TLC's cloudfront URL
        try:
            urllib.request.urlretrieve(tlc_url, local_file)
            print(f"Downloaded {file_name} to {local_file}")
            # Load the downloaded file
            df = pd.read_parquet(local_file)
            return df
        except Exception as e:
            print(f"Failed to download {file_name}: {e}")
            raise

def prepare_rides_per_day_data(trips):
    """
    Prepare data for plotting the number of rides per day, including extracting
    month/year, filtering trips, and grouping by day of the month.

    Parameters:
    - trips: DataFrame containing trip data with 'pickup_datetime' and 'pickup_day_of_month'.

    Returns:
    - rides_per_day: Series with the number of rides per day.
    - month_name: Name of the month (e.g., "July").
    - year: Year of the rides.
    - num_rides: Number of rides in the month
    """
    # Extract the month and year from the first 'pickup_datetime' entry
    first_trip_date = trips['pickup_datetime'].min()
    month_name = first_trip_date.strftime('%B')  # e.g., "July"
    year = first_trip_date.year
    num_rides = len(trips)

    # Filter trips for the specific month and year
    target_trips = trips[(trips['pickup_datetime'].dt.month == first_trip_date.month) &
                         (trips['pickup_datetime'].dt.year == year)]

    # Group by day of the month
    rides_per_day = target_trips.groupby('pickup_day_of_month').size()

    return rides_per_day, month_name, year, num_rides

# DsUtilFuncs.py
def plot_rides_per_day(
    rides_per_day, month_name, year, num_rides, cmap='Blues', bg_alpha=0.2, 
    ylim=None, yticks=None, ax=None, legend=True, show_xlabel=True
):
    """
    Plot a line chart showing the number of rides per day, with optional x-label control.

    Parameters:
    - rides_per_day: Series with the number of rides per day.
    - month_name: Name of the month (e.g., "July").
    - year: Year of the rides.
    - num_rides: Total number of rides in the month
    - cmap: Colormap to use for background fills by day of the week (default is 'Blues').
    - bg_alpha: Transparency for the background fill (default is 0.2).
    - ylim: Tuple (y_min, y_max) to set consistent y-axis limits.
    - yticks: Array-like of y-ticks to apply.
    - ax: Matplotlib axis to use for plotting (optional).
    - legend: Whether to display the legend (default is True).
    - show_xlabel: Whether to show the x-label (default is True).
    """

    # Get the colormap and generate 7 discrete colors (one for each weekday)
    colormap = plt.colormaps[cmap]
    colors = [colormap(i / 6) for i in range(7)]  # Normalize indices from 0 to 6

    # Create the plot if no axis is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the single line for total rides
    ax.plot(
        rides_per_day.index, rides_per_day.values, 
        marker='o', linestyle='-', color='black', label='Total Rides'
    )

    # Fill background based on the day of the week
    for day in range(1, rides_per_day.index.max() + 1):
        day_of_week = pd.Timestamp(f'{year}-{month_name[:3]}-{day:02}').dayofweek
        ax.axvspan(day - 0.5, day + 0.5, color=colors[day_of_week], alpha=bg_alpha)

    # Set y-axis limits and ticks if provided
    if ylim:
        ax.set_ylim(ylim)
    if yticks is not None:
        ax.set_yticks(yticks)

    # Optionally add the x-label
    if show_xlabel:
        ax.set_xlabel('Day of Month', fontsize=12)

    # Add the y-label
    ax.set_ylabel('Number of Rides', fontsize=12)
    ax.set_title(f'Daily Ride Count - {month_name} {year} - Total: {num_rides:,}', fontsize=12)

    # Set x-axis ticks for every day in the month
    ax.set_xticks(range(1, rides_per_day.index.max() + 1))

    # Add legend only if requested
    if legend:
        handles = [
            plt.Line2D([0], [0], color=colors[i], lw=4,
                       label=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][i])
            for i in range(7)
        ]
        ax.legend(handles=handles, title='Day of Week', fontsize=12)

    # Adjust layout and add grid
    ax.grid(True, linestyle='--', alpha=0.6)

class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to extract useful datetime features
    from 'pickup_datetime' and 'dropoff_datetime' columns.
    """

    def __init__(self):
        pass  # No parameters needed for now

    def fit(self, X, y=None):
        # Fit method doesn't need to do anything; it's a stateless transformer
        return self

    def transform(self, X):
        # Make a copy to avoid modifying the original dataframe
        X = X.copy()

        # Extract features from 'pickup_datetime'
        X['pickup_hour'] = X['pickup_datetime'].dt.hour
        X['pickup_day_of_week'] = X['pickup_datetime'].dt.day_of_week
        X['pickup_day_name'] = X['pickup_datetime'].dt.day_name()
        X['pickup_day_of_month'] = X['pickup_datetime'].dt.day

        # Extract features from 'dropoff_datetime'
        X['dropoff_hour'] = X['dropoff_datetime'].dt.hour
        X['dropoff_day_of_week'] = X['dropoff_datetime'].dt.day_of_week
        X['dropoff_day_name'] = X['dropoff_datetime'].dt.day_name()
        X['dropoff_day_of_month'] = X['dropoff_datetime'].dt.day

        return X