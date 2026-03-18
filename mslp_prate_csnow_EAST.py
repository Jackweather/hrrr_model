import os
import re
import requests
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time
import gc
from scipy import ndimage
from scipy.spatial import distance
from matplotlib import patheffects
from scipy.ndimage import maximum_filter, minimum_filter, label, generate_binary_structure
import geopandas as gpd
import shutil


EASTERN_TZ = ZoneInfo("America/New_York")


# --- Utility to fetch regional county/state geodata and compute extent/boundary ---
@lru_cache(maxsize=1)
def get_county_geodataframe():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    response = requests.get(url)
    response.raise_for_status()
    geojson = response.json()

    for feat in geojson.get("features", []):
        feat.setdefault("properties", {})
        feat["properties"]["fips"] = feat.get("id", "")

    gdf = gpd.GeoDataFrame.from_features(geojson["features"])
    gdf = gdf.set_crs("EPSG:4326")
    gdf["fips"] = gdf["fips"].astype(str)
    return gdf


@lru_cache(maxsize=1)
def get_census_states_geodataframe():
    census_states_url = "https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_500k.json"
    return gpd.read_file(census_states_url)


def get_region_geodata(state_names, region_fips, padding_frac=0.05):
    gdf = get_county_geodataframe()
    counties = gdf[gdf["fips"].str[:2].isin(region_fips)]
    if counties.empty:
        raise RuntimeError("No region counties found in GeoJSON.")
    minx, miny, maxx, maxy = counties.total_bounds
    pad_x = (maxx - minx) * padding_frac
    pad_y = (maxy - miny) * padding_frac
    extent = [minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y]

    states_census_gdf = get_census_states_geodataframe()
    states_census_gdf = states_census_gdf[states_census_gdf["NAME"].isin(state_names)]
    state_outline = states_census_gdf.unary_union
    return counties, extent, state_outline, states_census_gdf

NORTHEAST_STATE_NAMES = [
    "Maine", "New Hampshire", "Vermont", "Massachusetts", "Rhode Island", "Connecticut",
    "New York", "New Jersey", "Pennsylvania", "Ohio", "Virginia", "Maryland", "Delaware", "District of Columbia", "West Virginia"
]
NORTHEAST_STATE_FIPS = ["23", "33", "50", "25", "44", "09", "36", "34", "42", "39", "51", "24", "10", "11", "54"]
SOUTHEAST_STATE_NAMES = [
    "Virginia", "Kentucky", "Tennessee", "North Carolina", "South Carolina",
    "Georgia", "Florida", "Alabama", "Mississippi"
]
SOUTHEAST_STATE_FIPS = ["51", "21", "47", "37", "45", "13", "12", "01", "28"]
SOUTH_CENTRAL_STATE_NAMES = [
    "Texas", "Oklahoma", "Arkansas", "Louisiana", "New Mexico"
]
SOUTH_CENTRAL_STATE_FIPS = ["48", "40", "05", "22", "35"]
NORTH_CENTRAL_STATE_NAMES = [
    "North Dakota", "South Dakota", "Nebraska", "Kansas", "Minnesota",
    "Iowa", "Missouri", "Wisconsin", "Illinois", "Michigan"
]
NORTH_CENTRAL_STATE_FIPS = ["38", "46", "31", "20", "27", "19", "29", "55", "17", "26"]
WESTERN_STATE_NAMES = [
    "Washington", "Oregon", "California", "Nevada", "Idaho",
    "Montana", "Wyoming", "Utah", "Colorado", "Arizona"
]
WESTERN_STATE_FIPS = ["53", "41", "06", "32", "16", "30", "56", "49", "08", "04"]

# Acquire region geodata once (cache shapes in memory)
northeast_gdf, REGION_EXTENT, northeast_outline, northeast_states_gdf = get_region_geodata(
    NORTHEAST_STATE_NAMES,
    NORTHEAST_STATE_FIPS,
)
southeast_gdf, SOUTHEAST_EXTENT, southeast_outline, southeast_states_gdf = get_region_geodata(
    SOUTHEAST_STATE_NAMES,
    SOUTHEAST_STATE_FIPS,
)
south_central_gdf, SOUTH_CENTRAL_EXTENT, south_central_outline, south_central_states_gdf = get_region_geodata(
    SOUTH_CENTRAL_STATE_NAMES,
    SOUTH_CENTRAL_STATE_FIPS,
)
north_central_gdf, NORTH_CENTRAL_EXTENT, north_central_outline, north_central_states_gdf = get_region_geodata(
    NORTH_CENTRAL_STATE_NAMES,
    NORTH_CENTRAL_STATE_FIPS,
)
western_gdf, WESTERN_EXTENT, western_outline, western_states_gdf = get_region_geodata(
    WESTERN_STATE_NAMES,
    WESTERN_STATE_FIPS,
)
CONUS_EXTENT = [-127, -66, 24, 50]
TARGET_PLOT_ASPECT = (REGION_EXTENT[1] - REGION_EXTENT[0]) / (REGION_EXTENT[3] - REGION_EXTENT[2])
REGION_CONFIGS = {
    "northeast": {
        "label": "Northeast",
        "title": "Northeast/Mid-Atlantic US",
        "extent": REGION_EXTENT,
        "counties_gdf": northeast_gdf,
        "states_gdf": northeast_states_gdf,
    },
    "southeast": {
        "label": "Southeast",
        "title": "Southeast US",
        "extent": SOUTHEAST_EXTENT,
        "counties_gdf": southeast_gdf,
        "states_gdf": southeast_states_gdf,
    },
    "south_central": {
        "label": "South Central",
        "title": "South Central US",
        "extent": SOUTH_CENTRAL_EXTENT,
        "counties_gdf": south_central_gdf,
        "states_gdf": south_central_states_gdf,
    },
    "north_central": {
        "label": "North Central",
        "title": "North Central US",
        "extent": NORTH_CENTRAL_EXTENT,
        "counties_gdf": north_central_gdf,
        "states_gdf": north_central_states_gdf,
    },
    "western": {
        "label": "Western",
        "title": "Western US",
        "extent": WESTERN_EXTENT,
        "counties_gdf": western_gdf,
        "states_gdf": western_states_gdf,
    },
    "conus": {
        "label": "CONUS",
        "title": "CONUS",
        "extent": CONUS_EXTENT,
    },
}


# Set base directory for HRRR output
BASE_DIR = '/var/data'
RUN_ROOT_DIR = os.path.join(BASE_DIR, "mslp_prate_csnow_EAST")
RUNS_DIR = os.path.join(RUN_ROOT_DIR, "runs")
MAX_SAVED_RUNS = 7

# Output directories
grib_dir = os.path.join(RUN_ROOT_DIR, "grib")
os.makedirs(grib_dir, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

png_dirs = {}
processed_steps_file = None

# HRRR URL and variables
base_url_hrrr = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl"
variable_mslma = "MSLMA"  # Mean Sea Level Pressure
variable_prate = "PRATE"  # Precipitation Rate
variable_csnow = "CSNOW"  # Snowfall Rate
variable_cfrzr = "CFRZR"  # Freezing Rain Rate
variable_cicep = "CICEP"  # Sleet Rate
RUN_AVAILABILITY_DELAY_MINUTES = 50

# Adjust available hours to include 03Z, 09Z, 15Z, 21Z with a max forecast range of 18 hours
available_hours = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
max_forecast_hours = {1, 2, 3, 4, 5,7,8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23}  # These runs only go out to 18 hours

# HRRR run hours are in UTC. In America/New_York that means, for example:
# 00Z = 8 PM EDT / 7 PM EST on the previous local calendar day
# 01Z = 9 PM EDT / 8 PM EST
# 02Z = 10 PM EDT / 9 PM EST
# 03Z = 11 PM EDT / 10 PM EST
# The code below uses America/New_York directly so it automatically handles EST vs EDT.


def format_eastern_time(dt_utc):
    local_time = dt_utc.astimezone(EASTERN_TZ)
    return local_time.strftime("%Y-%m-%d %I:%M %p %Z")


def print_run_hour_mapping(reference_utc_time):
    print("HRRR run hour mapping for America/New_York:")
    for run_hour in available_hours:
        run_utc_time = reference_utc_time.replace(hour=run_hour, minute=0, second=0, microsecond=0)
        print(f"  {run_hour:02d}Z = {format_eastern_time(run_utc_time)}")


def floor_to_hour(dt_value):
    return dt_value.replace(minute=0, second=0, microsecond=0)


def get_run_id(run_time):
    return f"{run_time.strftime('%Y%m%d_%H')}z"


def prepare_run_output(run_time):
    run_id = get_run_id(run_time)
    run_dir = os.path.join(RUNS_DIR, run_id)
    run_png_root_dir = os.path.join(run_dir, "png")
    run_processed_steps_file = os.path.join(run_dir, "processed_steps.txt")

    if os.path.isdir(run_png_root_dir):
        try:
            shutil.rmtree(run_png_root_dir)
            print(f"Removed existing png directory for run {run_id}: {run_png_root_dir}")
        except Exception as e:
            print(f"Failed to remove png directory {run_png_root_dir}: {e}")

    os.makedirs(run_png_root_dir, exist_ok=True)

    run_png_dirs = {}
    for region_name in REGION_CONFIGS:
        region_png_dir = os.path.join(run_png_root_dir, region_name)
        os.makedirs(region_png_dir, exist_ok=True)
        run_png_dirs[region_name] = region_png_dir

    if os.path.exists(run_processed_steps_file):
        os.remove(run_processed_steps_file)

    return run_id, run_dir, run_png_dirs, run_processed_steps_file


def prune_old_runs(max_saved_runs, keep_run_id=None):
    run_names = []
    for name in os.listdir(RUNS_DIR):
        run_path = os.path.join(RUNS_DIR, name)
        if os.path.isdir(run_path) and re.fullmatch(r"\d{8}_\d{2}z", name):
            run_names.append(name)

    run_names.sort(reverse=True)
    keep_names = []
    for run_name in run_names:
        if run_name == keep_run_id or len(keep_names) < max_saved_runs:
            keep_names.append(run_name)

    for run_name in run_names:
        if run_name in keep_names:
            continue

        run_path = os.path.join(RUNS_DIR, run_name)
        try:
            shutil.rmtree(run_path)
            print(f"Removed archived run: {run_name}")
        except Exception as e:
            print(f"Failed to remove archived run {run_name}: {e}")

# Function to validate if a run time is accessible
def is_valid_run(run_time):
    """Check if HRRR data for the given run time is accessible."""
    test_date_str = run_time.strftime("%Y%m%d")
    test_hour_str = run_time.strftime("%H")
    test_file_name = f"hrrr.t{test_hour_str}z.wrfsfcf01.grib2"
    test_url = (
        f"{base_url_hrrr}?file={test_file_name}"
        f"&lev_surface=on&lev_mean_sea_level=on"
        f"&var_MSLMA=on"
        f"&dir=%2Fhrrr.{test_date_str}%2Fconus"
    )
    try:
        response = requests.head(test_url, timeout=5)
        return response.status_code == 200
    except Exception:
        return False

# Function to get the forecast steps based on the run hour
def get_forecast_steps(run_hour):
    if run_hour in {3, 9, 13, 14, 15, 16, 21}:  # Limit to 18-hour forecast for 03Z, 09Z, 13Z, 14Z, 15Z, 16Z, 21Z
        return list(range(1, 19))  # 18-hour forecast
    return list(range(1, 49, 1))  # Default 48-hour forecast in 3-hour steps

# Calculate the most recent HRRR run dynamically
now_utc = datetime.now(timezone.utc)
current_utc_time = floor_to_hour(now_utc)
current_eastern_time = current_utc_time.astimezone(EASTERN_TZ)
selection_reference_utc = floor_to_hour(now_utc - timedelta(minutes=RUN_AVAILABILITY_DELAY_MINUTES))
most_recent_run_time = None

print(f"Current time UTC: {current_utc_time.strftime('%Y-%m-%d %HZ')}")
print(f"Current time Eastern: {current_eastern_time.strftime('%Y-%m-%d %I:%M %p %Z')}")
print(
    f"Selecting from runs at least {RUN_AVAILABILITY_DELAY_MINUTES} minutes old: "
    f"{selection_reference_utc.strftime('%Y-%m-%d %HZ')} "
    f"({format_eastern_time(selection_reference_utc)})"
)
print_run_hour_mapping(current_utc_time)

# Iterate backward to find the most recent valid run, checking only runs old enough to be complete
for offset in range(24):  # Check up to 24 hours back
    candidate_time = selection_reference_utc - timedelta(hours=offset)
    if candidate_time.hour in available_hours and is_valid_run(candidate_time):
        most_recent_run_time = candidate_time
        break

# If no valid run found, fall back to previous run (6 hours earlier)
if most_recent_run_time is None:
    print("No valid run time found in the available hours. Searching for fallback run hour one hour at a time.")
    fallback_found = False
    for offset in range(1, 25):  # Search up to 24 hours back, one hour at a time
        candidate_time = selection_reference_utc - timedelta(hours=offset)
        candidate_hour = candidate_time.hour
        if candidate_hour in available_hours:
            candidate_time = floor_to_hour(candidate_time)
            if is_valid_run(candidate_time):
                most_recent_run_time = candidate_time
                print(
                    f"Using fallback run time: {most_recent_run_time.strftime('%Y-%m-%d %HZ')} "
                    f"({format_eastern_time(most_recent_run_time)})"
                )
                fallback_found = True
                break
    if not fallback_found:
        raise ValueError("No valid run time found, including fallback.")

print(
    f"Selected HRRR run: {most_recent_run_time.strftime('%Y-%m-%d %HZ')} "
    f"({format_eastern_time(most_recent_run_time)})"
)

current_run_id, current_run_dir, png_dirs, processed_steps_file = prepare_run_output(most_recent_run_time)
for region_name, region_png_dir in png_dirs.items():
    print(f"Writing {region_name} PNG output for run {current_run_id} to {region_png_dir}")

forecast_step_numbers = get_forecast_steps(most_recent_run_time.hour)

# Function to get date and hour strings for a given run time
def get_run_strings(run_time):
    return run_time.strftime("%Y%m%d"), run_time.strftime("%H")


def expand_extent_to_aspect(extent, target_aspect):
    min_lon, max_lon, min_lat, max_lat = extent
    width = max_lon - min_lon
    height = max_lat - min_lat
    current_aspect = width / height

    if abs(current_aspect - target_aspect) < 0.01:
        return extent

    if current_aspect > target_aspect:
        target_height = width / target_aspect
        extra_height = max(target_height - height, 0)
        min_lat -= extra_height / 2
        max_lat += extra_height / 2
    else:
        target_width = height * target_aspect
        extra_width = max(target_width - width, 0)
        min_lon -= extra_width / 2
        max_lon += extra_width / 2

    return [min_lon, max_lon, min_lat, max_lat]

# Levels and colormaps
mslp_levels = np.arange(960, 1050 + 2, 2)
prate_levels = [value * 5 for value in [0.1, 0.25, 0.5, 0.75, 1.5, 2, 2.5, 3, 4, 6, 10, 16, 24]]
prate_colors = [
    "#b6ffb6", "#54f354", "#19a319", "#016601", "#c9c938", "#f5f825",
    "#ffd700", "#ffa500", "#ff7f50", "#ff4500", "#ff1493", "#9400d3"
]
prate_cmap = LinearSegmentedColormap.from_list("prate_custom", prate_colors, N=len(prate_colors))
prate_norm = BoundaryNorm(prate_levels, prate_cmap.N)

# Forecast steps grouped into chunks of 24
forecast_steps = [forecast_step_numbers[i:i + 24] for i in range(0, len(forecast_step_numbers), 24)]

# Download functions
def download_grib(url, file_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded {os.path.basename(file_path)}")
        return file_path
    else:
        print(f"Failed to download {os.path.basename(file_path)} (Status Code: {response.status_code})")
        return None

def get_hrrr_grib(run_time, step, variable):
    date_str, hour_str = get_run_strings(run_time)
    file_name = f"hrrr.t{hour_str}z.wrfsfcf{step:02d}.grib2"
    file_path = os.path.join(grib_dir, f"{variable.lower()}_{file_name}")
    url = (
        f"{base_url_hrrr}?file={file_name}"
        f"&lev_surface=on&lev_mean_sea_level=on"
        f"&var_{variable}=on"
        f"&dir=%2Fhrrr.{date_str}%2Fconus"
    )
    return download_grib(url, file_path)

# --- Plotting function (uses adjusted NY extent) ---
def plot_combined(mslp_path, prate_path, step, run_time, region_name, csnow_path=None, cfrzr_path=None, cicep_path=None):
    try:
        region_config = REGION_CONFIGS[region_name]
        extent = region_config["extent"]
        plot_extent = expand_extent_to_aspect(extent, TARGET_PLOT_ASPECT) if region_name == "conus" else extent

        # Open datasets with dask chunking for lazy loading
        ds_mslp = xr.open_dataset(mslp_path, engine="cfgrib", chunks={})
        ds_prate = xr.open_dataset(prate_path, engine="cfgrib", chunks={})
        ds_csnow = xr.open_dataset(csnow_path, engine="cfgrib", chunks={}) if csnow_path else None
        ds_cfrzr = xr.open_dataset(cfrzr_path, engine="cfgrib", chunks={}) if cfrzr_path else None
        ds_cicep = xr.open_dataset(cicep_path, engine="cfgrib", chunks={}) if cicep_path else None

        # extract arrays
        mslp = ds_mslp.get('mslma')
        prate = ds_prate.get('prate')
        if mslp is None or prate is None:
            print("Required variables not in datasets")
            ds_mslp.close()
            ds_prate.close()
            if ds_csnow is not None:
                ds_csnow.close()
            if ds_cfrzr is not None:
                ds_cfrzr.close()
            if ds_cicep is not None:
                ds_cicep.close()
            return None
        mslp = mslp.values / 100.0  # Pa to hPa
        prate = prate.values * 3600  # mm/s to mm/hr

        lats = ds_mslp['latitude'].values
        lons = ds_mslp['longitude'].values
        lons_plot = np.where(lons > 180, lons - 360, lons)

        if lats.ndim == 1 and lons.ndim == 1:
            Lon2d, Lat2d = np.meshgrid(lons_plot, lats)
            mslp2d = mslp.squeeze()
            prate2d = prate.squeeze()
        else:
            Lon2d, Lat2d = lons_plot, lats
            mslp2d = mslp.squeeze()
            prate2d = prate.squeeze()

        # --- Ensure rate arrays exist before masking (prevent UnboundLocalError) ---
        snow_rate2d = None
        cfrzr_rate2d = None
        cicep_rate2d = None

        # compute snow_rate2d if csnow dataset present
        if ds_csnow is not None and "csnow" in ds_csnow:
            try:
                csnow = ds_csnow['csnow'].values * 3600
                csnow2d = csnow.squeeze()
                if csnow2d.shape == prate2d.shape:
                    snow_mask = (csnow2d > 0)
                    snow_rate2d = np.where(snow_mask, prate2d, np.nan)
            except Exception:
                snow_rate2d = None
            ds_csnow.close()

        # compute cfrzr_rate2d if cfrzr dataset present
        if ds_cfrzr is not None and "cfrzr" in ds_cfrzr:
            try:
                cfrzr = ds_cfrzr['cfrzr'].values * 3600
                cfrzr2d = cfrzr.squeeze()
                if cfrzr2d.shape == prate2d.shape:
                    cfrzr_mask = (cfrzr2d > 0)
                    cfrzr_rate2d = np.where(cfrzr_mask, prate2d, np.nan)
            except Exception:
                cfrzr_rate2d = None
            ds_cfrzr.close()

        # compute cicep_rate2d if cicep dataset present
        if ds_cicep is not None and "cicep" in ds_cicep:
            try:
                cicep = ds_cicep['cicep'].values * 3600
                cicep2d = cicep.squeeze()
                if cicep2d.shape == prate2d.shape:
                    cicep_mask = (cicep2d > 0)
                    cicep_rate2d = np.where(cicep_mask, prate2d, np.nan)
            except Exception:
                cicep_rate2d = None
            ds_cicep.close()

        # Do not mask weather data to region; plot full grid

        # --- Title/time calculation --- use timezone-aware conversion so DST is handled
        date_str, hour_str = get_run_strings(run_time)
        base_time_utc = run_time.replace(minute=0, second=0, microsecond=0)
        valid_time = base_time_utc + timedelta(hours=step)  # Forecast valid time in UTC
        # convert valid_time to America/New_York to get local hour and weekday (handles DST)
        local_valid = valid_time.astimezone(ZoneInfo('America/New_York'))
        local_time = local_valid.strftime('%I %p')
        day_of_week = local_valid.strftime('%A')


        title = (
            f"HRRR Precipitation Rate & Mean Sea Level Pressure — {region_config['title']}\n"
            f"Valid: {valid_time.strftime('%Y-%m-%d %HZ')} ({day_of_week}, {local_time})  "
            f"Run: {hour_str}Z  Forecast Hour: {step}"
        )

        # --- Plotting setup (use expanded region extent) ---
        colorbar_band_bottom = 0.035
        map_bottom = 0.12 if region_name != "conus" else 0.08
        fig = plt.figure(figsize=(13, 11), dpi=300, facecolor='white')
        fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=map_bottom)
        ax = plt.axes(projection=ccrs.PlateCarree(), facecolor='white')
        ax.set_extent(plot_extent, crs=ccrs.PlateCarree())
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

        # Base map limited to NY: make anything outside appear white
        ax.add_feature(cfeature.LAND, facecolor='white')
        ax.add_feature(cfeature.OCEAN, facecolor='white')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor='black')  # Added coastlines
        ax.add_feature(cfeature.LAKES, facecolor='lightblue', edgecolor='black')  # Added lakes
        ax.set_facecolor('white')  # ensure background is white

        # PRATE plotting
        mesh = ax.contourf(
            Lon2d, Lat2d, prate2d,
            levels=prate_levels,
            cmap=prate_cmap,
            norm=prate_norm,
            extend='max',
            transform=ccrs.PlateCarree(),
            alpha=0.8,
            zorder=1
        )

        # CFRZR, CICEP, CSNOW plotting (if present)
        if cfrzr_rate2d is not None:
            cfrzr_levels = [value * 3 for value in [0.1, 0.25, 0.5, 1, 2, 4, 8, 16]]
            cfrzr_colors = ["#fce4ec", "#f8bbd0", "#f48fb1", "#ec407a", "#d81b60", "#880e4f", "#560027"]
            cfrzr_cmap = LinearSegmentedColormap.from_list("cfrzr_cbar", cfrzr_colors, N=len(cfrzr_colors))
            cfrzr_norm = BoundaryNorm(cfrzr_levels, cfrzr_cmap.N)
            cfrzr_mesh = ax.contourf(Lon2d, Lat2d, cfrzr_rate2d, levels=cfrzr_levels, cmap=cfrzr_cmap, norm=cfrzr_norm, extend='max', transform=ccrs.PlateCarree(), alpha=0.85, zorder=3)

        if cicep_rate2d is not None:
            cicep_levels = [value * 3 for value in [0.1, 0.25, 0.5, 1, 2, 4, 8, 16]]
            cicep_colors = ["#f3e5f5", "#e1bee7", "#ce93d8", "#ba68c8", "#9c27b0", "#7b1fa2", "#4a148c", "#12005e"]
            cicep_cmap = LinearSegmentedColormap.from_list("cicep_cbar", cicep_colors, N=len(cicep_colors))
            cicep_norm = BoundaryNorm(cicep_levels, cicep_cmap.N)
            cicep_mesh = ax.contourf(Lon2d, Lat2d, cicep_rate2d, levels=cicep_levels, cmap=cicep_cmap, norm=cicep_norm, extend='max', transform=ccrs.PlateCarree(), alpha=0.85, zorder=3)

        # MSLP plotting (contours)
        cs = ax.contour(Lon2d, Lat2d, mslp2d, levels=mslp_levels, colors='black', linewidths=0.7, transform=ccrs.PlateCarree(), zorder=4)
        ax.clabel(cs, fmt='%d', fontsize=6, colors='black', inline=True)

        # Snow plotting
        if snow_rate2d is not None:
            snow_levels = [value * 3 for value in [0.1, 0.25, 0.5, 1, 2, 4, 8, 16]]
            snow_colors = ["#e3f2fd", "#bbdefb", "#90caf9", "#42a5f5", "#1e88e5", "#1565c0", "#0d47a1", "#002171"]
            snow_cmap = LinearSegmentedColormap.from_list("snow_cbar", snow_colors, N=len(snow_colors))
            snow_norm = BoundaryNorm(snow_levels, snow_cmap.N)
            snow_mesh = ax.contourf(Lon2d, Lat2d, snow_rate2d, levels=snow_levels, cmap=snow_cmap, norm=snow_norm, extend='max', transform=ccrs.PlateCarree(), alpha=0.85, zorder=3)

        plt.draw()
        if region_name != "conus":
            cbar_y = colorbar_band_bottom
        else:
            map_position = ax.get_position()
            cbar_y = max(map_position.y0 - 0.045, colorbar_band_bottom)
        cbar_h = 0.02   # smaller height
        cbar_w = 0.21   # width chosen so 4 bars + gaps fit within [0.05,0.95]
        gap = 0.02
        x0 = 0.05
        x1 = x0 + cbar_w + gap
        x2 = x1 + cbar_w + gap
        x3 = x2 + cbar_w + gap
        cax_prate = fig.add_axes([x0, cbar_y, cbar_w, cbar_h])
        cax_cfrzr = fig.add_axes([x1, cbar_y, cbar_w, cbar_h])
        cax_csnow = fig.add_axes([x2, cbar_y, cbar_w, cbar_h])
        cax_cicep = fig.add_axes([x3, cbar_y, cbar_w, cbar_h])

        # Ensure axes have white background and hide ticks for empty axes
        for cax in (cax_prate, cax_cfrzr, cax_csnow, cax_cicep):
            cax.set_facecolor('white')
            cax.tick_params(labelbottom=False, bottom=False)
            cax.set_xticks([])

        # helper to remove leading zero from decimals (e.g. "0.25" -> ".25")
        def fmt_tick(v):
            s = f"{v:g}"
            if s.startswith("-0."):
                return "-" + s[2:]
            if s.startswith("0."):
                return s[1:]
            return s

        # PRATE colorbar (always present)
        cbar = plt.colorbar(mesh, cax=cax_prate, orientation='horizontal', ticks=prate_levels[::2], boundaries=prate_levels)
        prate_tick_labels = [fmt_tick(v) for v in prate_levels[::2]]
        cbar.ax.set_xticklabels(prate_tick_labels)
        cbar.set_label("Precipitation Rate (mm/hr)", fontsize=7, labelpad=2)  # slightly smaller
        cbar.ax.tick_params(labelsize=6, length=1)
        cbar.ax.set_facecolor('white')
        cbar.outline.set_edgecolor('black')

        # CFRZR colorbar (only if plotted)
        if cfrzr_rate2d is not None:
            cbar_cfrzr = plt.colorbar(cfrzr_mesh, cax=cax_cfrzr, orientation='horizontal', ticks=cfrzr_levels, boundaries=cfrzr_levels)
            cbar_cfrzr.ax.set_xticklabels([fmt_tick(v) for v in cfrzr_levels])
            cbar_cfrzr.ax.tick_params(labelsize=6)
            cbar_cfrzr.set_label("Freezing Rain (mm/hr)", fontsize=6)
        else:
            cax_cfrzr.set_axis_off()

        # CSNOW colorbar (only if plotted)
        if snow_rate2d is not None:
            cbar_csnow = plt.colorbar(snow_mesh, cax=cax_csnow, orientation='horizontal', ticks=snow_levels, boundaries=snow_levels)
            cbar_csnow.ax.set_xticklabels([fmt_tick(v) for v in snow_levels])
            cbar_csnow.ax.tick_params(labelsize=6)
            cbar_csnow.set_label("Snow Rate (mm/hr)", fontsize=6)
        else:
            cax_csnow.set_axis_off()

        # CICEP colorbar (only if plotted)
        if cicep_rate2d is not None:
            cbar_cicep = plt.colorbar(cicep_mesh, cax=cax_cicep, orientation='horizontal', ticks=cicep_levels, boundaries=cicep_levels)
            cbar_cicep.ax.set_xticklabels([fmt_tick(v) for v in cicep_levels])
            cbar_cicep.ax.tick_params(labelsize=6)
            cbar_cicep.set_label("Sleet (mm/hr)", fontsize=6)
        else:
            cax_cicep.set_axis_off()


        # Detect highs and lows limited to region extent
        mask = (
            (Lon2d >= extent[0]) & (Lon2d <= extent[1]) &
            (Lat2d >= extent[2]) & (Lat2d <= extent[3])
        )
        data_masked = np.where(mask, mslp2d, np.nan)

        def find_valid_extrema(extrema_y, extrema_x, values, lon_margin, lat_margin, bounds, is_high=True):
            sorted_indices = np.argsort(values)[::-1] if is_high else np.argsort(values)
            for idx in sorted_indices:
                y, x = extrema_y[idx], extrema_x[idx]
                lon, lat = Lon2d[y, x], Lat2d[y, x]
                if (
                    bounds[0] + lon_margin <= lon <= bounds[1] - lon_margin
                    and bounds[2] + lat_margin <= lat <= bounds[3] - lat_margin
                ):
                    return lon, lat, values[idx]
            return None, None, None

        extent_width = extent[1] - extent[0]
        extent_height = extent[3] - extent[2]
        extrema_bounds = plot_extent if region_name == "conus" else extent
        extrema_lon_margin = max(extent_width * (0.04 if region_name == "conus" else 0.01), 0.1)
        extrema_lat_margin = max(extent_height * (0.05 if region_name == "conus" else 0.01), 0.1)

        # Plot one low if <= 1005 hPa
        min_filt = ndimage.minimum_filter(data_masked, size=25, mode='constant', cval=np.nan)
        lows = (data_masked == min_filt) & ~np.isnan(data_masked)
        low_y, low_x = np.where(lows)
        low_values = data_masked[low_y, low_x]

        low_lon, low_lat, low_value = find_valid_extrema(
            low_y,
            low_x,
            low_values,
            lon_margin=extrema_lon_margin,
            lat_margin=extrema_lat_margin,
            bounds=extrema_bounds,
            is_high=False,
        )
        if low_lon is not None and low_value <= 1005:  # Only plot if <= 1005 hPa
            ax.text(low_lon, low_lat, "L", color='red', fontsize=12, fontweight='bold', ha='center', va='center', transform=ccrs.PlateCarree(), zorder=6, path_effects=[patheffects.Stroke(linewidth=1, foreground='white'), patheffects.Normal()])
            ax.text(low_lon, low_lat - 0.2, f"{low_value:.0f}", color='red', fontsize=6, fontweight='bold', ha='center', va='top', transform=ccrs.PlateCarree(), zorder=6)

        # Plot one high if > 1029 hPa
        max_filt = ndimage.maximum_filter(data_masked, size=25, mode='constant', cval=np.nan)
        highs = (data_masked == max_filt) & ~np.isnan(data_masked)
        high_y, high_x = np.where(highs)
        high_values = data_masked[high_y, high_x]

        high_lon, high_lat, high_value = find_valid_extrema(
            high_y,
            high_x,
            high_values,
            lon_margin=extrema_lon_margin,
            lat_margin=extrema_lat_margin,
            bounds=extrema_bounds,
            is_high=True,
        )
        if high_lon is not None and high_value > 1029:  # Only plot if > 1029 hPa
            ax.text(high_lon, high_lat, "H", color='blue', fontsize=12, fontweight='bold', ha='center', va='center', transform=ccrs.PlateCarree(), zorder=6, path_effects=[patheffects.Stroke(linewidth=1, foreground='white'), patheffects.Normal()])
            ax.text(high_lon, high_lat - 0.2, f"{high_value:.0f}", color='blue', fontsize=6, fontweight='bold', ha='center', va='top', transform=ccrs.PlateCarree(), zorder=6)



        # Overlay region counties and state outlines
        try:
            counties_gdf = region_config.get("counties_gdf")
            states_gdf = region_config.get("states_gdf")
            if counties_gdf is not None and states_gdf is not None:
                counties_gdf.plot(ax=ax, facecolor="none", edgecolor="gray", linewidth=0.3, zorder=7)
                states_gdf.boundary.plot(ax=ax, edgecolor="#000000", linewidth=1.0, zorder=8)
            else:
                ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="#444444", zorder=7)
                ax.add_feature(cfeature.STATES, linewidth=0.35, edgecolor="#666666", zorder=8)
        except Exception as e:
            print(f"Error plotting overlays: {e}")


        margin_x = (extent[1] - extent[0]) * 0.01
        margin_y = (extent[3] - extent[2]) * 0.01
        text_x = plot_extent[1] - margin_x
        text_y_base = plot_extent[2] + margin_y
        line_spacing = (plot_extent[3] - plot_extent[2]) * 0.025
        ax.text(
            text_x, text_y_base + line_spacing, "Images by Jack Fordyce",
            fontsize=7, color="black", ha="right", va="bottom",
            fontweight="normal", alpha=0.85,
            transform=ccrs.PlateCarree(),
            zorder=20,
            path_effects=[patheffects.Stroke(linewidth=1, foreground='white'), patheffects.Normal()]
        )
        ax.text(
            text_x, text_y_base, "Truelocalwx.com",
            fontsize=7, color="black", ha="right", va="bottom",
            fontweight="normal", alpha=0.85,
            transform=ccrs.PlateCarree(),
            zorder=20,
            path_effects=[patheffects.Stroke(linewidth=1, foreground='white'), patheffects.Normal()]
        )

        # Save PNG
        png_path = os.path.join(png_dirs[region_name], f"hrrr_combined_{region_name}_{step:02d}.png")
        plt.savefig(png_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        ds_mslp.close()
        ds_prate.close()
        print(f"Generated PNG: {png_path}")
        gc.collect()
        return png_path

    except Exception as e:
        print(f"Error in plot_combined: {e}")

# Function to clear all files in a directory
def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for f in os.listdir(folder_path):
            file_path = os.path.join(folder_path, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"Cleared folder: {folder_path}")

# Clear the folders at the start of the script
clear_folder(grib_dir)

# Main process
for step_group in forecast_steps:
    for step in step_group:
        mslp_grib = get_hrrr_grib(most_recent_run_time, step, "MSLMA")
        prate_grib = get_hrrr_grib(most_recent_run_time, step, "PRATE")
        csnow_grib = get_hrrr_grib(most_recent_run_time, step, "CSNOW")
        cfrzr_grib = get_hrrr_grib(most_recent_run_time, step, "CFRZR")
        cicep_grib = get_hrrr_grib(most_recent_run_time, step, "CICEP")

        if mslp_grib and prate_grib:
            for region_name in REGION_CONFIGS:
                plot_combined(mslp_grib, prate_grib, step, most_recent_run_time, region_name, csnow_grib, cfrzr_grib, cicep_grib)
        else:
            print(
                f"Skipping forecast hour {step:02d} for run {most_recent_run_time.strftime('%Y-%m-%d %HZ')} "
                "because required files are not available."
            )

        # Delete GRIB files after processing or skip
        for grib_file in [mslp_grib, prate_grib, csnow_grib, cfrzr_grib, cicep_grib]:
            if grib_file and os.path.exists(grib_file):
                os.remove(grib_file)

        # Collect garbage
        gc.collect()

    prune_old_runs(MAX_SAVED_RUNS, keep_run_id=current_run_id)
print("HRRR processing complete.")
