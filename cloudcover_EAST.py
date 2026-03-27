import gc
import os
import re
import shutil
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests
import xarray as xr
from matplotlib import patheffects
from matplotlib.colors import ListedColormap
from select_east_run import RUN_SELECTION_FILE, read_selected_run_time

matplotlib.use("Agg")

EASTERN_TZ = ZoneInfo("America/New_York")

@lru_cache(maxsize=1)
def get_county_geodataframe():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    geojson = response.json()
    for feature in geojson.get("features", []):
        feature.setdefault("properties", {})
        feature["properties"]["fips"] = feature.get("id", "")
    geodataframe = gpd.GeoDataFrame.from_features(geojson["features"])
    geodataframe = geodataframe.set_crs("EPSG:4326")
    geodataframe["fips"] = geodataframe["fips"].astype(str)
    return geodataframe

@lru_cache(maxsize=1)
def get_census_states_geodataframe():
    census_states_url = "https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_500k.json"
    return gpd.read_file(census_states_url)

def get_region_geodata(state_names, region_fips, padding_frac=0.05):
    counties_gdf = get_county_geodataframe()
    counties = counties_gdf[counties_gdf["fips"].str[:2].isin(region_fips)]
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
    "New York", "New Jersey", "Pennsylvania", "Ohio", "Virginia", "Maryland", "Delaware", "District of Columbia", "West Virginia",
]
NORTHEAST_STATE_FIPS = ["23", "33", "50", "25", "44", "09", "36", "34", "42", "39", "51", "24", "10", "11", "54"]

northeast_gdf, REGION_EXTENT, northeast_outline, northeast_states_gdf = get_region_geodata(
    NORTHEAST_STATE_NAMES,
    NORTHEAST_STATE_FIPS,
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
    "conus": {
        "label": "CONUS",
        "title": "CONUS",
        "extent": CONUS_EXTENT,
    },
}
ACTIVE_REGION_NAMES = ("northeast",)

BASE_DIR = "/var/data"
RUN_ROOT_DIR = os.path.join(BASE_DIR, "cloudcover_EAST")
RUNS_DIR = os.path.join(RUN_ROOT_DIR, "runs")
MAX_SAVED_RUNS = 7
grib_dir = os.path.join(RUN_ROOT_DIR, "grib")
os.makedirs(grib_dir, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)
png_dirs = {}
processed_steps_file = None

base_url_hrrr = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl"
variables = ["LCDC", "MCDC", "HCDC"]
levels = ["low_cloud_layer", "middle_cloud_layer", "high_cloud_layer"]


def format_eastern_time(dt_utc):
    local_time = dt_utc.astimezone(EASTERN_TZ)
    return local_time.strftime("%Y-%m-%d %I:%M %p %Z")

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
        except Exception as exc:
            print(f"Failed to remove png directory {run_png_root_dir}: {exc}")
    os.makedirs(run_png_root_dir, exist_ok=True)
    run_png_dirs = {}
    for region_name in ACTIVE_REGION_NAMES:
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
        except Exception as exc:
            print(f"Failed to remove archived run {run_name}: {exc}")

def get_forecast_steps(run_hour):
    if run_hour in {3, 9, 13, 14, 15, 16, 21}:
        return list(range(1, 19))
    return list(range(1, 49))

most_recent_run_time = read_selected_run_time()
print(
    f"Selected HRRR run from {RUN_SELECTION_FILE}: {most_recent_run_time.strftime('%Y-%m-%d %HZ')} "
    f"({format_eastern_time(most_recent_run_time)})"
)
current_run_id, current_run_dir, png_dirs, processed_steps_file = prepare_run_output(most_recent_run_time)
for region_name, region_png_dir in png_dirs.items():
    print(f"Writing {region_name} PNG output for run {current_run_id} to {region_png_dir}")
forecast_step_numbers = get_forecast_steps(most_recent_run_time.hour)
forecast_steps = [forecast_step_numbers[i:i + 24] for i in range(0, len(forecast_step_numbers), 24)]

def get_run_strings(run_time):
    return run_time.strftime("%Y%m%d"), run_time.strftime("%H")

def download_grib(url, file_path):
    response = requests.get(url, stream=True, timeout=120)
    if response.status_code == 200:
        with open(file_path, "wb") as output_file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    output_file.write(chunk)
        print(f"Downloaded {os.path.basename(file_path)}")
        return file_path
    print(f"Failed to download {os.path.basename(file_path)} (Status Code: {response.status_code})")
    return None

def get_hrrr_grib(run_time, step, variable, level):
    date_str, hour_str = get_run_strings(run_time)
    file_name = f"hrrr.t{hour_str}z.wrfsfcf{step:02d}.grib2"
    file_path = os.path.join(grib_dir, f"{variable.lower()}_{level}_{file_name}")
    url = (
        f"{base_url_hrrr}?file={file_name}"
        f"&lev_{level}=on"
        f"&var_{variable}=on"
        f"&dir=%2Fhrrr.{date_str}%2Fconus"
    )
    return download_grib(url, file_path)

def get_cloud_variable(dataset):
    for variable_name in dataset.data_vars:
        return dataset[variable_name]
    return None

def get_lat_lon_fields(dataset):
    latitude = dataset.get("latitude")
    if latitude is None:
        latitude = dataset.get("lat")
    longitude = dataset.get("longitude")
    if longitude is None:
        longitude = dataset.get("lon")
    return latitude, longitude

def plot_cloud_cover(lcdc_path, mcdc_path, hcdc_path, step, run_time, region_name):
    lcdc_ds = mcdc_ds = hcdc_ds = None
    figure = None
    try:
        region_config = REGION_CONFIGS[region_name]
        extent = region_config["extent"]
        plot_extent = extent
        lcdc_ds = xr.open_dataset(lcdc_path, engine="cfgrib", chunks={})
        mcdc_ds = xr.open_dataset(mcdc_path, engine="cfgrib", chunks={})
        hcdc_ds = xr.open_dataset(hcdc_path, engine="cfgrib", chunks={})
        lcdc = get_cloud_variable(lcdc_ds)
        mcdc = get_cloud_variable(mcdc_ds)
        hcdc = get_cloud_variable(hcdc_ds)
        lat_field, lon_field = get_lat_lon_fields(lcdc_ds)
        if lcdc is None or mcdc is None or hcdc is None or lat_field is None or lon_field is None:
            print("Required cloud cover variables or coordinates are not present in the dataset.")
            return None
        lcdc_vals = np.clip(lcdc.values.squeeze(), 0, 100)
        mcdc_vals = np.clip(mcdc.values.squeeze(), 0, 100)
        hcdc_vals = np.clip(hcdc.values.squeeze(), 0, 100)
        lats = lat_field.values
        lons = lon_field.values
        lons_plot = np.where(lons > 180, lons - 360, lons)
        if lats.ndim == 1 and lons.ndim == 1:
            lon2d, lat2d = np.meshgrid(lons_plot, lats)
        else:
            lon2d, lat2d = lons_plot, lats
        valid_time = run_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=step)
        local_valid = valid_time.astimezone(EASTERN_TZ)
        local_time = local_valid.strftime("%I %p")
        day_of_week = local_valid.strftime("%A")
        _, hour_str = get_run_strings(run_time)
        title = (
            f"HRRR Cloud Cover - {region_config['title']}\n"
            f"Valid: {valid_time.strftime('%Y-%m-%d %HZ')} ({day_of_week}, {local_time})  "
            f"Run: {hour_str}Z  Forecast Hour: {step}"
        )
        figure = plt.figure(figsize=(13, 11), dpi=300, facecolor="white")
        bottom_margin = 0.12 if region_name != "conus" else 0.08
        figure.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=bottom_margin)
        axis = plt.axes(projection=ccrs.PlateCarree(), facecolor="white")
        axis.set_extent(plot_extent, crs=ccrs.PlateCarree())
        axis.set_title(title, fontsize=11, fontweight="bold", pad=10)
        axis.add_feature(cfeature.LAND, facecolor="white")
        axis.add_feature(cfeature.OCEAN, facecolor="white")
        axis.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="black")
        axis.add_feature(cfeature.LAKES, facecolor="lightblue", edgecolor="black")
        axis.set_facecolor("white")
        # Plot low (red), middle (green), high (blue) as overlays
        low_mesh = axis.contourf(lon2d, lat2d, lcdc_vals, levels=[20, 100], colors=["red"], alpha=0.4, transform=ccrs.PlateCarree(), zorder=2)
        mid_mesh = axis.contourf(lon2d, lat2d, mcdc_vals, levels=[20, 100], colors=["green"], alpha=0.4, transform=ccrs.PlateCarree(), zorder=3)
        high_mesh = axis.contourf(lon2d, lat2d, hcdc_vals, levels=[20, 100], colors=["blue"], alpha=0.4, transform=ccrs.PlateCarree(), zorder=4)

        # Add three prominent but slightly shorter color bars, centered below the plot
        import matplotlib.patches as mpatches
        bar_height = 0.035
        bar_width = 0.22
        bar_y = 0.02
        bar_gap = 0.04
        total_width = 3 * bar_width + 2 * bar_gap
        left_margin = (1 - total_width) / 2
        bar_labels = ["Low Cloud", "Mid Cloud", "High Cloud"]
        bar_colors = ["red", "green", "blue"]
        for i in range(3):
            bar_x = left_margin + i * (bar_width + bar_gap)
            cax = figure.add_axes([bar_x, bar_y, bar_width, bar_height])
            cax.set_xticks([])
            cax.set_yticks([])
            for spine in cax.spines.values():
                spine.set_visible(False)
            rect = mpatches.FancyBboxPatch((0, 0), 1, 1,
                                           boxstyle="round,pad=0.02,rounding_size=0.04",
                                           linewidth=1.5, edgecolor="black", facecolor=bar_colors[i],
                                           transform=cax.transAxes, clip_on=False)
            cax.add_patch(rect)
            cax.text(0.5, -0.5, bar_labels[i], ha="center", va="top", fontsize=12, color="black", fontweight="bold", transform=cax.transAxes)
        try:
            counties_gdf = region_config.get("counties_gdf")
            states_gdf = region_config.get("states_gdf")
            if counties_gdf is not None and states_gdf is not None:
                counties_gdf.plot(ax=axis, facecolor="none", edgecolor="gray", linewidth=0.3, zorder=7)
                states_gdf.boundary.plot(ax=axis, edgecolor="#000000", linewidth=1.0, zorder=8)
            else:
                axis.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="#444444", zorder=7)
                axis.add_feature(cfeature.STATES, linewidth=0.35, edgecolor="#666666", zorder=8)
        except Exception as exc:
            print(f"Error plotting overlays: {exc}")
        plt.draw()
        margin_x = (extent[1] - extent[0]) * 0.01
        margin_y = (extent[3] - extent[2]) * 0.01
        text_x = plot_extent[1] - margin_x
        text_y_base = plot_extent[2] + margin_y
        line_spacing = (plot_extent[3] - plot_extent[2]) * 0.025
        axis.text(
            text_x,
            text_y_base + line_spacing,
            "Images by Jack Fordyce",
            fontsize=7,
            color="black",
            ha="right",
            va="bottom",
            fontweight="normal",
            alpha=0.85,
            transform=ccrs.PlateCarree(),
            zorder=20,
            path_effects=[patheffects.Stroke(linewidth=1, foreground="white"), patheffects.Normal()],
        )
        axis.text(
            text_x,
            text_y_base,
            "Hrrrweathermodel.com",
            fontsize=9,
            color="black",
            ha="right",
            va="bottom",
            fontweight="normal",
            alpha=0.85,
            transform=ccrs.PlateCarree(),
            zorder=20,
            path_effects=[patheffects.Stroke(linewidth=1, foreground="white"), patheffects.Normal()],
        )
        png_path = os.path.join(png_dirs[region_name], f"hrrr_cloudcover_{region_name}_{step:02d}.png")
        plt.savefig(png_path, bbox_inches="tight", dpi=300)
        print(f"Generated PNG: {png_path}")
        return png_path
    except Exception as exc:
        print(f"Error in plot_cloud_cover: {exc}")
        return None
    finally:
        for ds in (lcdc_ds, mcdc_ds, hcdc_ds):
            if ds is not None:
                ds.close()
        if figure is not None:
            plt.close(figure)
        gc.collect()

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"Cleared folder: {folder_path}")

clear_folder(grib_dir)

for step_group in forecast_steps:
    for step in step_group:
        grib_paths = []
        for var, lev in zip(variables, levels):
            grib_path = get_hrrr_grib(most_recent_run_time, step, var, lev)
            grib_paths.append(grib_path)
        if all(grib_paths):
            for region_name in ACTIVE_REGION_NAMES:
                plot_cloud_cover(grib_paths[0], grib_paths[1], grib_paths[2], step, most_recent_run_time, region_name)
        else:
            print(
                f"Skipping forecast hour {step:02d} for run {most_recent_run_time.strftime('%Y-%m-%d %HZ')} "
                "because one or more cloud cover files are not available."
            )
        for grib_path in grib_paths:
            if grib_path and os.path.exists(grib_path):
                os.remove(grib_path)
        gc.collect()
    prune_old_runs(MAX_SAVED_RUNS, keep_run_id=current_run_id)

print("HRRR cloud cover processing complete.")
