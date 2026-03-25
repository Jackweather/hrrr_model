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
from matplotlib.colors import BoundaryNorm, ListedColormap

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
SOUTHEAST_STATE_NAMES = [
    "Virginia", "Kentucky", "Tennessee", "North Carolina", "South Carolina",
    "Georgia", "Florida", "Alabama", "Mississippi",
]
SOUTHEAST_STATE_FIPS = ["51", "21", "47", "37", "45", "13", "12", "01", "28"]
SOUTH_CENTRAL_STATE_NAMES = [
    "Texas", "Oklahoma", "Arkansas", "Louisiana", "New Mexico",
]
SOUTH_CENTRAL_STATE_FIPS = ["48", "40", "05", "22", "35"]
NORTH_CENTRAL_STATE_NAMES = [
    "North Dakota", "South Dakota", "Nebraska", "Kansas", "Minnesota",
    "Iowa", "Missouri", "Wisconsin", "Illinois", "Michigan",
]
NORTH_CENTRAL_STATE_FIPS = ["38", "46", "31", "20", "27", "19", "29", "55", "17", "26"]
WESTERN_STATE_NAMES = [
    "Washington", "Oregon", "California", "Nevada", "Idaho",
    "Montana", "Wyoming", "Utah", "Colorado", "Arizona",
]
WESTERN_STATE_FIPS = ["53", "41", "06", "32", "16", "30", "56", "49", "08", "04"]

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
ACTIVE_REGION_NAMES = ("northeast",)


BASE_DIR = "/var/data"
RUN_ROOT_DIR = os.path.join(BASE_DIR, "weasd_EAST")
RUNS_DIR = os.path.join(RUN_ROOT_DIR, "runs")
MAX_SAVED_RUNS = 7

grib_dir = os.path.join(RUN_ROOT_DIR, "grib")
os.makedirs(grib_dir, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

png_dirs = {}
processed_steps_file = None

base_url_hrrr = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl"
variable_weasd = "WEASD"
RUN_AVAILABILITY_DELAY_MINUTES = 50
PREFERRED_CYCLE_HOURS = (0, 6, 12, 18)
MAX_RUN_SEARCH_HOURS = 30

available_hours = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]


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


def get_preferred_cycle_start(reference_utc_time):
    preferred_hour = max(hour for hour in PREFERRED_CYCLE_HOURS if hour <= reference_utc_time.hour)
    return reference_utc_time.replace(hour=preferred_hour, minute=0, second=0, microsecond=0)


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


def is_valid_run(run_time):
    test_date_str = run_time.strftime("%Y%m%d")
    test_hour_str = run_time.strftime("%H")
    test_file_name = f"hrrr.t{test_hour_str}z.wrfsfcf01.grib2"
    test_url = (
        f"{base_url_hrrr}?file={test_file_name}"
        f"&lev_surface=on"
        f"&var_WEASD=on"
        f"&dir=%2Fhrrr.{test_date_str}%2Fconus"
    )
    try:
        response = requests.head(test_url, timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def get_forecast_steps(run_hour):
    if run_hour in {3, 9, 13, 14, 15, 16, 21}:
        return list(range(1, 19))
    return list(range(1, 49))


now_utc = datetime.now(timezone.utc)
current_utc_time = floor_to_hour(now_utc)
current_eastern_time = current_utc_time.astimezone(EASTERN_TZ)
selection_reference_utc = floor_to_hour(now_utc - timedelta(minutes=RUN_AVAILABILITY_DELAY_MINUTES))
selection_start_utc = get_preferred_cycle_start(selection_reference_utc)
most_recent_run_time = None

print(f"Current time UTC: {current_utc_time.strftime('%Y-%m-%d %HZ')}")
print(f"Current time Eastern: {current_eastern_time.strftime('%Y-%m-%d %I:%M %p %Z')}")
print(
    f"Selecting from runs at least {RUN_AVAILABILITY_DELAY_MINUTES} minutes old: "
    f"{selection_reference_utc.strftime('%Y-%m-%d %HZ')} "
    f"({format_eastern_time(selection_reference_utc)})"
)
print("Preferring anchor cycles: 00Z, 06Z, 12Z, 18Z")
print(
    f"Starting run search from preferred cycle: {selection_start_utc.strftime('%Y-%m-%d %HZ')} "
    f"({format_eastern_time(selection_start_utc)})"
)
print_run_hour_mapping(current_utc_time)

for offset in range(MAX_RUN_SEARCH_HOURS + 1):
    candidate_time = selection_start_utc - timedelta(hours=offset)
    if candidate_time.hour in available_hours and is_valid_run(candidate_time):
        most_recent_run_time = candidate_time
        break

if most_recent_run_time is None:
    raise ValueError("No valid run time found within the anchored search window.")

print(
    f"Selected HRRR run: {most_recent_run_time.strftime('%Y-%m-%d %HZ')} "
    f"({format_eastern_time(most_recent_run_time)})"
)

current_run_id, current_run_dir, png_dirs, processed_steps_file = prepare_run_output(most_recent_run_time)
for region_name, region_png_dir in png_dirs.items():
    print(f"Writing {region_name} PNG output for run {current_run_id} to {region_png_dir}")

forecast_step_numbers = get_forecast_steps(most_recent_run_time.hour)


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


snow_breaks = [
    0, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 36, 48, 56,
]
snow_tick_levels = snow_breaks
snow_colors = [
    "#ffffff", "#0d1a4a", "#1565c0", "#42a5f5", "#90caf9", "#e3f2fd",
    "#b39ddb", "#7e57c2", "#512da8", "#c2185b", "#f06292", "#81c784",
    "#388e3c", "#1b5e20", "#bdbdbd", "#757575", "#212121", "#000000",
]
snow_plot_levels = snow_breaks + [72]
snow_cmap = ListedColormap(snow_colors)
snow_norm = BoundaryNorm(snow_plot_levels, snow_cmap.N)

forecast_steps = [forecast_step_numbers[i:i + 24] for i in range(0, len(forecast_step_numbers), 24)]


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


def get_hrrr_grib(run_time, step, variable):
    date_str, hour_str = get_run_strings(run_time)
    file_name = f"hrrr.t{hour_str}z.wrfsfcf{step:02d}.grib2"
    file_path = os.path.join(grib_dir, f"{variable.lower()}_{file_name}")
    url = (
        f"{base_url_hrrr}?file={file_name}"
        f"&lev_surface=on"
        f"&var_{variable}=on"
        f"&dir=%2Fhrrr.{date_str}%2Fconus"
    )
    return download_grib(url, file_path)


def get_snow_variable(dataset):
    for variable_name in ("sdwe", "weasd", "snow", "sde"):
        if variable_name in dataset:
            return dataset[variable_name]

    for data_var_name in dataset.data_vars:
        return dataset[data_var_name]

    return None


def get_lat_lon_fields(dataset):
    latitude = dataset.get("latitude")
    if latitude is None:
        latitude = dataset.get("lat")

    longitude = dataset.get("longitude")
    if longitude is None:
        longitude = dataset.get("lon")

    return latitude, longitude


def plot_snowfall(weasd_path, step, run_time, region_name):
    dataset = None
    figure = None
    try:
        region_config = REGION_CONFIGS[region_name]
        extent = region_config["extent"]
        plot_extent = expand_extent_to_aspect(extent, TARGET_PLOT_ASPECT) if region_name == "conus" else extent

        dataset = xr.open_dataset(
            weasd_path,
            engine="cfgrib",
            chunks={},
            backend_kwargs={
                "filter_by_keys": {"stepType": "accum"},
                "indexpath": "",
            },
        )
        snow_field = get_snow_variable(dataset)
        latitude_field, longitude_field = get_lat_lon_fields(dataset)
        if snow_field is None or latitude_field is None or longitude_field is None:
            print("Required snowfall coordinates or variable are not present in the dataset.")
            return None

        snow_water_equivalent_mm = snow_field.values.squeeze()
        snowfall_inches = snow_water_equivalent_mm * 0.393700787

        lats = latitude_field.values
        lons = longitude_field.values
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
            f"HRRR Snowfall 10:1 - {region_config['title']}\n"
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

        mesh = axis.contourf(
            lon2d,
            lat2d,
            snowfall_inches,
            levels=snow_plot_levels,
            cmap=snow_cmap,
            norm=snow_norm,
            extend="max",
            transform=ccrs.PlateCarree(),
            alpha=0.9,
            zorder=1,
        )

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
        colorbar_y = 0.035 if region_name != "conus" else max(axis.get_position().y0 - 0.045, 0.035)
        colorbar_axis = figure.add_axes([0.18, colorbar_y, 0.64, 0.02])
        colorbar = plt.colorbar(mesh, cax=colorbar_axis, orientation="horizontal", ticks=snow_tick_levels)
        colorbar.set_label("Snowfall (inches, 10:1)", fontsize=7, labelpad=2)
        colorbar.ax.tick_params(labelsize=6, length=1)
        colorbar.ax.set_facecolor("white")
        colorbar.outline.set_edgecolor("black")

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

        png_path = os.path.join(png_dirs[region_name], f"hrrr_weasd_{region_name}_{step:02d}.png")
        plt.savefig(png_path, bbox_inches="tight", dpi=300)
        print(f"Generated PNG: {png_path}")
        return png_path
    except Exception as exc:
        print(f"Error in plot_snowfall: {exc}")
        return None
    finally:
        if dataset is not None:
            dataset.close()
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
        weasd_grib = get_hrrr_grib(most_recent_run_time, step, variable_weasd)

        if weasd_grib:
            for region_name in ACTIVE_REGION_NAMES:
                plot_snowfall(weasd_grib, step, most_recent_run_time, region_name)
        else:
            print(
                f"Skipping forecast hour {step:02d} for run {most_recent_run_time.strftime('%Y-%m-%d %HZ')} "
                "because the snowfall file is not available."
            )

        if weasd_grib and os.path.exists(weasd_grib):
            os.remove(weasd_grib)

        gc.collect()

    prune_old_runs(MAX_SAVED_RUNS, keep_run_id=current_run_id)

print("HRRR snowfall processing complete.")
