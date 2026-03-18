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
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

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
REGION_SUBSET_PADDING_DEGREES = 1.5
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


BASE_DIR = "/var/data"
RUN_ROOT_DIR = os.path.join(BASE_DIR, "tmp2m_EAST")
RUNS_DIR = os.path.join(RUN_ROOT_DIR, "runs")
MAX_SAVED_RUNS = 7

grib_dir = os.path.join(RUN_ROOT_DIR, "grib")
os.makedirs(grib_dir, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

png_dirs = {}
processed_steps_file = None

base_url_hrrr = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl"
variable_tmp = "TMP"
RUN_AVAILABILITY_DELAY_MINUTES = 50

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
        except Exception as exc:
            print(f"Failed to remove archived run {run_name}: {exc}")


def is_valid_run(run_time):
    test_date_str = run_time.strftime("%Y%m%d")
    test_hour_str = run_time.strftime("%H")
    test_file_name = f"hrrr.t{test_hour_str}z.wrfsfcf01.grib2"
    test_url = (
        f"{base_url_hrrr}?file={test_file_name}"
        f"&lev_2_m_above_ground=on"
        f"&var_TMP=on"
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
most_recent_run_time = None

print(f"Current time UTC: {current_utc_time.strftime('%Y-%m-%d %HZ')}")
print(f"Current time Eastern: {current_eastern_time.strftime('%Y-%m-%d %I:%M %p %Z')}")
print(
    f"Selecting from runs at least {RUN_AVAILABILITY_DELAY_MINUTES} minutes old: "
    f"{selection_reference_utc.strftime('%Y-%m-%d %HZ')} "
    f"({format_eastern_time(selection_reference_utc)})"
)
print_run_hour_mapping(current_utc_time)

for offset in range(24):
    candidate_time = selection_reference_utc - timedelta(hours=offset)
    if candidate_time.hour in available_hours and is_valid_run(candidate_time):
        most_recent_run_time = candidate_time
        break

if most_recent_run_time is None:
    print("No valid run time found in the available hours. Searching for fallback run hour one hour at a time.")
    fallback_found = False
    for offset in range(1, 25):
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


for region_name, region_config in REGION_CONFIGS.items():
    region_extent = region_config["extent"]
    region_config["plot_extent"] = (
        expand_extent_to_aspect(region_extent, TARGET_PLOT_ASPECT)
        if region_name == "conus"
        else region_extent
    )

    counties_gdf = region_config.get("counties_gdf")
    states_gdf = region_config.get("states_gdf")
    region_config["county_geometries"] = tuple(counties_gdf.geometry) if counties_gdf is not None else ()
    region_config["state_geometries"] = tuple(states_gdf.geometry) if states_gdf is not None else ()


temperature_levels = [-20, 0, 10, 20, 32, 40, 50, 60, 70, 80, 90, 100]
temperature_tick_levels = temperature_levels
temperature_cmap = LinearSegmentedColormap.from_list(
    "temp_cmap",
    [
        "#08306b", "#2171b5", "#6baed6", "#b3cde3", "#ffffff",
        "#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026",
    ],
    N=256,
)
temperature_norm = BoundaryNorm(temperature_levels, temperature_cmap.N)
temperature_contour_levels = np.arange(-20, 101, 10)
temperature_contour_smoothing_sigma = 1.1

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
        f"&lev_2_m_above_ground=on"
        f"&var_{variable}=on"
        f"&dir=%2Fhrrr.{date_str}%2Fconus"
    )
    return download_grib(url, file_path)


def get_temperature_variable(dataset):
    for variable_name in ("t2m", "t", "tmp", "2t"):
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


def load_temperature_plot_data(tmp_path):
    dataset = None
    try:
        dataset = xr.open_dataset(tmp_path, engine="cfgrib", chunks={})
        temperature_field = get_temperature_variable(dataset)
        latitude_field, longitude_field = get_lat_lon_fields(dataset)
        if temperature_field is None or latitude_field is None or longitude_field is None:
            print("Required temperature coordinates or variable are not present in the dataset.")
            return None

        temperature_kelvin = temperature_field.values.squeeze()
        temperature_fahrenheit = ((temperature_kelvin - 273.15) * 9.0 / 5.0) + 32.0
        temperature_contours = gaussian_filter(temperature_fahrenheit, sigma=temperature_contour_smoothing_sigma)

        lats = latitude_field.values
        lons = longitude_field.values
        lons_plot = np.where(lons > 180, lons - 360, lons)

        if lats.ndim == 1 and lons.ndim == 1:
            lon2d, lat2d = np.meshgrid(lons_plot, lats)
        else:
            lon2d, lat2d = lons_plot, lats

        return {
            "temperature_fahrenheit": temperature_fahrenheit,
            "temperature_contours": temperature_contours,
            "lon2d": lon2d,
            "lat2d": lat2d,
        }
    except Exception as exc:
        print(f"Error loading temperature data: {exc}")
        return None
    finally:
        if dataset is not None:
            dataset.close()


def get_region_plot_data(plot_data, extent, padding_degrees=REGION_SUBSET_PADDING_DEGREES):
    min_lon, max_lon, min_lat, max_lat = extent
    lon2d = plot_data["lon2d"]
    lat2d = plot_data["lat2d"]

    region_mask = (
        (lon2d >= (min_lon - padding_degrees))
        & (lon2d <= (max_lon + padding_degrees))
        & (lat2d >= (min_lat - padding_degrees))
        & (lat2d <= (max_lat + padding_degrees))
    )

    if not np.any(region_mask):
        return plot_data

    row_indices = np.where(np.any(region_mask, axis=1))[0]
    col_indices = np.where(np.any(region_mask, axis=0))[0]
    if row_indices.size == 0 or col_indices.size == 0:
        return plot_data

    row_slice = slice(row_indices[0], row_indices[-1] + 1)
    col_slice = slice(col_indices[0], col_indices[-1] + 1)
    return {
        "temperature_fahrenheit": plot_data["temperature_fahrenheit"][row_slice, col_slice],
        "temperature_contours": plot_data["temperature_contours"][row_slice, col_slice],
        "lon2d": plot_data["lon2d"][row_slice, col_slice],
        "lat2d": plot_data["lat2d"][row_slice, col_slice],
    }


def add_region_overlays(axis, region_config):
    county_geometries = region_config.get("county_geometries")
    state_geometries = region_config.get("state_geometries")

    if county_geometries and state_geometries:
        axis.add_geometries(
            county_geometries,
            ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="gray",
            linewidth=0.3,
            zorder=7,
        )
        axis.add_geometries(
            state_geometries,
            ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="#000000",
            linewidth=1.0,
            zorder=8,
        )
        return

    axis.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="#444444", zorder=7)
    axis.add_feature(cfeature.STATES, linewidth=0.35, edgecolor="#666666", zorder=8)


def plot_temperature(plot_data, step, run_time, region_name):
    figure = None
    try:
        region_config = REGION_CONFIGS[region_name]
        extent = region_config["extent"]
        plot_extent = region_config["plot_extent"]
        region_plot_data = get_region_plot_data(plot_data, plot_extent) if region_name != "conus" else plot_data

        valid_time = run_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=step)
        local_valid = valid_time.astimezone(EASTERN_TZ)
        local_time = local_valid.strftime("%I %p")
        day_of_week = local_valid.strftime("%A")
        _, hour_str = get_run_strings(run_time)

        title = (
            f"HRRR 2 m Temperature - {region_config['title']}\n"
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
            region_plot_data["lon2d"],
            region_plot_data["lat2d"],
            region_plot_data["temperature_fahrenheit"],
            levels=temperature_levels,
            cmap=temperature_cmap,
            norm=temperature_norm,
            extend="both",
            transform=ccrs.PlateCarree(),
            alpha=0.9,
            zorder=1,
        )
        contours = axis.contour(
            region_plot_data["lon2d"],
            region_plot_data["lat2d"],
            region_plot_data["temperature_contours"],
            levels=temperature_contour_levels,
            colors="black",
            linewidths=0.55,
            alpha=0.7,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
        axis.clabel(contours, fmt="%d", fontsize=6, colors="black", inline=True)

        try:
            add_region_overlays(axis, region_config)
        except Exception as exc:
            print(f"Error plotting overlays: {exc}")

        plt.draw()
        colorbar_y = 0.035 if region_name != "conus" else max(axis.get_position().y0 - 0.045, 0.035)
        colorbar_axis = figure.add_axes([0.18, colorbar_y, 0.64, 0.02])
        colorbar = plt.colorbar(mesh, cax=colorbar_axis, orientation="horizontal", ticks=temperature_tick_levels)
        colorbar.set_label("2 m Temperature (deg F)", fontsize=7, labelpad=2)
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

        png_path = os.path.join(png_dirs[region_name], f"hrrr_tmp2m_{region_name}_{step:02d}.png")
        plt.savefig(png_path, bbox_inches="tight", dpi=300)
        print(f"Generated PNG: {png_path}")
        return png_path
    except Exception as exc:
        print(f"Error in plot_temperature: {exc}")
        return None
    finally:
        if figure is not None:
            plt.close(figure)


def download_all_gribs(run_time, forecast_steps, variable):
    downloaded_gribs = {}
    print("Starting GRIB download phase.")

    for step_group in forecast_steps:
        for step in step_group:
            grib_path = get_hrrr_grib(run_time, step, variable)
            if grib_path:
                downloaded_gribs[step] = grib_path
            else:
                print(
                    f"Skipping forecast hour {step:02d} for run {run_time.strftime('%Y-%m-%d %HZ')} "
                    "because the temperature file is not available."
                )

    return downloaded_gribs


def generate_all_pngs(downloaded_gribs, run_time):
    print("Starting PNG generation phase.")

    for step_group in forecast_steps:
        for step in step_group:
            grib_path = downloaded_gribs.get(step)
            if not grib_path:
                continue

            plot_data = load_temperature_plot_data(grib_path)
            if plot_data is None:
                continue

            for region_name in REGION_CONFIGS:
                plot_temperature(plot_data, step, run_time, region_name)

            if os.path.exists(grib_path):
                os.remove(grib_path)

            gc.collect()

        prune_old_runs(MAX_SAVED_RUNS, keep_run_id=current_run_id)


def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"Cleared folder: {folder_path}")


clear_folder(grib_dir)

downloaded_gribs = download_all_gribs(most_recent_run_time, forecast_steps, variable_tmp)
generate_all_pngs(downloaded_gribs, most_recent_run_time)

print("HRRR 2 m temperature processing complete.")
