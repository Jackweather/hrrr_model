import os
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests


EASTERN_TZ = ZoneInfo("America/New_York")
BASE_DIR = os.environ.get("HRRR_DATA_ROOT", "/var/data")
RUN_SELECTION_FILE = os.path.join(BASE_DIR, "east_selected_run.txt")
RUN_AVAILABILITY_DELAY_MINUTES = 50
RUN_SELECTION_OVERRIDE_WINDOW_MINUTES = 59
LOCAL_RUN_SELECTION_OVERRIDES = (
    ((15, 45), 18),
    ((21, 45), 12),
    ((3, 45), 0),
    ((9, 45), 6),
)
PREFERRED_RUN_HOURS = (0, 6, 12, 18)
RUN_ID_PATTERN = re.compile(r"^\d{8}_\d{2}z$")


def format_eastern_time(dt_utc):
    return dt_utc.astimezone(EASTERN_TZ).strftime("%Y-%m-%d %I:%M %p %Z")


def floor_to_hour(dt_value):
    return dt_value.replace(minute=0, second=0, microsecond=0)


def get_run_id(run_time):
    return f"{run_time.strftime('%Y%m%d_%H')}z"


def get_forced_run_time(now_utc, now_local):
    for (start_hour, start_minute), run_hour in LOCAL_RUN_SELECTION_OVERRIDES:
        window_start_local = now_local.replace(
            hour=start_hour,
            minute=start_minute,
            second=0,
            microsecond=0,
        )
        window_end_local = window_start_local + timedelta(minutes=RUN_SELECTION_OVERRIDE_WINDOW_MINUTES)
        if window_start_local <= now_local < window_end_local:
            forced_run_time = now_utc.replace(hour=run_hour, minute=0, second=0, microsecond=0)
            if forced_run_time > now_utc:
                forced_run_time -= timedelta(days=1)
            return floor_to_hour(forced_run_time), window_start_local, window_end_local

    return None, None, None


def get_latest_preferred_run_anchor(reference_utc):
    for run_hour in reversed(PREFERRED_RUN_HOURS):
        if reference_utc.hour >= run_hour:
            return reference_utc.replace(hour=run_hour, minute=0, second=0, microsecond=0)

    previous_day = reference_utc - timedelta(days=1)
    return previous_day.replace(hour=PREFERRED_RUN_HOURS[-1], minute=0, second=0, microsecond=0)


def is_valid_run(run_time):
    date_str = run_time.strftime("%Y%m%d")
    hour_str = run_time.strftime("%H")
    url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.{date_str}/conus/hrrr.t{hour_str}z.wrfsfcf01.grib2"
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        return response.status_code == 200
    except Exception:
        return False


def select_run_time(now_utc=None):
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    current_utc_time = floor_to_hour(now_utc)
    current_eastern_time = current_utc_time.astimezone(EASTERN_TZ)
    selection_reference_utc = floor_to_hour(now_utc - timedelta(minutes=RUN_AVAILABILITY_DELAY_MINUTES))

    print(f"Current time UTC: {current_utc_time.strftime('%Y-%m-%d %HZ')}")
    print(f"Current time Eastern: {current_eastern_time.strftime('%Y-%m-%d %I:%M %p %Z')}")
    print(
        f"Selecting from runs at least {RUN_AVAILABILITY_DELAY_MINUTES} minutes old: "
        f"{selection_reference_utc.strftime('%Y-%m-%d %HZ')} "
        f"({format_eastern_time(selection_reference_utc)})"
    )

    # Try each hour from the reference hour down to 0, then previous day if needed
    for offset in range(0, 25):
        candidate_time = selection_reference_utc - timedelta(hours=offset)
        if is_valid_run(candidate_time):
            print(f"Selected run: {candidate_time.strftime('%Y-%m-%d %HZ')} ({format_eastern_time(candidate_time)})")
            return candidate_time

    raise ValueError("No valid run time found within the last 25 hours.")


def write_selected_run_file(run_time, file_path=RUN_SELECTION_FILE):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    run_id = get_run_id(run_time)
    with open(file_path, "w", encoding="utf-8") as output_file:
        output_file.write(f"{run_id}\n")
    return run_id


def read_selected_run_time(file_path=RUN_SELECTION_FILE):
    with open(file_path, "r", encoding="utf-8") as input_file:
        run_id = input_file.read().strip()

    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"Invalid run id in {file_path}: {run_id!r}")

    return datetime.strptime(run_id[:-1], "%Y%m%d_%H").replace(tzinfo=timezone.utc)


def main():
    selected_run_time = select_run_time()
    run_id = write_selected_run_file(selected_run_time)
    print(
        f"Selected HRRR run: {selected_run_time.strftime('%Y-%m-%d %HZ')} "
        f"({format_eastern_time(selected_run_time)})"
    )
    print(f"Wrote selected run to {RUN_SELECTION_FILE}: {run_id}")


if __name__ == "__main__":
    main()