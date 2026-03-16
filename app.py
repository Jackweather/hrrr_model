import os
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from flask import Flask, abort, jsonify, render_template, request, send_from_directory


app = Flask(__name__)
DATA_DIR = Path(os.environ.get("HRRR_DATA_DIR", "/var/data/mslp_prate_csnow_EAST")).expanduser().resolve()
RUNS_DIR = Path(os.environ.get("HRRR_RUNS_DIR", str(DATA_DIR / "runs"))).expanduser().resolve()
PNG_DIR = Path(os.environ.get("HRRR_PNG_DIR", str(DATA_DIR / "png"))).expanduser().resolve()
PNG_PATTERN = re.compile(r"_(\d+)\.png$", re.IGNORECASE)
RUN_ID_PATTERN = re.compile(r"^\d{8}_\d{2}z$", re.IGNORECASE)
LOG_DIR = Path(os.environ.get("HRRR_LOG_DIR", "logs")).expanduser().resolve()
EASTERN_TZ = ZoneInfo("America/New_York")
LEGACY_RUN_ID = "legacy"
DEFAULT_REGION_ID = "northeast"
REGIONS = [
    {"id": "northeast", "label": "Northeast"},
    {"id": "conus", "label": "CONUS"},
]
REGION_IDS = {region["id"] for region in REGIONS}


def parse_run_id(run_id: str) -> datetime | None:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        return None

    try:
        return datetime.strptime(run_id[:-1], "%Y%m%d_%H").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def format_run_label(run_id: str) -> str:
    if run_id == LEGACY_RUN_ID:
        return "Current images"

    run_time_utc = parse_run_id(run_id)
    if run_time_utc is None:
        return run_id

    run_time_eastern = run_time_utc.astimezone(EASTERN_TZ)
    return f"{run_time_utc.strftime('%HZ %b %d')} | {run_time_eastern.strftime('%a %I %p %Z')}"


def resolve_region_id(requested_region_id: str | None = None) -> str:
    if requested_region_id in REGION_IDS:
        return str(requested_region_id)

    return DEFAULT_REGION_ID


def get_run_directory(run_id: str, region_id: str = DEFAULT_REGION_ID) -> Path | None:
    resolved_region_id = resolve_region_id(region_id)

    if run_id == LEGACY_RUN_ID:
        if resolved_region_id != DEFAULT_REGION_ID:
            return None
        return PNG_DIR

    if parse_run_id(run_id) is None:
        return None

    run_png_dir = RUNS_DIR / run_id / "png"
    region_dir = run_png_dir / resolved_region_id
    if region_dir.is_dir():
        return region_dir

    if resolved_region_id == DEFAULT_REGION_ID and run_png_dir.is_dir():
        return run_png_dir

    return None


def list_runs() -> list[dict[str, str | int | bool]]:
    run_items: list[dict[str, str | int | bool]] = []

    if RUNS_DIR.is_dir():
        for run_dir in sorted(RUNS_DIR.iterdir(), key=lambda path: path.name, reverse=True):
            if not run_dir.is_dir() or parse_run_id(run_dir.name) is None:
                continue

            image_dir = get_run_directory(run_dir.name, DEFAULT_REGION_ID)
            if image_dir is None or not image_dir.exists():
                continue

            image_count = sum(1 for _ in image_dir.glob("*.png"))
            run_items.append(
                {
                    "id": run_dir.name,
                    "label": format_run_label(run_dir.name),
                    "image_count": image_count,
                }
            )

    if not run_items and PNG_DIR.is_dir():
        image_count = sum(1 for _ in PNG_DIR.glob("*.png"))
        run_items.append(
            {
                "id": LEGACY_RUN_ID,
                "label": format_run_label(LEGACY_RUN_ID),
                "image_count": image_count,
            }
        )

    for index, item in enumerate(run_items):
        item["is_latest"] = index == 0

    return run_items


def resolve_run_id(requested_run_id: str | None = None) -> str | None:
    run_items = list_runs()
    valid_run_ids = {item["id"] for item in run_items}

    if requested_run_id in valid_run_ids:
        return requested_run_id

    if run_items:
        return str(run_items[0]["id"])

    return None


def list_images(run_id: str | None = None, region_id: str | None = None) -> tuple[list[dict[str, str | int]], str | None, str]:
    resolved_run_id = resolve_run_id(run_id)
    resolved_region_id = resolve_region_id(region_id)
    image_dir = get_run_directory(resolved_run_id, resolved_region_id) if resolved_run_id else None

    if image_dir is None or not image_dir.is_dir():
        return [], resolved_run_id, resolved_region_id

    image_items = []
    for image_path in sorted(image_dir.glob("*.png"), key=lambda path: path.name):
        match = PNG_PATTERN.search(image_path.name)
        frame = int(match.group(1)) if match else -1
        stat = image_path.stat()
        image_items.append(
            {
                "filename": image_path.name,
                "frame": frame,
                "label": f"Hour {frame:02d}",
                "url": f"/images/{resolved_run_id}/{resolved_region_id}/{image_path.name}?v={int(stat.st_mtime)}",
            }
        )

    return sorted(image_items, key=lambda item: (item["frame"], item["filename"])), resolved_run_id, resolved_region_id


def run_scripts(scripts: list[tuple[str, str]], task_number: int) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"task{task_number}.log"

    with log_path.open("a", encoding="utf-8") as log_file:
        for script_path, working_dir in scripts:
            log_file.write(f"Starting {script_path}\n")
            log_file.flush()

            try:
                result = subprocess.run(
                    [sys.executable, script_path],
                    cwd=working_dir,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.stdout:
                    log_file.write(result.stdout)
                if result.stderr:
                    log_file.write(result.stderr)
                log_file.write(f"Finished {script_path} with exit code {result.returncode}\n")
            except Exception as exc:
                log_file.write(f"Failed to run {script_path}: {exc}\n")

            log_file.write("\n")
            log_file.flush()


@app.route("/")
def index():
    runs = list_runs()
    selected_run = resolve_run_id(request.args.get("run"))
    selected_region = resolve_region_id(request.args.get("region"))
    images, resolved_run_id, resolved_region_id = list_images(selected_run, selected_region)
    image_dir = get_run_directory(resolved_run_id, resolved_region_id) if resolved_run_id else RUNS_DIR
    return render_template(
        "index.html",
        images=images,
        image_count=len(images),
        png_dir=str(image_dir),
        runs=runs,
        regions=REGIONS,
        selected_run=resolved_run_id,
        selected_region=resolved_region_id,
    )


@app.route("/api/runs")
def api_runs():
    return jsonify({"runs": list_runs()})


@app.route("/api/images")
def api_images():
    images, resolved_run_id, resolved_region_id = list_images(request.args.get("run"), request.args.get("region"))
    return jsonify({"run_id": resolved_run_id, "region_id": resolved_region_id, "images": images})


@app.route("/run-task1")
def run_task1():
    scripts = [
        ("/opt/render/project/src/mslp_prate_csnow_EAST.py", "/opt/render/project/src"),
        
    ]
    threading.Thread(target=run_scripts, args=(scripts, 1), daemon=True).start()
    return "Task started in background! Check logs folder for output.", 200


@app.route("/images/<run_id>/<region_id>/<path:filename>")
def serve_image(run_id: str, region_id: str, filename: str):
    image_dir = get_run_directory(run_id, region_id)
    if image_dir is None or not image_dir.is_dir():
        abort(404)

    return send_from_directory(image_dir, filename)


@app.route("/images/<run_id>/<path:filename>")
def serve_legacy_image(run_id: str, filename: str):
    image_dir = get_run_directory(run_id, DEFAULT_REGION_ID)
    if image_dir is None or not image_dir.is_dir():
        abort(404)

    return send_from_directory(image_dir, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
