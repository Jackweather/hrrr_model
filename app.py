import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from flask import Flask, abort

from routes import register_routes


app = Flask(__name__)
APP_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("HRRR_DATA_ROOT", "/var/data")).expanduser().resolve()
PNG_PATTERN = re.compile(r"_(\d+)\.png$", re.IGNORECASE)
RUN_ID_PATTERN = re.compile(r"^\d{8}_\d{2}z$", re.IGNORECASE)
LOG_DIR = Path(os.environ.get("HRRR_LOG_DIR", "logs")).expanduser().resolve()
EASTERN_TZ = ZoneInfo("America/New_York")
LEGACY_RUN_ID = "legacy"
DEFAULT_REGION_ID = "northeast"
DEFAULT_PRODUCT_ID = "mslp"
REGIONS = [
    {"id": "northeast", "label": "Northeast"},
]
REGION_IDS = {region["id"] for region in REGIONS}
PRODUCTS = [
    {
        "id": "mslp",
        "label": "MSLP / Precip",
        "data_dir": Path(os.environ.get("HRRR_MSLP_DATA_DIR", str(DATA_ROOT / "mslp_prate_csnow_EAST"))).expanduser().resolve(),
        "script_path": APP_ROOT / "mslp_prate_csnow_EAST.py",
    },
    {
        "id": "tmp2m",
        "label": "2 m Temperature",
        "data_dir": Path(os.environ.get("HRRR_TMP2M_DATA_DIR", str(DATA_ROOT / "tmp2m_EAST"))).expanduser().resolve(),
        "script_path": APP_ROOT / "tmp2m_EAST.py",
    },
    {
        "id": "vis",
        "label": "Visibility",
        "data_dir": Path(os.environ.get("HRRR_VIS_DATA_DIR", str(DATA_ROOT / "vis_EAST"))).expanduser().resolve(),
        "script_path": APP_ROOT / "vis_EAST.py",
    },
    {
        "id": "weasd",
        "label": "Snowfall 10:1",
        "data_dir": Path(os.environ.get("HRRR_WEASD_DATA_DIR", str(DATA_ROOT / "weasd_EAST"))).expanduser().resolve(),
        "script_path": APP_ROOT / "weasd_EAST.py",
    },
]
PRODUCT_IDS = {product["id"] for product in PRODUCTS}
PRODUCTS_BY_ID = {product["id"]: product for product in PRODUCTS}


def resolve_product_id(requested_product_id: str | None = None) -> str:
    if requested_product_id in PRODUCT_IDS:
        return str(requested_product_id)

    return DEFAULT_PRODUCT_ID


def get_product_config(product_id: str | None = None) -> dict:
    return PRODUCTS_BY_ID[resolve_product_id(product_id)]


def get_product_runs_dir(product_id: str | None = None) -> Path:
    product_config = get_product_config(product_id)
    return product_config["data_dir"] / "runs"


def get_product_png_dir(product_id: str | None = None) -> Path:
    product_config = get_product_config(product_id)
    return product_config["data_dir"] / "png"


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


def get_run_directory(product_id: str, run_id: str, region_id: str = DEFAULT_REGION_ID) -> Path | None:
    resolved_product_id = resolve_product_id(product_id)
    resolved_region_id = resolve_region_id(region_id)
    runs_dir = get_product_runs_dir(resolved_product_id)
    png_dir = get_product_png_dir(resolved_product_id)

    if run_id == LEGACY_RUN_ID:
        if resolved_region_id != DEFAULT_REGION_ID:
            return None
        return png_dir

    if parse_run_id(run_id) is None:
        return None

    run_png_dir = runs_dir / run_id / "png"
    region_dir = run_png_dir / resolved_region_id
    if region_dir.is_dir():
        return region_dir

    if resolved_region_id == DEFAULT_REGION_ID and run_png_dir.is_dir():
        return run_png_dir

    return None


def list_runs(product_id: str | None = None) -> list[dict[str, str | int | bool]]:
    resolved_product_id = resolve_product_id(product_id)
    runs_dir = get_product_runs_dir(resolved_product_id)
    png_dir = get_product_png_dir(resolved_product_id)
    run_items: list[dict[str, str | int | bool]] = []

    if runs_dir.is_dir():
        for run_dir in sorted(runs_dir.iterdir(), key=lambda path: path.name, reverse=True):
            if not run_dir.is_dir() or parse_run_id(run_dir.name) is None:
                continue

            image_dir = get_run_directory(resolved_product_id, run_dir.name, DEFAULT_REGION_ID)
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

    if not run_items and png_dir.is_dir():
        image_count = sum(1 for _ in png_dir.glob("*.png"))
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


def resolve_run_id(requested_run_id: str | None = None, product_id: str | None = None) -> str | None:
    run_items = list_runs(product_id)
    valid_run_ids = {item["id"] for item in run_items}

    if requested_run_id in valid_run_ids:
        return requested_run_id

    if run_items:
        return str(run_items[0]["id"])

    return None


def list_images(
    product_id: str | None = None,
    run_id: str | None = None,
    region_id: str | None = None,
) -> tuple[list[dict[str, str | int]], str | None, str, str]:
    resolved_product_id = resolve_product_id(product_id)
    resolved_run_id = resolve_run_id(run_id, resolved_product_id)
    resolved_region_id = resolve_region_id(region_id)
    image_dir = get_run_directory(resolved_product_id, resolved_run_id, resolved_region_id) if resolved_run_id else None

    if image_dir is None or not image_dir.is_dir():
        return [], resolved_run_id, resolved_region_id, resolved_product_id

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
                "url": f"/images/{resolved_product_id}/{resolved_run_id}/{resolved_region_id}/{image_path.name}?v={int(stat.st_mtime)}",
            }
        )

    return (
        sorted(image_items, key=lambda item: (item["frame"], item["filename"])),
        resolved_run_id,
        resolved_region_id,
        resolved_product_id,
    )


def run_scripts(scripts: list[tuple[str, str]], task_number: int, parallel: bool = False) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"task{task_number}.log"
    timing_path = LOG_DIR / f"task{task_number}_timing.txt"
    batch_started_at = datetime.now(timezone.utc)
    batch_started_perf = time.perf_counter()

    def format_duration(total_seconds: float) -> str:
        rounded_seconds = max(int(round(total_seconds)), 0)
        hours, remainder = divmod(rounded_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def write_timing_summary() -> None:
        total_seconds = time.perf_counter() - batch_started_perf
        completed_at = datetime.now(timezone.utc)
        summary_lines = [
            f"Task {task_number} timing summary",
            f"Start time UTC: {batch_started_at.strftime('%H:%M:%S')}",
            f"End time UTC: {completed_at.strftime('%H:%M:%S')}",
            f"Time taken: {format_duration(total_seconds)}",
        ]

        timing_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    with log_path.open("a", encoding="utf-8") as log_file:
        try:
            if parallel:
                processes: list[tuple[str, subprocess.Popen[str], float]] = []

                for script_path, working_dir in scripts:
                    log_file.write(f"Starting {script_path}\n")
                    log_file.flush()

                    started_perf = time.perf_counter()
                    try:
                        process = subprocess.Popen(
                            [sys.executable, script_path],
                            cwd=working_dir,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                        )
                        processes.append((script_path, process, started_perf))
                    except Exception as exc:
                        log_file.write(f"Failed to run {script_path}: {exc}\n\n")
                        log_file.flush()

                for script_path, process, started_perf in processes:
                    stdout, stderr = process.communicate()
                    if stdout:
                        log_file.write(stdout)
                    if stderr:
                        log_file.write(stderr)
                    log_file.write(f"Finished {script_path} with exit code {process.returncode}\n\n")
                    log_file.flush()

                return

            for script_path, working_dir in scripts:
                log_file.write(f"Starting {script_path}\n")
                log_file.flush()

                started_perf = time.perf_counter()
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
        finally:
            write_timing_summary()


@app.route("/run-task1")
def run_task1():
    scripts = [
        ("/opt/render/project/src/mslp_prate_csnow_EAST.py", "/opt/render/project/src"),
        ("/opt/render/project/src/tmp2m_EAST.py", "/opt/render/project/src"),
        ("/opt/render/project/src/vis_EAST.py", "/opt/render/project/src"),
        ("/opt/render/project/src/weasd_EAST.py", "/opt/render/project/src"),
    ]
    threading.Thread(target=run_scripts, args=(scripts, 3), kwargs={"parallel": True}, daemon=True).start()
    return "Task started in background! Check logs folder for output.", 200


@app.route("/run-task/<product_id>")
def run_task(product_id: str):
    resolved_product_id = resolve_product_id(product_id)
    product_config = PRODUCTS_BY_ID.get(resolved_product_id)
    if product_config is None:
        abort(404)

    script_path = product_config.get("script_path")
    if script_path is None:
        abort(404)

    scripts = [
        (str(script_path), str(Path(script_path).parent)),
    ]
    threading.Thread(target=run_scripts, args=(scripts, 1), daemon=True).start()
    return f"Started {resolved_product_id} task in background! Check logs folder for output.", 200


register_routes(
    app,
    default_region_id=DEFAULT_REGION_ID,
    default_product_id=DEFAULT_PRODUCT_ID,
    products=PRODUCTS,
    regions=REGIONS,
    get_product_png_dir=get_product_png_dir,
    get_run_directory=get_run_directory,
    list_images=list_images,
    list_runs=list_runs,
    resolve_product_id=resolve_product_id,
    resolve_region_id=resolve_region_id,
    resolve_run_id=resolve_run_id,
    run_scripts=run_scripts,
)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
