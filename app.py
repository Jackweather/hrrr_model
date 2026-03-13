import os
import re
import subprocess
import sys
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, send_from_directory


app = Flask(__name__)
PNG_DIR = Path(os.environ.get("HRRR_PNG_DIR", "/var/data/mslp_prate_csnow_EAST/png")).expanduser().resolve()
PNG_PATTERN = re.compile(r"_(\d+)\.png$", re.IGNORECASE)
LOG_DIR = Path(os.environ.get("HRRR_LOG_DIR", "logs")).expanduser().resolve()


def list_images() -> list[dict[str, str | int]]:
    if not PNG_DIR.is_dir():
        return []

    image_items = []
    for image_path in sorted(PNG_DIR.glob("*.png"), key=lambda path: path.name):
        match = PNG_PATTERN.search(image_path.name)
        frame = int(match.group(1)) if match else -1
        stat = image_path.stat()
        image_items.append(
            {
                "filename": image_path.name,
                "frame": frame,
                "label": f"Hour {frame:02d}",
                "url": f"/images/{image_path.name}?v={int(stat.st_mtime)}",
            }
        )

    return sorted(image_items, key=lambda item: (item["frame"], item["filename"]))


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
    images = list_images()
    return render_template("index.html", images=images, image_count=len(images), png_dir=str(PNG_DIR))


@app.route("/api/images")
def api_images():
    return jsonify(list_images())


@app.route("/run-task1")
def run_task1():
    scripts = [
        ("/opt/render/project/src/mslp_prate_csnow_EAST.py", "/opt/render/project/src"),
        
    ]
    threading.Thread(target=run_scripts, args=(scripts, 1), daemon=True).start()
    return "Task started in background! Check logs folder for output.", 200


@app.route("/images/<path:filename>")
def serve_image(filename: str):
    return send_from_directory(PNG_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)