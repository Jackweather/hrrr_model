from pathlib import Path
from typing import Callable

from flask import abort, jsonify, render_template, request, send_from_directory


def register_routes(
    app,
    *,
    default_region_id: str,
    regions: list[dict[str, str]],
    runs_dir: Path,
    get_run_directory: Callable,
    list_images: Callable,
    list_runs: Callable,
    resolve_region_id: Callable,
    resolve_run_id: Callable,
    run_scripts: Callable,
):
    @app.route("/")
    def index():
        runs = list_runs()
        selected_run = resolve_run_id(request.args.get("run"))
        selected_region = resolve_region_id(request.args.get("region"))
        images, resolved_run_id, resolved_region_id = list_images(selected_run, selected_region)
        image_dir = get_run_directory(resolved_run_id, resolved_region_id) if resolved_run_id else runs_dir
        return render_template(
            "index.html",
            images=images,
            image_count=len(images),
            png_dir=str(image_dir),
            runs=runs,
            regions=regions,
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
        import threading

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
        image_dir = get_run_directory(run_id, default_region_id)
        if image_dir is None or not image_dir.is_dir():
            abort(404)

        return send_from_directory(image_dir, filename)