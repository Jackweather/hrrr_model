from typing import Callable

from flask import abort, jsonify, render_template, request, send_from_directory


def register_routes(
    app,
    *,
    default_region_id: str,
    default_product_id: str,
    products: list[dict[str, str]],
    regions: list[dict[str, str]],
    get_product_png_dir: Callable,
    get_run_directory: Callable,
    list_images: Callable,
    list_runs: Callable,
    resolve_product_id: Callable,
    resolve_region_id: Callable,
    resolve_run_id: Callable,
    run_scripts: Callable,
):
    @app.route("/")
    def index():
        selected_product = resolve_product_id(request.args.get("product"))
        product_items = [{"id": product["id"], "label": product["label"]} for product in products]
        runs = list_runs(selected_product)
        selected_run = resolve_run_id(request.args.get("run"), selected_product)
        selected_region = resolve_region_id(request.args.get("region"))
        images, resolved_run_id, resolved_region_id, resolved_product_id = list_images(
            selected_product,
            selected_run,
            selected_region,
        )
        image_dir = (
            get_run_directory(resolved_product_id, resolved_run_id, resolved_region_id)
            if resolved_run_id
            else get_product_png_dir(resolved_product_id)
        )
        return render_template(
            "index.html",
            images=images,
            image_count=len(images),
            png_dir=str(image_dir),
            products=product_items,
            runs=runs,
            regions=regions,
            selected_product=resolved_product_id,
            selected_run=resolved_run_id,
            selected_region=resolved_region_id,
        )

    @app.route("/api/runs")
    def api_runs():
        product_id = resolve_product_id(request.args.get("product"))
        return jsonify({"product_id": product_id, "runs": list_runs(product_id)})

    @app.route("/api/images")
    def api_images():
        images, resolved_run_id, resolved_region_id, resolved_product_id = list_images(
            request.args.get("product"),
            request.args.get("run"),
            request.args.get("region"),
        )
        return jsonify(
            {
                "product_id": resolved_product_id,
                "run_id": resolved_run_id,
                "region_id": resolved_region_id,
                "images": images,
            }
        )

    @app.route("/images/<product_id>/<run_id>/<region_id>/<path:filename>")
    def serve_image(product_id: str, run_id: str, region_id: str, filename: str):
        image_dir = get_run_directory(product_id, run_id, region_id)
        if image_dir is None or not image_dir.is_dir():
            abort(404)

        return send_from_directory(image_dir, filename)

    @app.route("/images/<run_id>/<region_id>/<path:filename>")
    def serve_default_product_image(run_id: str, region_id: str, filename: str):
        image_dir = get_run_directory(default_product_id, run_id, region_id)
        if image_dir is None or not image_dir.is_dir():
            abort(404)

        return send_from_directory(image_dir, filename)

    @app.route("/images/<run_id>/<path:filename>")
    def serve_legacy_image(run_id: str, filename: str):
        image_dir = get_run_directory(default_product_id, run_id, default_region_id)
        if image_dir is None or not image_dir.is_dir():
            abort(404)

        return send_from_directory(image_dir, filename)
