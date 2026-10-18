"""Serve the production site under the GitHub Pages project subpath.

The server only exposes the allowlisted static files. It is a local preview,
not a deployment.
"""

from __future__ import annotations

import argparse
import shutil
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.config import PROJECT_ROOT
from src.pages.schema import PROJECT_SUBPATH, allowed_relative_files
from src.pages.verify import verify_site

PREVIEW_ROOT = PROJECT_ROOT / ".pages-preview"


def assemble_preview(site_dir: Path, destination_root: Path) -> Path:
    """Copy the allowlisted site into ``destination_root/<project-subpath>/``."""
    verify_site(site_dir)
    target = destination_root / PROJECT_SUBPATH
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for relative in allowed_relative_files():
        source = site_dir / relative
        if not source.is_file():
            raise FileNotFoundError(relative)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return destination_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview the Pages site locally")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--site", type=Path, default=PROJECT_ROOT / "site")
    args = parser.parse_args()
    root = assemble_preview(args.site, PREVIEW_ROOT)
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/{PROJECT_SUBPATH}/"
    print(url, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
