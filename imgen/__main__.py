"""CLI entry: `python -m imgen`."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import webbrowser

from . import __version__
from .constants import APP_NAME, DEFAULT_HOST, DEFAULT_PORT


def _enable_windows_utf8() -> None:
    if sys.platform != "win32":
        return
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="imgen",
        description="IMGEN — local visual studio for Qwen-Image-2.1",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="HTTP port (default 9100)")
    parser.add_argument("--demo", action="store_true", help="UI demo mode, no GPU / weights required")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the system browser")
    parser.add_argument("--check", action="store_true", help="Print device and model status, then exit")
    parser.add_argument("--version", action="store_true")
    return parser


def _check() -> int:
    from .config import ConfigStore
    from .device import probe
    from .hub import model_status
    from .paths import AppPaths

    paths = AppPaths()
    cfg = ConfigStore(paths)
    cfg.load()
    print(f"{APP_NAME} {__version__}")
    print(f"home     {paths.home}")
    device = probe()
    print(f"python   {device['python']}")
    print(f"torch    {device['torch']}")
    print(f"device   {device['device']}  {device['device_name']}")
    if device.get("vram_gb"):
        print(f"vram     {device['vram_gb']} GiB")
    hub = cfg._data.get("hub") or "huggingface"
    print(f"hub      {hub}")
    for row in model_status(hub):
        mark = "yes" if row["downloaded"] else "no"
        print(f"model    {row['label']:18} downloaded={mark:3}  {row['repo']}")
    for warning in device.get("warnings") or []:
        print(f"note     {warning}")
    return 0


def main(argv: list[str] | None = None) -> int:
    _enable_windows_utf8()
    args = build_parser().parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    if args.check:
        return _check()
    if args.demo:
        os.environ["IMGEN_DEMO"] = "1"

    import uvicorn

    from .app import create_app

    app = create_app(demo=args.demo)
    url = f"http://{args.host}:{args.port}"

    if not args.no_browser:

        def _open() -> None:
            time.sleep(1.1)
            try:
                webbrowser.open(url)
            except Exception:
                pass

        threading.Thread(target=_open, daemon=True).start()

    print(f"{APP_NAME} {__version__}")
    print(f"Open {url}")
    if args.demo:
        print("Demo mode: generation is simulated; no model weights are loaded.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
