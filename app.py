"""One-command launcher:  python app.py

Starts the FastAPI backend and the Streamlit front end, then opens the browser. Ctrl+C stops both.
"""
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import requests

from backend.config.settings import get_settings

ROOT = Path(__file__).resolve().parent


def main() -> int:
    s = get_settings()
    env = os.environ.copy()
    env["API_URL"] = f"http://127.0.0.1:{s.api_port}"
    api = subprocess.Popen([sys.executable, str(ROOT / "main.py")], cwd=ROOT, env=env)
    for _ in range(60):                       # wait for the API (model loading takes a few seconds)
        try:
            if requests.get(f"{env['API_URL']}/health", timeout=1).ok:
                break
        except requests.RequestException:
            time.sleep(1)
    ui = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "frontend" / "streamlit_app.py"),
                           "--server.port", str(s.streamlit_port), "--server.address", "127.0.0.1",
                           "--browser.gatherUsageStats", "false", "--server.headless", "true"], cwd=ROOT, env=env)
    time.sleep(3)
    webbrowser.open(f"http://127.0.0.1:{s.streamlit_port}")
    try:
        ui.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in (ui, api):
            p.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
