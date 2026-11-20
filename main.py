"""Start the backend API only:  python main.py   (docs at http://127.0.0.1:8000/docs)

Binds to 127.0.0.1 by default. To serve other PCs on the studio LAN set API_HOST=0.0.0.0 in .env *and*
put authentication in front of it first (see README, "Future studio integration").
"""
import uvicorn

from backend.config.settings import get_settings

if __name__ == "__main__":
    s = get_settings()
    uvicorn.run("backend.api.main:app", host=s.api_host, port=s.api_port, log_level="info")
