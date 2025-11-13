"""WSGI entry point for the Render deployment.

Render's default Gunicorn command looks for `app:app`, so this module exposes
the Flask application defined in `api.py` under that path.
"""

import os

from api import app


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

