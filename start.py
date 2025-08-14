# start.py
import sys
# Force Python to use the modern SQLite from pysqlite3-binary
sys.modules["sqlite3"] = __import__("pysqlite3")

import os, pathlib, subprocess, sys as _sys
PERSIST_DIR = pathlib.Path("chroma-db")
HAS_INDEX = PERSIST_DIR.exists() and any(PERSIST_DIR.rglob("*"))

if not HAS_INDEX:
    print("No index found → running ingest.py once…")
    env = os.environ.copy()
    # Optional: cap ingestion for faster first boot. Tune/remove later.
    env.setdefault("INGEST_LIMIT", "800")
    subprocess.call([_sys.executable, "ingest.py"], env=env)

# On Streamlit Cloud do NOT force a port; let the platform choose it.
subprocess.call([_sys.executable, "-m", "streamlit", "run", "app.py"])

