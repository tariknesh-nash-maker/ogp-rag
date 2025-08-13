# start.py
import os, pathlib, subprocess, sys

PERSIST_DIR = pathlib.Path("chroma-db")
HAS_INDEX = PERSIST_DIR.exists() and any(PERSIST_DIR.rglob("*"))

if not HAS_INDEX:
    print("No index found → running ingest.py once…")
    subprocess.call([sys.executable, "ingest.py"])

subprocess.call([sys.executable, "-m", "streamlit", "run", "app.py",
                 "--server.port=7860", "--server.address=0.0.0.0"])

