#!/usr/bin/env python3
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Add current directory to path for module imports
sys.path.insert(0, os.path.dirname(__file__))

print("Starting KG-RAG Server...")
print("Python version:", sys.version)

try:
    print("Importing FastAPI...")
    import fastapi
    print("✓ FastAPI imported")

    print("Importing uvicorn...")
    import uvicorn
    print("✓ Uvicorn imported")

    print("Importing application...")
    from app import app
    print("✓ Application imported successfully")

    print("Launching server on http://0.0.0.0:8004 (Press Ctrl+C to stop)")
    uvicorn.run(app, host="0.0.0.0", port=8004, reload=False)

except Exception as e:
    print(f"❌ Startup error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
