"""Run the whole pipeline end to end (about 10 minutes; the scenario step is the slow one)."""
import subprocess, sys
from pathlib import Path
src = Path(__file__).parent/"src"
for step in ["datacube.py", "heat_model.py", "scenarios.py", "build_dashboard.py"]:
    print(f"\n=== {step} ===", flush=True)
    subprocess.run([sys.executable, step], cwd=src, check=True)
print("\nDone. Open outputs/dashboard.html in a browser.")
