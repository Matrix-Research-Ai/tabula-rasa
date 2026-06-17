"""Test retrain directly in the running process."""
import sys, json, time, subprocess
from pathlib import Path

root = Path(__file__).parent
port = "8005"

# Start server
proc = subprocess.Popen(
    [sys.executable, "egefalos/tabula_rasa.py", "--port", port],
    cwd=str(root),
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True
)

# Wait for startup
time.sleep(8)

# Send 5 hello queries
for i in range(5):
    try:
        import urllib.request
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/ask",
            data=json.dumps({"question": "hello"}).encode(),
            headers={"Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(req, timeout=5)
        body = json.loads(resp.read())
        print(f"Q{i+1}: knows={body.get('knows')} answer={str(body.get('answer',''))[:30]}")
    except Exception as e:
        print(f"Q{i+1}: ERROR {e}")
    time.sleep(2)

# Give it time for output to flush
time.sleep(3)
proc.terminate()
stdout = proc.stdout.read() if proc.stdout else ""
# Print only retrace lines
for line in stdout.split("\n"):
    if "RETRACE" in line or "retraining" in line.lower() or "train:" in line:
        print(line)
