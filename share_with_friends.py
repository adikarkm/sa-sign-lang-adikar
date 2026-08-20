import os
import sys
import time
import subprocess
import threading

def run_https_tunnel():
    time.sleep(2)
    print("\n" + "="*65)
    print("  🚀 GENERATING FREE HTTPS LINK FOR MOBILE PHONES & FRIENDS...")
    print("="*65 + "\n")
    
    # Launch Pinggy HTTPS tunnel via built-in Windows SSH
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-p", "443", "-R", "0:localhost:5000", "qr@a.pinggy.io"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in iter(proc.stdout.readline, ''):
            if line:
                print(line.strip())
    except Exception as e:
        print("Tunnel error:", e)

if __name__ == "__main__":
    # Start HTTPS tunnel in background thread
    t = threading.Thread(target=run_https_tunnel, daemon=True)
    t.start()

    # Launch Flask App
    from app_flask import app
    app.run(host="0.0.0.0", port=5000, debug=False)
