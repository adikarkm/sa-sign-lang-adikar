import sys
import subprocess

def main():
    print("Launching Sign Language Web App locally...")
    
    # Try importing pyngrok for public URL sharing
    try:
        from pyngrok import ngrok
        # Launch public tunnel on port 7860
        public_url = ngrok.connect(7860)
        print(f"\n==========================================")
        print(f"  PUBLIC INTERNET URL: {public_url}")
        print(f"  Share this link with anyone to test your app online!")
        print(f"==========================================\n")
    except Exception as e:
        print(f"Note: ngrok tunnel skipped ({e}). App running on localhost.")

    # Run app.py
    subprocess.run([sys.executable, "app.py"])

if __name__ == "__main__":
    main()
