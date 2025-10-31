# client.py
# pip install requests pandas

import argparse, json, sys                 # CLI parsing, JSON pretty print, stderr
import pandas as pd                        # CSV reading
import requests                            # HTTP client

def csv_to_sessions(csv_path: str) -> list[dict]:
    """Read CSV and convert rows into a list of dicts suitable for API JSON."""
    df = pd.read_csv(csv_path)             # Load CSV into DataFrame
    df = df.where(pd.notnull(df), None)    # Convert NaN to None for JSON serialization
    sessions = df.to_dict(orient="records")# Each row -> dict
    return sessions                        # Return list of row dicts

def pretty_print(obj):
    """Helper to print JSON with indentation and UTF-8 preserved."""
    print(json.dumps(obj, ensure_ascii=False, indent=2))  # Pretty JSON to stdout

def do_train(url: str, models_dir: str, csv_path: str):
    """Send /train request built from a CSV file."""
    sessions = csv_to_sessions(csv_path)   # Build session list from CSV
    payload = {                            # Compose request body
        "models_dir": models_dir,
        "sessions": sessions
    }
    print("=== TRAIN payload ===")         # Show the request JSON (debug)
    pretty_print(payload)
    r = requests.post(f"{url.rstrip('/')}/train", json=payload, timeout=60)  # POST to /train
    try:
        r.raise_for_status()               # Raise on HTTP error
    except Exception as e:
        print("Server error:", r.status_code, r.text, file=sys.stderr)  # Log error response
        raise                              # Re-raise to fail fast
    print("=== TRAIN response ===")        # Show server response JSON
    pretty_print(r.json())

def do_eval(url: str, models_dir: str, csv_path: str, username: str, contract: str, k: float, explain: bool, topN: int):
    """Send /eval request for a specific user key with sessions from CSV."""
    sessions = csv_to_sessions(csv_path)   # Build session list from CSV
    payload = {                            # Compose request body
        "models_dir": models_dir,
        "username": username,
        "contract": contract,
        "k": k,
        "explain": explain,
        "topN": topN,
        "sessions": sessions
    }
    print("=== EVAL payload ===")          # Show the request JSON (debug)
    pretty_print(payload)
    r = requests.post(f"{url.rstrip('/')}/eval", json=payload, timeout=60)  # POST to /eval
    try:
        r.raise_for_status()               # Raise on HTTP error
    except Exception as e:
        print("Server error:", r.status_code, r.text, file=sys.stderr)  # Log error response
        raise                              # Re-raise to fail fast
    resp = r.json()                        # Parse JSON response
    print("=== EVAL response (compact) ===")   # Show compact section
    pretty_print(resp.get("compact"))
    print("=== EVAL response (features) ===")  # Show features section
    pretty_print(resp.get("features"))
    if "explanations" in resp and resp["explanations"] is not None:
        print("=== EVAL response (explanations) ===")
        pretty_print(resp["explanations"])

def main():
    """CLI entrypoint for client usage."""
    ap = argparse.ArgumentParser()         # Create parser
    ap.add_argument("--url", default="https://behavioural-analytics.onrender.com/", help="FastAPI base URL")  # Server URL
    ap.add_argument("--models-dir", default="./models", help="Models directory on server")  # Models dir
    sub = ap.add_subparsers(dest="mode", required=True)  # Subcommands: train or eval

    ap_tr = sub.add_parser("train", help="Send CSV sessions to /train")  # Train command
    ap_tr.add_argument("--csv", required=True, help="CSV file with sessions")  # CSV path

    ap_ev = sub.add_parser("eval", help="Send CSV sessions to /eval")    # Eval command
    ap_ev.add_argument("--csv", required=True, help="CSV file with sessions")  # CSV path
    ap_ev.add_argument("--username", required=True)                      # Target username
    ap_ev.add_argument("--contract", required=True)                      # Target contract
    ap_ev.add_argument("--k", type=float, default=1.0)                   # Score slope parameter
    ap_ev.add_argument("--explain", action="store_true", help="Return per-feature contributions")
    ap_ev.add_argument("--topN", type=int, default=8)

    args = ap.parse_args()            # Parse command line
    if args.mode == "train":          # Dispatch by subcommand
        do_train(args.url, args.models_dir, args.csv)
    else:
        do_eval(args.url, args.models_dir, args.csv, args.username, args.contract, args.k, args.explain, args.topN)

if __name__ == "__main__":            # Executed as script
    main()                            # Run CLI entrypoint