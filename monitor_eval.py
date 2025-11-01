import time, requests

URL = "https://behavioural-analytics.onrender.com/healthz"
MIN_INTERVAL_S = 300  # 5 minutes
TIMEOUT_S = 5

while True:  # continuous monitor with spacing
    start = time.time()
    try:
        r = requests.get(URL, timeout=TIMEOUT_S, headers={"User-Agent": "health-check/1.0"})
        ok = 200 <= r.status_code < 300
        print(time.strftime("%Y-%m-%d %H:%M:%S"), r.status_code, "OK" if ok else "FAIL")
        # honor Retry-After if present
        retry_after = r.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            sleep_s = max(MIN_INTERVAL_S, int(retry_after))
        else:
            sleep_s = MIN_INTERVAL_S
    except requests.RequestException as e:
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "ERROR", e)
        sleep_s = MIN_INTERVAL_S
    elapsed = time.time() - start
    time.sleep(max(0, sleep_s - elapsed))
