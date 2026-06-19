import time
import urllib.request
import json

def main():
    url_start = "http://127.0.0.1:8000/api/training/start"
    payload = {
        "rounds": 2,
        "min_clients": 2,
        "dummy_model": False
    }
    
    headers = {"Content-Type": "application/json"}
    
    print("Triggering live RT-DETR-L training via API...")
    req = urllib.request.Request(
        url_start, 
        data=json.dumps(payload).encode("utf-8"), 
        headers=headers, 
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as res:
            response_data = json.loads(res.read().decode())
            print("Response:", json.dumps(response_data, indent=2))
    except Exception as e:
        print("Failed to start training:", e)
        return

    # Now monitor the training state
    url_state = "http://127.0.0.1:8000/api/training/state"
    for _ in range(120): # Monitor for up to 120 seconds
        time.sleep(2)
        try:
            with urllib.request.urlopen(url_state) as res:
                state = json.loads(res.read().decode())
                phase = state.get("phase")
                current_round = state.get("current_round")
                connected = state.get("connected_clients")
                last_error = state.get("last_error")
                print(f"Phase: {phase} | Round: {current_round}/2 | Connected Clients: {connected}")
                if last_error:
                    print(f"Error encountered: {last_error}")
                if phase in ["completed", "failed"]:
                    print(f"Training finished with phase: {phase}")
                    break
        except Exception as e:
            print("Error fetching state:", e)
            
    # Get last logs
    url_logs = "http://127.0.0.1:8000/api/logs?n=50"
    try:
        with urllib.request.urlopen(url_logs) as res:
            logs = json.loads(res.read().decode())
            print("\n--- Last 50 Log Lines ---")
            for line in logs.get("lines", []):
                print(line)
    except Exception as e:
        print("Failed to fetch logs:", e)

if __name__ == "__main__":
    main()
