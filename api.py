import os
import requests
from datetime import datetime

API_URL = "https://amused-top-sole.ngrok-free.app/events/detection" 

def report_detection_event(group, baby, event_type, event_date=None):
    if event_date is None:
        event_date = datetime.utcnow().isoformat()
    payload = {
        "group": group,
        "baby": baby,
        "type": event_type,
        "date": event_date
    }
    print(f"[api] Reportando evento: {payload}")
    try:
        resp = requests.post(API_URL, json=payload, timeout=5)
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"[api] Error al reportar evento: {e}")
        return False
