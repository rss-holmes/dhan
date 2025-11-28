import http.client
import json
import os

import dotenv

dotenv.load_dotenv()

conn = http.client.HTTPSConnection("api.dhan.co")

CLIENT_ID = os.getenv("CLIENT_ID", "")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", "")

payload = {
    "securityId": "1333",
    "exchangeSegment": "NSE_EQ",
    "instrument": "INDEX",
    "expiryCode": -2147483648,
    "fromDate": "2025-08-24",
    "toDate": "2025-10-24",
}

json_payload = json.dumps(payload)

headers = {
    "client-id": CLIENT_ID,
    "access-token": ACCESS_TOKEN,
    "Content-Type": "application/json",
    "Accept": "application/json",
}

conn.request("POST", "/v2/charts/historical", json_payload, headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))
