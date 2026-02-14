import socket
import json

HOST = "127.0.0.1"
PORT = 5200

def get_coretemp_data():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    s.connect((HOST, PORT))

    data = s.recv(8192)
    s.close()

    payload = data.decode().strip()
    return json.loads(payload)

data = get_coretemp_data()

cpu = data["CpuInfo"]

core_temps = cpu["fTemp"]
core_loads = cpu["uiLoad"]

result = {
    "core_max": max(core_temps),
    "core_avg": sum(core_temps) / len(core_temps),
    "power_w": cpu.get("fPower", [None])[0],
    "tjmax": cpu["uiTjMax"][0],
    "core_count": cpu["uiCoreCnt"]
}

print(result)
