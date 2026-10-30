"""Fetch the unchanged Peng (2022) firm workbook and verify its deposited hash."""
from pathlib import Path
import hashlib
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = "https://data.mendeley.com/public-files/datasets/wx8nwsrrbr/files/7f8e2241-ffc8-44ea-8921-02e356494c66/file_downloaded"
NAME = "wx8nwsrrbr__Roe and revenue growth rate of 3270 listed companies in Shanghai and Shenzhen stock markets from 2009 to 2020.xlsx"
SHA256 = "9cb04bd1d5ed44383d7e54351c8e8c7271797ccd74ecac3934e06302141dd75f"
destination = ROOT / "data/raw/public" / NAME
if destination.exists():
    payload = destination.read_bytes()
else:
    request = urllib.request.Request(URL, headers={"User-Agent": "AcademicDataReproduction/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
assert hashlib.sha256(payload).hexdigest() == SHA256, "Dataset checksum does not match frozen release"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_bytes(payload)
print(destination)
