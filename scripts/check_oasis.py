# scripts/check_oasis.py
import os
oasis_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OASIS")
for d in sorted(os.listdir(oasis_dir)):
    path = os.path.join(oasis_dir, d)
    if os.path.isdir(path):
        print(f"{d}: {len(os.listdir(path))} images")