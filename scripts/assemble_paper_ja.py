# -*- coding: utf-8 -*-
import os
import shutil

parts = ["scripts/paper_part1.txt", "scripts/paper_part2.txt", "scripts/paper_part3.txt", "scripts/paper_part4.txt"]

full_text = []
for p in parts:
    with open(p, "r", encoding="utf-8") as f:
        full_text.append(f.read().strip())

assembled = "\n\n".join(full_text) + "\n"

# Write to usir_v2_camera_ready_paper_ja.md
with open("usir_v2_camera_ready_paper_ja.md", "w", encoding="utf-8") as f:
    f.write(assembled)
print(f"Wrote usir_v2_camera_ready_paper_ja.md ({len(assembled)} chars, {len(assembled.splitlines())} lines)")

# Copy to usir_v3_camera_ready_paper_ja.md
shutil.copyfile("usir_v2_camera_ready_paper_ja.md", "usir_v3_camera_ready_paper_ja.md")
print("Copied to usir_v3_camera_ready_paper_ja.md")

# Verify forbidden author strings
forbidden = ["志賀", "川端", "三島", "Shiga", "Kawabata", "Mishima", "城の崎", "雪国", "金閣寺"]
found = [w for w in forbidden if w in assembled]
if found:
    print(f"ERROR: Found forbidden keywords: {found}")
else:
    print("Cleanroom Check: PASS (0 forbidden author/novel keywords found)")
