"""docs/ARENA_REPORT.md = tools/arena_report_template.md with the generated tables (tools/report_tables.py) inserted.

  docker run --rm -v $PWD:/workspace/g1_stairs g1-arena tools/build_report.py
"""
import datetime
import os
import subprocess
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tables = subprocess.run([sys.executable, os.path.join(root, "tools/report_tables.py")], capture_output=True,
                        text=True, check=True).stdout
secs = {}
for part in tables.split("### ")[1:]:
    title, body = part.split("\n", 1)
    secs[title.strip()] = body.strip()
out = open(os.path.join(root, "tools/arena_report_template.md")).read()
out = out.replace("{{DATE}}", datetime.date.today().isoformat())
for k, v in secs.items():
    out = out.replace("{{T:" + k + "}}", v)
assert "{{" not in out, [l for l in out.splitlines() if "{{" in l]
open(os.path.join(root, "docs/ARENA_REPORT.md"), "w").write(out)
print("wrote docs/ARENA_REPORT.md", len(out.splitlines()), "lines")
