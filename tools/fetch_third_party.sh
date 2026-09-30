#!/usr/bin/env bash
# Fetch the pinned third-party repos and Hugging Face weights listed in third_party.yaml into third_party/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party && cd third_party
python3 - <<'PY' > /tmp/g1_fetch_plan.sh
import yaml
m = yaml.safe_load(open("../third_party.yaml"))
for name, r in m["repos"].items():
    print(f'[ -d {name} ] || (GIT_LFS_SKIP_SMUDGE=1 git clone -q --filter=blob:limit=20m {r["url"]}.git {name} && '
          f'git -C {name} checkout -q {r["commit"]})')
    for pat in r.get("lfs", []):
        print(f'git -C {name} lfs pull --include="{pat}"')
for repo, h in m["huggingface"].items():
    base = ("datasets/" if h.get("dataset") else "") + repo
    for f in h["files"]:
        src, rel = (f["src"], f["dst"]) if isinstance(f, dict) else (f, f)
        dst = f'hf/{h.get("dir", repo.split("/")[1])}/{rel}'
        print(f'[ -s "{dst}" ] || (mkdir -p "$(dirname "{dst}")" && curl -sfL -o "{dst}" https://huggingface.co/{base}/resolve/main/{src})')
PY
bash -e /tmp/g1_fetch_plan.sh
echo "third_party ready"
