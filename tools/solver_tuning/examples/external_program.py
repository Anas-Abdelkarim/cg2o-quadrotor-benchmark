"""A standalone executable implementing the request/result JSON protocol."""

import json
import sys
from pathlib import Path

request = json.loads(Path(sys.argv[1]).read_text())
x = request["parameters"]["gain"]
loss = 1 + (x - 2) ** 2
Path(sys.argv[2]).write_text(
    json.dumps({"metrics": {"loss": loss, "violation": max(0, x - 4)}})
)
