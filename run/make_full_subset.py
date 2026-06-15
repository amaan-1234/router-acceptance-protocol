"""Regenerate the frontier subset over ALL ChaosNLI items (not the 450 stratified sample)."""
import json
from cac import config
from cac.data import chaosnli

items = chaosnli.load("snli_mnli")
out = []
for i, it in enumerate(items):
    out.append({
        "idx": i,
        "uid": it.uid,
        "premise": it.premise,
        "hypothesis": it.hypothesis,
        "human_dist": [round(float(x), 6) for x in it.human_dist],
        "entropy": float(it.entropy),
    })
path = config.OUTPUTS_DIR / "chaosnli_frontier_subset_full.json"
path.write_text(json.dumps(out, indent=2))
print(f"wrote {len(out)} items -> {path}")
