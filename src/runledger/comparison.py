"""A standalone, offline view of paired prediction changes."""

from __future__ import annotations

from html import escape
from pathlib import Path

from runledger.util import canonical_json


def render_comparison(comparison: dict, output: Path) -> Path:
    """Write an inspectable comparison without embedding prompts or contacting a server."""
    data = canonical_json(comparison).replace("<", "\\u003c").replace(">", "\\u003e")
    delta = comparison["accuracy_delta_right_minus_left"] * 100
    low, high = [value * 100 for value in comparison["paired_bootstrap_95"]]
    rows = "".join(
        f'<tr data-outcome="{escape(row["outcome"], quote=True)}">'
        f'<td class="id">{escape(row["sample_id"])}</td>'
        f"<td>{row['target'] + 1}</td><td>{row['left_prediction'] + 1}</td>"
        f"<td>{row['right_prediction'] + 1}</td>"
        f'<td class="{escape(row["outcome"], quote=True)}">{escape(row["outcome"])}</td></tr>'
        for row in comparison["changed_samples"]
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RunLedger · Prediction drift</title><style>
:root{{color-scheme:dark;font-family:system-ui,sans-serif;background:#080d19;color:#e9eef8}}
body{{margin:0;padding:40px 24px}}main{{max-width:1100px;margin:auto}}
.eyebrow{{font:12px monospace;color:#7ce0ee;letter-spacing:3px}}h1{{font-size:clamp(30px,5vw,52px);letter-spacing:-2px;margin-bottom:12px}}
.muted{{color:#9daec5;line-height:1.6}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:32px 0}}
.card{{padding:24px;background:#10192b;border:1px solid #26354f;border-radius:12px}}
.value{{font-size:32px;font-weight:650;display:block;margin:8px 0}}.label{{font-size:13px;color:#9daec5}}
.receipt{{border-left:2px solid #a998f4;padding:12px 20px;background:#14172b;overflow-wrap:anywhere;font:12px/1.8 monospace}}
.controls{{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0 12px}}input,select{{background:#10192b;color:inherit;border:1px solid #32435f;border-radius:6px;padding:10px;font:inherit}}
input{{flex:1;min-width:180px}}.scroll{{overflow:auto;max-height:600px}}table{{width:100%;border-collapse:collapse;font-size:13px;text-align:left}}th{{background:#162037;position:sticky;top:0}}th,td{{padding:13px;border-bottom:1px solid #26354f}}.id{{font-family:monospace;overflow-wrap:anywhere}}.improved{{color:#79e8c4}}.regressed{{color:#ff9caa}}.changed{{color:#c7b6ff}}code{{font-size:12px}}[hidden]{{display:none}}
</style></head><body><main><p class="eyebrow">RUNLEDGER / PAIRED COMPARISON</p>
<h1>The score moved. What changed?</h1><p class="muted">Same ordered examples, two verified evidence bundles. Inspect prediction changes, including wrong-to-wrong flips that accuracy hides.</p>
<div class="cards"><div class="card"><span class="label">Accuracy delta · right − left</span><span class="value">{delta:+.2f} pp</span><span class="label">Paired bootstrap 95%: [{low:+.2f}, {high:+.2f}] pp</span></div>
<div class="card"><span class="label">Changed predictions</span><span class="value">{comparison["prediction_changes"]}</span><span class="label">of {comparison["samples"]} paired examples</span></div>
<div class="card"><span class="label">Improved / regressed</span><span class="value">{comparison["improved"]} / {comparison["regressed"]}</span><span class="label">Correctness transitions</span></div></div>
<div class="receipt">LEFT &nbsp; {escape(comparison["left_run"])}<br>RIGHT {escape(comparison["right_run"])}<br>Identical scoring configuration: {comparison["same_scoring_configuration"]}</div>
<h2>Changed examples</h2><p class="muted">Choice positions below are 1-based. Raw stored indices are 0-based. Unchanged predictions are omitted.</p>
<div class="controls"><input id="query" type="search" aria-label="Search sample IDs" placeholder="Search a sample ID"><select id="outcome" aria-label="Filter outcome"><option value="all">All changes</option><option value="improved">Improved</option><option value="regressed">Regressed</option><option value="changed">Wrong → wrong</option></select></div>
<p id="count" class="muted" aria-live="polite">{comparison["prediction_changes"]} changed examples</p>
<div class="scroll"><table><thead><tr><th scope="col">Sample ID</th><th scope="col">Target</th><th scope="col">Left</th><th scope="col">Right</th><th scope="col">Outcome</th></tr></thead><tbody>{rows}</tbody></table></div>
<noscript><p>All changed examples are shown. JavaScript only enables filtering.</p></noscript>
<h2>Reading this result</h2><p class="muted">{escape(comparison["interpretation"])} The interval resamples correctness differences on these examples. It is not a forecast of performance on new data, and a scoring change is not necessarily a model improvement.</p>
<details><summary>Complete comparison JSON</summary><pre style="overflow:auto" id="json"></pre></details>
<script type="application/json" id="data">{data}</script><script>
const data=JSON.parse(document.getElementById('data').textContent);
document.getElementById('json').textContent=JSON.stringify(data,null,2);
const query=document.getElementById('query'), outcome=document.getElementById('outcome');
function filter(){{let shown=0;document.querySelectorAll('tbody tr').forEach(row=>{{
row.hidden=!(row.cells[0].textContent.toLowerCase().includes(query.value.toLowerCase())&&(outcome.value==='all'||row.dataset.outcome===outcome.value));if(!row.hidden)shown++;
}});document.getElementById('count').textContent=shown+' of '+data.prediction_changes+' changed examples';}}
query.addEventListener('input',filter);outcome.addEventListener('change',filter);
</script></main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(document)
    return output
