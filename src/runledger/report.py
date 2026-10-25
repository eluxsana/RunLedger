"""Self-contained HTML evidence viewer. Untrusted strings enter the DOM as text only."""

from __future__ import annotations

import html
import json
from pathlib import Path

import pyarrow.parquet as pq

from runledger.util import canonical_json

STYLE = """
:root{color-scheme:light;--ink:#152b33;--muted:#586c74;--line:#d4dedf;--paper:#f3f6f4;--accent:#087b72;--soft:#e0efeb;--red:#a2403e}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;line-height:1.55}a{color:var(--accent);text-underline-offset:3px}button,input,select{font:inherit}button{cursor:pointer}button:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid #e2a43e;outline-offset:3px}.skip{position:absolute;top:-100px}.skip:focus{top:8px;background:white;padding:12px;z-index:5}
nav{background:var(--ink);color:#dfeae8;padding:15px max(24px,calc((100vw - 1180px)/2));display:flex;justify-content:space-between;align-items:center;gap:16px}.brand{font-size:19px;letter-spacing:-.5px;font-weight:750;color:white}.mark{display:inline-block;background:#78d8bd;width:11px;height:19px;margin-right:9px;vertical-align:-2px}nav small{font-size:12px;text-transform:uppercase;letter-spacing:1.3px}main{max-width:1228px;margin:auto;padding:42px 24px 50px}.eyebrow{font-size:12px;letter-spacing:1.8px;text-transform:uppercase;color:var(--accent);font-weight:750}h1{font-size:clamp(32px,4vw,48px);letter-spacing:-1.8px;line-height:1.13;margin:13px 0 16px;max-width:800px}h2{font-size:22px;letter-spacing:-.5px;line-height:1.2;margin:0}h3{font-size:14px;margin:0 0 10px}p{margin:8px 0}.intro{max-width:750px;color:var(--muted);font-size:16px}.status{display:inline-block;padding:5px 10px;border:1px solid #badbd0;background:var(--soft);border-radius:4px;color:#07665e;font-size:12px;font-weight:700}.hero-meta{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:22px 0 26px}.mono,code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.id{font-size:12px;color:var(--muted);overflow-wrap:anywhere}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:25px 0}.card,.panel{background:white;border:1px solid var(--line);border-radius:9px}.card{padding:20px;min-height:140px}.label{font-size:12px;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);font-weight:650}.value{font-size:36px;letter-spacing:-1.5px;line-height:1.15;margin:11px 0 7px;font-weight:650}.unit{font-size:18px;color:var(--muted);letter-spacing:-.5px}.note{font-size:12px;color:var(--muted)}.notice{padding:13px 17px;border-left:3px solid var(--accent);background:#e9f0ed;font-size:13px;margin-bottom:28px}.section-title{display:flex;justify-content:space-between;gap:15px;align-items:baseline;margin:35px 0 16px}.grid{display:grid;grid-template-columns:1.12fr 1fr;gap:18px}.panel{padding:23px;min-width:0}.key-value{display:grid;grid-template-columns:110px 1fr;gap:10px 16px;font-size:13px;margin:0}.key-value dt{color:var(--muted)}.key-value dd{margin:0;overflow-wrap:anywhere}.bar-row{display:grid;grid-template-columns:110px 1fr 45px;gap:12px;align-items:center;font-size:12px;margin:12px 0}.bar-track{height:8px;background:#e6edea;border-radius:5px;overflow:hidden}.bar-fill{height:100%;background:var(--accent);border-radius:5px}.divider{height:1px;background:var(--line);margin:20px 0}.controls{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:15px}.controls input{flex:1;min-width:180px}.controls input,.controls select,.controls button,.paging button{padding:8px 11px;border:1px solid #bacacb;border-radius:5px;background:white;color:var(--ink);font-size:13px}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:12px;text-align:left}th{color:var(--muted);font-weight:600;background:#f4f7f5;border-top:1px solid var(--line)}td,th{padding:11px 13px;border-bottom:1px solid #e3eae6;white-space:nowrap}td.score{font-variant-numeric:tabular-nums;font-family:ui-monospace,monospace}.pill{display:inline-block;border-radius:3px;font-weight:600;font-size:11px;padding:3px 7px;background:var(--soft);color:#10695d}.pill.bad{color:var(--red);background:#f8eae6}.paging{display:flex;justify-content:space-between;align-items:center;margin-top:15px;gap:12px}.paging button:disabled{opacity:.4;cursor:default}.paging div{display:flex;gap:7px}.event{display:grid;grid-template-columns:12px 1fr auto;gap:10px;align-items:baseline;border-bottom:1px solid #ecf0ec;padding:10px 0;font-size:12px}.dot{width:7px;height:7px;background:#149580;border-radius:50%;display:inline-block}.event small{color:var(--muted)}.event-list{max-height:247px;overflow:auto}.limits{font-size:13px;color:var(--muted);padding-left:20px}.limits li{margin:8px 0}details{margin-top:18px}summary{cursor:pointer;font-size:13px;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px;background:#f1f5f2;padding:17px;border-radius:5px;max-height:380px;overflow:auto}.footer{border-top:1px solid var(--line);margin-top:35px;padding-top:20px;display:flex;justify-content:space-between;gap:20px;font-size:12px;color:var(--muted)}.legend{display:flex;gap:12px;margin:14px 0 0}.strong{color:var(--ink);font-weight:650}.empty{padding:30px;color:var(--muted)}
@media(max-width:820px){.cards{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}main{padding-top:28px}nav small{display:none}.footer{display:block}}@media(max-width:440px){main{padding:24px 14px}.card{padding:15px}.value{font-size:29px}.key-value{grid-template-columns:90px 1fr}.section-title{display:block}.panel{padding:17px}.paging{align-items:flex-start}}@media print{nav,.controls,.paging{display:none}.panel,.card{break-inside:avoid}body{background:white}main{padding:0}.event-list{max-height:none}.cards{grid-template-columns:repeat(4,1fr)}a{color:inherit}}
"""

SCRIPT = r"""
const data = JSON.parse(document.getElementById('run-data').textContent);
const $ = id => document.getElementById(id);
const fmt = n => n === null ? '—' : n.toFixed(2);
let page = 0;
function cell(row, text, cls='') { const el=document.createElement('td');el.textContent=text;el.className=cls;row.appendChild(el);return el; }
function render() {
 const query=$('search').value.toLowerCase(); const filter=$('filter').value;
 let rows=data.rows.filter(r=>r.sample_id.toLowerCase().includes(query) && (filter==='all'||String(r.correct)===filter));
 if($('sort').value==='margin')rows.sort((a,b)=>margin(a)-margin(b));
 const pages=Math.max(1,Math.ceil(rows.length/12));page=Math.min(page,pages-1);
 $('rows').replaceChildren();
 rows.slice(page*12,(page+1)*12).forEach(r=>{
  const tr=document.createElement('tr');cell(tr,r.sample_id,'mono');cell(tr,String(r.prediction+1));cell(tr,String(r.target+1));
  const td=cell(tr,'');const pill=document.createElement('span');pill.className='pill'+(r.correct?'':' bad');pill.textContent=r.correct?'Correct':'Incorrect';td.appendChild(pill);
  cell(tr,fmt(Math.max(...r.choice_scores)),'score');cell(tr,fmt(margin(r)),'score');cell(tr,r.token_counts.join(' / '),'score');$('rows').appendChild(tr);
 });
 if(!rows.length){const tr=document.createElement('tr');const td=cell(tr,'No samples match these filters.','empty');td.colSpan=7;$('rows').appendChild(tr);}
 $('page-label').textContent=`${rows.length? page*12+1:0}–${Math.min((page+1)*12,rows.length)} of ${rows.length} preview samples`;
 $('prev').disabled=page===0;$('next').disabled=page>=pages-1;
}
function margin(r){const s=[...r.choice_scores].sort((a,b)=>b-a);return s[0]-s[1];}
['search','filter','sort'].forEach(id=>$(id).addEventListener('input',()=>{page=0;render();}));
$('prev').addEventListener('click',()=>{page--;render();});$('next').addEventListener('click',()=>{page++;render();});
$('config').textContent=JSON.stringify(data.summary.spec,null,2);
render();
"""


def render_report(bundle_dir: Path) -> Path:
    summary = json.loads((bundle_dir / "summary.json").read_text())
    events = json.loads((bundle_dir / "events.json").read_text())
    preview = []
    for batch in pq.ParquetFile(bundle_dir / "results.parquet").iter_batches(batch_size=256):
        preview.extend(batch.to_pylist())
        if len(preview) >= 1000:
            break
    preview = preview[:1000]
    e = html.escape
    metrics, ops, config = summary["metrics"], summary["operations"], summary["spec"]["scorer"]
    quality = summary["model_quality_evidence"]
    accuracy = f"{100 * metrics['accuracy']:.1f}" if quality else "Fixture"
    interval = metrics["accuracy_wilson_95"]
    accuracy_note = (
        f"95% Wilson interval: {100 * interval[0]:.1f}–{100 * interval[1]:.1f}%"
        if quality
        else "Hash scores test infrastructure only"
    )
    dataset = summary["dataset"]["provenance"]
    bars = []
    for group in metrics["by_target"]:
        pct = 100 * group["correct"] / group["samples"]
        bars.append(
            f'<div class="bar-row"><span>Target {group["target"] + 1} · n={group["samples"]}</span>'
            f'<div class="bar-track" aria-hidden="true"><div class="bar-fill" style="width:{pct:.2f}%"></div></div>'
            f"<span>{pct:.0f}%</span></div>"
        )
    event_html = "".join(
        f'<div class="event"><span class="dot"></span><span>{e(event["kind"].replace("_", " "))}</span>'
        f"<small>job {event['job_id'] if event['job_id'] is not None else '—'}</small></div>"
        for event in reversed(events[-100:])
    )
    data = (
        canonical_json({"summary": summary, "rows": preview})
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    limits = "".join(f"<li>{e(limit)}</li>" for limit in summary["limits"])
    elapsed = ops["execution_seconds"]
    p95 = "—" if ops["job_p95_seconds"] is None else f"{ops['job_p95_seconds']:.3f} s"
    scoring_label = (
        f"{config['normalization']} continuation log likelihood · {config['dtype']} · CPU"
        if quality
        else f"Deterministic hash fixture · seed {config['seed']}"
    )
    label = "Model evaluation" if quality else "Infrastructure validation"
    notice = (
        "This pinned, small-model evaluation demonstrates the scoring pipeline. The selected subset and prompt protocol do not establish a leaderboard result."
        if quality
        else "Synthetic workload + deterministic hash scorer. These results demonstrate orchestration and recovery; accuracy has no model-quality meaning."
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>RunLedger · {e(label)} evidence</title><style>{STYLE}</style></head>
<body><a class="skip" href="#main">Skip to evidence</a><nav><span class="brand"><span class="mark"></span>RunLedger</span><small>Versioned inputs. Inspectable outcomes.</small></nav>
<main id="main"><div class="eyebrow">Evidence / {e(label)}</div><h1>Every result has a receipt.</h1>
<p class="intro">A reproducible evaluation with pinned inputs, durable results, and an auditable execution history. Explore the measurements and the limits behind this run.</p>
<div class="hero-meta"><span class="status">✓ COMPLETE · {metrics["samples"]:,} / {summary["dataset"]["row_count"]:,} SAMPLES</span><span class="id mono">run {e(summary["run_id"][:16])}</span></div>
<div class="cards"><div class="card"><div class="label">{"Subset accuracy" if quality else "Scoring backend"}</div><div class="value">{accuracy}<span class="unit">{"%" if quality else ""}</span></div><div class="note">{accuracy_note}</div></div>
<div class="card"><div class="label">Committed samples</div><div class="value">{metrics["samples"]:,}</div><div class="note">{metrics["unique_sample_ids"]:,} unique IDs · {ops["jobs"]} atomic jobs</div></div>
<div class="card"><div class="label">Execution time</div><div class="value">{elapsed:.1f}<span class="unit"> s</span></div><div class="note">Includes workers, model load, and retries</div></div>
<div class="card"><div class="label">Retried jobs</div><div class="value">{ops["retried_jobs"]}</div><div class="note">{ops["attempts"]} total attempts · fenced commits</div></div></div>
<div class="notice">{notice}</div>
<div class="grid"><section class="panel"><h2>Reproducibility receipt</h2><div class="divider"></div><dl class="key-value">
<dt>Model</dt><dd class="strong">{e(config["model_id"])}</dd><dt>Revision</dt><dd class="mono">{e(config["revision"])}</dd>
<dt>Dataset</dt><dd>{e(dataset["name"])} · {e(dataset["split"])}</dd><dt>Selection</dt><dd>{e(dataset["description"])}</dd>
<dt>Scoring</dt><dd>{e(scoring_label)}</dd>
<dt>Result SHA-256</dt><dd class="mono">{e(summary["result_sha256"])}</dd></dl>
<details><summary>View complete run specification</summary><pre id="config"></pre></details></section>
<section class="panel"><h2>{"Correctness by answer position" if quality else "Fixture results by answer position"}</h2><p class="note">Descriptive slices; small groups can vary substantially.</p>{"".join(bars)}<div class="divider"></div>
<dl class="key-value"><dt>Job p95</dt><dd>{p95} · successful attempts</dd><dt>Runtime</dt><dd>Python {e(summary["spec"]["environment"]["python"])} / {e(summary["spec"]["environment"]["platform"])} {e(summary["spec"]["environment"]["arch"])}</dd><dt>Provenance</dt><dd>{e(dataset["license"])}</dd></dl></section></div>
<div class="section-title"><h2>Inspect individual results</h2><span class="note">Preview: first {len(preview):,} of {metrics["samples"]:,}; full data in results.parquet</span></div>
<section class="panel"><div class="controls"><input id="search" type="search" placeholder="Search sample ID…" aria-label="Search sample IDs"><select id="filter" aria-label="Filter correctness"><option value="all">All outcomes</option><option value="true">Correct</option><option value="false">Incorrect</option></select><select id="sort" aria-label="Sort samples"><option value="original">Dataset order</option><option value="margin">Smallest score margin</option></select></div>
<div class="table-wrap"><table><thead><tr><th scope="col">Sample ID</th><th scope="col">Predicted¹</th><th scope="col">Target¹</th><th scope="col">Outcome</th><th scope="col">Top score</th><th scope="col">Margin²</th><th scope="col">Answer tokens</th></tr></thead><tbody id="rows"></tbody></table></div>
<div class="paging"><span class="note" id="page-label" aria-live="polite"></span><div><button id="prev">← Previous</button><button id="next">Next →</button></div></div><p class="note">¹ Displayed as 1-based choice positions; stored indices are 0-based. ² Best minus second-best score; not calibrated confidence.</p><noscript><p>Enable JavaScript for the interactive preview. Complete results and metrics are available in the adjacent JSON and Parquet files.</p></noscript></section>
<div class="section-title"><h2>Execution &amp; interpretation</h2></div><div class="grid"><section class="panel"><h3>Latest execution events</h3><p class="note">Most recent first · showing up to 100 events</p><div class="event-list">{event_html}</div></section>
<section class="panel"><h3>What this evidence supports</h3><ul class="limits">{limits}</ul></section></div>
<footer class="footer"><span>RunLedger · generated {e(summary["created_at"][:19].replace("T", " "))} UTC</span><span><a href="summary.json">Summary JSON</a> · <a href="bundle.json">File checksums</a> · <a href="results.parquet">Results Parquet</a></span></footer></main>
<script type="application/json" id="run-data">{data}</script><script>{SCRIPT}</script></body></html>"""
    output = bundle_dir / "index.html"
    output.write_text(document, encoding="utf-8")
    return output
