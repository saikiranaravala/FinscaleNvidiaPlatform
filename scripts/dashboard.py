#!/usr/bin/env python3
"""Live demo dashboard: both nodes' GPU utilization, the training loss curve, and a
excerpt / base-output / fine-tuned-output / human-label comparison table.

Run during the live demo, in a separate terminal from training:
  python scripts/dashboard.py --log out/train_peft.log --eval-results out/heldout/eval_results.json

Each node's scripts/gpu_monitor.py writes its own CSV; point --gpu-csv at both:
  python scripts/dashboard.py --gpu-csv out/gpu_node0.csv out/gpu_node1.csv

The page polls GET /api/state every 2s and re-renders; nothing is pushed from the server,
so a flaky venue network just means a stale read, not a broken page.
"""
import argparse, csv, json, re, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LOSS_RE = re.compile(r"'loss':\s*([0-9.eE+-]+)")

STATE = {}


def read_loss_curve(log_path):
    if not log_path or not log_path.exists():
        return []
    losses = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LOSS_RE.search(line)
        if m:
            try:
                losses.append(float(m.group(1)))
            except ValueError:
                pass
    return losses[-200:]  # keep the page light


def read_gpu_csvs(paths):
    by_node = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        rows = list(csv.DictReader(p.open(newline="", encoding="utf-8")))
        if rows:
            last = rows[-1]
            by_node[last.get("node_id", p.stem)] = {
                "utilization_pct": last.get("utilization_pct"),
                "memory_used_mib": last.get("memory_used_mib"),
                "memory_total_mib": last.get("memory_total_mib"),
                "samples": len(rows),
            }
    return by_node


def read_eval_examples(eval_results_path, limit=12):
    if not eval_results_path or not eval_results_path.exists():
        return []
    data = json.loads(eval_results_path.read_text(encoding="utf-8"))
    base_detail = data.get("base", {}).get("per_example", [])
    tuned_detail = data.get("fine_tuned", {}).get("per_example", [])
    rows = []
    for i, (b, t) in enumerate(zip(base_detail, tuned_detail)):
        if i >= limit:
            break
        rows.append({
            "gold_category": t.get("gold_category"), "gold_severity": t.get("gold_severity"),
            "base_category": b.get("pred_category"), "base_severity": b.get("pred_severity"),
            "tuned_category": t.get("pred_category"), "tuned_severity": t.get("pred_severity"),
        })
    return rows


def refresh_state(args):
    STATE["loss_curve"] = read_loss_curve(args.log)
    STATE["gpu_nodes"] = read_gpu_csvs(args.gpu_csv)
    STATE["eval_examples"] = read_eval_examples(args.eval_results)


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>FinScale live dashboard</title>
<style>
body{font-family:system-ui,sans-serif;background:#0b1220;color:#e2e8f0;margin:0;padding:24px}
h1{font-size:20px;margin:0 0 16px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{background:#111827;border:1px solid #1f2937;border-radius:10px;padding:16px}
.card h2{font-size:14px;color:#94a3b8;margin:0 0 10px;text-transform:uppercase;letter-spacing:.05em}
table{width:100%;border-collapse:collapse;font-size:13px}
td,th{padding:4px 8px;border-bottom:1px solid #1f2937;text-align:left}
.gpu-bar{height:10px;background:#1f2937;border-radius:6px;overflow:hidden;margin-top:4px}
.gpu-fill{height:100%;background:#0F9D8A}
canvas{width:100%;height:160px}
.mismatch{color:#f87171}
.match{color:#34d399}
</style></head><body>
<h1>FinScale live dashboard</h1>
<div class="grid">
  <div class="card"><h2>GPU utilization by node</h2><div id="gpus"></div></div>
  <div class="card"><h2>Training loss</h2><canvas id="loss"></canvas></div>
  <div class="card" style="grid-column:1/-1"><h2>Excerpt vs base vs fine-tuned vs gold</h2>
    <table id="examples"><thead><tr><th>#</th><th>Gold</th><th>Base</th><th>Fine-tuned</th></tr></thead><tbody></tbody></table>
  </div>
</div>
<script>
function render(state){
  const gpus = document.getElementById('gpus');
  gpus.innerHTML = Object.entries(state.gpu_nodes||{}).map(([node, g]) => `
    <div style="margin-bottom:10px">
      <div>node ${node}: ${g.utilization_pct||0}% util, ${g.memory_used_mib||0}/${g.memory_total_mib||'?'} MiB</div>
      <div class="gpu-bar"><div class="gpu-fill" style="width:${g.utilization_pct||0}%"></div></div>
    </div>`).join('') || '<div>no scripts/gpu_monitor.py CSVs found yet</div>';

  const canvas = document.getElementById('loss');
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.clientWidth; canvas.height = 160;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const losses = state.loss_curve || [];
  if (losses.length > 1) {
    const max = Math.max(...losses), min = Math.min(...losses);
    ctx.strokeStyle = '#0F9D8A'; ctx.lineWidth = 2; ctx.beginPath();
    losses.forEach((v,i) => {
      const x = (i/(losses.length-1)) * canvas.width;
      const y = canvas.height - ((v-min)/((max-min)||1)) * (canvas.height-10) - 5;
      i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    });
    ctx.stroke();
  } else {
    ctx.fillStyle = '#64748B'; ctx.fillText('no loss values parsed from the log yet', 8, 20);
  }

  const tbody = document.querySelector('#examples tbody');
  tbody.innerHTML = (state.eval_examples||[]).map((e,i) => {
    const cls = (pred) => pred === e.gold_category ? 'match' : 'mismatch';
    return `<tr><td>${i+1}</td><td>${e.gold_category}/${e.gold_severity}</td>
      <td class="${cls(e.base_category)}">${e.base_category}/${e.base_severity}</td>
      <td class="${cls(e.tuned_category)}">${e.tuned_category}/${e.tuned_severity}</td></tr>`;
  }).join('') || '<tr><td colspan="4">no eval_results.json found yet</td></tr>';
}
async function poll(){
  try { render(await (await fetch('/api/state')).json()); } catch(e) {}
  setTimeout(poll, 2000);
}
poll();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/api/state":
            body = json.dumps(STATE).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", type=Path, default=Path("out/train_peft.log"))
    ap.add_argument("--eval-results", type=Path, default=Path("out/heldout/eval_results.json"))
    ap.add_argument("--gpu-csv", nargs="*", default=[])
    ap.add_argument("--port", type=int, default=7777)
    args = ap.parse_args()

    refresh_state(args)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Dashboard at http://localhost:{args.port} (Ctrl+C to stop)", file=sys.stderr)

    import threading, time

    def refresher():
        while True:
            refresh_state(args)
            time.sleep(2)

    threading.Thread(target=refresher, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
