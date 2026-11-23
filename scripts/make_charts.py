import html
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
data = json.loads((root / "results/summary.json").read_text())

OURS = "#2a78d6"
OTHER = "#a9a8a2"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

W = 1040
NAME_W = 250
GAP = 30
BAR_W = 230
ROW_H = 30
BAR_H = 16


def esc(s):
    return html.escape(str(s), quote=True)


def fmt(v):
    return f"{v:,.0f}" if v >= 100 else f"{v:.1f}"


def base_of(section, key):
    rows = section["rows"]
    return rows[0][key]


def bar_block(y, section, kind):
    out = []
    rows = section["rows"]
    xs = {"prefill": NAME_W + GAP, "decode": NAME_W + GAP + BAR_W + 150}
    labels = {"prefill": section.get("prefill_label", "Prefill, 2K tokens") + " (tok/s)", "decode": "Decode (tok/s)"}
    out.append(f'<text x="24" y="{y}" font-size="17" font-weight="600" fill="{INK}">{esc(section["machine"])}</text>')
    y += 26
    for key, x0 in xs.items():
        out.append(f'<text x="{x0}" y="{y}" font-size="13" fill="{INK2}">{esc(labels[key])}</text>')
    y += 12
    top = y
    for key, x0 in xs.items():
        vmax = max(r[key] for r in rows) * 1.0
        h = len(rows) * ROW_H
        out.append(f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{top + h}" stroke="{MUTED}" stroke-width="1"/>')
        for i, r in enumerate(rows):
            yy = top + i * ROW_H + (ROW_H - BAR_H) / 2
            w = max(2, BAR_W * r[key] / vmax)
            color = OTHER if (kind == "engines" and not r.get("ours")) else OURS
            if kind == "steps" and i < len(rows) - 1:
                color = "#9cc0ec"
            d = f"M{x0},{yy} h{w - 4} a4,4 0 0 1 4,4 v{BAR_H - 8} a4,4 0 0 1 -4,4 h{-(w - 4)} z"
            out.append(f'<path d="{d}" fill="{color}"><title>{esc(r["name"])}: {fmt(r[key])} tok/s</title></path>')
            label = fmt(r[key])
            extra = ""
            if kind == "engines" and r.get("ours"):
                extra = f" · {r[key] / base_of(section, key):.1f}×"
            if kind == "steps" and r.get("gain_" + key):
                extra = f" · {r['gain_' + key]}"
            weight = "600" if (r.get("ours") or (kind == "steps" and i == len(rows) - 1)) else "400"
            out.append(f'<text x="{x0 + w + 6}" y="{yy + BAR_H - 3}" font-size="13" font-weight="{weight}" fill="{INK}">{esc(label)}<tspan fill="{INK2}" font-weight="400">{esc(extra)}</tspan></text>')
    for i, r in enumerate(rows):
        yy = top + i * ROW_H + ROW_H / 2 + 5
        name = r["name"] + (" †" if r.get("other_session") else "")
        weight = "600" if r.get("ours") else "400"
        out.append(f'<text x="{NAME_W + GAP - 12}" y="{yy}" font-size="13.5" text-anchor="end" font-weight="{weight}" fill="{INK}">{esc(name)}</text>')
    y = top + len(rows) * ROW_H + 18
    for line in wrap(section["note"], 150):
        out.append(f'<text x="24" y="{y}" font-size="12" fill="{MUTED}">{esc(line)}</text>')
        y += 16
    return out, y + 22


def wrap(text, n):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines


def svg():
    body = []
    y = 44
    body.append(f'<text x="24" y="{y}" font-size="24" font-weight="700" fill="{INK}">Inkling 2 Mini on Apple Silicon</text>')
    y += 24
    body.append(f'<text x="24" y="{y}" font-size="14" fill="{INK2}">Tokens per second, higher is better. Blue: this patch. Grey: other engines.</text>')
    y += 20
    body.append(f'<text x="24" y="{y}" font-size="14" fill="{INK2}">×: speed-up over LM Studio in the same run. %: gain over the step before, measured in the same session.</text>')
    y += 40
    body.append(f'<text x="24" y="{y}" font-size="13" font-weight="700" letter-spacing="1" fill="{INK2}">AGAINST OTHER ENGINES</text>')
    y += 30
    for s in data["engines"]:
        part, y = bar_block(y, s, "engines")
        body += part
    body.append(f'<line x1="24" y1="{y - 18}" x2="{W - 24}" y2="{y - 18}" stroke="{GRID}"/>')
    y += 10
    body.append(f'<text x="24" y="{y}" font-size="13" font-weight="700" letter-spacing="1" fill="{INK2}">STEP BY STEP</text>')
    y += 30
    for s in data["steps"]:
        part, y = bar_block(y, s, "steps")
        body += part
    h = y
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" width="{W}" height="{h}" '
            f'font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif">'
            f'<rect width="{W}" height="{h}" fill="{SURFACE}" rx="12"/>' + "".join(body) + "</svg>\n")


def html_page():
    sections = []
    for kind, title in (("engines", "Against other engines"), ("steps", "Step by step")):
        cards = []
        for s in data[kind]:
            charts = []
            for key in ("prefill", "decode"):
                label = (s.get("prefill_label", "Prefill, 2K tokens") if key == "prefill" else "Decode")
                vmax = max(r[key] for r in s["rows"])
                rows = []
                for i, r in enumerate(s["rows"]):
                    last = i == len(s["rows"]) - 1
                    cls = "ours" if (r.get("ours") or (kind == "steps" and last)) else ("prev" if kind == "steps" else "other")
                    extra = ""
                    if kind == "engines" and r.get("ours"):
                        extra = f"{r[key] / base_of(s, key):.1f}× LM Studio"
                    if kind == "steps" and r.get("gain_" + key):
                        extra = f"{r['gain_' + key]} same session"
                    tip = f"{r['name']}: {fmt(r[key])} tok/s" + (f" ({extra})" if extra else "") + (" · other session" if r.get("other_session") else "")
                    rows.append(
                        f'<div class="row {cls}" tabindex="0" data-tip="{esc(tip)}">'
                        f'<div class="name">{esc(r["name"])}{" †" if r.get("other_session") else ""}</div>'
                        f'<div class="track"><div class="bar" style="width:calc((100% - 170px) * {r[key] / vmax:.4f})"></div>'
                        f'<span class="val">{fmt(r[key])}<em>{esc(extra)}</em></span></div></div>')
                charts.append(f'<div class="chart"><h4>{esc(label)} <small>tok/s</small></h4>{"".join(rows)}</div>')
            cards.append(f'<section class="card"><h3>{esc(s["machine"])}</h3><div class="pair">{"".join(charts)}</div><p class="note">{esc(s["note"])}</p></section>')
        sections.append(f'<h2>{title}</h2>' + "".join(cards))
    table = []
    for kind in ("engines", "steps"):
        for s in data[kind]:
            for r in s["rows"]:
                table.append(f"<tr><td>{esc(s['machine'])}</td><td>{esc(r['name'])}</td><td>{fmt(r['prefill'])}</td><td>{fmt(r['decode'])}</td></tr>")
    return TEMPLATE.replace("{SECTIONS}", "".join(sections)).replace("{TABLE}", "".join(table))


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Inkling Speed Results</title>
<style>
:root{--surface:#f6f5f1;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--line:#e4e3df;--ours:#2a78d6;--prev:#9cc0ec;--other:#a9a8a2}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--surface:#121211;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#8f8e88;--line:#2c2c2a;--ours:#3987e5;--prev:#2f5585;--other:#5d5c58}}
:root[data-theme="dark"]{--surface:#121211;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#8f8e88;--line:#2c2c2a;--ours:#3987e5;--prev:#2f5585;--other:#5d5c58}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--ink);font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:26px;margin:0 0 4px}.lede{color:var(--ink2);margin:0 0 20px}
h2{font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink2);margin:28px 0 10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;margin:0 0 14px}
h3{font-size:17px;margin:0 0 10px}h4{font-size:13px;font-weight:600;color:var(--ink2);margin:0 0 8px}h4 small{font-weight:400;color:var(--muted)}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}@media (max-width:700px){.pair{grid-template-columns:1fr}}
.row{margin:0 0 9px;outline:none}.name{font-size:13px;color:var(--ink);margin-bottom:3px}
.row.ours .name{font-weight:600}.track{display:flex;align-items:center;gap:8px}
.bar{height:14px;border-radius:0 4px 4px 0;background:var(--other);min-width:2px;flex:none}
.ours .bar{background:var(--ours)}.prev .bar{background:var(--prev)}
.val{font-size:13px;font-weight:600;white-space:nowrap}.val em{font-style:normal;font-weight:400;color:var(--ink2);margin-left:6px}
.row:hover .bar,.row:focus .bar{filter:brightness(1.12)}
.note{font-size:12.5px;color:var(--muted);margin:10px 0 0}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--card);font-size:12.5px;padding:6px 9px;border-radius:7px;opacity:0;transition:opacity .1s;max-width:280px;z-index:9}
details{margin-top:24px}summary{cursor:pointer;color:var(--ink2)}table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}
td,th{text-align:left;padding:5px 6px;border-bottom:1px solid var(--line)}td:nth-child(n+3),th:nth-child(n+3){text-align:right;font-variant-numeric:tabular-nums}
a{color:var(--ours)}
</style></head><body><main>
<h1>Inkling 2 Mini on Apple Silicon</h1>
<p class="lede">Tokens per second, higher is better. <b style="color:var(--ours)">Blue</b> is this patch, grey is other engines. Source: <a href="https://github.com/vasanthsreeram/inkling-apple-silicon">github.com/vasanthsreeram/inkling-apple-silicon</a></p>
{SECTIONS}
<p class="note">† Other session. Each machine's engines ran back to back; numbers from different harnesses (llama-server vs llama-bench) are not comparable with each other.</p>
<details><summary>All numbers as a table</summary><table><thead><tr><th>Machine</th><th>Run</th><th>Prefill</th><th>Decode</th></tr></thead><tbody>{TABLE}</tbody></table></details>
</main><div id="tip"></div>
<script>
var tip=document.getElementById('tip');
function show(e,el){tip.textContent=el.dataset.tip;tip.style.opacity=1;var r=el.getBoundingClientRect();var x=(e&&e.clientX)||r.left+20,y=(e&&e.clientY)||r.top;tip.style.left=Math.min(x+12,innerWidth-tip.offsetWidth-8)+'px';tip.style.top=(y-tip.offsetHeight-10)+'px';}
document.querySelectorAll('.row').forEach(function(el){el.addEventListener('mousemove',function(e){show(e,el)});el.addEventListener('focus',function(){show(null,el)});el.addEventListener('mouseleave',function(){tip.style.opacity=0});el.addEventListener('blur',function(){tip.style.opacity=0});});
</script></body></html>
"""

if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "results"
    (out / "comparison.svg").write_text(svg())
    (out / "comparison.html").write_text(html_page())
    print(out / "comparison.svg", out / "comparison.html")
