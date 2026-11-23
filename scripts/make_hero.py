import html
import json
import shutil
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
data = json.loads((root / "results/summary.json").read_text())

SHORT = {
    "LM Studio, Q4_K_M": "LM Studio",
    "nanoattention, 4-bit": "nanoattention",
    "nanoattention, original weights": "nanoattention",
    "This patch, same Q4_K_M": "This patch · Q4_K_M",
    "This patch, tie64 + int8": "This patch · tie64 + int8",
    "This patch, all-4-bit": "This patch · all-4-bit",
}


def esc(s):
    return html.escape(str(s), quote=True)


def fmt(v):
    return f"{v:,.0f}" if v >= 100 else f"{v:.1f}"


def chart(section, key, title):
    rows = section["rows"]
    top = max(r[key] for r in rows)
    base = rows[0][key]
    out = [f'<div class="chart"><div class="ctitle">{esc(title)}</div>']
    for r in rows:
        cls = "ours" if r.get("ours") else "other"
        best = r[key] == max(x[key] for x in rows if x.get("ours")) and r.get("ours")
        ratio = f'<span class="x">{r[key] / base:.1f}×</span>' if r.get("ours") else ""
        mark = "<sup>†</sup>" if r.get("other_session") else ""
        out.append(
            f'<div class="row {cls}{" best" if best else ""}">'
            f'<div class="lab">{esc(SHORT.get(r["name"], r["name"]))}{mark}</div>'
            f'<div class="track"><div class="bar" style="width:{100 * r[key] / top:.2f}%"></div></div>'
            f'<div class="num">{fmt(r[key])}{ratio}</div></div>')
    out.append("</div>")
    return "".join(out)


def column(section):
    rows = section["rows"]
    ours = [r for r in rows if r.get("ours")]
    bp = max(r["prefill"] for r in ours) / rows[0]["prefill"]
    bd = max(r["decode"] for r in ours) / rows[0]["decode"]
    return (f'<div class="col"><div class="machine">{esc(section["machine"])}</div>'
            f'<div class="stats"><div class="stat"><b>{bp:.1f}×</b><span>prefill vs LM Studio</span></div>'
            f'<div class="stat"><b>{bd:.1f}×</b><span>decode vs LM Studio</span></div></div>'
            + chart(section, "prefill", "Prefill · 2K-token prompt · tok/s")
            + chart(section, "decode", "Decode · tok/s")
            + "</div>")


def page():
    cols = "".join(column(s) for s in data["engines"])
    return f"""<!doctype html><html><head><meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:1280px;background:#f4f2ec;font-family:Inter,-apple-system,Helvetica,Arial,sans-serif;color:#141412}}
.wrap{{padding:56px 64px 44px}}
.kicker{{font-size:15px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:#2a78d6}}
h1{{font-size:46px;font-weight:800;letter-spacing:-.02em;margin-top:10px}}
.sub{{font-size:19px;color:#5a5852;margin-top:10px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:28px;margin-top:38px}}
.col{{background:#fdfcf9;border:1px solid #e3e0d7;border-radius:22px;padding:30px 30px 26px;box-shadow:0 1px 0 rgba(0,0,0,.03),0 12px 30px -18px rgba(40,35,20,.18)}}
.machine{{font-size:22px;font-weight:700}}
.stats{{display:flex;gap:14px;margin-top:18px}}
.stat{{flex:1;background:#eef4fc;border-radius:14px;padding:14px 16px}}
.stat b{{display:block;font-size:40px;font-weight:800;color:#1f63b8;letter-spacing:-.02em;line-height:1}}
.stat span{{display:block;font-size:14px;color:#4a5a70;margin-top:6px;font-weight:500}}
.chart{{margin-top:24px}}
.ctitle{{font-size:13px;font-weight:600;color:#7a776e;letter-spacing:.04em;text-transform:uppercase;margin-bottom:12px}}
.row{{display:grid;grid-template-columns:190px 1fr 92px;align-items:center;gap:12px;margin-bottom:10px}}
.lab{{font-size:15px;color:#3b3a36;white-space:nowrap}}
.ours .lab{{font-weight:600;color:#141412}}
.track{{height:20px;background:#efede6;border-radius:6px;overflow:hidden}}
.bar{{height:100%;border-radius:6px;background:#b5b2a8}}
.ours .bar{{background:#7fb0ea}}
.best .bar{{background:linear-gradient(90deg,#2a78d6,#3d8ef0)}}
.num{{font-size:16px;font-weight:700;font-variant-numeric:tabular-nums;white-space:nowrap}}
.other .num{{font-weight:500;color:#5a5852}}
.x{{font-size:13px;font-weight:600;color:#2a78d6;margin-left:6px}}
sup{{font-size:11px;color:#8a877e;margin-left:2px}}
.foot{{margin-top:26px;font-size:14px;color:#7a776e;line-height:1.55;display:flex;justify-content:space-between;gap:30px}}
.foot b{{color:#3b3a36;font-weight:600}}
</style></head><body><div class="wrap">
<div class="kicker">llama.cpp Metal patch</div>
<h1>Inkling 2 Mini, up to 2.9× faster on a Mac</h1>
<div class="sub">Same machine, same Q4_K_M file, back to back against LM Studio and nanoattention. Higher is better.</div>
<div class="grid">{cols}</div>
<div class="foot"><div>Measured through <b>llama-server</b>: fresh 2,048-token prompts, 128-token greedy decodes. Air on AC, 24 Nov; Studio 23 Nov.<br>† nanoattention on the Air ran in a separate session, where LM Studio scored 165 / 24.0. tie64 + int8 matches the unpatched model on 97.3% of next tokens.</div>
<div style="text-align:right;white-space:nowrap"><b>github.com/vasanthsreeram/inkling-apple-silicon</b></div></div>
</div></body></html>
"""


def browser():
    for name in ("chromium", "chromium-browser", "google-chrome"):
        if shutil.which(name):
            return name
    mac = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    return mac if Path(mac).exists() else None


if __name__ == "__main__":
    out = root / "results"
    src = out / "hero.html"
    src.write_text(page())
    exe = browser()
    if not exe:
        sys.exit("wrote results/hero.html; no Chromium found for the PNG")
    subprocess.run([exe, "--headless", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
                    "--force-device-scale-factor=2", "--window-size=1280,860", "--virtual-time-budget=4000",
                    f"--screenshot={out / 'hero.png'}", src.as_uri()], check=True, capture_output=True)
    print(out / "hero.png")
