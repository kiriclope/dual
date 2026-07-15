#!/usr/bin/env python3
"""Live figure gallery for /home/leon/dual — replaces sshfs for looking at PNGs.

Browse by FOLDER: the landing page lists every figure directory (newest-active first) with a cover
thumbnail and count; click one to render just that folder's figures. Nothing loads until you open a
folder, so the ~2000-figure repo stays snappy. Regenerate a figure, refresh, it's there — no mount.

Usage (on the remote box, once):
    /home/leon/mambaforge/envs/dual/bin/python serve_figures.py            # port 8000
    /home/leon/mambaforge/envs/dual/bin/python serve_figures.py --port 9001

Then on your LAPTOP (once per ssh session, or via LocalForward in ~/.ssh/config):
    ssh -L 8000:localhost:8000 <this-box>
and open http://localhost:8000.

Binds 127.0.0.1 only — reachable exclusively through the SSH tunnel.
"""
import argparse, html, os
from collections import defaultdict
from urllib.parse import quote, unquote, urlsplit
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

BASE = os.path.dirname(os.path.abspath(__file__))
EXTS = ('.png',)                       # PNG only; SVG twins share the folder and view rasters here


def scan():
    """dir_rel -> [(name, mtime)] for every figure directory under BASE."""
    groups = defaultdict(list)
    for root, dirs, files in os.walk(BASE):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'svg']
        for f in files:
            if f.lower().endswith(EXTS):
                try:
                    mt = os.stat(os.path.join(root, f)).st_mtime
                except OSError:
                    continue
                groups[os.path.relpath(root, BASE)].append((f, mt))
    return groups


def label(dir_rel):
    """Friendly folder label: drop a trailing /png, keep the last 2-3 meaningful parts."""
    parts = [p for p in dir_rel.split(os.sep) if p not in ('figures', 'png')]
    dedup = [p for i, p in enumerate(parts) if i == 0 or p != parts[i - 1]]   # drop repeated 'overlaps/overlaps'
    return ' / '.join(dedup[-3:]) if dedup else dir_rel


HEAD = """<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;font:14px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;background:#0f1113;color:#dde}
a{color:inherit;text-decoration:none}
header{position:sticky;top:0;background:#16191ccc;backdrop-filter:blur(8px);padding:11px 16px;
  border-bottom:1px solid #262b30;display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:5}
header h1{font-size:15px;margin:0;font-weight:600;color:#fff;letter-spacing:.01em}
header h1 .crumb{color:#5b8;font-weight:500}
#q{flex:1;min-width:180px;padding:7px 10px;background:#1c2024;border:1px solid #333a40;border-radius:6px;color:#eee;font-size:14px}
#q:focus{outline:none;border-color:#3a8}
#count{color:#7a8590;font-size:12px;white-space:nowrap;font-variant-numeric:tabular-nums}
.back{padding:6px 11px;background:#1c2024;border:1px solid #333a40;border-radius:6px;color:#cde;font-size:13px}
.back:hover{border-color:#3a8}
button{padding:7px 11px;background:#1c2024;border:1px solid #333a40;border-radius:6px;color:#ccd;cursor:pointer;font-size:13px}
button.on{background:#2f9e6f;border-color:#2f9e6f;color:#04120a}
main{display:grid;gap:14px;padding:16px}
.folders{grid-template-columns:repeat(auto-fill,minmax(260px,1fr))}
.figs{grid-template-columns:repeat(auto-fill,minmax(300px,1fr))}
.card{margin:0;background:#171a1d;border:1px solid #24292e;border-radius:9px;overflow:hidden;display:flex;flex-direction:column}
.card:hover{border-color:#356}
.card img{width:100%;display:block;background:#fff;min-height:60px}
.folder .thumb{height:150px;background:#fff center/cover no-repeat}
figcaption,.meta{padding:8px 11px;display:flex;flex-direction:column;gap:2px;font-size:12px}
.lbl{color:#6ec7a0;font-weight:600}
.nm{color:#aab3bb;word-break:break-all}
.sub{color:#6b757e;font-size:11px;font-variant-numeric:tabular-nums}
time{color:#6b757e;font-size:11px}
.hidden{display:none}
</style>"""


def render_index(groups):
    order = sorted(groups.items(), key=lambda kv: max(m for _, m in kv[1]), reverse=True)
    cards = []
    for d, items in order:
        newest_name, newest_mt = max(items, key=lambda t: t[1])
        cover = quote(f'{d}/{newest_name}') + f'?v={int(newest_mt)}'   # mtime busts the browser cache
        cards.append(
            f'<a class="card folder" href="/view?d={quote(d)}" data-h="{html.escape((label(d)+" "+d).lower())}">'
            f'<div class="thumb" style="background-image:url(/{cover})"></div>'
            f'<div class="meta"><span class="lbl">{html.escape(label(d))}</span>'
            f'<span class="sub">{len(items)} figures · <time data-ts="{newest_mt:.0f}"></time></span></div></a>')
    body = (f'<header><h1>dual figures</h1>'
            f'<input id=q placeholder="filter folders (e.g. cosine, story, flow)" autofocus>'
            f'<span id=count></span></header>'
            f'<main class="folders" id=grid>{"".join(cards)}</main>')
    return page(body, total=len(order), unit='folders')


def render_folder(d, items):
    items = sorted(items, key=lambda t: t[1], reverse=True)
    cards = []
    for name, mt in items:
        src = quote(f'{d}/{name}') + f'?v={int(mt)}'                   # mtime busts the browser cache
        cards.append(
            f'<figure class=card data-h="{html.escape(name.lower())}">'
            f'<a href="/{src}" target="_blank"><img loading="lazy" src="/{src}"></a>'
            f'<figcaption><span class="nm">{html.escape(name)}</span>'
            f'<time data-ts="{mt:.0f}"></time></figcaption></figure>')
    body = (f'<header><a class="back" href="/">← folders</a>'
            f'<h1><span class="crumb">{html.escape(label(d))}</span></h1>'
            f'<input id=q placeholder="filter figures in this folder" autofocus>'
            f'<span id=count></span></header>'
            f'<main class="figs" id=grid>{"".join(cards)}</main>')
    return page(body, total=len(items), unit='figures')


def page(body, total, unit):
    js = f"""
const items=[...document.querySelectorAll('#grid>*')], q=document.getElementById('q'),
      cnt=document.getElementById('count');
const fmt=ts=>{{const d=new Date(ts*1000),p=n=>String(n).padStart(2,'0');
  return (d.getMonth()+1)+'/'+d.getDate()+' '+p(d.getHours())+':'+p(d.getMinutes());}};
document.querySelectorAll('time[data-ts]').forEach(t=>t.textContent=fmt(+t.dataset.ts));
function apply(){{let re=null;try{{re=new RegExp(q.value.trim(),'i');}}catch(e){{}}
  let n=0;items.forEach(el=>{{const h=el.dataset.h||'';
    const ok=!q.value.trim()||(re?re.test(h):h.includes(q.value.toLowerCase()));
    el.classList.toggle('hidden',!ok);if(ok)n++;}});
  cnt.textContent=n+' / {total} {unit}';}}
q.addEventListener('input',apply);apply();"""
    return f"<!doctype html><html><head>{HEAD}<title>dual figures</title></head><body>{body}<script>{js}</script></body></html>"


class Handler(SimpleHTTPRequestHandler):
    def _send(self, body):
        b = body.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(b)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlsplit(self.path)
        if u.path in ('/', '/index.html'):
            self._send(render_index(scan()))
            return
        if u.path == '/view':
            d = unquote(u.query[2:]) if u.query.startswith('d=') else ''
            self._send(render_folder(d, scan().get(d, [])))
            return
        super().do_GET()

    def translate_path(self, path):
        rel = unquote(urlsplit(path).path).lstrip('/')
        return os.path.join(BASE, *[p for p in rel.split('/') if p not in ('', '.', '..')])

    def log_message(self, format, *args):
        pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8000)
    a = ap.parse_args()
    g = scan()
    print(f'serving {sum(len(v) for v in g.values())} figures in {len(g)} folders from {BASE}')
    print(f'  remote:  http://127.0.0.1:{a.port}  (localhost only)')
    print(f'  laptop:  ssh -L {a.port}:localhost:{a.port} <this-box>  then open http://localhost:{a.port}')
    ThreadingHTTPServer(('127.0.0.1', a.port), Handler).serve_forever()
