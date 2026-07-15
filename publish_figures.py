#!/usr/bin/env python3
"""Build a SELF-CONTAINED figure gallery (images embedded as data: URIs) for cloud publishing.

Unlike serve_figures.py (live, local, SSH-tunnel), this bakes a chosen subset of PNGs into one
standalone HTML file with no external references — suitable for hosting as a shareable page.

    python publish_figures.py --match cosine --out /tmp/gallery.html --title "overlaps cosine"

--match is a regex tested against each figure's repo-relative path (case-insensitive). Only PNGs
are embedded (SVGs are vector but bloat data URIs). Keep the set small — data URIs are ~1.35x the
raw byte size, so a few dozen figures is the practical ceiling for a single page.
"""
import argparse, base64, html, os, re

BASE = os.path.dirname(os.path.abspath(__file__))


def collect(pattern):
    rx = re.compile(pattern, re.I) if pattern else None
    out = []
    for root, dirs, files in os.walk(BASE):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'svg']
        for f in files:
            if not f.lower().endswith('.png'):
                continue
            rel = os.path.relpath(os.path.join(root, f), BASE)
            if rx is None or rx.search(rel):
                out.append((rel, os.stat(os.path.join(root, f)).st_mtime))
    out.sort(key=lambda t: t[1], reverse=True)
    return [r for r, _ in out]


def build(rels, title):
    cards = []
    for rel in rels:
        with open(os.path.join(BASE, rel), 'rb') as fh:
            b64 = base64.b64encode(fh.read()).decode('ascii')
        grp = html.escape(rel.rsplit('/png/', 1)[0].split('/')[-1] if '/png/' in rel else os.path.dirname(rel))
        cards.append(f'<figure class=card><img src="data:image/png;base64,{b64}">'
                     f'<figcaption><b>{grp}</b><br>{html.escape(os.path.basename(rel))}</figcaption></figure>')
    return f"""<h1>{html.escape(title)}</h1>
<p class=sub>{len(rels)} figures · embedded snapshot</p>
<main>{''.join(cards)}</main>
<style>
body{{margin:0;background:#111;color:#ddd;font:14px -apple-system,Segoe UI,sans-serif}}
h1{{padding:16px 16px 0;font-size:18px;color:#fff}} .sub{{padding:0 16px;color:#888}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px;padding:16px}}
.card{{margin:0;background:#1b1b1b;border:1px solid #2c2c2c;border-radius:8px;overflow:hidden}}
.card img{{width:100%;display:block;background:#fff}}
figcaption{{padding:8px 10px;font-size:12px}} figcaption b{{color:#6ca}}
</style>"""


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--match', default='', help='regex on repo-relative path (case-insensitive)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--title', default='dual figures')
    a = ap.parse_args()
    rels = collect(a.match)
    with open(a.out, 'w') as fh:
        fh.write(build(rels, a.title))
    print(f'embedded {len(rels)} figures -> {a.out}  ({os.path.getsize(a.out)/1e6:.1f} MB)')
    for r in rels:
        print('  ', r)
