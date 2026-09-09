"""build_variant_page.py SUF LOG TITLE OUT_HTML DEFINITION — one page with Figs 2/3/4/6 of a variant build.

Durable home (2026-09-09): this and run_axis_variant.sh used to live in a Claude scratchpad, which is
cleaned up between sessions. They are now tracked in the repo root beside make_submission_figs.py.

  ./run_axis_variant.sh _t1 54-65 decision_t1 _tewin 2>&1 | tee .variant_tmp/variant_t1.log
  python build_variant_page.py _t1 .variant_tmp/variant_t1.log \
         "Choice axis = test odor + 1 s (bins 54-65)" .variant_tmp/variant_t1.html \
         "Sample/GNG axis bins 36-38 (6.0-6.5 s); choice/test axis bins 54-65 (9.0-11.0 s)."
  then publish with the Artifact tool: file_path = the html, url = that variant's artifact.

The five variant artifacts (sample/GNG axis 36-38 in all; test odor is 9.0-10.0 s) — republish in place,
never create new ones:
  A _te  54-59  test odor only        https://claude.ai/code/artifact/52ee577b-46d5-4f07-91ff-77d60b2f5e55
  C _tc  54-62  test + 0.5 s CANONICAL https://claude.ai/code/artifact/9ac2ba19-8875-4c85-8fe8-b8fc46fa4483
  B _t1  54-65  test + 1 s            https://claude.ai/code/artifact/e2134a9b-74d1-4cdc-baf9-53c3929768af
  D _td  57-62                        https://claude.ai/code/artifact/1fb757d6-f332-4d8f-b00e-43f6adad26a6
  E _f2  57-65                        https://claude.ai/code/artifact/b7fbf2e1-92de-49e6-8425-223e56144a09
  index  Axis-window variants         https://claude.ai/code/artifact/9a161ae8-f525-402e-ab31-d1bde23e2f5b

The per-figure blurbs are SCRAPED from the run log, so a change to a script's printed format silently
empties one. Every miss is now reported on stderr — check the run output, do not trust a quiet page.
"""
import base64, io, os, re, sys
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
os.chdir('/home/leon/dual')
SUF, LOG, TITLE, OUT, DEFN = sys.argv[1:6]
log = open(LOG, encoding='utf-8').read()


MISSED = []


def grab(pat, default='n/a'):
    m = re.search(pat, log)
    if not m:
        MISSED.append(pat)
    return m.group(1) if m else default


def findall(pat):
    out = re.findall(pat, log)
    if not out:
        MISSED.append(pat)
    return out


push = grab(r'A depth \[mixed model \(9 mice, 36 obs\)\] (β=[^ ]+ p=[^ ]+)')
pm = grab(r'A depth per-mouse: (ΔA=[^)]+\))')
cdpa = grab(r'B\[Δ DPA \] per-mouse n=9 (Spearman ρ=[^ ]+ p=[^ \n]+)')
cgng = grab(r'B\[Δ GNG \] per-mouse n=9 (Spearman ρ=[^ ]+ p=[^ \n]+)')
dprime = grab(r'D action-code (d′ Naive=[^\n]+)')
egen = findall(r'E-gen: (\w+)\s+Expert within \[([^\]]+)\]\s+transferred frac \[([^\]]+)\]\s+mean ([\d.]+)\s+PS ([\d.]+)', log)
fstat = findall(r'F: (\w+)\s+per-mouse cross ([\d.]+) -> ([\d.]+)\s+p=([\d.]+)', log)
cosb = findall(r'b: (\w+) sample-action ([\d.]+)\s+sample-distr ([\d.]+)\s+action-distr ([\d.]+)\s+\(reliab \[([^\]]+)\]', log)
cdec = findall(r'C-dec: decision\s+(\w+)\s+(\w+)\s+E ([\d.]+) \(n95 [\d.]+\)\s*\*?\s*N ([\d.]+)', log)
opto = findall(r'(d_dpa|d_gng|trade-off): corr r=([^ ]+) p=([^ ]+) ρ=([^ ]+) p=([^ ]+)\s+\|\s+LMM β=([^ ]+) p=([^ ]+)', log)
figs = [
    ('Figure 2', f'pca/figures/pseudo/dimensionality/png/fig_dimensionality_main{SUF}.png',
     ('Decision-state decodability (expert / naïve): ' + '; '.join(f'{s} {v} {e}/{n}' for s, v, e, n in cdec) + '. ' if cdec else '') +
     ('Panel e: ' + '; '.join(f'{v} within {w.strip()} · transfer {t.strip()} (mean {m}) · PS {p}' for v, w, t, m, p in egen) + '. ' if egen else '') +
     ('Panel f: ' + '; '.join(f'{v} {a}→{b} p={p}' for v, a, b, p in fstat) if fstat else '')),
    ('Figure 3', f'pca/figures/pseudo/dimensionality/png/fig_manifold_main{SUF}.png',
     ('Cosines (corrected): ' + '; '.join(f'{st} sample×choice {a}, sample×GNG {b}, choice×GNG {c} (rel {r})' for st, a, b, c, r in cosb) if cosb else '')),
    ('Figure 4', f'overlaps/figures/overlaps/main/png/fig_overlaps_main_ab_dpaact{SUF}.png',
     f'Push: mixed model {push}; per-animal {pm}. Coupling ΔDPA: {cdpa}; ΔGNG: {cgng}. Choice-code {dprime}.'),
    ('Figure 6', f'overlaps/figures/overlaps/behavior/png/behavior_opto_main{SUF}.png',
     ('; '.join(f'{k}: r={r} p={p} (clustered β={b} p={pb})' for k, r, p, _, _, b, pb in opto) if opto else '')),
]


def enc(p):
    im = Image.open(p).convert('RGB'); im.thumbnail((2600, 2600), Image.LANCZOS)
    im = im.quantize(256, method=Image.MEDIANCUT); b = io.BytesIO(); im.save(b, 'PNG', optimize=True)
    return base64.b64encode(b.getvalue()).decode()


parts = [f'<title>{TITLE}</title>',
         '<style>:root{--bg:#FBFBF9;--ink:#1C1C26;--muted:#66666F;--line:#E3E3E8}@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#16161C;--ink:#E7E7EC;--muted:#9C9CA8;--line:#2C2C36}}:root[data-theme="dark"]{--bg:#16161C;--ink:#E7E7EC;--muted:#9C9CA8;--line:#2C2C36}body{background:var(--bg);color:var(--ink);font-family:Helvetica,Arial,sans-serif;font-size:15px;line-height:1.5;margin:0}main{max-width:1180px;margin:0 auto;padding:40px 24px 60px}h1{font-size:24px;margin:0 0 6px}p.sub{color:var(--muted);margin:0 0 24px}section{padding:28px 0;border-bottom:1px solid var(--line)}h2{font-size:18px;margin:0 0 4px}p.claim{color:var(--muted);margin:0 0 12px;font-size:13.5px}div.wrap{background:#fff;border:1px solid #D8D8DF;border-radius:4px;padding:8px;overflow-x:auto}img{display:block;max-width:100%;height:auto;margin:0 auto}</style>',
         f'<main><h1>{TITLE}</h1><p class="sub">{DEFN} Panel annotations carry this build’s own statistics; caption entries still quote the canonical build.</p>']
for t, p, d in figs:
    if not os.path.exists(p):
        print('MISSING', p); continue
    parts.append(f'<section><h2>{t}</h2><p class="claim">{d}</p><div class="wrap"><img src="data:image/png;base64,{enc(p)}" alt="{t}"></div></section>')
parts.append('</main>')
html = '\n'.join(parts); open(OUT, 'w', encoding='utf-8').write(html)
print(f'{OUT}: {len(html)/1e6:.2f} MB')
if MISSED:
    print(f'\n{len(MISSED)} log pattern(s) matched NOTHING — the page is missing those numbers.', file=sys.stderr)
    for m in MISSED:
        print('   ' + m, file=sys.stderr)
