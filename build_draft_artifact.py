"""build_draft_artifact.py — render the shared DRAFT artifact page from the two manuscript markdown files.

  cd /home/leon/dual && /home/leon/mambaforge/envs/dual/bin/python build_draft_artifact.py

then publish the result with the Artifact tool, passing the URL so it updates in place:
  file_path = figures/paper_share/artifact_build/mpfc_dual_draft_v1.html
  url       = https://claude.ai/code/artifact/d338b195-af51-4f54-9ea9-0a2ef1f99235

Inputs   docs/paper/results_draft.md · docs/paper/discussion_draft.md
Template figures/paper_share/artifact_build/draft_template.html  (page head, CSS, pagehead + {{PLACEHOLDERS}})
Output   figures/paper_share/artifact_build/mpfc_dual_draft_v1.html

The Results markdown is split into the page's four parts at three boundaries that MUST stay in the file:
    intro     — everything above the `<!-- RESULTS -->` marker (banners, title, Abstract, untitled openers)
    results   — `<!-- RESULTS -->` up to `## Methods`
    methods   — `## Methods` up to `## Extended Data Figures` (includes Figure legends + References)
    appendix  — `## Extended Data Figures` to the end (shown as the working appendix)
The Discussion file is one part. Version strings in the pagehead are read from the drafts themselves (the
first `> **vNN.NN` banner in the Results, the `draft vN.N` in the Discussion's title line), so a new banner
is enough — nothing to bump here.

Rebuilt 2026-09-09 after the original lived in a Claude job tmp dir that was cleaned up. If the page itself is
ever lost, recover it with the Artifact tool's `action: "read"` on the URL above and strip the
`<!doctype …><body>` skeleton (it is re-added at publish time); that is how the template was recovered.
"""
import re
import subprocess

ROOT = '/home/leon/dual'
BUILD = f'{ROOT}/figures/paper_share/artifact_build'
TPL = f'{BUILD}/draft_template.html'
OUT = f'{BUILD}/mpfc_dual_draft_v1.html'
RES_MD = f'{ROOT}/docs/paper/results_draft.md'
DIS_MD = f'{ROOT}/docs/paper/discussion_draft.md'

res = open(RES_MD, encoding='utf-8').read().split('\n')
dis = open(DIS_MD, encoding='utf-8').read().split('\n')


def boundary(pred, what):
    hits = [i for i, ln in enumerate(res) if pred(ln)]
    assert hits, f'boundary missing from results_draft.md: {what}'
    return hits[0]


i_res = boundary(lambda ln: ln.strip() == '<!-- RESULTS -->', '<!-- RESULTS --> marker')
i_met = boundary(lambda ln: ln.startswith('## Methods'), '## Methods')
i_app = boundary(lambda ln: ln.startswith('## Extended Data Figures'), '## Extended Data Figures')
assert i_res < i_met < i_app, 'results_draft.md sections are out of order'

chunks = {'intro': res[:i_res], 'results': res[i_res:i_met], 'methods': res[i_met:i_app],
          'appendix': res[i_app:], 'discussion': dis}

html = {}
for k, lines in chunks.items():
    html[k] = subprocess.run(['pandoc', '-f', 'gfm', '-t', 'html'], input='\n'.join(lines),
                             capture_output=True, text=True, check=True).stdout
    print(f'{k:10s} {len(lines):4d} md lines -> {len(html[k]):7d} html chars')

m_res = re.search(r'^> \*\*(v[\d.]+)', '\n'.join(res), re.M)
m_dis = re.search(r'draft (v[\d.]+)', dis[0])
assert m_res and m_dis, 'could not read the version banners'

page = open(TPL, encoding='utf-8').read()
for key, val in [('RES_V', m_res.group(1)), ('DIS_V', m_dis.group(1)), ('INTRO', html['intro']),
                 ('RESULTS', html['results']), ('DISCUSSION', html['discussion']),
                 ('METHODS', html['methods']), ('APPENDIX', html['appendix'])]:
    ph = '{{%s}}' % key
    assert ph in page, f'placeholder {ph} missing from the template'
    page = page.replace(ph, val)
assert '{{' not in page, 'unfilled placeholder left in the page'

open(OUT, 'w', encoding='utf-8').write(page)
print(f'draft html written {len(page)} chars ({m_res.group(1)} / {m_dis.group(1)}) -> {OUT}')
