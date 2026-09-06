#!/usr/bin/env python
"""make_comment_copy.py — build the collaborator "comment copy" of the manuscript.

Assembles MANUSCRIPT TEXT ONLY (no version banners, no working appendix) from
`docs/paper/results_draft.md` + `docs/paper/discussion_draft.md`, in submission order:

    Abstract → Introduction → Results → Discussion → Methods → Figure legends → References

and writes two outputs to --outdir:

  comment_copy.docx   the same document with the five main figures embedded above their
                      legends. THIS is the one to put in Google Drive: drag it into the
                      shared folder and Drive converts it to a Google Doc with the figures
                      intact.
  comment_copy.md     text-only (legends kept, images replaced by a pointer line). This is
                      what the Google Drive connector can upload directly.

WHY TWO OUTPUTS (2026-09-06). The Drive connector takes file content inline in the request,
so an upload is capped at roughly 100 KB. Five main figures do not compress below ~360 KB
even at JPEG q40 / 700 px (measured), i.e. ~540 KB of base64 — an order of magnitude over.
Anything that would fit is ~7 KB per figure, an illegible thumbnail. So the connector gets
the text and the .docx carries the figures.

The Google Doc CANNOT be content-updated in place (the connector's update_file is metadata
only), so every refresh creates a NEW Doc and the previous one should be trashed.

Usage:
    /home/leon/mambaforge/envs/dual/bin/python make_comment_copy.py [--outdir DIR] [--width PX]
"""
import argparse, io, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGS = {1: 'Fig1_behaviour', 2: 'Fig2_geometry', 3: 'Fig3_one_manifold',
        4: 'Fig4_learning', 6: 'Fig6_opto'}
HEADER = ("Comment copy ({date}) of `results_draft.md` and `discussion_draft.md`. The repository "
          "markdown is canonical: comments left here are harvested and applied to the source, and "
          "the shared claude.ai pages are refreshed. {figline} [Author Year] tags resolve in the "
          "References section at the end.")


def _lines(path):
    return open(path, encoding='utf-8').read().split('\n')


def _at(lines, prefix, start=0):
    for i in range(start, len(lines)):
        if lines[i].startswith(prefix):
            return i
    raise SystemExit(f'marker not found: {prefix!r}')


def _strip_notes(text):
    """Drop the working-note blockquotes; the comment copy carries manuscript text only."""
    return re.sub(r'(?m)^>.*\n?', '', text).strip()


def build_sections():
    L = _lines(os.path.join(ROOT, 'docs/paper/results_draft.md'))
    i_abs = _at(L, '## Abstract')
    i_hr = _at(L, '---', i_abs)
    i_res = _at(L, '<!-- RESULTS -->')          # Intro/Results boundary marker — keep it in the md
    i_met = _at(L, '## Methods')
    i_leg = _at(L, '## Figure legends')
    i_ref = _at(L, '## References')
    i_ed = _at(L, '## Extended Data Figures')
    sec = dict(
        abstract='\n'.join(L[i_abs + 1:i_hr]).strip(),
        intro='\n'.join(L[i_hr + 1:i_res]).strip(),
        results='\n'.join(L[i_res + 1:i_met]).strip(),
        methods=_strip_notes('\n'.join(L[i_met + 1:i_leg])),
        legends=_strip_notes('\n'.join(L[i_leg + 1:i_ref])),
        refs=_strip_notes('\n'.join(L[i_ref + 1:i_ed])),
    )
    D = _lines(os.path.join(ROOT, 'docs/paper/discussion_draft.md'))
    while D and (D[0].startswith('#') or not D[0].strip()):
        D.pop(0)
    sec['discussion'] = _strip_notes('\n'.join(D))
    return sec


def shrink_figures(outdir, width):
    """Downscale the share-copy PNGs so the .docx stays a few MB rather than 60."""
    from PIL import Image
    figdir = os.path.join(outdir, 'figs')
    os.makedirs(figdir, exist_ok=True)
    paths = {}
    for n, name in FIGS.items():
        src = os.path.join(ROOT, 'figures/paper_share', name + '.png')
        im = Image.open(src).convert('RGB')
        im.thumbnail((width, width * 3), Image.LANCZOS)
        dst = os.path.join(figdir, name + '.png')
        im.save(dst, optimize=True)
        paths[n] = dst
        print(f'  Fig {n}: {im.size[0]}x{im.size[1]}  {os.path.getsize(dst) / 1e6:.2f} MB')
    return paths


def render_legends(legends, paths=None):
    """Bold each legend's opening sentence; insert the figure above it when paths are given."""
    out = []
    for para in legends.split('\n\n'):
        m = re.match(r'Figure (\d) \|', para.strip())
        if not m:
            out.append(para)
            continue
        n = int(m.group(1))
        if paths:
            out.append('![](%s){width=6.4in}' % paths[n])
        else:
            out.append('*[Figure %d — on the shared claude.ai figures page]*' % n)
        out.append('**' + ' '.join(para.split()) + '**')
    return '\n\n'.join(out)


def assemble(sec, legends_block, figline, date):
    return f"""# Compositional learning by geometric editing — draft for comments

{HEADER.format(date=date, figline=figline)}

---

## Abstract

{sec['abstract']}

---

## Introduction

{sec['intro']}

## Results

{sec['results']}

## Discussion

{sec['discussion']}

## Methods

{sec['methods']}

## Figure legends

{legends_block}

## References

{sec['refs']}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default=os.environ.get('CLAUDE_JOB_DIR', '/tmp') + '/comment_copy')
    ap.add_argument('--width', type=int, default=1600, help='max figure dimension in the .docx')
    ap.add_argument('--date', default=None)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    date = a.date or __import__('datetime').date.today().isoformat()

    sec = build_sections()
    print('sections:', {k: len(v.split()) for k, v in sec.items()})

    md_text = assemble(sec, render_legends(sec['legends']),
                       'Figure legends are included below; the figures themselves are on the '
                       'shared claude.ai figures page.', date)
    p_md = os.path.join(a.outdir, 'comment_copy.md')
    open(p_md, 'w', encoding='utf-8').write(md_text)
    print(f'\ntext-only  -> {p_md}  ({len(md_text) / 1e3:.0f} KB — upload this via the Drive connector)')

    print('\nfigures:')
    paths = shrink_figures(a.outdir, a.width)
    md_fig = assemble(sec, render_legends(sec['legends'], paths),
                      'The five main figures are embedded below, each above its legend.', date)
    p_fig_md = os.path.join(a.outdir, '_with_figures.md')
    open(p_fig_md, 'w', encoding='utf-8').write(md_fig)
    p_docx = os.path.join(a.outdir, 'comment_copy.docx')
    subprocess.run(['pandoc', '-f', 'markdown', '-t', 'docx', '-o', p_docx, p_fig_md], check=True)
    print(f'\nwith figures -> {p_docx}  ({os.path.getsize(p_docx) / 1e6:.2f} MB — '
          f'drag into Drive to get a Doc with the figures)')


if __name__ == '__main__':
    main()
