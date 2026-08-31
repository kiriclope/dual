"""Justified in-figure captions (shared by fig_dimensionality_main.py / fig_manifold_main.py).

matplotlib has no native text justification, so the caption is typeset word-by-word: word widths
are measured with the Agg renderer, lines are greedily wrapped to the text column, and on every
non-final line the slack is distributed evenly across the word gaps so both margins sit flush;
each paragraph's last line stays flush-left (print convention).

NB: the words land in the SVG as separate text elements — edit wording in the caller's paragraph
list and re-render rather than editing the SVG.
"""


def draw_justified(fig, paragraphs, fontsize=7.2, x0=0.012, x1=0.988, y0=-0.012,
                   color='k', linespacing=1.42):
    """Draw `paragraphs` (list of str) justified below the figure, starting at figure-fraction y0
    (negative = below the axes region; bbox_inches='tight' extends the canvas)."""
    fig.canvas.draw()                                  # Agg metrics for width measurement
    rend = fig.canvas.get_renderer()
    fw = fig.bbox.width
    cache = {}

    def wordw(s):
        if s not in cache:
            t = fig.text(0, 2, s, fontsize=fontsize)
            cache[s] = t.get_window_extent(renderer=rend).width / fw
            t.remove()
        return cache[s]

    spw = wordw('x x') - wordw('xx')                   # width of one space
    lh = fontsize * linespacing / 72.0 / fig.get_size_inches()[1]
    tw = x1 - x0
    y = y0
    for para in paragraphs:
        words = para.split()
        lines, cur, curw = [], [], 0.0
        for wd in words:
            add = wordw(wd) + (spw if cur else 0.0)
            if cur and curw + add > tw:
                lines.append(cur); cur, curw = [wd], wordw(wd)
            else:
                cur.append(wd); curw += add
        lines.append(cur)
        for li, lw in enumerate(lines):
            if li == len(lines) - 1 or len(lw) == 1:   # paragraph-final line: flush left
                fig.text(x0, y, ' '.join(lw), ha='left', va='top', fontsize=fontsize, color=color)
            else:                                      # justified: spread slack across the gaps
                ww = sum(wordw(w) for w in lw)
                gap = (tw - ww) / (len(lw) - 1)
                x = x0
                for wd in lw:
                    fig.text(x, y, wd, ha='left', va='top', fontsize=fontsize, color=color)
                    x += wordw(wd) + gap
            y -= lh
    return y
