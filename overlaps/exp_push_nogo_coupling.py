"""exp_push_nogo_coupling.py — does the no-lick push (per-mouse Δdepth, Expert−Naive, Fig 4b/c) relate
to the change in NoGo-trial accuracy specifically? (Leon, 2026-09-04: NoGo trials are the only GNG arm
that improves with learning, Fig 1c.) Same estimator as Fig 4c — main_panels._panelC_coupling, the
between-mouse n=9 Spearman on per-mouse means (A/B aggregated within mouse) — with the GNG performance
column (odr_perf) restricted to Go-only or NoGo-only trials. Run from overlaps/ (canonical axis/norm)."""
import sys, os
sys.path.insert(0, '/home/leon/dual/'); os.chdir('/home/leon/dual/overlaps'); sys.path.insert(0, '/home/leon/dual/overlaps')
import numpy as np
from scipy.stats import pearsonr
import main_panels as MP
y = MP.y
tasks = sorted(y.tasks.unique()); print('tasks:', tasks)
go = [t for t in tasks if 'go' in t.lower() and 'nogo' not in t.lower()][0]
nogo = [t for t in tasks if 'nogo' in t.lower()][0]
rows = [('DPA, DPA trials', 'performance', y.tasks == 'DPA'), ('DPA, dual trials', 'performance', y.tasks != 'DPA'),
        ('DPA, Go trials', 'performance', y.tasks == go), ('DPA, NoGo trials', 'performance', y.tasks == nogo),
        ('GNG (Go+NoGo)', 'odr_perf', y.tasks != 'DPA'),
        ('Go only', 'odr_perf', y.tasks == go), ('NoGo only', 'odr_perf', y.tasks == nogo)]   # 2026-09-09: DPA arm by trial set
for name, col, mask in rows:
    d = MP._perf_delta_by_sample(col, mask)
    rho, p, n, mx, my = MP._panelC_coupling(d)
    r, pr = pearsonr(mx, my)
    print(f'{name:17s} n={n}  Spearman rho={rho:+.2f} p={p:.3f}   Pearson r={r:+.2f} p={pr:.3f}   '
          f'mean Δacc={my.mean():+.3f}  per-mouse Δacc={np.round(my,2).tolist()}')
