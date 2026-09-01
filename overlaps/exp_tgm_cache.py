"""TGM cache — temporal-generalisation matrices (King & Dehaene 2014) per code, for the ED
supplement (user routing 2026-09-01: "cross-temporal matrices go in supp").

The overlaps tensor already contains the raw material: X[row, 1, train_bin, test_bin] is the
cross-validated decision function of the decoder TRAINED at train_bin, read at test_bin, for the
trial in that row. The TGM is then just balanced accuracy of sign(d) over trials, per
(train_bin, test_bin) pair — no refitting.

Codes (rows of the label table, per target): sample (DPA correct, A vs B) · lick/choice (DPA,
behavioural lick vs no-lick) · test (DPA correct, C vs D) · gng (dual, Go vs NoGo). Correct
laser-off trials of the named task set, per stage. 84x84 matrices.

Merges {'TGM' (+'_pca20' with --pca)} into pca results.pkl: TGM[(stage, code)] = (84, 84).
Run:  cd /home/leon/dual/overlaps && /home/leon/mambaforge/envs/dual/bin/python exp_tgm_cache.py
"""
import sys, os, warnings, pickle
warnings.filterwarnings('ignore'); sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/leon/dual/overlaps')
import numpy as np
from src.pca.io import pkl_load                                # house loader (as main_panels)
from src.common.options import set_options

TSUF = '_pca20' if '--pca' in sys.argv[1:] else ''
options = set_options()
DUM = 'log_generalizing_overlaps_none_l1_ratio_0.0'
if '--pca' in sys.argv[1:]:
    DUM += '_pca_20'                                            # mirrors main_panels' FILE naming
BDUM = f'{DUM}_raw_targets_choice-gng-sample-test'
DATA_IN = '../data/overlaps'

print('loading tensor …', flush=True)
X = pkl_load(f'X_{BDUM}', path=DATA_IN)
y = pkl_load(f'labels_{BDUM}', path=DATA_IN)
print('  X', X.shape, flush=True)

CODES = [('sample', 'sample', 'sample_odor', 'DPA', True),
         ('choice', 'choice', 'choice', 'DPA', False),          # behavioural lick: all perf
         ('test', 'test', 'test_odor', 'DPA', True),
         ('gng', 'gng', 'gng', 'Dual', True)]

TGM = {}
for code, target, col, taskset, correct_only in CODES:
    rows = (y.target == target).to_numpy()
    Xc = X[rows][:, 1, :, :]                                    # (trials, train_bin, test_bin)
    yc = y[rows].reset_index(drop=True)
    for stage in ['Naive', 'Expert']:
        m = ((yc.laser == 0).to_numpy() & (yc.learning == stage).to_numpy())
        m &= ((yc.tasks == 'DPA').to_numpy() if taskset == 'DPA'
              else (yc.tasks != 'DPA').to_numpy())
        if correct_only:
            m &= (yc.performance == 1).to_numpy()
        lab = yc[col].to_numpy()[m].astype(float)
        D = Xc[m]
        acc = np.zeros((84, 84))
        for cls, sgn in ((1, 1), (0, -1)):
            sel = lab == cls
            if sel.sum():
                acc += 0.5 * np.mean(sgn * D[sel] > 0, axis=0)
        TGM[(stage, code)] = acc
        di = np.diag(acc)
        print(f'  {stage:6s} {code:7s} n={int(m.sum()):4d}  diag max {di.max():.2f} '
              f'@bin {int(di.argmax())}  offdiag mean {acc[~np.eye(84, dtype=bool)].mean():.2f}',
              flush=True)

RES = '/home/leon/dual/pca/figures/pseudo/dimensionality/results.pkl'
d = pickle.load(open(RES, 'rb'))
d['TGM' + TSUF] = TGM
pickle.dump(d, open(RES, 'wb'))
print('merged TGM' + TSUF, 'into', RES)
