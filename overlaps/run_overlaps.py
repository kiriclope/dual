"""
Run cross-condition generalised decoding (CCGD) for all mice and save results.

For each (mouse, stage, target, context) the script fits a time-generalising
logistic decoder via ccgd_validation and stacks the per-trial decision-function
projections into a single array.

Two-phase pipeline
------------------
Phase 1 (--rebuild):  load raw fluorescence per mouse, build the shared
                      pseudo-population array, and save it.
Phase 2 (default):    load the pre-built pseudo-population and run CCGD.

Pass --rebuild to run both phases; omit it to skip phase 1 and use an
existing X_all_nan_<scale>.pkl.

Inputs  (phase 1)
-----------------
Raw per-mouse .mat files, accessed via get_X_y_days.

Inputs  (phase 2)
-----------------
<data-in>/X_all_nan_<scale>.pkl   (n_trials, n_neurons, n_time)
<data-in>/y_all_nan_<scale>.pkl   DataFrame

Outputs
-------
<data-in>/X_all_nan_<scale>.pkl          written by --rebuild
<data-out>/X_<dum>.pkl                   (n_trials, 2, n_train, n_test)
<data-out>/labels_<dum>.pkl              DataFrame aligned with X

Examples
--------
# default run (loads existing X_all_nan_.pkl)
python run_overlaps.py

# rebuild pseudo-population then run CCGD
python run_overlaps.py --rebuild

# rebuild with std normalisation, 'all' days, then run
python run_overlaps.py --rebuild --scale std --days all

# quick test: one mouse, Expert only
python run_overlaps.py --mice JawsM01 --stages Expert

# sliding estimator, ElasticNet, correct trials, 3 parallel jobs
python run_overlaps.py --mode sliding --l1-ratio 0.5 --correct --n-jobs 3

# weight-shuffle null
python run_overlaps.py --null-type weight --n-shuffles 200
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import os
import sys
import warnings

sys.path.insert(0, '/home/leon/dual/')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from time import perf_counter
from sklearn.model_selection import RepeatedStratifiedKFold

from src.common.options import set_options
from src.common.get_data import get_X_y_days
from src.pca.io import pkl_load, pkl_save
from src.overlaps.ccgd import ccgd_validation
from src.overlaps.data import dataloader
from src.overlaps.estimator import get_estimator

# ── CLI ───────────────────────────────────────────────────────────────────────

ALL_MICE = [
    'JawsM01', 'JawsM06', 'JawsM12', 'JawsM15', 'JawsM18',
    'ChRM04', 'ChRM23', 'ACCM03', 'ACCM04',
]

parser = argparse.ArgumentParser(
    description='Build pseudo-population and/or run CCGD overlaps.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# ── phase control ─────────────────────────────────────────────────────────────
phase = parser.add_argument_group('phase control')
phase.add_argument('--rebuild', action='store_true',
                   help='Rebuild X_all_nan from raw data before running CCGD')

# ── data I/O ──────────────────────────────────────────────────────────────────
io = parser.add_argument_group('data I/O')
io.add_argument('--data-in',  default='../data/pca',
                help='Directory for X_all_nan_.pkl (read and written by --rebuild)')
io.add_argument('--data-out', default='../data/overlaps',
                help='Output directory for CCGD results')

# ── pseudo-population build (phase 1) ─────────────────────────────────────────
build = parser.add_argument_group('pseudo-population build  (--rebuild only)')
build.add_argument('--scale', default='',
                   choices=['', 'std'],
                   help="Extra scaling applied per neuron/day after BL correction. "
                        "'' = none (notebook default), 'std' = divide by clipped std. "
                        "Also used as the filename tag for X_all_nan_<scale>.pkl")
build.add_argument('--data-type', default='dF', dest='data_type',
                   help='Fluorescence data type passed to get_X_y_days')
build.add_argument('--scaler-bl', default='standard_BL', dest='scaler_bl',
                   choices=['standard_BL', 'center_BL', 'robust_BL'],
                   help='Baseline scaler applied inside get_X_y_days')
build.add_argument('--days', nargs='+', default=['first', 'last'],
                   metavar='DAY',
                   help="Days to include, e.g. first last  or  all")
build.add_argument('--n-neurons', type=int, default=None, dest='n_neurons',
                   help='Total neuron slots for the pseudo-population. '
                        'Defaults to the sum of neurons across all selected mice '
                        '(computed from data on first pass).')
build.add_argument('--reload', type=int, default=0, choices=[0, 1],
                   help='Force raw-data reload inside get_X_y_days (0=use cache)')

# ── subset selection ──────────────────────────────────────────────────────────
subset = parser.add_argument_group('subset selection')
subset.add_argument('--mice',    nargs='+', default=ALL_MICE,
                    metavar='MOUSE',
                    help='Mice to process')
subset.add_argument('--stages',  nargs='+', default=['Naive', 'Expert'],
                    metavar='STAGE',
                    help='Learning stages')
subset.add_argument('--targets', nargs='+', default=['sample', 'choice', 'test'],
                    metavar='TARGET',
                    help='Decode targets (sample choice test distractor licks)')
subset.add_argument('--contexts', nargs='+', default=['all'],
                    metavar='CTX',
                    help='Task contexts (all DPA DualGo DualNoGo)')

# ── decoder ───────────────────────────────────────────────────────────────────
dec = parser.add_argument_group('decoder')
dec.add_argument('--mode',    default='generalizing',
                 choices=['sliding', 'generalizing'],
                 help='MNE estimator mode')
dec.add_argument('--scaler',  default='standard',
                 choices=['standard', 'center', 'none'],
                 help='Per-fold feature scaler (none = no scaling)')
dec.add_argument('--l1-ratio', type=float, default=0.0, dest='l1_ratio',
                 help='ElasticNet L1 ratio (0=ridge, 1=lasso)')
dec.add_argument('--n-splits',  type=int, default=5,  dest='n_splits',
                 help='Number of CV folds')
dec.add_argument('--n-repeats', type=int, default=1,  dest='n_repeats',
                 help='Number of CV repeats')
dec.add_argument('--n-jobs',    type=int, default=-1, dest='n_jobs',
                 help='Parallel jobs for the estimator (-1 = all cores)')

# ── decoder flags ─────────────────────────────────────────────────────────────
flags = parser.add_argument_group('decoder flags')
flags.add_argument('--no-raw', dest='raw', action='store_false',
                   help='Disable raw-space weight back-projection')
flags.add_argument('--correct', action='store_true',
                   help='Restrict to correct trials only')
flags.add_argument('--fit-param-epoch', action='store_true',
                   dest='fit_param_epoch',
                   help='Select C via epoch-averaged selector before fitting')

# ── null distribution ─────────────────────────────────────────────────────────
null = parser.add_argument_group('null distribution')
null.add_argument('--null-type', default=None,
                  choices=['weight', 'label'], dest='null_type',
                  help='Null distribution type (omit to skip)')
null.add_argument('--n-shuffles', type=int, default=100, dest='n_shuffles',
                  help='Shuffles (used only with --null-type)')

# ── misc ──────────────────────────────────────────────────────────────────────
misc = parser.add_argument_group('misc')
misc.add_argument('--tag', default='',
                  help='Extra string appended to the run-id (dum)')
misc.add_argument('--random-state', type=int, default=0, dest='random_state',
                  help='Global random seed')

parser.set_defaults(raw=True)
args = parser.parse_args()

# ── phase 1: rebuild pseudo-population ───────────────────────────────────────

def build_X_all(args, options_base, tag=None):
    """Load per-mouse raw data and assemble the shared pseudo-population array.

    Mirrors the 'Loading Raw Data' section of singleOverlaps.org.
    Saves X_all_nan_<tag>.pkl and y_all_nan_<tag>.pkl to args.data_in.
    """
    if tag is None:
        tag = args.scale
    print('\n── Building X_all ──────────────────────────────────────────────')
    opts = dict(options_base)   # shallow copy; set_options is called per-mouse

    # First pass: collect per-mouse neuron counts to set n_neurons if not given
    if args.n_neurons is None:
        n_neurons = 0
        for mouse in args.mice:
            opts['mouse'] = mouse
            opts['reload'] = args.reload
            o = set_options(**opts)
            X, _ = get_X_y_days(**o)
            n_neurons += X.shape[1]
        print(f'Auto n_neurons = {n_neurons}')
    else:
        n_neurons = args.n_neurons

    X_trials, y_trials = [], []
    counter = 0

    for mouse in args.mice:
        opts['mouse'] = mouse
        opts['reload'] = args.reload
        o = set_options(**opts)
        X, y = get_X_y_days(**o)
        print(f'{mouse}  X {X.shape}  y {y.shape}  n_days {o["n_days"]}')

        X_scale = X.copy()
        std_ = 1
        for day in range(1, o['n_days'] + 1):
            idx0 = (y.day == day) & (y.laser == 0)
            mean_ = 0.0

            if args.scale != '':
                mean_ = np.nanmean(X[idx0], axis=0, keepdims=True)
            if args.scale == 'std':
                std_ = np.nanstd(X[idx0], axis=(0, 2), ddof=0, keepdims=True)
                lo, hi = np.percentile(std_.ravel(), [5, 95])
                std_ = np.clip(std_, lo, hi)

            idx = (y.day == day)
            X_scale[idx] = (X[idx] - mean_) / std_

        X_trial = np.full((X.shape[0], n_neurons, X.shape[-1]), np.nan)
        X_trial[:, counter:counter + X.shape[1], :] = X_scale
        counter += X.shape[1]

        y['mouse'] = mouse
        X_trials.append(X_trial)
        y_trials.append(y)

    X_all = np.concatenate(X_trials, axis=0)
    y_all = pd.concat(y_trials, ignore_index=True)
    print(f'X_all {X_all.shape}  y_all {y_all.shape}')

    pkl_save(X_all, f'X_all_nan_{tag}', path=args.data_in)
    pkl_save(y_all, f'y_all_nan_{tag}', path=args.data_in)
    return X_all, y_all


# ── shared options (used by both phases) ──────────────────────────────────────

options_kwargs = {
    'mice': args.mice, 'tasks': ['Dual'],
    'mouse': args.mice[0], 'laser': 0,
    'trials': '', 'reload': args.reload if args.rebuild else 0,
    'data_type': args.data_type,
    'prescreen': None, 'pval': 0.05,
    'preprocess': None, 'scaler_BL': args.scaler_bl,
    'avg_noise': False, 'unit_var_BL': False,
    'random_state': None, 'T_WINDOW': 0.0,
    'l1_ratio': 0.95,
    'n_comp': 3, 'pca': 'pca', 'scaler': None,
    'bootstrap': 1, 'n_boots': 128,
    'n_splits': 5, 'n_repeats': 10,
    'class_weight': 0, 'multilabel': 0,
    'mne_estimator': 'generalizing', 'n_jobs': 64,
}
options_kwargs['days'] = args.days
options_base = set_options(**options_kwargs)

# Tag encodes scale + mouse subset so a partial --rebuild never overwrites the
# full-population file.
_mice_tag = ('' if sorted(args.mice) == sorted(ALL_MICE)
             else '_mice_' + '-'.join(sorted(args.mice)))
X_all_tag = args.scale + _mice_tag

if args.rebuild:
    X_all, y_all = build_X_all(args, options_base, tag=X_all_tag)
else:
    print(f'Loading X_all_nan_{X_all_tag} ...')
    X_all = pkl_load(f'X_all_nan_{X_all_tag}', path=args.data_in)
    y_all = pkl_load(f'y_all_nan_{X_all_tag}', path=args.data_in)

y_all['sample']     = y_all.sample_odor
y_all['distractor'] = y_all.dist_odor
y_all['test']       = y_all.test_odor
y_all['licks']      = (y_all.odr_choice + y_all.choice) > 0
print(f'X_all {X_all.shape}  y_all {y_all.shape}')

# ── phase 2: CCGD ────────────────────────────────────────────────────────────

dec_scaler = None if args.scaler == 'none' else args.scaler
options = set_options(**{**options_kwargs, 'days': args.days})

folds = RepeatedStratifiedKFold(
    n_splits=args.n_splits, n_repeats=args.n_repeats,
)
bins_epoch = options['bins_CHOICE']   # used only when --fit-param-epoch

estimator = get_estimator(
    clf='logcv', scoring='accuracy', mode=args.mode,
    scaler=dec_scaler, l1_ratio=args.l1_ratio, n_jobs=args.n_jobs,
)
selector = get_estimator(
    clf='logcv', scoring='accuracy', mode=None,
    scaler=dec_scaler, l1_ratio=args.l1_ratio, n_jobs=args.n_jobs,
)

# ── run-id string ─────────────────────────────────────────────────────────────

dum = 'log'
dum += '_' + args.mode
dum += '_overlaps'
dum += '_' + args.scaler
if args.scale:
    dum += f'_scale_{args.scale}'
if args.scaler_bl != 'standard_BL':
    dum += '_' + args.scaler_bl
if args.days != ['first', 'last']:
    dum += '_days_' + '-'.join(args.days)
if args.correct:
    dum += '_correct'
dum += f'_l1_ratio_{args.l1_ratio}'
if args.raw:
    dum += '_raw'
if args.fit_param_epoch:
    dum += '_fit_epoch'
if args.null_type:
    dum += f'_null_{args.null_type}'
if sorted(args.mice) != sorted(ALL_MICE):
    dum += '_mice_' + '-'.join(sorted(args.mice))
if sorted(args.stages) != sorted(['Naive', 'Expert']):
    dum += '_stages_' + '-'.join(sorted(args.stages))
if sorted(args.targets) != sorted(['sample', 'choice', 'test']):
    dum += '_targets_' + '-'.join(sorted(args.targets))
if args.contexts != ['all']:
    dum += '_ctx_' + '-'.join(args.contexts)
if args.n_splits != 5 or args.n_repeats != 1:
    dum += f'_cv_{args.n_splits}x{args.n_repeats}'
if args.random_state != 0:
    dum += f'_seed_{args.random_state}'
if args.tag:
    dum += '_' + args.tag
print('dum:', dum)
print(f'mice={args.mice}  stages={args.stages}  '
      f'targets={args.targets}  contexts={args.contexts}')

# ── main loop ─────────────────────────────────────────────────────────────────

t0 = perf_counter()
X_mice, y_mice = [], []

for mouse in args.mice:
    X_stage_list, y_stage_list = [], []

    for stage in args.stages:
        idx = (y_all.mouse == mouse) & (y_all.learning == stage)
        X_df = X_all[idx]
        y_df = y_all.loc[idx].reset_index(drop=True)

        valid = ~np.all(np.isnan(X_df), axis=(0, 2))
        X_df = X_df[:, valid, :]

        Z_ctx_list, y_ctx_list = [], []

        for context in args.contexts:
            Z_tar_list, y_tar_list = [], []

            for target in args.targets:
                print(f'mouse={mouse}  stage={stage}  target={target}  '
                      f'context={context}')

                X_y_data = dataloader(
                    X_df, y_df,
                    target=target, stage=stage, context=context,
                    correct=args.correct, strata=True,
                )

                probas, dfs, y_, null_info = ccgd_validation(
                    X_y_data,
                    estimator,
                    selector=selector,
                    cv=folds,
                    fit_param_epoch=args.fit_param_epoch,
                    bins_epoch=bins_epoch,
                    null_type=args.null_type,
                    null_reduction='zscore',
                    n_shuffles=args.n_shuffles,
                    raw=args.raw,
                    signed=False,
                    random_state=args.random_state,
                )

                print(f'  probas {probas.shape}  dfs {dfs.shape}')
                Z_ = np.stack([probas, dfs], axis=1)   # (n_trials, 2, T_train, T_test)

                y_ = y_.copy()
                y_['target']  = target
                y_['context'] = context

                Z_tar_list.append(Z_)
                y_tar_list.append(y_)

            Z_ctx_list.append(np.vstack(Z_tar_list))
            y_ctx_list.append(pd.concat(y_tar_list, ignore_index=True))

        Z_mouse = np.vstack(Z_ctx_list)
        y_mouse = pd.concat(y_ctx_list, ignore_index=True)
        y_mouse['stage'] = stage

        print(f'  stage done: Z {Z_mouse.shape}  y {y_mouse.shape}')

        X_stage_list.append(Z_mouse)
        y_stage_list.append(y_mouse)

    X_mice.append(np.vstack(X_stage_list))
    y_mice.append(pd.concat(y_stage_list, ignore_index=True))

h = int((perf_counter() - t0) // 3600)
m = int(((perf_counter() - t0) % 3600) // 60)
s = int((perf_counter() - t0) % 60)
print(f'CCGD loop done in {h}h {m}m {s}s')

# ── stack and save ────────────────────────────────────────────────────────────

X_single = np.vstack(X_mice)
y_single = pd.concat(y_mice, ignore_index=True)
print(f'X_single {X_single.shape}  y_single {y_single.shape}')

pkl_save(X_single, f'X_{dum}',      path=args.data_out)
pkl_save(y_single, f'labels_{dum}', path=args.data_out)
print('Results saved to', args.data_out)
