"""log_timings.py — keep a durable record of what each rebuild costs.

Rebuild costs are the thing you need BEFORE deciding whether to re-run something, and they were being
re-derived by hand every time (and, on 2026-09-09, guessed wrong by an order of magnitude: a variant
rebuild was estimated at "a couple of hours" when the logged truth was 33 minutes). Every measured run
now lands in `docs/rebuild_timings.md` as one row per step.

Three ways in:

  # parse a run log whose steps are marked `== <date> [<tag>] <step>` (run_axis_variant.sh does this)
  python log_timings.py parse .variant_tmp/variant_t1.log --run "axis variant _t1"

  # time a command and record it
  python log_timings.py time --run "ED figures" --step "make_ed_figures.py" -- \
      /home/leon/mambaforge/envs/dual/bin/python make_ed_figures.py

  # record something already measured elsewhere (say so in --source)
  python log_timings.py add --run "stale-cache refresh" --step "exp_dimensionality_ci.py" \
      --minutes 10 --source "approx, reported by the agent that ran it"

Rows are append-only. `--source` defaults to "measured"; anything not measured in place MUST say so, so a
later reader can tell a stopwatch from a recollection.
"""
import argparse
import datetime as dt
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'docs', 'rebuild_timings.md')
HEAD = """# Rebuild timings

Append-only record of what each rebuild actually costs, written by `log_timings.py` (see its docstring).
Consult it before estimating; do not guess. `source` says whether the number came from a stopwatch in
place or from somewhere less direct.

| finished | run | step | minutes | source |
|---|---|---|---|---|
"""
ROW = '| {when} | {run} | {step} | {minutes} | {source} |\n'


def append(rows):
    if not os.path.exists(OUT):
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        open(OUT, 'w', encoding='utf-8').write(HEAD)
    with open(OUT, 'a', encoding='utf-8') as f:
        for r in rows:
            f.write(ROW.format(**r))
    for r in rows:
        print(f'{r["minutes"]:>7} min  {r["run"]} / {r["step"]}')
    print(f'-> {OUT}')


def _clean(s):
    return s.replace('|', '/').strip()


def parse(a):
    """Steps are the gaps between consecutive `== <date> [<tag>] <step>` marks in a run log."""
    marks = []
    for ln in open(a.logfile, encoding='utf-8'):
        m = re.match(r'== (.+?) \[(\S+)\] (.+)', ln.strip())
        if not m:
            continue
        p = subprocess.run(['date', '-d', m.group(1), '+%s'], capture_output=True, text=True)
        if p.returncode == 0 and p.stdout.strip():
            marks.append((int(p.stdout.strip()), m.group(3)))
    if len(marks) < 2:
        sys.exit(f'{a.logfile}: found {len(marks)} step marks, need at least 2')
    rows = []
    for (t0, step), (t1, _) in zip(marks, marks[1:]):
        rows.append(dict(when=dt.datetime.fromtimestamp(t1).strftime('%Y-%m-%d %H:%M'),
                         run=_clean(a.run), step=_clean(step), minutes=f'{(t1 - t0) / 60:.1f}',
                         source=_clean(a.source)))
    rows.append(dict(when=dt.datetime.fromtimestamp(marks[-1][0]).strftime('%Y-%m-%d %H:%M'),
                     run=_clean(a.run), step='**TOTAL**',
                     minutes=f'{(marks[-1][0] - marks[0][0]) / 60:.1f}', source=_clean(a.source)))
    append(rows)


def timeit(a):
    if not a.cmd:
        sys.exit('nothing to run — put the command after --')
    t0 = time.time()
    rc = subprocess.run(a.cmd).returncode
    mins = (time.time() - t0) / 60
    src = a.source if rc == 0 else f'{a.source} (command FAILED, exit {rc})'
    append([dict(when=dt.datetime.now().strftime('%Y-%m-%d %H:%M'), run=_clean(a.run),
                 step=_clean(a.step), minutes=f'{mins:.1f}', source=_clean(src))])
    sys.exit(rc)


def add(a):
    append([dict(when=(a.when or dt.datetime.now().strftime('%Y-%m-%d %H:%M')), run=_clean(a.run),
                 step=_clean(a.step), minutes=f'{a.minutes:.1f}', source=_clean(a.source))])


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = ap.add_subparsers(dest='mode', required=True)
p = sub.add_parser('parse', help='split a run log at its `== <date> [tag] step` marks')
p.add_argument('logfile'); p.add_argument('--run', required=True)
p.add_argument('--source', default='measured, parsed from the run log'); p.set_defaults(fn=parse)
p = sub.add_parser('time', help='run a command and record how long it took')
p.add_argument('--run', required=True); p.add_argument('--step', required=True)
p.add_argument('--source', default='measured'); p.add_argument('cmd', nargs=argparse.REMAINDER)
p.set_defaults(fn=timeit)
p = sub.add_parser('add', help='record a duration measured elsewhere')
p.add_argument('--run', required=True); p.add_argument('--step', required=True)
p.add_argument('--minutes', type=float, required=True); p.add_argument('--when')
p.add_argument('--source', required=True); p.set_defaults(fn=add)
a = ap.parse_args()
if a.mode == 'time' and a.cmd and a.cmd[0] == '--':
    a.cmd = a.cmd[1:]
a.fn(a)
