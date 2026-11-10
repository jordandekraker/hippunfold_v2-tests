#!/usr/bin/env python3
"""Self-contained v2.4/T1w noise experiment: --check, --submit, --status, --plot.
Reuses frozen degraded images and the exact clean images that generated them.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

SCRIPT_DIR = Path(os.environ.get('NOISE_EXPERIMENT_DIR', Path(__file__).resolve().parent))
REPO = SCRIPT_DIR.parent
DATA = Path('/host/bb-comp/tank/data/BIDS_PNI_degraded')
# Deliberately frozen original image provenance, not the new analysis baseline.
SOURCE = Path('/host/bb-comp/tank/data/BIDS_PNI/derivatives/hippunfold_v2.0.0')
RESULTS = DATA / 'derivatives/hippunfold_v2.4_T1w'
WORK = Path('/host/bb-comp/export02/tmp/jordand/hippunfold-noise-v2.4-T1w')
ASSETS = WORK / 'assets'
CODE = WORK / 'code'
SOURCE_ASSETS = REPO / '.cache/hippunfold-v2.4'
MODEL = 'T1w'
SUBJECTS = ['PNC003', 'PNC007', 'PNC011', 'PNC014', 'PNC019', 'PNC022', 'PNC026', 'PNC020', 'PNC024', 'PNC027']
PIXI = Path.home() / '.pixi/bin/pixi'
PIXI_CACHE = '/host/bb-compxg-01/export03/data/tmp/hippunfold-v2-tests-pixi/cache'
QUEUE = 'mica.q@bb-comp.bic-node.mni.mcgill.ca'
FIX_COMMIT = '2ba95432605d7ba697ae8025f4d5735edb376e31'  # upstream PR #604


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def inventory():
    conditions = sorted(p for p in DATA.iterdir() if p.is_dir() and p.name.startswith(('noise-L', 'blur-L', 'blurnoise-')))
    if len(conditions) != 8:
        raise RuntimeError(f'Expected eight frozen conditions, found {len(conditions)}')
    images = {'clean': {s: SOURCE / f'sub-{s}/ses-01/anat/sub-{s}_ses-01_desc-preproc_T1w.nii.gz' for s in SUBJECTS}}
    for condition in conditions:
        images[condition.name] = {s: condition / f'sub-{s}/anat/sub-{s}_T1w.nii.gz' for s in SUBJECTS}
    for condition, subjects in images.items():
        for subject, image in subjects.items():
            if not image.is_file():
                raise FileNotFoundError(f'{condition}/{subject}: {image}')
    return images


def stage_inputs(run, images):
    rows = []
    for condition, subjects in images.items():
        root = run / 'inputs' / condition
        root.mkdir(parents=True)
        write_json(root / 'dataset_description.json', {'Name': f'Frozen noise experiment: {condition}', 'BIDSVersion': '1.8.0'})
        for subject, source in subjects.items():
            target = root / f'sub-{subject}/anat/sub-{subject}_T1w.nii.gz'
            target.parent.mkdir(parents=True)
            target.symlink_to(source)
        for start in range(0, len(SUBJECTS), 5):
            rows.append({'task': len(rows) + 1, 'condition': condition, 'subjects': SUBJECTS[start:start + 5]})
    return rows


def worker_env():
    scratch = WORK / 'runtime' / f"{os.environ.get('JOB_ID', 'manual')}-{os.environ.get('SGE_TASK_ID', '0')}"
    scratch.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.pop('PYTHONHOME', None)
    env.update(PYTHONPATH=str(CODE), PYTHONNOUSERSITE='1', MPLBACKEND='Agg', CUDA_VISIBLE_DEVICES='',
               HIPPUNFOLD_CACHE_DIR=str(ASSETS), SNAKEMAKE_CONDA_PREFIX=str(ASSETS / 'conda'),
               CONDA_PKGS_DIRS=str(WORK / 'conda-pkgs'), XDG_CACHE_HOME=str(scratch / 'cache'),
               TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch), OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=os.environ.get('NSLOTS', '32'))
    return env


def hu(run, condition, output, level='participant', subjects=None, extra=()):
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-m', 'hippunfold.run', str(run / 'inputs' / condition), str(output), level,
           '--participant_label', *(subjects or SUBJECTS), '--modality', 'T1w', '--force_tissuepred_model', MODEL,
           '--cores', os.environ.get('NSLOTS', '32'), '--notemp', '--keep-going', '--rerun-incomplete', *extra]
    print('RUN:', ' '.join(cmd), flush=True)
    return subprocess.run(cmd, env=worker_env(), cwd=WORK).returncode


def prepare(run):
    WORK.mkdir(parents=True, exist_ok=True)
    package = Path(importlib.util.find_spec('hippunfold').origin).parent
    target = CODE / 'hippunfold'
    if target.exists():
        raise RuntimeError(f'Existing preparation code: {target}; inspect before replacing')
    shutil.copytree(package, target, ignore=shutil.ignore_patterns('__pycache__'))
    quality = target / 'workflow/lib/surface_quality.py'
    original = quality.read_text()
    fixed = original.replace('def _worse_faces(points, faces, normals, min_angle):', 'def _worse_faces(points, old_points, faces, normals, min_angle):')
    fixed = fixed.replace('    flipped = np.sum(_face_normals(points, faces) * normals, axis=1) <= 0',
                          '    moved = np.any(points != old_points, axis=1)[faces].any(axis=1)\n    flipped = np.sum(_face_normals(points, faces) * normals, axis=1) <= 0')
    fixed = fixed.replace('    return flipped | (face_geometry(points, faces)[1] < min_angle)',
                          '    return moved & (flipped | (face_geometry(points, faces)[1] < min_angle))')
    fixed = fixed.replace('_worse_faces(new, faces, fn, floor)', '_worse_faces(new, points, faces, fn, floor)')
    if fixed == original or fixed.count('_worse_faces(new, points, faces, fn, floor)') != 2:
        raise RuntimeError('Installed code does not match the expected v2.4 backport target')
    quality.write_text(fixed)
    write_json(run / 'code-provenance.json', {'package': 'hippunfold 2.4.0', 'model': MODEL,
               'backport': 'https://github.com/khanlab/hippunfold/pull/604', 'commit': FIX_COMMIT,
               'original_sha256': hashlib.sha256(original.encode()).hexdigest(),
               'patched_sha256': hashlib.sha256(fixed.encode()).hexdigest()})
    # Exact regression: the collinear face previously made relaxation spin forever.
    test = '''import sys, numpy as np
sys.path.insert(0, sys.argv[1])
from lib.surface_quality import relax_vertices
p=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[.4,.6,0],[.5,0,0]],dtype=float)
f=np.array([[0,5,4],[5,1,4],[1,2,4],[2,3,4],[3,0,4],[0,1,5]])
q=relax_vertices(p,f,lambda x:x)
assert np.isfinite(q).all() and np.allclose(q[:4],p[:4])
print('PR #604 degenerate-face regression passed')
'''
    subprocess.run([sys.executable, '-c', test, str(target / 'workflow')], check=True, timeout=30)
    # Reuse downloads; recreate inference environments to preserve valid prefixes.
    for name in ['model', 'template', 'atlases', 'atlases_dl']:
        dest = ASSETS / name
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(['rsync', '-a', '--stats', str(SOURCE_ASSETS / name) + '/', str(dest) + '/'], check=True)
    output = WORK / 'prepare-output'
    if hu(run, 'clean', output, 'download', [SUBJECTS[0]]) != 0:
        raise RuntimeError('Model/template preparation failed')
    if hu(run, 'clean', output, subjects=[SUBJECTS[0]], extra=['--conda-create-envs-only']) != 0:
        raise RuntimeError('Inference environment preparation failed')
    if hu(run, 'clean', output, subjects=[SUBJECTS[0]], extra=['--dry-run', '--quiet']) != 0:
        raise RuntimeError('Participant dry-run failed')
    (run / 'prepare.complete').touch()
    print('Noise preparation completed.', flush=True)


def batch(run):
    if not (run / 'prepare.complete').is_file():
        raise RuntimeError('Preparation failed; inspect its log')
    task = int(os.environ['SGE_TASK_ID'])
    spec = next(x for x in json.loads((run / 'manifest.json').read_text())['batches'] if x['task'] == task)
    root = RESULTS / spec['condition']
    output = root / '.batches' / f'task-{task}'
    code = hu(run, spec['condition'], output, subjects=spec['subjects'])
    for subject in spec['subjects']:
        source = output / f'sub-{subject}'
        dest = root / source.name
        relative = Path('.batches') / f'task-{task}' / source.name
        if dest.is_symlink():
            if dest.readlink() != relative:
                raise RuntimeError(f'Unexpected existing link: {dest}')
        elif source.is_dir():
            dest.symlink_to(relative, target_is_directory=True)
    # Severe degradation can legitimately fail. Preserve every return code and
    # count missing hemispheres in the analysis, never silently drop subjects.
    write_json(run / f'task-{task}.json', dict(spec, returncode=code, output=str(output)))
    print(f'Task {task} finished; HippUnfold return code {code}', flush=True)


def plot_results():
    import numpy as np
    import nibabel as nib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import re
    images = inventory()
    noise = sorted(t for t in images if t.startswith('noise-L'))
    blur = sorted(t for t in images if t.startswith('blur-L'))
    combos = {tuple(map(int, re.search(r'N(\d+)B(\d+)', t).groups())): t for t in images if t.startswith('blurnoise-')}
    def xyz(condition, subject, hemi):
        parts = []
        for label in ['hipp', 'dentate']:
            p = RESULTS / condition / f'sub-{subject}/surf/sub-{subject}_hemi-{hemi}_space-T1w_den-8k_label-{label}_midthickness.surf.gii'
            if not p.is_file():
                return None
            parts.append(nib.load(p).get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data.astype(float))
        v = np.vstack(parts).ravel()
        return v if np.isfinite(v).all() else None
    clean = {(s, h): xyz('clean', s, h) for s in SUBJECTS for h in ['L', 'R']}
    if any(v is None for v in clean.values()):
        raise RuntimeError('Clean controls are incomplete; cannot interpret noise failures')
    means = np.full((3, 3), np.nan)
    valid = np.zeros((3, 3), dtype=int)
    failed = np.zeros((3, 3), dtype=int)
    rows = []
    for i in range(3):
        for j in range(3):
            condition = 'clean' if (i, j) == (0, 0) else noise[i-1] if j == 0 else blur[j-1] if i == 0 else combos[(i, j)]
            correlations = []
            for subject in SUBJECTS:
                for hemi in ['L', 'R']:
                    a = clean[(subject, hemi)]
                    b = a if condition == 'clean' else xyz(condition, subject, hemi)
                    ok = b is not None and a.shape == b.shape and a.std() > 0 and b.std() > 0
                    r = float(np.corrcoef(a, b)[0, 1]) if ok else float('nan')
                    rows.append(dict(condition=condition, subject=subject, hemi=hemi, correlation=r, passed=bool(np.isfinite(r))))
                    if np.isfinite(r):
                        correlations.append(r)
            valid[i, j] = len(correlations)
            failed[i, j] = 2 * len(SUBJECTS) - len(correlations)
            if correlations:
                means[i, j] = np.mean(correlations)
    plots = SCRIPT_DIR / 'plots'
    plots.mkdir(exist_ok=True)
    with (plots / 'xyz_correlations_per_subject.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(means, cmap='viridis')
    im.cmap.set_bad((.2, .2, .2, 1))
    ax.tick_params(bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    for i in range(3):
        for j in range(3):
            value = f'{means[i,j]:.4f}' if np.isfinite(means[i,j]) else 'NA'
            ax.text(j, i, f'{value}\n(n={valid[i,j]}, fail={failed[i,j]})', ha='center', va='center', color='red', fontsize=8)
    ax.set_title('XYZ Coordinate Correlation (vs Clean)')
    fig.tight_layout()
    fig.savefig(plots / 'xyz_correlation_matrix.png', dpi=200)
    plt.close(fig)
    write_json(plots / 'noise_v2.4_summary.json', {'pipeline': '2.4.0 + PR604 loop fix', 'model': MODEL,
               'result_root': str(RESULTS), 'means': means.tolist(), 'valid': valid.tolist(), 'failed': failed.tolist()})
    subprocess.run([sys.executable, str(SCRIPT_DIR / 'noise_view.py')], cwd=SCRIPT_DIR, check=True,
                   env=dict(os.environ, MPLBACKEND='Agg'))
    print('Noise figures regenerated in', plots, flush=True)


def finalize(run):
    reports = []
    for spec in json.loads((run / 'manifest.json').read_text())['batches']:
        p = run / f"task-{spec['task']}.json"
        if not p.is_file():
            raise RuntimeError(f"Task {spec['task']} did not reach reporting; inspect its log")
        reports.append(json.loads(p.read_text()))
    write_json(run / 'task-summary.json', reports)
    plot_results()
    (run / 'analysis.complete').touch()


def submit():
    images = inventory()
    control = REPO / '.sge/noise_v2.4'
    if control.exists():
        for p in control.glob('*/manifest.json'):
            record = json.loads(p.read_text())
            if record.get('jobs') and not (p.parent / 'analysis.complete').exists():
                raise RuntimeError(f'Existing unfinished submission: {p}')
    run = control / datetime.now().strftime('%Y%m%d-%H%M%S')
    run.mkdir(parents=True)
    (run / 'logs').mkdir()
    batches = stage_inputs(run, images)
    driver = run / 'driver.py'
    driver.write_text(Path(__file__).read_text())
    worker = run / 'worker.sh'
    worker.write_text(f'''#!/usr/bin/env bash
set -euo pipefail
export NOISE_EXPERIMENT_DIR={SCRIPT_DIR}
export PIXI_CACHE_DIR={PIXI_CACHE}
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
exec {PIXI} run --manifest-path {REPO}/pixi.toml --locked python -u {driver} --run-dir {run} --worker "$1"
''')
    record = {'version': '2.4.0', 'model': MODEL, 'subjects': SUBJECTS, 'batches': batches, 'jobs': {}}
    path = run / 'manifest.json'
    write_json(path, record)
    base = ['/opt/sge/bin/lx-amd64/qsub', '-terse', '-S', '/bin/bash', '-q', QUEUE,
            '-cwd', '-j', 'y', '-o', str(run / 'logs')]
    def enqueue(name, slots, mode, options=()):
        cmd = base + ['-N', name, '-pe', 'smp', str(slots), '-l', 'h_vmem=4G,h_rt=72:00:00', *options, str(worker), mode]
        job = subprocess.check_output(cmd, text=True).strip().split('.')[0]
        record['jobs'][mode] = {'id': job, 'command': cmd}
        write_json(path, record)
        print(name, job, flush=True)
        return job
    prep = enqueue('noise24-prep', 4, 'prepare')
    array = enqueue('noise24-T1w', 32, 'batch', ['-hold_jid', prep, '-t', f'1-{len(batches)}', '-tc', '30'])
    enqueue('noise24-plots', 4, 'finalize', ['-hold_jid', array])
    print('Submission:', path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--check', action='store_true')
    group.add_argument('--submit', action='store_true')
    group.add_argument('--status', action='store_true')
    group.add_argument('--plot', action='store_true')
    group.add_argument('--worker', choices=['prepare', 'batch', 'finalize'])
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.run_dir is None:
            parser.error('--worker requires --run-dir')
        mpl_cache = WORK / 'runtime' / os.environ.get('JOB_ID', 'manual') / 'matplotlib'
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ.update(MPLBACKEND='Agg', MPLCONFIGDIR=str(mpl_cache))
        print('Host:', os.uname().nodename, 'mode:', args.worker, flush=True)
        {'prepare': prepare, 'batch': batch, 'finalize': finalize}[args.worker](args.run_dir)
    elif args.submit:
        submit()
    elif args.plot:
        plot_results()
    elif args.status:
        for p in sorted((REPO / '.sge/noise_v2.4').glob('*/manifest.json')):
            record = json.loads(p.read_text())
            print(p.parent, {k: v['id'] for k, v in record['jobs'].items()},
                  'reported batches:', len(list(p.parent.glob('task-[0-9]*.json'))),
                  'figures complete:', (p.parent / 'analysis.complete').exists())
    else:
        images = inventory()
        print(json.dumps({'model': MODEL, 'pipeline': '2.4.0 with upstream PR604 loop fix', 'conditions': list(images),
                          'subjects_per_condition': len(SUBJECTS), 'input_images': sum(map(len, images.values())),
                          'batches': 18, 'queue': QUEUE, 'result_root': str(RESULTS)}, indent=2))


if __name__ == '__main__':
    main()
