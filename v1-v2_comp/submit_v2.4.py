#!/usr/bin/env python3
"""Submit the MICs/PNI v2.4 comparison as a prepared SGE job array."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--prepare-only', action='store_true', help='Write manifests and commands without submitting.')
args = parser.parse_args()
repo = Path(__file__).resolve().parent.parent
run_dir = repo / '.sge' / 'v2.4' / datetime.now().strftime('%Y%m%d-%H%M%S')
run_dir.mkdir(parents=True)
(run_dir / 'logs').mkdir()
rows = []
input_count = 0
for dataset in ('MICs', 'PNI'):
    subjects = (repo / 'v1-v2_comp' / f'BIDS_{dataset}' / 'participants.txt').read_text().split()
    if not subjects or len(set(subjects)) != len(subjects):
        raise SystemExit(f'Empty or duplicate participant list for {dataset}')
    # Index only selected images, not the large micapipe derivative trees.
    source = Path(f'/data/mica3/BIDS_{dataset}/derivatives/micapipe_v0.2.0')
    staged = run_dir / 'inputs' / dataset
    staged.mkdir(parents=True)
    description = source / 'dataset_description.json'
    if description.exists():
        (staged / description.name).write_text(description.read_text())
    else:
        (staged / 'dataset_description.json').write_text(json.dumps({'Name': f'{dataset} nativepro inputs', 'BIDSVersion': '1.8.0'}) + '\n')
    sessions = ('01', '02') if dataset == 'MICs' else ('01', '02', '03')
    for subject in subjects:
        for session in sessions:
            anat = source / f'sub-{subject}' / f'ses-{session}' / 'anat'
            images = list(anat.glob('*space-nativepro_T1w.nii*'))
            if len(images) != 1:
                raise SystemExit(f'Expected exactly one input for {dataset}/{subject}/{session}: {images}')
            image = images[0]
            target_anat = staged / f'sub-{subject}' / f'ses-{session}' / 'anat'
            target_anat.mkdir(parents=True)
            (target_anat / image.name).symlink_to(image)
            sidecar = image.with_name(image.name.removesuffix('.gz').removesuffix('.nii') + '.json')
            if sidecar.exists():
                (target_anat / sidecar.name).symlink_to(sidecar)
            input_count += 1
    output = repo / 'v1-v2_comp' / f'BIDS_{dataset}' / 'hippunfold_v2.4'
    for subject in subjects:
        if (output / f'sub-{subject}').exists() or (output / f'sub-{subject}').is_symlink():
            raise SystemExit(f'Results already exist for {dataset}/{subject}; inspect before resubmission.')
    for start in range(0, len(subjects), 5):
        rows.append((len(rows) + 1, dataset, subjects[start:start + 5]))
(run_dir / 'batches.tsv').write_text(''.join(f'{task}\t{ds}\t{" ".join(subs)}\n' for task, ds, subs in rows))
# Snapshot the worker so later edits cannot alter queued tasks.
worker = run_dir / 'worker.sh'
worker.write_text((repo / 'v1-v2_comp' / 'sge_v2.4.sh').read_text())
qsub = '/opt/sge/bin/lx-amd64/qsub'
base = [qsub, '-terse', '-S', '/bin/bash', '-cwd', '-q', 'mica.q', '-j', 'y', '-o', str(run_dir / 'logs')]
prep = base + ['-N', 'hu24-prepare', '-pe', 'smp', '4', '-l', 'h_vmem=8G,h_rt=24:00:00', str(worker), str(run_dir), 'prepare']
array = base + ['-N', 'hu24-T1w', '-pe', 'smp', '32', '-l', 'h_vmem=4G,h_rt=72:00:00', '-t', f'1-{len(rows)}', '-tc', '30']
record = {'run_directory': str(run_dir), 'batch_size': 5, 'max_concurrent': 30, 'tasks': len(rows),
          'subjects': sum(len(s) for _, _, s in rows), 'input_images': input_count, 'prepare_command': prep, 'array_command': array}
record_path = run_dir / 'submission.json'
def save():
    record_path.write_text(json.dumps(record, indent=2) + '\n')
save()
print(json.dumps(record, indent=2), flush=True)
if not args.prepare_only:
    prep_id = subprocess.check_output(prep, text=True).strip().split('.')[0]
    record['prepare_job_id'] = prep_id
    save()
    command = array + ['-hold_jid', prep_id, str(worker), str(run_dir), 'batch']
    record['array_command'] = command
    save()
    record['array_job_id'] = subprocess.check_output(command, text=True).strip().split('.')[0]
    save()
    print(f"Submitted preparation {prep_id}; array {record['array_job_id']} ({len(rows)} tasks).", flush=True)
