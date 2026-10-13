# @title Infrastructure: isolated environment, stage runner and display helpers
UV_URL = 'https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'
UV_SHA256 = 'aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60'
with urllib.request.urlopen(UV_URL, timeout=90) as response:
    wheel = response.read(20081405)
if len(wheel) != 20081404 or hashlib.sha256(wheel).hexdigest() != UV_SHA256:
    raise RuntimeError('uv wheel size/hash mismatch')
import io
with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
    member = next(n for n in archive.namelist() if n.endswith('.data/scripts/uv'))
    UV = ROOT / 'uv'
    UV.write_bytes(archive.read(member))
UV.chmod(0o700)
ENV = dict(os.environ, HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1')
ENV.pop('HF_TOKEN', None)
ENV.pop('HUGGING_FACE_HUB_TOKEN', None)
print('Creating the isolated Python 3.12.12 environment and installing the hash-locked packages;'
      ' this can take a few minutes on a fresh runtime.', flush=True)
subprocess.run([str(UV), 'venv', '--managed-python', '--python', '3.12.12', str(ROOT / 'env')], env=ENV, check=True)
PYTHON = ROOT / 'env/bin/python'
subprocess.run([str(UV), 'pip', 'install', '--python', str(PYTHON), '--require-hashes', '--only-binary', ':all:',
                '--index-url', 'https://pypi.org/simple', '-r', str(ROOT / 'requirements.txt')], env=ENV, check=True)
versions = subprocess.run([str(PYTHON), '-c', 'import sys, torch, transformers; print("Python", sys.version); '
                           'print("torch", torch.__version__, "built for CUDA", torch.version.cuda); '
                           'print("transformers", transformers.__version__); assert torch.cuda.is_available()'],
                          env=ENV, capture_output=True, text=True)
print(versions.stdout.strip())
if versions.returncode:
    print(versions.stderr)
    raise RuntimeError('The isolated environment failed its version/CUDA check. Read the output above.')
print('Dependencies installed and verified in the isolated environment.')
BOOTSTRAP_SECONDS = time.perf_counter() - SESSION_START

def run_stage(stage, *options, root=None):
    selected_root = ROOT if root is None else root
    log = selected_root / (stage + '.log')
    print('Running', stage, '— one model per process; log:', log, flush=True)
    with log.open('w', encoding='utf-8') as output:
        result = subprocess.Popen([str(PYTHON), '-u', str(ROOT / 'workshop.py'), '--root', str(selected_root), '--stage', stage, *options],
                                  env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in result.stdout:
            output.write(line)
            output.flush()
            if line.strip() and len(line) < 2000:
                print(line.rstrip(), flush=True)
        result.wait()
    if result.returncode == 65:
        # The worker refused the learner's input and printed the reason once; nothing to repeat or raise.
        return False
    if result.returncode:
        # Lines are streamed above; on failure repeat the end of the log, which holds the traceback.
        print(log.read_text(encoding='utf-8')[-7000:])
        raise RuntimeError(f'{stage} failed with exit {result.returncode}. Read the diagnostics above.')
    return True

def show_json(relative, root=None):
    value = json.loads(((ROOT if root is None else root) / relative).read_text())
    print(json.dumps(value, indent=2))
    return value

def show_training(root=None):
    history = json.loads(((ROOT if root is None else root) / 'outputs/training_history.json').read_text())
    changes = {u['epoch']: u['max_abs_decoder_change'] for u in history.get('observed_epoch_updates', [])}
    print('epoch | train loss | validation mean IoU | max decoder change')
    for entry in history['history']:
        loss = '-' if entry['train_loss'] is None else format(entry['train_loss'], '.4f')
        val = (entry.get('val') or {}).get('miou')
        val = '-' if val is None else format(val, '.4f')
        print(f"{entry['epoch']:>5} | {loss:>10} | {val:>18} | {changes.get(entry['epoch'], 0.0):.2e}")
    frozen = ' (the frozen decoder)' if history['best_epoch'] == 0 else ''
    print(f"Selected epoch: {history['best_epoch']}{frozen}; optimizer steps: {history.get('optimizer_steps')}; "
          f"trainable decoder parameters: {history.get('n_trainable')} of {history.get('n_total')}")
    return history

def show_reload(root=None):
    path = (ROOT if root is None else root) / 'outputs/reload_parity.json'
    parity = json.loads(path.read_text())
    records = parity['records']
    worst = max(r['max_abs_probability_difference'] for r in records)
    print(f"Fresh reload {'passed' if parity['passed'] else 'FAILED'}: {len(records)} records scored by a new process "
          f"(PID {parity.get('pid')}); maximum probability difference {worst:.2e} (tolerance {parity['atol']}); "
          f"identical masks on every record: {all(r['identical_masks'] for r in records)}; "
          f"metric parity: {parity.get('metric_parity')}. Per-record parity: {os.path.relpath(path)}")
    return parity

def show_previews(pattern):
    from IPython.display import Image, display
    for path in sorted((ROOT / 'outputs').glob(pattern))[:6]:
        print(path.name)
        display(Image(filename=str(path)))

def show_score_table(filenames, models=None, root=None):
    from IPython.display import Markdown, display
    groups = {}
    for filename in filenames:
        for row in json.loads(((ROOT if root is None else root) / 'outputs' / filename).read_text()):
            if row['role'] != 'test' or (models is not None and row['model'] not in models):
                continue
            key = (row['model'], row['protocol'])
            groups.setdefault(key, []).append(row)

    def mean(rows, key):
        values = [r[key] for r in rows if r[key] is not None]
        return format(sum(values) / len(values), '.4f') if values else 'n/a'

    table = ['| Model | Prompt protocol | Test images | Mean target IoU | Dice | Precision* | Recall | Predicted area |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for (model, protocol), rows in groups.items():
        table.append(f"| {model} | {protocol} | {len(rows)} | {mean(rows, 'iou')} | {mean(rows, 'dice')} | "
                     f"{mean(rows, 'precision')} | {mean(rows, 'recall')} | {mean(rows, 'predicted_area_fraction')} |")
    table += ['', '*Per-image means. Precision is averaged over images with any predicted foreground; '
              'an empty prediction has no precision (n/a). Predicted area is the fraction of the image predicted as target.']
    display(Markdown('\n'.join(table)))

def show_box_experiment(root=None):
    from IPython.display import Markdown, display
    entries = json.loads(((ROOT if root is None else root) / 'outputs/box_experiment.json').read_text())
    lines = ['| Expansion per side | Images | Improved | Unchanged | Worsened | Clipped by an image edge | Whole-image box |',
             '|---:|---:|---:|---:|---:|---:|---:|']
    per_image = {}
    for entry in entries:
        lines.append(f"| {entry['expansion']:.0%} | {len(entry['records'])} | {entry['improved']} | {entry['unchanged']} | "
                     f"{entry['worsened']} | {entry['clipped_records']} | {entry['whole_image_records']} |")
        for record in entry['records']:
            per_image.setdefault(record['id'], {'phrase': record['phrase'], 'tight': record['iou_tight_box']})[entry['expansion']] = record
    lines += ['', '| Expansion per side | Boxes clipped by an edge: improved / unchanged / worsened | '
              'Boxes not clipped: improved / unchanged / worsened |', '|---:|---|---|']
    for entry in entries:
        split = [entry['by_clipping'][key] for key in ('clipped', 'not_clipped')]
        lines.append(f"| {entry['expansion']:.0%} | " + ' | '.join(
            f"{s['improved']} / {s['unchanged']} / {s['worsened']} (of {s['n']})" for s in split) + ' |')
    lines += ['', '*The clipped / not-clipped split is descriptive: the groups also differ in ingredient, '
              'target size and surroundings, so it cannot establish that clipping caused a change.*']
    lines += ['', '| Image | Phrase | IoU, tight box | IoU, +10% | Clipped sides, +10% | IoU, +25% | Clipped sides, +25% |',
              '|---|---|---:|---:|---|---:|---|']
    for image_id, value in per_image.items():
        cells = []
        for fraction in (0.1, 0.25):
            record = value[fraction]
            sides = ', '.join(record['clipped_sides']) or 'none'
            if record['whole_image_box']:
                sides += ' (whole image)'
            cells += [format(record['iou_expanded_box'], '.3f'), sides]
        lines.append(f"| {image_id} | {value['phrase']} | {value['tight']:.3f} | " + ' | '.join(cells) + ' |')
    display(Markdown('\n'.join(lines)))
    return entries
