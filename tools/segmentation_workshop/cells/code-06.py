run_stage('prepare')
print('Frozen roles:', json.loads((ROOT / 'sample.json').read_text())['roles'])
sample_records = json.loads((ROOT / 'data/records.json').read_text())['records']
train_classes = {r['prompt'] for r in sample_records if r['role'] == 'train'}
print('role       | images | distinct classes | classes also in train')
for role_name in ('train', 'validation', 'test', 'activity'):
    role_classes = {r['prompt'] for r in sample_records if r['role'] == role_name}
    role_images = sum(r['role'] == role_name for r in sample_records)
    print(f'{role_name:<10} | {role_images:>6} | {len(role_classes):>16} | {len(role_classes & train_classes):>21}')
show_previews('reference*.png')
