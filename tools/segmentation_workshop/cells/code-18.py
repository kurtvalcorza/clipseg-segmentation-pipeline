run_stage('report')
from IPython.display import FileLink, Markdown, display
display(Markdown((ROOT / 'outputs/summary.md').read_text()))
show_previews('figures/*.png')
bundle = os.path.relpath(ROOT / 'outputs/segmentation_results.zip')
print('Results bundle:', bundle)
print('In Colab: open the Files sidebar (folder icon), browse to that path, then choose Download.')
display(FileLink(bundle))
session = {'bootstrap_seconds': BOOTSTRAP_SECONDS, 'elapsed_seconds': time.perf_counter() - SESSION_START,
           'source_files': CARRIED_HASHES, 'qualification': 'Candidate until exact hosted run reviewed'}
(ROOT / 'session_summary.json').write_text(json.dumps(session, indent=2))
print(session['elapsed_seconds'], 'seconds since setup began')
