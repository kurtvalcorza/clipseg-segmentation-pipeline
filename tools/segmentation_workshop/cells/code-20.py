RUN_BYOD = False  # @param {type:"boolean"}
BYOD_PATH = ''  # @param {type:"string"}
if RUN_BYOD and not BYOD_PATH.strip():
    print('Refused: BYOD_PATH is empty. Set it to the directory containing manifest.json, images and masks,'
          ' and run this cell again.')
elif RUN_BYOD:
    if run_stage('byod', '--byod', str(Path(BYOD_PATH).resolve())):
        byod_root = Path(json.loads((ROOT / 'byod_output.json').read_text())['path'])
        from IPython.display import FileLink, Markdown, display
        display(Markdown((byod_root / 'outputs/summary.md').read_text()))
        show_training(byod_root)
        show_score_table(['baselines.json', 'clipseg_frozen.json', 'clipseg_reloaded.json'],
                         models={'empty', 'clipseg_frozen', 'clipseg_reloaded'}, root=byod_root)
        show_score_table(['baselines.json', 'sam_metrics.json', 'sam2_metrics.json'],
                         models={'oracle_filled_box', 'sam', 'sam2'}, root=byod_root)
        bundle = os.path.relpath(byod_root / 'outputs/segmentation_results.zip')
        print('Your results bundle (metrics and manifests; no images):', bundle)
        print('In Colab: open the Files sidebar (folder icon), browse to that path, then choose Download.')
        display(FileLink(bundle))
else:
    print('Optional BYOD disabled; default workflow complete.')
