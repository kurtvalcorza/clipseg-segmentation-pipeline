RUN_UNLABELLED = False  # @param {type:"boolean"}
IMAGE_PATH = ''  # @param {type:"string"}
TEXT_PROMPT = 'bread'  # @param {type:"string"}
GEOMETRIC_PROMPT_KIND = 'point'  # @param ['point', 'box']
GEOMETRIC_PROMPT = '[100, 100]'  # @param {type:"string"}
if RUN_UNLABELLED:
    if not IMAGE_PATH.strip():
        raise ValueError('Set IMAGE_PATH and choose coordinates inside the original image.')
    run_stage('infer', '--image', str(Path(IMAGE_PATH).resolve()), '--prompt', TEXT_PROMPT,
              '--' + GEOMETRIC_PROMPT_KIND, GEOMETRIC_PROMPT)
    inference_root = Path(json.loads((ROOT / 'unlabelled_output.json').read_text())['path'])
    print((inference_root / 'results.json').read_text())
    from IPython.display import Image as DisplayImage
    for mask_path in sorted(inference_root.glob('*.png')):
        print(mask_path.name)
        display(DisplayImage(filename=str(mask_path)))
else:
    print('Optional unlabelled inference disabled.')
