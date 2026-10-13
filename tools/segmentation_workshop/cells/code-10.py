run_stage('clipseg')
show_training()
show_score_table(['baselines.json', 'clipseg_frozen.json', 'clipseg_selected_live.json'],
                 models={'empty', 'clipseg_frozen', 'clipseg_selected_live'})
show_previews('clipseg*.png')
