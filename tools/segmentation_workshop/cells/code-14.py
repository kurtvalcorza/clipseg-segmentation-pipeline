run_stage('sam')
run_stage('sam2')
show_score_table(['baselines.json', 'sam_metrics.json', 'sam2_metrics.json'], models={'oracle_filled_box', 'sam', 'sam2'})
show_previews('sam*.png')
