## Correction eval

```
python policies/pi0_family/run_correction_eval.py     --corrections policies/pi0_family/prolific_deployment_1_feedback_clean.csv     --survey-tasks policies/pi0_family/prolific_1_survey_tasks.csv     --policy pi05 --headless
```

Can be used to validate whether the participant's corrections actually steer it towards success.

Only `main_task_*` rows are evaluated. Each is forked from the original rollout listed in `prolific_1_survey_tasks.csv` (`main_task_1` → `run_1.hdf5`, `main_task_2` → `run_7.hdf5`, `main_task_3` → `run_10.hdf5`, `main_task_4` → `run_14.hdf5`). Video timestamps are converted to policy steps at 15 Hz. Visual click annotations and "No correction needed" rows are skipped.

A spreadsheet `correction_eval_results.csv` is written next to `episode_results.jsonl`, with success/failure, correction text, timestamp, output video filename, and participant ID.