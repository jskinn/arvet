#!/bin/sh
# Reset the file permissions for all the relevant files
chmod +x reset_permissions.sh
chmod -x README.md .gitignore
find . -name "*.py" -exec chmod -x {} \;
chmod +x add_initial_entities.py
chmod +x scheduler.py
chmod +x task_train_system.py
chmod +x task_run_system.py
chmod +x task_benchmark_result.py
chmod +x task_compare_trials.py
chmod +x task_compare_benchmark_results.py
chmod +x generate_simulated_dataset.py
chmod +x recalculate_bounding_boxes.py
chmod +x verify_bounding_boxes_manually.py
