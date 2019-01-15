#!/bin/sh
# Reset the file permissions for all the relevant files
chmod +x reset_permissions.sh
chmod -R o-w .
chmod -R g-w .
chmod -x arvet/simulation/tests/test-depth-left.npy arvet/simulation/tests/test-depth-right.npy
chmod -x README.rst .gitignore requirements.txt LICENSE environment.yml
find . -name "*.py" -exec chmod -x {} \;
chmod +x arvet/scheduler.py
chmod +x arvet/plot_results.py
chmod +x arvet/verify_bounding_boxes_manually.py
chmod +x arvet/verify_database.py
chmod +x arvet/invalidate_data.py
chmod +x arvet/batch_analysis/scripts/run_task.py
