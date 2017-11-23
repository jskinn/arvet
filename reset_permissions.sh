#!/bin/sh
# Reset the file permissions for all the relevant files
chmod +x reset_permissions.sh
chmod -x README.md .gitignore requirements.txt LICENSE
find . -name "*.py" -exec chmod -x {} \;
chmod +x add_initial_entities.py
chmod +x scheduler.py
chmod +x run_task.py
chmod +x plot_results.py
chmod +x verify_bounding_boxes_manually.py
chmod +x verify_database.py
chmod +x invalidate_data.py
