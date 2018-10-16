#!/bin/sh
# Reset the file permissions for all the relevant files
chmod +x reset_permissions.sh
chmod -R o-w .
chmod -R g-w .
chmod -x README.rst .gitignore requirements.txt LICENSE
find . -name "*.py" -exec chmod -x {} \;
chmod +x arvet/scheduler.py
chmod +x arvet/plot_results.py
chmod +x arvet/verify_bounding_boxes_manually.py
chmod +x arvet/verify_database.py
chmod +x arvet/invalidate_data.py
