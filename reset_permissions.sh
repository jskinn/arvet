#!/bin/sh
# Reset the file permissions for all the relevant files
chmod +x reset_permissions.sh
chmod -x README.md .gitignore
find . -name "*.py" -exec chmod -x {} \;

