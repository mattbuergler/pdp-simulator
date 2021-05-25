#!/bin/bash

# this script shall be executed to set all the git configurations
# accordingly for this project

# Checking your commits conforms to coding guidelines
git config --local core.whitespace "space-before-tab, tab-in-indent, trailing-space, tabwidth=4"

# Install a Git pre-commit hook to automatically check for tab and whitespace errors before committing.
mv .git/hooks/pre-commit.sample .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Configure Git to use a rebase instead of a merge strategy when pulling a branch that has diverged
git config --global branch.autosetuprebase always
git config --global pull.rebase true

# Make Git remember merge conflict resolutions
git config --global rerere.enabled true