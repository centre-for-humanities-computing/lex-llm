#!/bin/bash

# Configuration
REPO_DIR="/apps/lex-llm-staging/"
DEPLOY_SCRIPT="./scripts/deploy-staging.sh"
GIT_BRANCH="main"

# --- Script Logic ---

# Go to the repository directory
cd "$REPO_DIR" || { echo "ERROR: Could not change directory to $REPO_DIR"; exit 1; }

# Fetch remote changes without merging
# This updates the remote-tracking branches (e.g., origin/main)
echo "Fetching remote changes..."
git fetch origin

# Get the SHA of the local branch head and the remote branch head
LOCAL_SHA=$(git rev-parse HEAD)
REMOTE_SHA=$(git rev-parse origin/$GIT_BRANCH)

echo "Local SHA: $LOCAL_SHA"
echo "Remote SHA: $REMOTE_SHA"

# Compare SHAs
if [ "$LOCAL_SHA" != "$REMOTE_SHA" ]; then
    echo "--- NEW CHANGES DETECTED! Starting deployment... ---"
    
    # 1. Pull the changes
    echo "Pulling changes from $GIT_BRANCH..."
    git pull origin $GIT_BRANCH
    
    # Check if the pull was successful before installing
    if [ $? -ne 0 ]; then
        echo "ERROR: Git pull failed. Aborting deployment."
        exit 1
    fi

    # 2. Build and install the latest version
    echo "Building and installing the latest version..."
    make install

    # Check if the build/install was successful
    if [ $? -ne 0 ]; then
        echo "ERROR: Build/install failed. Aborting deployment."
        exit 1
    fi

    # 3. Run the deployment script
    echo "Deploying latest version to staging..."
    $DEPLOY_SCRIPT
    
    # Check the deploy script's exit status
    if [ $? -eq 0 ]; then
        echo "--- Deployment SUCCESSFUL! ---"
    else
        echo "--- Deployment FAILED! Check $DEPLOY_SCRIPT for errors. ---"
    fi
else
    echo "No new changes on $GIT_BRANCH. Staging environment is up to date."
fi

exit 0