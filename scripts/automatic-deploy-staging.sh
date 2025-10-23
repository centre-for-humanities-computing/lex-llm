#!/bin/bash

# Configuration
REPO_DIR="/apps/lex-llm-staging/"
DEPLOY_SCRIPT="./scripts/deploy-staging.sh"

# --- Script Logic ---
# --- Add Start Timestamp ---
echo "============================================================"
echo "STARTING DEPLOYMENT CHECK at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
# Go to the repository directory
cd "$REPO_DIR" || { echo "ERROR: Could not change directory to $REPO_DIR"; exit 1; }

# Fetch remote changes without merging
# This updates the remote-tracking branches (e.g., origin/main)
echo "Fetching remote changes..."
git fetch origin

# Get the SHA of the local main branch and the remote main branch head
LOCAL_SHA=$(git rev-parse main)
REMOTE_SHA=$(git rev-parse origin/main)

echo "Local main SHA: $LOCAL_SHA"
echo "Remote main SHA: $REMOTE_SHA"

# Compare SHAs
if [ "$LOCAL_SHA" != "$REMOTE_SHA" ]; then
    echo "--- NEW CHANGES DETECTED! Starting deployment... ---"
    
    # 1. Pull the changes
    echo "Pulling changes from main..."
    git pull origin main
    
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
    echo "--- Deployment FINISHED at $(date '+%Y-%m-%d %H:%M:%S') ---"
else
    echo "No new changes on main. Staging environment is up to date."
fi
echo "Check FINISHED at $(date '+%Y-%m-%d %H:%M:%S')"
echo "" # Add a blank line for separation
exit 0