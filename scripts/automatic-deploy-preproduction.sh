#!/bin/bash

# Set PATH for cron environment
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Configuration
REPO_DIR="/apps/lex-llm-preproduction/"
DEPLOY_SCRIPT="./scripts/deploy-preproduction.sh"
LOCK_FILE="/tmp/lex-llm-preproduction-deploy.lock"

# --- Script Logic ---
# Check for lock file to prevent concurrent runs
if [ -f "$LOCK_FILE" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Deployment already in progress (lock file exists). Exiting."
    exit 0
fi

# Create lock file
touch "$LOCK_FILE"

# Ensure lock file is removed on exit
trap "rm -f $LOCK_FILE" EXIT

# Go to the repository directory
cd "$REPO_DIR" || { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Could not change directory to $REPO_DIR"; exit 1; }

# Fetch remote changes without merging
if ! git fetch origin >/dev/null 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Git fetch failed. Check network/authentication."
    exit 1
fi

# Get latest tag name
LATEST_TAG=$(git describe --tags $(git rev-list --tags --max-count=1))
if [ -z "$LATEST_TAG" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: No tags found in repository. Cannot deploy to preproduction."
    exit 1
fi
# Get the SHA of the local main branch and the remote main branch head
LOCAL_SHA=$(git rev-parse main)
REMOTE_SHA=$(git rev-parse "$LATEST_TAG")

# Compare SHAs
if [ "$LOCAL_SHA" != "$REMOTE_SHA" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] STARTING DEPLOYMENT"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Latest version: $LATEST_TAG"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Local SHA: $LOCAL_SHA"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Remote SHA: $REMOTE_SHA"
    
    # 1. Get the latest changes from main
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Resetting to tag $LATEST_TAG..."
    git checkout main
    git reset --hard "$LATEST_TAG"
    
    # Check if the reset was successful before installing
    if [ $? -ne 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Git reset failed. Aborting deployment."
        exit 1
    fi

    # 2. Build and install the latest version
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Building and installing..."
    make install

    # Check if the build/install was successful
    if [ $? -ne 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Build/install failed. Aborting deployment."
        exit 1
    fi

    # 3. Run the deployment script
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Deploying to preproduction..."
    $DEPLOY_SCRIPT
    
    # Check the deploy script's exit status
    if [ $? -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Deployment SUCCESSFUL"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Deployment FAILED"
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Deployment FINISHED"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Preproduction up to date"
fi
exit 0