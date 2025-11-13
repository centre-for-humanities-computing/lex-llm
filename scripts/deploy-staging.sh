#!/bin/bash
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Reloading systemctl daemon"
sudo systemctl daemon-reload
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Enabling lex-llm-staging service"
sudo systemctl enable lex-llm-staging.service
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restarting lex-llm-staging service"
sudo systemctl restart lex-llm-staging.service
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking lex-llm-staging service status"
sudo systemctl status lex-llm-staging.service 
exit 0