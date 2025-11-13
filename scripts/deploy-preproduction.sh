#!/bin/bash
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Reloading systemctl daemon"
sudo systemctl daemon-reload
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Enabling lex-llm-preproduction service"
sudo systemctl enable lex-llm-preproduction.service
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restarting lex-llm-preproduction service"
sudo systemctl restart lex-llm-preproduction.service
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking lex-llm-preproduction service status"
sudo systemctl status lex-llm-preproduction.service 
exit 0