sudo systemctl daemon-reload
sudo systemctl enable lex-llm-staging.service
sudo systemctl restart lex-llm-staging.service
sudo systemctl status lex-llm-staging.service --no-pager