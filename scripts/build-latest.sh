#!/bin/bash
git fetch
git checkout main
git pull origin main
make install
exit 0