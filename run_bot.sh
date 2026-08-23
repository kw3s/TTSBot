#!/bin/bash
# Supervisor: keeps the bot alive while Termux itself is alive.
cd "$(dirname "$0")"
while true; do
    python3 bot.py >> bot.log 2>&1
    code=$?
    echo "$(date '+%F %T') bot exited (code $code), restarting in 5s" >> bot.log
    sleep 5
done
