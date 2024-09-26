#!/bin/bash

NAME=${NAME:-app}
WORKERS=${WORKERS:-1}
WORKER_CLASS=uvicorn.workers.UvicornWorker
LOG_LEVEL=${LOG_LEVEL:-info}
APP_FILE=${APP_FILE:-main}
PORT=${PORT:-5000}

exec gunicorn $APP_FILE:app \
  --name $NAME \
  --workers $WORKERS \
  --worker-class $WORKER_CLASS \
  --log-level=$LOG_LEVEL \
  --bind=0000:$PORT \
  --log-file=-
