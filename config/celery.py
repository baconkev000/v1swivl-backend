"""
Celery app for Swivl backend.

Usage:
  celery -A config worker -l info
  celery -A config beat -l info   # if using periodic tasks
"""
from celery import Celery

app = Celery("config")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
