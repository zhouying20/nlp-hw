# config.py
import os
import gevent.monkey
gevent.monkey.patch_all()


# debug = True
loglevel = 'debug'
bind = "0.0.0.0:5000"
pidfile = "log/gunicorn.pid"
accesslog = "log/access.log"
errorlog = "log/error.log"
daemon = True

# 启动的进程数
workers = 1
worker_class = 'gevent'
x_forwarded_for_header = 'X-FORWARDED-FOR'