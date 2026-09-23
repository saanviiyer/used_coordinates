#!/usr/bin/env python3
"""Run a command in its own session so it survives the parent's death.

macOS has no ``setsid``.  Re-parenting to init with ``nohup cmd &`` inside a
subshell gives ppid 1 but leaves the process in the launching terminal's
session, and a session teardown still takes it down.  This does the real
double fork with ``os.setsid`` between them, which is what makes the process
immune to the parent going away.

Usage:  python3 src/detach.py <logfile> <command> [args...]
"""
from __future__ import annotations

import os
import sys


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: detach.py <logfile> <command> [args...]")
    log, cmd = sys.argv[1], sys.argv[2:]
    if os.fork() > 0:
        os._exit(0)
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
