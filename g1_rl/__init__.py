"""G1 training in Isaac Lab: multi-teacher distillation of pretrained policies into one body controller."""

import os


def chown_to_host(*paths):
    """Containers run as root; hand outputs back to the host user (HOST_UID/HOST_GID, set in docker-compose)."""
    uid, gid = os.environ.get("HOST_UID"), os.environ.get("HOST_GID")
    if uid is None:
        return
    for top in paths:
        if not os.path.exists(top):
            continue
        os.chown(top, int(uid), int(gid or uid))
        for root, dirs, files in os.walk(top):
            for name in dirs + files:
                os.chown(os.path.join(root, name), int(uid), int(gid or uid))
