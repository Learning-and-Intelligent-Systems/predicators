#!/bin/bash
# Container entrypoint: initialize the network firewall, then run the command.
#
# The firewall requires NET_ADMIN / NET_RAW capabilities:
#   docker run --cap-add=NET_ADMIN --cap-add=NET_RAW ...
#
# The `node` user has passwordless sudo for init-firewall.sh only
# (configured in the Dockerfile via /etc/sudoers.d/node-firewall).
set -e

sudo /usr/local/bin/init-firewall.sh

exec "$@"
