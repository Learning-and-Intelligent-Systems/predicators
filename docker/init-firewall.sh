#!/bin/bash
# Network firewall for the robocode-sandbox container.
#
# Adapted from the official Claude Code devcontainer:
# https://github.com/anthropics/claude-code/blob/main/.devcontainer/init-firewall.sh
#
# Restricts outbound traffic to only the services needed by Claude:
#   - api.anthropic.com  (Claude API)
#   - GitHub IPs         (git operations, public docs)
#   - statsig / sentry   (Claude telemetry)
#
# Requires --cap-add=NET_ADMIN --cap-add=NET_RAW on docker run.
set -euo pipefail
IFS=$'\n\t'

# Extract Docker's internal DNS NAT rules BEFORE flushing anything.
DOCKER_DNS_RULES=$(iptables-save -t nat | grep "127.0.0.11" || true)

# Flush all existing rules and ipsets.
iptables -F
iptables -X
iptables -t nat -F
iptables -t nat -X
iptables -t mangle -F
iptables -t mangle -X
ipset destroy allowed-domains 2>/dev/null || true

# Restore Docker's internal DNS NAT rules so container DNS still resolves.
if [ -n "$DOCKER_DNS_RULES" ]; then
    iptables -t nat -N DOCKER_OUTPUT 2>/dev/null || true
    iptables -t nat -N DOCKER_POSTROUTING 2>/dev/null || true
    echo "$DOCKER_DNS_RULES" | xargs -L 1 iptables -t nat
fi

# Allow outbound DNS (UDP 53) and inbound DNS responses.
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
iptables -A INPUT  -p udp --sport 53 -j ACCEPT

# Allow outbound SSH and established inbound SSH.
iptables -A OUTPUT -p tcp --dport 22 -j ACCEPT
iptables -A INPUT  -p tcp --sport 22 -m state --state ESTABLISHED -j ACCEPT

# Allow loopback.
iptables -A INPUT  -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# Create the allowed-domains ipset (CIDR support).
ipset create allowed-domains hash:net

# Add GitHub IP ranges (web + api + git).
gh_ranges=$(curl -s https://api.github.com/meta)
echo "$gh_ranges" | jq -r '(.web + .api + .git)[]' | aggregate -q | while read -r cidr; do
    if echo "$cidr" | grep -qE '^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}/[0-9]{1,2}$'; then
        ipset add allowed-domains "$cidr"
    fi
done

# Resolve and add specific domains required by Claude.
for domain in \
    "api.anthropic.com" \
    "sentry.io" \
    "statsig.anthropic.com" \
    "statsig.com"; do
    ips=$(dig +noall +answer A "$domain" | awk '$4 == "A" {print $5}')
    if [ -z "$ips" ]; then
        echo "WARNING: could not resolve $domain" >&2
        continue
    fi
    while IFS= read -r ip; do
        if [ -n "$ip" ] && echo "$ip" | grep -qE '^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$'; then
            ipset add allowed-domains "$ip"
        fi
    done <<< "$ips"
done

# Allow host network (/24 of the default gateway).
HOST_IP=$(ip route | grep default | awk '{print $3}' | head -1)
HOST_NETWORK=$(echo "$HOST_IP" | sed 's/\.[0-9]*$/.0\/24/')
iptables -A INPUT  -s "$HOST_NETWORK" -j ACCEPT
iptables -A OUTPUT -d "$HOST_NETWORK" -j ACCEPT

# Set default-deny policies.
iptables -P INPUT   DROP
iptables -P FORWARD DROP
iptables -P OUTPUT  DROP

# Allow established / related connections.
iptables -A INPUT  -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow outbound to whitelisted IPs only.
iptables -A OUTPUT -m set --match-set allowed-domains dst -j ACCEPT

# Reject everything else immediately.
iptables -A OUTPUT -j REJECT --reject-with icmp-admin-prohibited

# Sanity check: example.com must be blocked.
if curl --connect-timeout 5 -s https://example.com > /dev/null 2>&1; then
    echo "ERROR: firewall misconfigured — example.com is reachable" >&2
    exit 1
fi

echo "Firewall initialized: outbound restricted to whitelisted domains."
