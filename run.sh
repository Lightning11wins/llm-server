#!/usr/bin/env bash
# Run the server, confined by the llm-server AppArmor profile.
#
# Installs or reloads the profile first if it is missing, stale, or unloaded,
# asking before anything that needs sudo. The rules themselves are in
# apparmor/llm-server, with a comment on each explaining why it is there.
#
# Usage: ./run.sh [--unconfined]
set -uo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd -P) || exit 1
cd "$ROOT" || exit 1

PROFILE=llm-server
AA="AppArmor profile '$PROFILE'"
SRC=apparmor/$PROFILE
INSTALLED=/etc/apparmor.d/$PROFILE
PYTHON=venv/bin/python
AA_FEATURES=/sys/kernel/security/apparmor/features

# torch resolves a cache directory from tempfile.gettempdir() when server.py is
# imported, so the profile has to grant a writable temp dir. Keeping it in the
# project means the profile does not need access to the shared /tmp.
export TMPDIR="$ROOT/tmp"
mkdir -p "$TMPDIR" || exit 1

# The profile grants no write access to the code it runs, and that includes
# __pycache__: a process that can rewrite .pyc files on its import path can
# make itself come back after a restart. Compiling backends/ in memory on each
# start costs a few milliseconds.
export PYTHONDONTWRITEBYTECODE=1

die()  { printf 'run.sh: %s\n' "$*" >&2; exit 1; }
note() { printf 'run.sh: %s\n' "$*" >&2; }

# Ask before doing anything privileged. Declining, or having no terminal to ask
# from, is a failure rather than a silent fall back to running unconfined.
confirm() {
	local answer
	if ! { : < /dev/tty; } 2> /dev/null; then
		note "$* (no terminal to ask on)"
		return 1
	fi
	read -r -p "run.sh: $* [y/N] " answer < /dev/tty || return 1
	[[ $answer == [yY]* ]]
}

# A root-owned copy, not a symlink into this checkout. apparmor.service loads
# everything under /etc/apparmor.d at boot, and a symlink there would let
# anyone who can write to this directory author system-wide policy as root.
#
# Copied first, then loaded from the copy, so the kernel and /etc/apparmor.d
# hold the same bytes even if $SRC changes underneath. The copy is staged
# under a dotfile name, which the parser skips when apparmor.service loads the
# directory at boot, and moved into place only once the parser has accepted
# it: a rejected profile must not end up matching $SRC, which would hide that
# the kernel still has the previous version. The unprivileged -Q dry run
# catches most of that before sudo is even asked for.
install_profile() {
	local staged
	staged=$(dirname "$INSTALLED")/.$PROFILE.new
	apparmor_parser -Q --skip-cache "$SRC" || return 1
	sudo install -m 0644 -o root -g root "$SRC" "$staged" || return 1
	if sudo apparmor_parser -r --skip-cache "$staged"; then
		sudo mv "$staged" "$INSTALLED"
	else
		sudo rm -f "$staged"
		return 1
	fi
}

# The label the kernel gives a process entered into the profile, with its mode:
# "llm-server (enforce)". Empty when the profile is not loaded. Unlike
# /sys/kernel/security/apparmor/profiles, this does not need root to read.
loaded_label() {
	aa-exec -p "$PROFILE" -- cat /proc/self/attr/current 2> /dev/null
}

# Create the venv and install requirements if they are not there yet. torch is
# deliberately left out: which wheel is right depends on the local CUDA
# version, and guessing installs a 2.5 GB package that may not work.
ensure_venv() {
	if [ ! -x "$PYTHON" ]; then
		confirm "no venv in $ROOT. Create one and install requirements.txt (downloads packages)?" \
			|| die "declined; create the venv per README.md Setup"
		python3 -m venv venv || die "could not create the venv"
		"$PYTHON" -m pip install -q -r requirements.txt || die "could not install requirements.txt"
		note "venv ready"
	fi
	"$PYTHON" -c "import torch" 2> /dev/null || die "torch is not installed. Pick the line matching your CUDA version (check with nvidia-smi):
    $PYTHON -m pip install torch                                                     # CPU only
    $PYTHON -m pip install torch --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
    $PYTHON -m pip install torch --index-url https://download.pytorch.org/whl/cu124  # CUDA 12.4"
}
ensure_venv

if [ "${1:-}" = "--unconfined" ]; then
	note "starting without AppArmor confinement"
	exec "$PYTHON" server.py
fi
[ $# -eq 0 ] || die "unknown argument '$1'; usage: ./run.sh [--unconfined]"

# --- AppArmor usable at all? ------------------------------------------------
command -v aa-exec > /dev/null || die "aa-exec not found. Install the apparmor package, or run ./run.sh --unconfined"
[ "$(aa-enabled 2> /dev/null)" = "Yes" ] || die "AppArmor is not enabled on this kernel (aa-enabled: $(aa-enabled 2>&1)). Run ./run.sh --unconfined to start without it"
[ -e "$SRC" ] || die "$SRC is missing from this checkout"

# The profile's ip= and peer= network rules need the kernel to mediate inet
# addresses. Without that the parser loads them as plain `network inet
# stream`, and the profile no longer keeps the server on loopback.
if [ ! -e "$AA_FEATURES/network_v9/af_inet" ]; then
	note "warning: this kernel cannot restrict sockets by address, so the $AA allows any TCP peer, not only loopback"
fi

# The profile hardcodes the checkout path. A mismatch makes every file rule
# silently deny instead of match, which looks like a pile of unrelated bugs.
profile_dir=$(sed -n 's/^@{LLM_DIR}[[:space:]]*=[[:space:]]*//p' "$SRC")
if [ "$profile_dir" != "$ROOT" ]; then
	confirm "$SRC sets @{LLM_DIR} to '$profile_dir' but this checkout is at '$ROOT'. Update it?" \
		|| die "declined; fix the @{LLM_DIR} line in $SRC"
	sed -i "s|^@{LLM_DIR}[[:space:]]*=.*|@{LLM_DIR} = $ROOT|" "$SRC" || die "could not edit $SRC"
fi

# --- profile installed and current -----------------------------------------
# The installed file is a copy, so a difference from $SRC means an edit that
# has not been installed and loaded yet.
reason=""
if [ ! -e "$INSTALLED" ]; then
	reason="the $AA is not installed"
elif [ -L "$INSTALLED" ]; then
	# Left by an earlier run.sh that symlinked instead of copying.
	reason="$INSTALLED is a symlink rather than a root-owned copy"
elif ! cmp -s "$SRC" "$INSTALLED"; then
	# Show what is about to be loaded as system policy. $SRC is writable by
	# this user, so an edit here is the one place a change can slip in.
	diff -u "$INSTALLED" "$SRC" >&2
	reason="$INSTALLED does not match $SRC (diff above)"
fi
if [ -n "$reason" ]; then
	confirm "$reason. Install it to $INSTALLED and load it (needs sudo)?" \
		|| die "declined; nothing to run under"
	install_profile || die "could not load the profile"
fi

# --- profile loaded into the kernel, in enforce mode -----------------------
case "$(loaded_label)" in
	"$PROFILE (enforce)")
		;;
	"")
		confirm "the $AA is installed but not loaded into the kernel. Load it (needs sudo)?" \
			|| die "declined; nothing to run under"
		install_profile || die "load failed"
		;;
	*)
		# Complain mode is what run_tests.sh leaves behind when it could not
		# restore enforce mode on its way out.
		confirm "the $AA is loaded in complain mode, so violations would be logged but not blocked. Reload it in enforce mode (needs sudo)?" \
			|| die "declined; nothing to run under"
		install_profile || die "reload failed"
		;;
esac

# The profile denying something Python needs at startup would show up here as
# an unexplained exit, so name the log to look in.
aa-exec -p "$PROFILE" -- "$PYTHON" -c pass 2> /dev/null \
	|| die "could not start python under the $AA; it is denying something needed at startup:
    sudo journalctl -k --since '1 min ago' | grep apparmor"

exec aa-exec -p "$PROFILE" -- "$PYTHON" server.py
