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
STAMP=$ROOT/.apparmor-loaded
PYTHON=venv/bin/python
KERNEL_PROFILES=/sys/kernel/security/apparmor/profiles

# torch resolves a cache directory from tempfile.gettempdir() when server.py is
# imported, so the profile has to grant a writable temp dir. Keeping it in the
# project means the profile does not need access to the shared /tmp.
export TMPDIR="$ROOT/tmp"
mkdir -p "$TMPDIR" || exit 1

die()  { printf 'run.sh: %s\n' "$*" >&2; exit 1; }
note() { printf 'run.sh: %s\n' "$*" >&2; }

# Ask before doing anything privileged. Declining, or having no terminal to ask
# from, is a failure rather than a silent fall back to running unconfined.
confirm() {
	local answer
	if [ ! -t 1 ]; then
		note "$* (no terminal to ask on)"
		return 1
	fi
	read -r -p "run.sh: $* [y/N] " answer < /dev/tty || return 1
	[[ $answer == [yY]* ]]
}

# Symlink rather than copy, so editing apparmor/llm-server updates the
# installed profile and only the reload is a separate step. The stamp records
# what was last pushed into the kernel; see the reload check below.
install_profile() {
	sudo ln -sfn "$ROOT/$SRC" "$INSTALLED" \
		&& sudo apparmor_parser -r "$INSTALLED" \
		&& profile_hash > "$STAMP"
}

profile_hash() { sha256sum "$SRC" | cut -d' ' -f1; }

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

# --- profile installed, current, and pushed into the kernel ----------------
reason=""
if [ ! -e "$INSTALLED" ]; then
	reason="the $AA is not installed"
elif ! cmp -s "$SRC" "$INSTALLED"; then
	# A stale copy from before the symlink install.
	reason="$INSTALLED does not match $SRC"
elif [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$(profile_hash)" ]; then
	# The install is a symlink, so an edit to $SRC changes both sides of the
	# cmp above and cannot be spotted that way. The only way to notice an edit
	# that was never pushed into the kernel is to remember what was last
	# loaded, which is what $STAMP is for.
	reason="$SRC has changed since it was last loaded into the kernel"
fi
if [ -n "$reason" ]; then
	confirm "$reason. Install it to $INSTALLED and load it (needs sudo)?" \
		|| die "declined; nothing to run under"
	install_profile || die "could not load the profile"
fi

# The profile hardcodes the checkout path. A mismatch makes every file rule
# silently deny instead of match, which looks like a pile of unrelated bugs.
profile_dir=$(sed -n 's/^@{LLM_DIR}[[:space:]]*=[[:space:]]*//p' "$SRC")
if [ "$profile_dir" != "$ROOT" ]; then
	confirm "$SRC sets @{LLM_DIR} to '$profile_dir' but this checkout is at '$ROOT'. Update it and reload (needs sudo)?" \
		|| die "declined; fix the @{LLM_DIR} line in $SRC"
	sed -i "s|^@{LLM_DIR}[[:space:]]*=.*|@{LLM_DIR} = $ROOT|" "$SRC" || die "could not edit $SRC"
	install_profile || die "reload failed"
fi

# --- profile loaded into the kernel ----------------------------------------
# Reading the kernel's profile list needs root, so when it is unreadable, fall
# back to entering the profile and seeing whether Python starts.
if loaded=$(cat "$KERNEL_PROFILES" 2> /dev/null); then
	mode=$(awk -v p="$PROFILE" '$1 == p { print $2; exit }' <<< "$loaded")
	if [ -z "$mode" ]; then
		confirm "the $AA is installed but not loaded into the kernel. Load it (needs sudo)?" \
			|| die "declined; nothing to run under"
		install_profile || die "load failed"
	elif [ "$mode" != "(enforce)" ]; then
		note "warning: the $AA is loaded in $mode mode, so violations are logged but not blocked"
	fi
elif ! aa-exec -p "$PROFILE" -- "$PYTHON" -c pass 2> /dev/null; then
	die "could not start python under the $AA. Either it is not loaded:
    sudo apparmor_parser -r '$INSTALLED'
or it is denying something needed at startup:
    sudo journalctl -k --since '1 min ago' | grep apparmor"
fi

exec aa-exec -p "$PROFILE" -- "$PYTHON" server.py
