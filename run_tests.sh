#!/usr/bin/env bash
# Run the end-to-end tests: load, run, evict and unload every model in models/.
#
# When the AppArmor profile is loaded, the tests run against it in complain
# mode, so a rule that is too tight shows up as a report at the end instead of
# a confusing failure in the middle. Enforce mode is restored on the way out.
#
# Usage: ./run_tests.sh [model ...]      (default: every directory in models/)
set -uo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd -P) || exit 1
cd "$ROOT" || exit 1

PROFILE=llm-server
AA="AppArmor profile '$PROFILE'"
INSTALLED=/etc/apparmor.d/$PROFILE

note() { printf 'run_tests.sh: %s\n' "$*" >&2; }

confirm() {
	local answer
	if ! { : < /dev/tty; } 2> /dev/null; then
		note "$* (no terminal to ask on)"
		return 1
	fi
	read -r -p "run_tests.sh: $* [y/N] " answer < /dev/tty || return 1
	[[ $answer == [yY]* ]]
}

# Reloading with -C puts every profile in the file, the llama_server child
# included, into complain mode without touching the file. aa-complain would do
# the same but rewrites the profile file and is a separate package.
restore_enforce() {
	if sudo apparmor_parser -r "$INSTALLED" > /dev/null; then
		note "$AA back in enforce mode"
	else
		note "WARNING: could not put the $AA back in enforce mode. It is still in complain mode; run:"
		note "    sudo apparmor_parser -r $INSTALLED"
	fi
}

# --- put the profile in complain mode for the run --------------------------
complaining=0
if ! aa-exec -p "$PROFILE" -- true 2> /dev/null; then
	note "the $AA is not loaded, so the tests run unconfined (see ./run.sh)"
elif [ ! -e "$INSTALLED" ]; then
	note "the $AA is loaded but $INSTALLED is missing; running with it as-is, so a too-tight rule will look like a test failure"
elif ! cmp -s apparmor/$PROFILE "$INSTALLED"; then
	note "apparmor/$PROFILE differs from $INSTALLED; the tests would run against the installed copy. Run ./run.sh once to install and load it, then rerun"
	exit 1
elif confirm "run with the $AA in complain mode, so denied accesses are reported rather than blocked (needs sudo)?"; then
	if sudo apparmor_parser -r -C --skip-cache "$INSTALLED" > /dev/null; then
		complaining=1
		trap restore_enforce EXIT
	else
		note "warning: could not switch to complain mode; running as-is"
	fi
else
	note "running with the $AA as-is; a too-tight rule will look like a test failure"
fi

# --- run --------------------------------------------------------------------
started=$(date '+%Y-%m-%d %H:%M:%S')
scripts/e2e.sh "$@"
rc=$?

# --- report anything the profile would have blocked ------------------------
if [ $complaining -eq 1 ]; then
	echo
	echo "--- AppArmor accesses denied during the run:"
	if pgrep -x auditd > /dev/null; then
		echo "  (auditd is running, so denials went to /var/log/audit/audit.log rather than the kernel log read here)"
	fi
	# ALLOWED lines are what complain mode logs in place of a denial. Keep the
	# fields that say what was asked for: profile, operation, path or socket
	# family, and the permission mask.
	denied=$(sudo journalctl -k --since "$started" 2> /dev/null \
		| awk -v p="$PROFILE" '
			index($0, "apparmor=\"ALLOWED\"") && index($0, "profile=\"" p) {
				out = ""
				for (i = 1; i <= NF; i++)
					if ($i ~ /^(profile|operation|name|family|requested_mask)=/) out = out " " $i
				print out
			}' \
		| sort -u)
	if [ -n "$denied" ]; then
		echo "$denied"
		echo
		echo "Each line is an access the profile does not grant. Widen the rule in"
		echo "apparmor/llm-server, or walk them interactively with: sudo aa-logprof"
		rc=1
	else
		echo "  (none)"
	fi
fi

exit $rc
