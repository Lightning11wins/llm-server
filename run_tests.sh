#!/usr/bin/env bash
# Run the end-to-end tests: load, run, evict and unload every model in models/.
#
# When the AppArmor profile is installed, the tests run against it in complain
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
	if [ ! -t 1 ]; then
		note "$* (no terminal to ask on)"
		return 1
	fi
	read -r -p "run_tests.sh: $* [y/N] " answer < /dev/tty || return 1
	[[ $answer == [yY]* ]]
}

# --- put the profile in complain mode for the run --------------------------
complaining=0
if [ -e "$INSTALLED" ] && command -v aa-complain > /dev/null; then
	if confirm "run with the $AA in complain mode, so denied accesses are reported rather than blocked (needs sudo)?"; then
		if sudo aa-complain "$INSTALLED" > /dev/null; then
			complaining=1
			trap 'sudo aa-enforce "$INSTALLED" > /dev/null && note "$AA back in enforce mode"' EXIT
		else
			note "warning: could not switch to complain mode; running as-is"
		fi
	else
		note "running with the $AA as-is; a too-tight rule will look like a test failure"
	fi
else
	note "the $AA is not installed, so the tests run unconfined (see ./run.sh)"
fi

# --- run --------------------------------------------------------------------
started=$(date '+%Y-%m-%d %H:%M:%S')
scripts/e2e.sh "$@"
rc=$?

# --- report anything the profile would have blocked ------------------------
if [ $complaining -eq 1 ]; then
	echo
	echo "--- AppArmor accesses denied during the run:"
	# ALLOWED lines are what complain mode logs in place of a denial.
	denied=$(sudo journalctl -k --since "$started" 2> /dev/null \
		| grep -oE 'apparmor="ALLOWED"[^]]*' \
		| grep -E "profile=\"$PROFILE" \
		| sed -E 's/.*operation="([^"]*)".*(name|profile)="([^"]*)".*/  \1 \3/' \
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
