#!/usr/bin/env bash
set -euo pipefail

PORT=8080
BASE="http://localhost:$PORT"

usage() {
  echo "Usage:"
  echo "  $0 list [--loaded true|false|any]"
  echo "  $0 load --model <name> [--ttl <s>]"
  echo "  $0 run  --model <name> --prompt <text> [--autoload --ttl <s>]"
  echo "          [--max-tokens <n>] [--temperature <f>] [--top-p <f>] [--repetition-penalty <f>]"
  exit 1
}

[[ $# -eq 0 ]] && usage
CMD=$1; shift

case "$CMD" in
  list)
    LOADED="any"
    while [[ $# -gt 0 ]]; do
      case "$1" in --loaded) LOADED="$2"; shift 2 ;; *) usage ;; esac
    done
    curl -s "$BASE/list?loaded=$LOADED" | python3 -m json.tool
    ;;

  load)
    MODEL=""; TTL=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --ttl)   TTL="$2";   shift 2 ;;
        *) usage ;;
      esac
    done
    [[ -z "$MODEL" ]] && usage
    BODY=$(python3 -c "
import json, sys
d = {'model': sys.argv[1]}
if sys.argv[2]: d['ttl'] = float(sys.argv[2])
print(json.dumps(d))
" "$MODEL" "$TTL")
    curl -s -X POST "$BASE/load" -H "Content-Type: application/json" -d "$BODY"
    echo
    ;;

  run)
    MODEL=""; PROMPT=""; TTL=""; AUTOLOAD=""; MAX_TOKENS=""; TEMPERATURE=""; TOP_P=""; REP_PENALTY=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --model)              MODEL="$2";        shift 2 ;;
        --prompt)             PROMPT="$2";       shift 2 ;;
        --ttl)                TTL="$2";          shift 2 ;;
        --autoload)           AUTOLOAD="true";   shift   ;;
        --max-tokens)         MAX_TOKENS="$2";   shift 2 ;;
        --temperature)        TEMPERATURE="$2";  shift 2 ;;
        --top-p)              TOP_P="$2";        shift 2 ;;
        --repetition-penalty) REP_PENALTY="$2";  shift 2 ;;
        *) usage ;;
      esac
    done
    [[ -z "$MODEL" || -z "$PROMPT" ]] && usage
    BODY=$(python3 -c "
import json, sys
model, prompt, ttl, autoload, max_tokens, temperature, top_p, rep = sys.argv[1:]
d = {'model': model, 'prompt': prompt}
if ttl:         d['ttl']                = float(ttl)
if autoload:    d['autoload']           = True
if max_tokens:  d['max_tokens']         = int(max_tokens)
if temperature: d['temperature']        = float(temperature)
if top_p:       d['top_p']              = float(top_p)
if rep:         d['repetition_penalty'] = float(rep)
print(json.dumps(d))
" "$MODEL" "$PROMPT" "$TTL" "$AUTOLOAD" "$MAX_TOKENS" "$TEMPERATURE" "$TOP_P" "$REP_PENALTY")
    curl -sN -X POST "$BASE/run" \
      -H "Content-Type: application/json" \
      -H "Accept: text/event-stream" \
      -d "$BODY" | python3 -c "
import sys, json
buf = []
for line in sys.stdin:
    line = line.rstrip('\n')
    if line.startswith('data: '):
        buf = None
        data = line[6:]
        if data == '[DONE]':
            print()
            break
        try:
            d = json.loads(data)
            if 'token' in d:
                print(d['token'], end='', flush=True)
            elif 'error' in d:
                print('\nError: ' + d['error'], file=sys.stderr)
        except json.JSONDecodeError:
            pass
    elif buf is not None and line.strip():
        buf.append(line)
if buf:
    raw = ' '.join(buf)
    try:
        print('Error: ' + json.loads(raw).get('error', raw), file=sys.stderr)
    except json.JSONDecodeError:
        print('Error: ' + raw, file=sys.stderr)
    sys.exit(1)
"
    ;;

  *)
    usage
    ;;
esac
