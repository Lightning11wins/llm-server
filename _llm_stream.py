#!/usr/bin/env python3
import sys
import json

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
