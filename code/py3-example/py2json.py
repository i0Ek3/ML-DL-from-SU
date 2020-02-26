#!/usr/bin/env python3

import json

d = dict(name='a', age=16, s='100')
print(json.dumps(d, ensure_ascii=True))


