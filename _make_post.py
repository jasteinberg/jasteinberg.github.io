#!/usr/bin/env python3
from pathlib import Path

src = Path('/Users/steinbergj/public-repos/interp-foundations/docs/dyck_findings.md')
dst = Path('/Users/steinbergj/public-repos/website/_posts/2026-06-10-dyck-circuits.md')

s = src.read_text()
title = '# Dyck-(k, m) from-scratch \u2014 empirical findings'
assert s.count(title) == 1
s = s.replace(title, '').lstrip()

front = '''---
layout: post
title: "What makes a transformer use both of its layers? Circuit formation in Dyck-(k, m)"
date: 2026-06-10
---

*A small-scale study of when 2-layer attention-only transformers learn
shortcut circuits versus cleanly factored two-stage circuits, using
bracket languages as the test bed. The arc: a hypothesis, a negative
result that turned out to be a flaw in the experiment rather than the
hypothesis, a refined test, and activation patching that identified a
causal head the attention patterns alone would have missed. Code and
executed notebooks: [interp-foundations](REPO_URL_INTERP).*

'''

tail = '''

---

*Code, notebooks, and per-head diagnostic implementations are in
[interp-foundations](REPO_URL_INTERP). Drafted with the assistance of
Claude (Anthropic).*
'''

assert s.count('Drafted with the assistance of Claude (Anthropic).') == 1
s = s.replace('Drafted with the assistance of Claude (Anthropic).', '')
dst.write_text(front + s.rstrip() + tail)
print('post written:', dst, len((front + s).splitlines()), 'lines')
