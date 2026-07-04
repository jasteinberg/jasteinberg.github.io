#!/usr/bin/env python3
"""Quick local preview: renders a markdown file (or _posts/ index) with
marked.js + MathJax in the browser. Not a Jekyll replacement -- just for
proofreading prose and math before pushing.
Usage: python3 preview.py [file.md]   (default: newest post)"""
import sys, http.server, socketserver, webbrowser
from pathlib import Path

root = Path(__file__).parent
if len(sys.argv) > 1:
    md = Path(sys.argv[1])
else:
    md = sorted((root/'_posts').glob('*.md'))[-1]

text = md.read_text()
if text.startswith('---'):           # strip front matter
    text = text.split('---', 2)[2]

# Match the live Jekyll/kramdown convention so this preview agrees with the
# built site: inline $$...$$ renders inline, only standalone $$ blocks are
# display. (Without this, marked.js treats every $$...$$ as display and breaks
# inline math.) Protect display blocks, then rewrite inline $$...$$ to \(...\).
import re
_disp = []
text = re.compile(r'^\$\$$.*?^\$\$$', re.M | re.S).sub(
    lambda m: _disp.append(m.group(0)) or f'\x00{len(_disp)-1}\x00', text)
text = re.sub(r'\$\$(.+?)\$\$', r'\\(\1\\)', text, flags=re.S)
text = re.sub(r'\x00(\d+)\x00', lambda m: _disp[int(m.group(1))], text)

PAGE = '''<!doctype html><meta charset="utf-8">
<title>preview</title>
<style>body{max-width:46rem;margin:3rem auto;font:17px/1.6 Georgia,serif;
padding:0 1rem}code,pre{font:14px Menlo,monospace;background:#f6f6f6}
pre{padding:.7rem;overflow-x:auto}table{border-collapse:collapse}
td,th{border:1px solid #ccc;padding:.3rem .6rem}</style>
<script>MathJax={tex:{inlineMath:[['$','$'],['\\\\(','\\\\)']],
displayMath:[['$$','$$'],['\\\\[','\\\\]']],processEscapes:true}};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<div id="c"></div>
<script>document.getElementById('c').innerHTML =
  marked.parse(%s, {gfm:true});
window.addEventListener('load',()=>MathJax.typesetPromise());</script>'''

import json
html = PAGE % json.dumps(text)
out = root/'_preview.html'
out.write_text(html)
webbrowser.open('file://' + str(out))
print('opened', out)
