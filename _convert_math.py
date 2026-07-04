import re, pathlib, shutil

p = pathlib.Path('_posts/2026-06-10-dyck-circuits.md')
shutil.copy(p, pathlib.Path('..') / 'dyck-circuits.md.backup')
s = p.read_text()

assert s.startswith('---')
parts = s.split('---', 2)          # ['', frontmatter, body]
fm, body = parts[1], parts[2]

# 1. Stash fenced code blocks and inline code so we never touch $ inside them
stash = []
def _stash(m):
    stash.append(m.group(0))
    return f'\x00{len(stash)-1}\x00'
body = re.sub(r'```.*?```', _stash, body, flags=re.S)
body = re.sub(r'`[^`\n]*`', _stash, body)

# 2. Protect existing $$ -> \x01
body = body.replace('$$', '\x01')

# 3. Single $...$  -> \x01...\x01   (content cannot contain $ or \x01)
before_single = body.count('$')
body = re.sub(r'\$([^$\x01]*?)\$', lambda m: '\x01' + m.group(1) + '\x01', body,
              flags=re.S)
leftover = body.count('$')
print('single-$ before:', before_single, ' leftover single-$ (want 0):', leftover)

# 4. Restore \x01 -> $$
body = body.replace('\x01', '$$')

# 5. Restore code
body = re.sub(r'\x00(\d+)\x00', lambda m: stash[int(m.group(1))], body)

# 6. Ensure a blank line before each opening standalone $$ and after each
#    closing standalone $$ (so kramdown emits a display <div>, not inline).
lines = body.split('\n')
out, indisp = [], False
for i, ln in enumerate(lines):
    if ln.strip() == '$$':
        if not indisp:                       # opening delimiter
            if out and out[-1].strip() != '':
                out.append('')
            out.append(ln); indisp = True
        else:                                # closing delimiter
            out.append(ln); indisp = False
            nxt = lines[i+1] if i+1 < len(lines) else ''
            if nxt.strip() != '':
                out.append('')
    else:
        out.append(ln)
body = '\n'.join(out)

p.write_text('---' + fm + '---' + body)
print('display $$-delimiter lines:', sum(1 for l in body.split(chr(10)) if l.strip()=='$$'))
print('done')
