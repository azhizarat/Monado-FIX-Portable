#!/usr/bin/env python3
"""Statistics per shader from shaderlab's stderr with RADV_DEBUG=shaders (ACO's final code), read as a stream.
   MESA_SHADER_CACHE_DISABLE=true RADV_DEBUG=shaders ./shaderlab cache/spirv.pack 2>&1 >/dev/null | python3 rank.py out.csv
Per shader: instructions, highest VGPR / SGPR used (+1), scratch accesses, the translator's state machine (sm: OpSwitch in a
loop) and its 100000-iteration loop guard in the final code. Then a summary and the worst shaders.
   python3 rank.py --selftest"""
import csv, re, sys

VREG = re.compile(r'\bv\[(\d+)(?:-(\d+))?\]')
SREG = re.compile(r'\bs\[(\d+)(?:-(\d+))?\]')


def finish(cur, code):
    vg = sg = scratch = 0
    for ln in code:
        for m in VREG.finditer(ln): vg = max(vg, int(m.group(2) or m.group(1)) + 1)
        for m in SREG.finditer(ln): sg = max(sg, int(m.group(2) or m.group(1)) + 1)
        if 'scratch_' in ln: scratch += 1
    cur.update(instr=len(code), vgpr=vg, sgpr=sg, scratch=scratch, guard=int(any('0x186a0' in ln for ln in code)))
    return cur


def parse(lines):
    """Yields one dict per shader. Only the last 'After lowering to hw instructions' block of a shader counts."""
    cur, code, grab = None, [], False
    for ln in lines:
        if ln.startswith('@@ '):
            if cur and code: yield finish(cur, code)
            p = ln.split()
            if p[1] == 'FAIL':
                cur = None
                continue
            cur = {'hash': p[1], 'stage': p[2], 'sm': int(p[3][3:]), 'words': int(p[4][6:])}
            code, grab = [], False
        elif ln.startswith('After lowering to hw instructions'):
            code, grab = [], True
        elif grab and cur is not None:
            s = ln.strip()
            if s.startswith(('BB', '/*', 'ACO shader stage')) or not s: continue
            code.append(s)
            if s == 's_endpgm': grab = False
    if cur and code: yield finish(cur, code)


def selftest():
    sample = ['@@ 00000000000000aa fs sm=1 words=100', 'After RA:', '\t v1: %1:v[40] = junk',
              'After lowering to hw instructions:', 'BB0', '\tv_add_f32 v[3], v[1], v[2]', '\tscratch_store_dword v[7-9]',
              '\ts_cmp_ge_u32 s[4-5], 0x186a0', '\ts_endpgm', '@@ 00000000000000bb vs sm=0 words=50',
              'After lowering to hw instructions:', '\tv_mov_b32 v[0], 1.0', '\ts_endpgm']
    r = list(parse(sample))
    assert len(r) == 2, r
    assert r[0]['vgpr'] == 10 and r[0]['sgpr'] == 6 and r[0]['scratch'] == 1 and r[0]['guard'] == 1 and r[0]['instr'] == 4, r[0]
    assert r[1]['vgpr'] == 1 and r[1]['sm'] == 0 and r[1]['guard'] == 0, r[1]
    print('ok')


def main():
    rows = []
    with open(sys.argv[1], 'w', newline='') as f:
        w = None
        for r in parse(sys.stdin):
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(r)); w.writeheader()
            w.writerow(r); rows.append(r)
            if len(rows) % 1000 == 0: print(f'{len(rows)} shaders', file=sys.stderr, flush=True)
    if not rows: return

    def avg(rs, k): return sum(r[k] for r in rs) / max(1, len(rs))
    for st in ('vs', 'fs', 'gs'):
        rs = [r for r in rows if r['stage'] == st]
        if not rs: continue
        g = [r for r in rs if r['guard']]; n = [r for r in rs if not r['guard']]
        print(f'{st}: {len(rs)} shaders, {len(g)} with the state-machine loop guard | '
              f'guard: {avg(g, "instr"):.0f} instr {avg(g, "vgpr"):.0f} vgpr {avg(g, "scratch"):.1f} scratch | '
              f'no guard: {avg(n, "instr"):.0f} instr {avg(n, "vgpr"):.0f} vgpr {avg(n, "scratch"):.1f} scratch')
    print(f'scratch users: {sum(1 for r in rows if r["scratch"])} of {len(rows)}')
    for key in ('scratch', 'vgpr', 'instr'):
        print(f'-- worst by {key}')
        for r in sorted(rows, key=lambda r: -r[key])[:10]:
            print(f'   {r["hash"]} {r["stage"]} instr {r["instr"]} vgpr {r["vgpr"]} sgpr {r["sgpr"]} scratch {r["scratch"]} guard {r["guard"]}')


if __name__ == '__main__':
    selftest() if sys.argv[1:] == ['--selftest'] else main()
