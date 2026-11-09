#!/usr/bin/env python3
"""GPU time per fragment shader from a LARECOMP_GPUPROF_AT + XCDE_GPUPROF_DRAWS=1 log.
   python3 gpuprof_sum.py build/last_game.log [top]"""
import re, sys
from collections import defaultdict

ROW = re.compile(r'^\[gpuprof\]\s+([\d.]+) ms\s+[\d.]+%\s+(.*)$')


def summarize(lines):
    per, cnt, other, total = defaultdict(float), defaultdict(int), 0.0, 0.0
    for line in lines:
        m = ROW.match(line)
        if not m:
            continue
        ms, label = float(m.group(1)), m.group(2)
        total += ms
        d = re.match(r'\s*draw fs ([0-9a-f]{16})', label)
        if d:
            per[d.group(1)] += ms
            cnt[d.group(1)] += 1
        else:
            other += ms  # pass starts (clears / loads / barriers), copies
    return per, cnt, other, total


def main():
    per, cnt, other, total = summarize(open(sys.argv[1], errors='replace'))
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    print(f'total {total:.2f} ms, outside draws {other:.2f} ms, {len(per)} fragment shaders')
    for h, ms in sorted(per.items(), key=lambda kv: -kv[1])[:top]:
        print(f'{ms:8.3f} ms {100 * ms / (total or 1):5.1f}%  fs {h}  {cnt[h]} draws')


if __name__ == '__main__':
    if len(sys.argv) == 1:  # self-check
        p, c, o, t = summarize(['[gpuprof]  1.500 ms  10.0%  draw fs 00000000000000aa',
                                '[gpuprof]  0.500 ms   5.0%  draw fs 00000000000000aa',
                                '[gpuprof]  2.000 ms  20.0%  pass: 1 color 1280x720 fmt 0x35, depth yes'])
        assert p['00000000000000aa'] == 2.0 and c['00000000000000aa'] == 2 and o == 2.0 and t == 4.0
        print('ok')
    else:
        main()
