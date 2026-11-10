#!/usr/bin/env python
"""Minimal symbolic reader for the DXIL dumps the probe captures.

The probe writes each program the provider decoded as a DXBC/DXIL container and
`trace/disasm.ps1` turns it into LLVM IR text with `dxc -dumpbin`. The two-times
model's flow programs are straight-line bodies wrapped in count-controlled loops
(staging weights or iterating input slices), so a small SSA reader plus an
affine evaluator can recover every index they compute:

    evaluate()      one expression for one concrete (tig, gid) point
    affine()        split an index expression into const + coefficients over
                    (threadIdInGroup, groupId); every index in these bodies is
                    affine, which is what makes the numpy replay cheap
    linearize()     flatten a chain of dx.op.dot4AddPacked accumulations
    strip_requant() peel min/max(fptosi(fadd(fmul(sitofp(x), s), b)), lo, hi)

Only the opcodes these programs use are implemented; anything else raises, so a
program this reader cannot handle fails loudly instead of being guessed at.
"""
import re

# Instruction forms the reader understands beyond the dx.op.* calls.
_BINARY = {
    'add': lambda a, b: (a + b) & 0xFFFFFFFF,
    'sub': lambda a, b: (a - b) & 0xFFFFFFFF,
    'mul': lambda a, b: (a * b) & 0xFFFFFFFF,
    'shl': lambda a, b: (a << (b & 31)) & 0xFFFFFFFF,
    'lshr': lambda a, b: a >> (b & 31),
    'ashr': lambda a, b: a >> (b & 31),
    'and': lambda a, b: a & b,
    'or': lambda a, b: a | b,
    'xor': lambda a, b: a ^ b,
    'sdiv': lambda a, b: int(a / b) if b else 0,
    'srem': lambda a, b: a - int(a / b) * b if b else 0,
}
_CASTS = ('sitofp', 'fptosi', 'fptoui', 'bitcast', 'zext', 'sext', 'trunc')


def split_args(text):
    """Split a call's argument list on top-level commas."""
    out, depth, cur = [], 0, ''
    for ch in text:
        if ch in '({[':
            depth += 1
        elif ch in ')}]':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def value(token):
    """Last word of a typed operand: `i32 %12` -> `%12`."""
    token = token.strip()
    return token.split()[-1] if token else token


def signed(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


class IR:
    """Straight-line view of one disassembled program."""

    def __init__(self, path):
        self.defs = {}
        self.stores = []
        self.loads = []
        for raw in open(path, encoding='utf-8'):
            line = raw.split(';', 1)[0].strip()
            if not line:
                continue
            if line.startswith('call void @dx.op.'):
                if 'textureStore' in line:
                    self.stores.append(line)
                continue
            m = re.match(r'^%([\w.]+) = (.*)$', line)
            if m:
                self.defs[m.group(1)] = m.group(2)
                if 'dx.op.textureLoad' in m.group(2):
                    self.loads.append(m.group(1))

    @staticmethod
    def call(rhs):
        m = re.match(r'^call ([^@]+)@([\w.]+)\((.*)\)$', rhs)
        if not m:
            return None
        return m.group(2), split_args(m.group(3))

    def expr(self, token, depth=0):
        if depth > 500:
            raise RuntimeError('expression too deep')
        token = value(token)
        if token.startswith('%'):
            name = token[1:]
            if name not in self.defs:
                return ('unknown', name)
            return self.rhs(self.defs[name], depth + 1)
        if token in ('undef', 'null', 'true', 'false'):
            return ('c', 0)
        if token.startswith('0x') or re.match(r'^-?\d+$', token):
            return ('c', int(token, 0) & 0xFFFFFFFF)
        return ('unknown', token)

    def rhs(self, rhs, depth):
        if rhs.startswith('call '):
            parsed = self.call(rhs)
            if parsed is None:
                return ('unknown', rhs)
            name, args = parsed
            return self._call(name, args, depth)
        if rhs.startswith('extractvalue'):
            m = re.match(r'extractvalue \S+ %([\w.]+), (\d+)', rhs)
            base = self.expr('%' + m.group(1), depth)
            index = int(m.group(2))
            if base[0] == 'res-tex':
                return ('tex', base[1], index)
            if base[0] == 'res-buf':
                return ('w', base[1][1], index, base[1][2])
            if base[0] == 'cbuf':
                return ('cbuf', base[1], index)
            return ('unknown', 'extract')
        m = re.match(r'^getelementptr \[[\d x i]+\], [^,]+, i32 0, i32 (\S+)', rhs)
        if m:
            return ('gep', self.expr(m.group(1), depth))
        m = re.match(r'^load i32, i32 addrspace\(3\)\* ([^\s,]+)', rhs)
        if m:
            ptr = self.expr(m.group(1), depth)
            if ptr[0] == 'gep':
                index = ptr[1]
                return ('w', ('lshr', index, ('c', 2)), ('and', index, ('c', 3)), 'i32')
            return ('unknown', 'load')
        m = re.match(r'^load float, float addrspace\(3\)\* ([^\s,]+)', rhs)
        if m:
            ptr = self.expr(m.group(1), depth)
            if ptr[0] == 'gep':
                index = ptr[1]
                return ('wf', ('lshr', index, ('c', 2)), ('and', index, ('c', 3)))
            return ('unknown', 'load')
        m = re.match(r'^(add|sub|mul|shl|lshr|ashr|and|or|xor|sdiv|srem)\b\s*'
                     r'(?:nsw\s*|nuw\s*)?(?:i\d+\s+)?(\S+), (\S+)', rhs)
        if m:
            return (m.group(1), self.expr(m.group(2), depth),
                    self.expr(m.group(3), depth))
        m = re.match(r'^(sitofp|fptosi|fptoui|bitcast|zext|sext|trunc)\b\s*\S+\s+(\S+)', rhs)
        if m:
            return (m.group(1), self.expr(m.group(2), depth))
        m = re.match(r'^(fmul|fadd|fsub|fdiv)\b\s*(?:fast\s*)?(?:float\s+)?'
                     r'(\S+), (\S+)', rhs)
        if m:
            return (m.group(1), self.expr(m.group(2), depth),
                    self.expr(m.group(3), depth))
        return ('unknown', rhs)

    def _call(self, name, args, depth):
        if name == 'dx.op.annotateHandle':
            return self.expr(args[1], depth)
        if name == 'dx.op.createHandleFromBinding':
            if 'zeroinitializer' in args[1]:
                return ('handle', 0, False)
            nums = re.findall(r'-?\d+', args[1])
            return ('handle', int(nums[0]), int(nums[3]) != 0)
        if name == 'dx.op.threadId.i32':
            return ('tid', int(args[1].split()[-1]))
        if name == 'dx.op.threadIdInGroup.i32':
            return ('tig', int(args[1].split()[-1]))
        if name == 'dx.op.groupId.i32':
            return ('gid', int(args[1].split()[-1]))
        if name in ('dx.op.textureLoad.i32', 'dx.op.textureLoad.f32'):
            return ('res-tex', [self.expr(args[1], depth), self.expr(args[3], depth),
                                self.expr(args[4], depth), self.expr(args[5], depth)])
        if name == 'dx.op.cbufferLoadLegacy.i32':
            # (opcode, handle, regIndex): only the register index is needed.
            return ('cbuf', int(args[2].split()[-1]))
        if name in ('dx.op.rawBufferLoad.i32', 'dx.op.rawBufferLoad.f32'):
            kind = 'f32' if name.endswith('f32') else 'i32'
            return ('res-buf', [self.expr(args[1], depth), self.expr(args[2], depth), kind])
        if name == 'dx.op.dot4AddPacked.i32':
            return ('dot', self.expr(args[1], depth), self.expr(args[2], depth),
                    self.expr(args[3], depth))
        if name == 'dx.op.binary.i32':
            code = int(args[0].split()[-1])
            a, b = self.expr(args[1], depth), self.expr(args[2], depth)
            if code == 37:
                return ('max', a, b)
            if code == 38:
                return ('min', a, b)
            return ('bin%d' % code, a, b)
        if name == 'dx.op.pack4x8.i32':
            return ('pack4x8', [self.expr(a, depth) for a in args[2:6]])
        return ('call', name, args)


def show(e, depth=0, limit=6):
    """Compact, human-readable rendering; used in diagnostics."""
    if not isinstance(e, tuple):
        return str(e)
    if depth > limit:
        return '...'
    kind = e[0]
    if kind == 'c':
        return str(e[1])
    if kind in ('tid', 'tig', 'gid'):
        return '%s.%d' % (kind, e[1])
    if kind == 'dot':
        return 'dot(%s,%s)' % (show(e[2], depth + 1, limit), show(e[3], depth + 1, limit))
    if kind == 'tex':
        h, x, y, z = e[1]
        return 'T%d[%s,%s,%s].w%d' % (h[1], show(x, depth + 1, limit),
                                      show(y, depth + 1, limit),
                                      show(z, depth + 1, limit), e[2])
    if kind in ('w', 'wf'):
        return '%s[%s].%s' % (kind.upper(), show(e[1], depth + 1, limit),
                              show(e[2], depth + 1, limit))
    if kind == 'gep':
        return 'S[%s]' % show(e[1], depth + 1, limit)
    if kind in _BINARY:
        return '(%s %s %s)' % (show(e[1], depth + 1, limit), kind,
                               show(e[2], depth + 1, limit))
    if kind in ('max', 'min'):
        return '%s(%s,%s)' % (kind, show(e[1], depth + 1, limit),
                              show(e[2], depth + 1, limit))
    if kind in ('sitofp', 'fptosi'):
        return '%s(%s)' % (kind, show(e[1], depth + 1, limit))
    if kind in ('fmul', 'fadd'):
        return '%s(%s,%s)' % (kind, show(e[1], depth + 1, limit),
                              show(e[2], depth + 1, limit))
    if kind == 'pack4x8':
        return 'pack(' + ','.join(show(x, depth + 1, limit) for x in e[1]) + ')'
    if kind == 'handle':
        return 'h%d%s' % (e[1], 'u' if e[2] else '')
    return '<%s>' % (kind,)


def evaluate(e, env):
    """Evaluate one expression at a concrete (tig, gid) point."""
    if not isinstance(e, tuple):
        return e
    kind = e[0]
    if kind == 'c':
        return e[1]
    if kind in ('tid', 'tig', 'gid'):
        return env[kind][e[1]]
    a = evaluate(e[1], env)
    if kind in _CASTS:
        return a
    b = evaluate(e[2], env)
    if kind in _BINARY:
        return _BINARY[kind](a, b)
    if kind == 'max':
        return max(a, b)
    if kind == 'min':
        return min(a, b)
    raise RuntimeError('cannot evaluate %s' % kind)


def affine(e, nthreads):
    """Split an expression into (constant, [coef per tig/gid axis])."""
    axes = [('tig', 0), ('tig', 1), ('tig', 2), ('gid', 0), ('gid', 1), ('gid', 2)]

    def env(tig=(0, 0, 0), gid=(0, 0, 0)):
        return {'tig': tig, 'gid': gid,
                'tid': tuple(tig[i] + gid[i] * nthreads[i] for i in range(3))}

    base = signed(evaluate(e, env()))
    coefs = []
    for name, i in axes:
        point = {'tig': [0, 0, 0], 'gid': [0, 0, 0]}
        point[name][i] = 1
        v = signed(evaluate(e, env(tuple(point['tig']), tuple(point['gid']))))
        coefs.append(v - base)
    return base, coefs


def linearize(e):
    """Flatten a dx.op.dot4AddPacked accumulation chain into (base, terms)."""
    terms = []
    while e[0] == 'dot':
        terms.append((e[2], e[3]))
        e = e[1]
    terms.reverse()
    return e, terms


def strip_requant(e):
    """Return (sum, scale, bias, lo, hi) from the requantise/clamp chain."""
    lo = hi = None
    if e[0] == 'min' and e[2][0] == 'c':
        hi = signed(e[2][1])
        e = e[1]
    if e[0] == 'max' and e[2][0] == 'c':
        lo = signed(e[2][1])
        e = e[1]
    if e[0] == 'fptosi':
        e = e[1]
    if e[0] == 'fadd':
        for scale, bias in ((e[1], e[2]), (e[2], e[1])):
            if scale[0] == 'fmul':
                for a, b in ((scale[1], scale[2]), (scale[2], scale[1])):
                    if a[0] == 'sitofp':
                        return a[1], b, bias, lo, hi
    return None, None, None, None, None


def unpack_word(e):
    """Split `texel.word + const` into (texel expression, constant)."""
    if e[0] == 'tex':
        return e, 0
    if e[0] == 'add' and e[1][0] == 'tex' and e[2][0] == 'c':
        return e[1], e[2][1]
    raise RuntimeError('unexpected input word ' + show(e, 0, 3))


def num_threads(path):
    for line in open(path, encoding='utf-8'):
        m = re.search(r'NumThreads=\((\d+),(\d+),(\d+)\)', line)
        if m:
            return tuple(int(x) for x in m.groups())
    raise RuntimeError('no NumThreads in ' + path)
