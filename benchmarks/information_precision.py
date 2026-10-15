"""Independent original-information arithmetic for small relative Schur pivots.

No whitening coefficients from either library are used. Ordinary matrices keep
historical float operations; ill-conditioned matrices use 70 decimal digits on
the exact original binary64 entries. This helper expects SPD SE2 information.
"""
from decimal import Decimal, localcontext
from functools import lru_cache
import math


@lru_cache(maxsize=4096)
def precise_information(info):
    a, b, c, d, e, f = info
    l00 = math.sqrt(a)
    l10, l20 = b / l00, c / l00
    second = d - l10 * l10
    needs = second <= 0. or second / d < 2. ** -26
    if not needs:
        l11 = math.sqrt(second)
        l21 = (e - l20 * l10) / l11
        third = f - l20 * l20 - l21 * l21
        needs = third <= 0. or third / f < 2. ** -26
    return tuple(Decimal.from_float(v) for v in info) if needs else None


def energy_weighted_residual(r, info):
    precision = precise_information(tuple(info))
    if precision is None:
        a, b, c, d, e, f = info
        x, y, z = r
        w = (a*x+b*y+c*z, b*x+d*y+e*z, c*x+e*y+f*z)
        return .5 * sum(v*u for v, u in zip(r, w)), w
    with localcontext() as context:
        context.prec = 70
        a, b, c, d, e, f = precision
        x, y, z = map(Decimal.from_float, r)
        w = (a*x+b*y+c*z, b*x+d*y+e*z, c*x+e*y+f*z)
        return float((x*w[0]+y*w[1]+z*w[2])/2), tuple(float(v) for v in w)


def column_energy(column, info):
    return 2 * energy_weighted_residual(column, info)[0]
