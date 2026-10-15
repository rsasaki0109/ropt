"""Independent high-precision references for the Cauchy reduction regression."""
from decimal import Decimal, localcontext
import json
import math

cases = [
    (1e155, 5e154, 1., math.log(2.)),
    (1e300, 0., 1e-100, 9.210340371976183e-198),
    (1., 0., 1e200, 0.5),
    (2e154, 1e154, 2e154, 9.400072584914712e307),
    (2e154, 0., 2e154, 1.3862943611198907e308),
    (1., 0., 1e-162, 3.7e-322),
    (1e155, 1e155 * (1. - 1e-12), 1., 9.99958899574909e-13),
    (1e154, 1e154 * (1. - 1e-12), 1e200, 1.0000184422022033e296),
]
results = []
with localcontext() as context:
    context.prec = 1100
    for before, after, scale, expected in cases:
        b, c, a = map(Decimal.from_float, (before, after, scale))
        reference = a * a / 2 * ((1 + (b / a) ** 2).ln() - (1 + (c / a) ** 2).ln())
        rounded = float(reference)
        assert rounded == expected, (rounded, expected)
        results.append({
            'before': before,
            'after': after,
            'scale': scale,
            'exact_binary64_inputs': [value.hex() for value in (before, after, scale)],
            'reduction_rounded_to_binary64': rounded,
        })
print(json.dumps({'decimal_precision': 1100, 'cases': results}, indent=2))
