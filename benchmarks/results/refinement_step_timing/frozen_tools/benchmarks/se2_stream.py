"""Shared nonlinear input and independent original-energy/gradient oracle."""
import math
from information_precision import energy_weighted_residual
from pathlib import Path


def wrap(t):
    return math.atan2(math.sin(t), math.cos(t))


def compose(a, b):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0]+c*b[0]-s*b[1], a[1]+s*b[0]+c*b[1], wrap(a[2]+b[2]))


def inverse(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-c*a[0]-s*a[1], s*a[0]-c*a[1], -a[2])


def exp(v):
    t = v[2]
    a = math.sin(t)/t if t else 1.
    b = 2*math.sin(t/2)**2/t if t else 0.
    return (a*v[0]-b*v[1], b*v[0]+a*v[1], t)


def fixture(n):
    root = (2., -1., .4)
    truth = [root]
    frames = []
    info = (400., 30., 10., 500., -15., 2500.)
    for k in range(1, n+1):
        inc = exp((.6, .02, .04+.005*math.sin(k*.07)))
        truth.append(compose(truth[-1], inc))
        measured = compose(inc, exp((.008*math.sin(k*.37), .006*math.cos(k*.21), .002*math.sin(k*.19))))
        seed = compose(truth[k], exp((.02*math.cos(k), .01*math.sin(k), .002)))
        edges = [(k-1, k, *measured, *info)]
        if k % 25 == 0:
            for a, b in ((k//3, k-5), (k//5, k)):
                z = compose(compose(inverse(truth[a]), truth[b]), exp((.003*math.sin(k), .002*math.cos(k), .0005*math.sin(k))))
                edges.append((a, b, *z, *(4*v for v in info)))
        frames.append((seed, edges))
    return root, frames


def write(path, n):
    root, frames = fixture(n)
    lines = ['ROOT '+' '.join(format(v,'.17g') for v in root)]
    for k, (seed, edges) in enumerate(frames, 1):
        lines.append(f'FRAME {k} '+ ' '.join(format(v,'.17g') for v in seed)+f' {len(edges)}')
        lines.extend('EDGE '+' '.join(str(v) if isinstance(v,int) else format(v,'.17g') for v in edge) for edge in edges)
    Path(path).write_text('\n'.join(lines)+'\n')


def energy_gradient(poses, edges):
    gradient = [[0.,0.,0.] for _ in poses]
    cost = 0.
    for a,b,zx,zy,zt,i11,i12,i13,i22,i23,i33 in edges:
        x,y,t = poses[a]; bx,by,bt = poses[b]
        c,s,cz,sz = math.cos(t),math.sin(t),math.cos(zt),math.sin(zt)
        qx,qy = c*(bx-x)+s*(by-y), -s*(bx-x)+c*(by-y)
        r = (cz*(qx-zx)+sz*(qy-zy), -sz*(qx-zx)+cz*(qy-zy), wrap(bt-t-zt))
        energy, w = energy_weighted_residual(r,(i11,i12,i13,i22,i23,i33))
        cost += energy
        # Differentiate in world translation, then rotate into each body's tangent.
        d = t+zt; cd,sd = math.cos(d),math.sin(d)
        gx,gy = cd*w[0]-sd*w[1],sd*w[0]+cd*w[1]
        for key, sign in ((a,-1.),(b,1.)):
            ck,sk = math.cos(poses[key][2]),math.sin(poses[key][2])
            gradient[key][0] += sign*(ck*gx+sk*gy)
            gradient[key][1] += sign*(-sk*gx+ck*gy)
        gradient[a][2] += (cz*qy-sz*qx)*w[0]+(-sz*qy-cz*qx)*w[1]-w[2]
        gradient[b][2] += w[2]
    return cost, max(abs(v) for g in gradient[1:] for v in g), gradient
