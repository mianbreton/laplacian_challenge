import numpy as np

def wrap(grid, axis, pm):
    if pm == 'plus':
        return np.roll(grid, shift=1, axis=axis)
    elif pm == 'minus':
        return np.roll(grid, shift=-1, axis=axis)

def brute_force_vectorize(out, delta):
    N = len(delta[:,0,0])
    six = np.float32(6.0)
    invh2 = np.float32(N**2)

    shift_xm1 = wrap(delta, 0, 'minus')
    shift_xp1 = wrap(delta, 0, 'plus')

    shift_ym1 = wrap(delta, 1, 'minus')
    shift_yp1 = wrap(delta, 1, 'plus')

    shift_zm1 = wrap(delta, 2, 'minus')
    shift_zp1 = wrap(delta, 2, 'plus')
    out[:,:,:] = (shift_xm1 + shift_xp1 + shift_ym1 + shift_yp1 + shift_zm1 + shift_zp1 - six*delta) * invh2