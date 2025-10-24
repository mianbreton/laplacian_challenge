import numpy as np
import numpy.typing as npt
import time
import matplotlib.pyplot as plt


def brute_force_loop(out, delta):
    N = len(delta[:,0,0])
    invh2 = N**2

    for i_counter in range(0, N):
        i_m1 = (i_counter-1) % N
        i_p1 = (i_counter+1) % N

        for j_counter in range(0, N):
            j_m1 = (j_counter-1) % N
            j_p1 = (j_counter+1) % N
            for k_counter in range(0, N):
                k_m1 = (k_counter-1) % N
                k_p1 = (k_counter+1) % N

                # Comupute the seven-point stencil
                out[i_counter,j_counter,k_counter] = (delta[i_m1,j_counter,k_counter] + delta[i_p1,j_counter,k_counter] + delta[i_counter,j_m1,k_counter] + delta[i_counter,j_p1,k_counter] + delta[i_counter,j_counter,k_m1] + delta[i_counter,j_counter,k_p1] - 6.*delta[i_counter,j_counter,k_counter]) * invh2

def wrap(grid, axis, pm):
    if pm == 'plus':
        return np.roll(grid, shift=1, axis=axis)
    elif pm == 'minus':
        return np.roll(grid, shift=-1, axis=axis)

def brute_force_vectorize(out, delta):
    N = len(delta[:,0,0])
    invh2 = N**2

    shift_xm1 = wrap(delta, 0, 'minus')
    shift_xp1 = wrap(delta, 0, 'plus')

    shift_ym1 = wrap(delta, 1, 'minus')
    shift_yp1 = wrap(delta, 1, 'plus')

    shift_zm1 = wrap(delta, 2, 'minus')
    shift_zp1 = wrap(delta, 2, 'plus')

    out[:,:,:] = (shift_xm1 + shift_xp1 + shift_ym1 + shift_yp1 + shift_zm1 + shift_zp1 - 6.*delta) * invh2

def timing():
    time_array = np.zeros((2, 5, 10))
    rng = np.random.default_rng(42)

    N_list = [16, 32, 64, 128, 256]
    for N_counter, N in enumerate(N_list):
        print(N)
        for i in range(10):
            if N > 64:
                print(i)
            delta = rng.uniform(size=(N,N,N))
            out = np.zeros_like(delta)

            tick = time.perf_counter()
            brute_force_loop(out, delta)
            out1 = out
            print(out1[:3,:3,0])
            tock = time.perf_counter()
            out = np.zeros_like(delta)
            tick1 = time.perf_counter()
            brute_force_vectorize(out, delta)
            out2 = out
            print(out2[:3,:3,0])
            exit()
            tock1 = time.perf_counter()


            time_array[0,N_counter, i] = tock-tick
            time_array[1,N_counter, i] = tock1-tick1

    return time_array


def main():
    times = timing()
    med_times = np.median(times, axis=-1)

    plt.plot(med_times[0,:], label='loop')
    plt.plot(med_times[1,:], label='vec')
    plt.yscale('log')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
