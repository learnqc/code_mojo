from timeit import repeat
import time
import numpy as np

n = 14
# s = [1 +1j*0 if i == 1 else 0 for i in range(1<<n)]
signal = [np.random.rand() +1j*np.random.rand() for _ in range(1<<n)]
norm = np.sqrt(sum(abs(a) ** 2 for a in signal))
signal = [a / norm for a in signal]

def main():
    iters = 100
    t = 1000*sum(repeat('np.fft.fft(signal)', repeat=iters, number=1, globals=globals()))/iters

    start = time.perf_counter_ns()
    # Code to be timed
    for _ in range(iters):
        np.fft.fft(signal)
    end = time.perf_counter_ns()
    print(f"Average execution time: {(end - start)/iters/1000000:.6f} ms")

    print('numpy    %7.10f'  % t)

if __name__ == "__main__":
    main()
