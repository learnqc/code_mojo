import time
import numpy as np

if __name__ == "__main__":
    n = 14
    signal = [np.random.rand() +1j*np.random.rand()
              for _ in range(1<<n)]
    norm = np.sqrt(sum(abs(a) ** 2 for a in signal))
    signal = [a / norm for a in signal]

    iters = 100

    start = time.perf_counter_ns()
    for _ in range(iters):
        np.fft.fft(signal)
    end = time.perf_counter_ns()
    print(f"Average FFT execution time for {n} digits:"
          f" {(end - start)/iters/1000000:.6f} ms")


