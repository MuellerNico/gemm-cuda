import matplotlib.pyplot as plt
import numpy as np

matrix_sizes = [1024, 2048, 4096, 8192, 16384]
cpu_times = [8.202699, 93.049888, 0, 0, 0]
cuda_16 = [0.034854, 0.155287, 0.815158, 6.515443, 55.756577]
cuda_32 = [0.107921, 0.328719, 2.502125, 19.904485, 160.352764]
cuda_gemm = [0.071456, 0.280413, 2.231324, 17.803508, 164.176662]
cuda_8 = [0.017204, 0.135684, 0.633295, 4.215392, 34.358248]
cuda_wmma = [0.005829, 0.025793, 0.149792, 0.646806, 4.114238]

bar_width = 0.
index = np.arange(len(matrix_sizes))
plt.bar(index + 1*bar_width, cuda_8, bar_width, label='8x8 tiles')
plt.bar(index + 2*bar_width, cuda_16, bar_width, label='16x16 tiles')
plt.bar(index + 3*bar_width, cuda_32, bar_width, label='32x32 tiles')
plt.bar(index + 4*bar_width, cuda_gemm, bar_width, label='gemm')
plt.bar(index + 5*bar_width, cpu_times, bar_width, label='CPU')
plt.bar(index, cuda_wmma, bar_width, label='wmma')
plt.yscale('log')
plt.xlabel('Matrix Size')
plt.ylabel('Time (s)')
plt.xticks(index + 2*bar_width, matrix_sizes)
plt.legend()

plt.savefig('performance.pdf')

# debug
for i in range(len(matrix_sizes)):
    print(cuda_8[i] / cuda_wmma[i])