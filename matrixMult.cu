#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define DataType float

// naive CPU matrix multiplication
void gemm_CPU(DataType *A, DataType *B, DataType *C, int numARows,
        int numAColumns, int numBRows, int numBColumns) {
  for (int i = 0; i < numARows; i++) {
    for (int j = 0; j < numBColumns; j++) {
      DataType sum = 0;
      for (int k = 0; k < numAColumns; k++) {
        sum += A[i * numAColumns + k] * B[k * numBColumns + j];
      }
      C[i * numBColumns + j] = sum;
    }
  }
}

//Naive GPU matrix multiplication
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
           int numAColumns, int numBRows, int numBColumns) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row >= numARows || col >= numBColumns) {
    return;
  }

  DataType sum = 0;
  for (int i = 0; i < numAColumns; i++) {
    sum += A[row * numAColumns + i] * B[i * numBColumns + col];
  }
  C[row * numBColumns + col] = sum;
}

// Tiled GPU matrix multiplication
__global__ void tiled_gemm(DataType *A, DataType *B, DataType *C, int numARows,
               int numAColumns, int numBRows, int numBColumns, int TILE_SIZE) {
  int blockRow = blockIdx.x;
  int blockCol = blockIdx.y;
  int row = threadIdx.x;
  int col = threadIdx.y;
  int globalRow = blockRow * TILE_SIZE + row;
  int globalCol = blockCol * TILE_SIZE + col;

  extern __shared__ DataType shared[];  // dynamically allocated shared memory
  DataType *As = shared;
  DataType *Bs = shared + TILE_SIZE * TILE_SIZE;

  DataType sum = 0;

  // loop over tiles
  int numTiles = (numAColumns + TILE_SIZE - 1) / TILE_SIZE;

  for (int t = 0; t < numTiles; t++) {
    int tileIdx = t * TILE_SIZE;

    // load A tile into shared memory
    if (globalRow < numARows && (tileIdx + col) < numAColumns) {
      As[row * TILE_SIZE + col] = A[globalRow * numAColumns + (tileIdx + col)];
    } else {
      As[row * TILE_SIZE + col] = 0;
    }

    // load B tile into shared memory
    if ((tileIdx + row) < numBRows && globalCol < numBColumns) {
      Bs[row * TILE_SIZE + col] = B[(tileIdx + row) * numBColumns + globalCol];
    } else {
      Bs[row * TILE_SIZE + col] = 0;
    }

    __syncthreads(); // continue after all threads have loaded their tiles

    // compute the tile
    for (int k = 0; k < TILE_SIZE; k++) {
      sum += As[row * TILE_SIZE + k] * Bs[k * TILE_SIZE + col];
    }

    __syncthreads(); // finish computation before loading next tile
  }

  // Write result to global memory
  if (globalRow < numARows && globalCol < numBColumns) {
    C[globalRow * numBColumns + globalCol] = sum;
  }
}

// Helper kernel to convert single precision to half precision
__global__ void convertFP32ToFP16(float* in, half* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);
    }
}

// wmma GPU matrix multiplication with tensor cores
__global__ void wmma_gemm(half* A, half* B, float* C, 
                         int M, int N, int K) {
    // Each warp computes a 16x16 output tile
    //int warpID = threadIdx.x / 32;
    //int warpM = blockIdx.x * (blockDim.x / 32) + warpID;
    //int warpN = blockIdx.y;

    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    int aRow = warpM * 16;
    int bCol = warpN * 16;

    if (warpM >= M/16 || warpN >= N/16) return;

    // Create fragments and accumulator
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over K dimension
    for (int i = 0; i < K; i += 16) {
        int aCol = i;
        int bRow = i;

        //if (aRow < M && i < N)
        //{
          // Load inputs and compute
          wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
          wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
          wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        //}
    }

    int cRow = warpM * 16;
    int cCol = warpN * 16;
    if (cRow < M && cCol < N)
    {
      wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}

// Helper function to launch wmma kernel
void launchWMMAKernel(float* A, float* B, float* C,
                      int M, int N, int K) {
    // Allocate device memory for FP16 matrices
    half *d_A_half, *d_B_half;
    float *d_C_float;
    
    cudaMalloc(&d_A_half, M * K * sizeof(half));
    cudaMalloc(&d_B_half, K * N * sizeof(half));
    cudaMalloc(&d_C_float, M * N * sizeof(float));

    // Convert input to half precision
    int threadsPerBlock = 256;
    int blocksA = (M * K + threadsPerBlock - 1) / threadsPerBlock;
    int blocksB = (K * N + threadsPerBlock - 1) / threadsPerBlock;

    convertFP32ToFP16<<<blocksA, threadsPerBlock>>>(A, d_A_half, M * K);
    convertFP32ToFP16<<<blocksB, threadsPerBlock>>>(B, d_B_half, K * N);
    
    cudaMemset(d_C_float, 0, M * N * sizeof(float)); // init output to 0

    // Set dimensions for WMMA kernel
    //dim3 block(128); // 4 warps per block
    //dim3 grid((M + 31) / 32, (N + 31) / 32);

    dim3 block(128, 4);
    //dim block(32, 1);
    //dim block(16, 16);
    dim3 grid(
      (N + (16 * block.x / 32) - 1) / (16 * block.x / 32),
      (M + 16 * block.y - 1) / (16 * block.y));

    wmma_gemm<<<grid, block>>>(d_A_half, d_B_half, d_C_float, M, N, K);
    
    // Copy result back to output matrix
    cudaMemcpy(C, d_C_float, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A_half);
    cudaFree(d_B_half);
    cudaFree(d_C_float);
}

// max relative error between two matrices 
DataType relativeError(float *A, float *B, int rows, int cols) {
    float maxRelErr = 0.0f;
    for (int i = 0; i < rows * cols; ++i) {
        float relErr = fabs(A[i] - B[i]) / (fabs(B[i]) + 1e-10f);
        maxRelErr = fmax(maxRelErr, relErr);
    }
    return maxRelErr;
}

// infinity norm error
DataType inftyNorm(DataType *A, DataType *B, int rows, int cols) {
  DataType diff = 0;
  DataType maxDiff = 0;
  for (int i = 0; i < rows * cols; ++i) {
    diff = abs(A[i] - B[i]);
    if (diff > maxDiff) {
      maxDiff = diff;
    }
  }
  return maxDiff;
}

// helper to switch between error types easier
DataType checkResult(DataType *A, DataType *B, int rows, int cols) {
  return inftyNorm(A, B, rows, cols);
}

// Returns the current time in seconds.
__host__ double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

int main(int argc, char **argv) {
  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  // read matrix dimensions from input
  numARows = strtol(argv[1], NULL, 10);
  numAColumns = strtol(argv[2], NULL, 10);
  numBRows = strtol(argv[2], NULL, 10);
  numBColumns = strtol(argv[3], NULL, 10);
  numCRows = numARows;
  numCColumns = numBColumns;

  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  // allocate Host memory for input and output
  hostA = (DataType*)malloc(numARows * numAColumns * sizeof(DataType));
  hostB = (DataType*)malloc(numBRows * numBColumns * sizeof(DataType));
  hostC = (DataType*)malloc(numCRows * numCColumns * sizeof(DataType));
  resultRef = (DataType*)malloc(numCRows * numCColumns * sizeof(DataType));

  // initialize hostA and hostB to random numbers
  for (int i = 0; i < numARows * numAColumns; i++) {
    hostA[i] = 1.0 * rand() / RAND_MAX;
  }
  for (int i = 0; i < numBRows * numBColumns; ++i) {
    hostB[i] = 1.0 * rand() / RAND_MAX;
  }

  // allocate GPU memory
  double t0 = cpuSecond();
  cudaMalloc((void**)&deviceA, numARows * numAColumns * sizeof(DataType));
  cudaMalloc((void**)&deviceB, numBRows * numBColumns * sizeof(DataType));
  cudaMalloc((void**)&deviceC, numCRows * numCColumns * sizeof(DataType));
  double time_alloc = cpuSecond() - t0;

  // Copy A and B to GPU
  t0 = cpuSecond();
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  double time_copy = cpuSecond() - t0;

  // Compute reference result on CPU 
  printf("\nCPU reference\n");
  t0 = cpuSecond();
  gemm_CPU(hostA, hostB, resultRef, numARows, numAColumns, numBRows, numBColumns);
  double time_cpu = cpuSecond() - t0;
  printf("Timing: %f\n", time_cpu);

  // tile sizes for tiled kernel (0 for normal gpu gemm)
  int tileSizes[] = {0, 32, 64};
  int numTileSizes = 3;

  for (int i = 0; i < numTileSizes; i++) {
    int tileSize = tileSizes[i];
    if (tileSize == 0) {
      // Launch the naive GPU kernel
      printf("\nCUDA gemm\n");
      dim3 TPB(32, 32);
      dim3 BPG((numCRows + TPB.x - 1) / TPB.x, (numCColumns + TPB.y - 1) / TPB.y);
      t0 = cpuSecond();
      gemm<<<BPG, TPB>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
      cudaDeviceSynchronize();
      double time_kernel = cpuSecond() - t0;
      printf("Timing: %f\n", time_kernel);
    } else {
      // Launch the tiled GPU kernel here
      printf("\nCUDA tiled gemm with tile [%d, %d]\n", tileSize, tileSize);
      dim3 TPB(tileSize, tileSize);
      dim3 BPG((numCRows + TPB.x - 1) / TPB.x, (numCColumns + TPB.y - 1) / TPB.y);
      t0 = cpuSecond();
      tiled_gemm<<<BPG, TPB, 2 * tileSize * tileSize * sizeof(DataType)>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, tileSize);
      cudaDeviceSynchronize();
      double time_kernel = cpuSecond() - t0;
      printf("Timing: %f\n", time_kernel);
    }
    // Copy GPU memory back to CPU (same for every kernel)
    cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // compare to reference and print error
    DataType error = checkResult(hostC, resultRef, numCRows, numCColumns);
    printf("Error: %f\n", error);
  }
  // launch wmma kernel
  printf("\nCUDA WMMA gemm\n");
  t0 = cpuSecond();
  launchWMMAKernel(deviceA, deviceB, hostC, numARows, numAColumns, numBColumns);
  cudaDeviceSynchronize();
  double time_wmma = cpuSecond() - t0;
  printf("Timing: %f\n", time_wmma);
  float wmma_error = checkResult(hostC, resultRef, numCRows, numCColumns);
  printf("WMMA Error: %f\n", wmma_error);

  // Free GPU memory
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  // Free CPU memory
  free(hostA);
  free(hostB);
  free(hostC);
  free(resultRef);

  return 0;
}
