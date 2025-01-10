#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define DataType float

/**
 * @brief Performs matrix multiplication of two matrices A and B on the CPU.
 *
 * This function computes the general matrix multiplication (GEMM) operation:
 * C = A * B
 */
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

/**
 * @brief Naive GPU matrix multiplication
 *
 * This function computes the general matrix multiplication (GEMM) operation:
 * C = alpha * A * B + beta * C
 * where A, B, and C are matrices, and alpha and beta are scalars.
 */
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

/**
 * @brief Performs a tiled matrix multiplication (GEMM) on the GPU.
 *
 * This function computes the product of two matrices A and B, and stores the result in matrix C.
 * The computation is performed using a tiled approach to optimize memory access patterns and improve performance.
 * 
 * @param TILE_SIZE Size of the tile used for the computation.
 */
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


// Convert single precision to half precision
__global__ void convertFP32ToFP16(float* in, half* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);
    }
}

// WMMA GEMM kernel TODO: replace float with DataType
__global__ void wmma_gemm(half* A, half* B, float* C, 
                         int M, int N, int K) {
    // WMMA fragment declarations
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Calculate thread block position
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // M and N must be multiples of 16
    if (warpM >= M/16 || warpN >= N/16) return;

    // Initialize accumulator fragment
    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over K dimension
    for (int i = 0; i < K; i += 16) {
        int aRow = warpM * 16;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * 16;

        // Load the inputs
        wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
        wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);

        // Perform matrix multiplication
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store the output
    int cRow = warpM * 16;
    int cCol = warpN * 16;
    wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
}

// Helper function to launch WMMA kernel
void launchWMMAKernel(float* A, float* B, float* C,
                      int M, int N, int K) {
    // Allocate device memory for FP16 matrices
    half *d_A_half, *d_B_half;
    float *d_C_float;
    
    cudaMalloc(&d_A_half, M * K * sizeof(half));
    cudaMalloc(&d_B_half, K * N * sizeof(half));
    cudaMalloc(&d_C_float, M * N * sizeof(float));

    // Convert input matrices to FP16
    int threadsPerBlock = 256;
    int blocksA = (M * K + threadsPerBlock - 1) / threadsPerBlock;
    int blocksB = (K * N + threadsPerBlock - 1) / threadsPerBlock;

    convertFP32ToFP16<<<blocksA, threadsPerBlock>>>(A, d_A_half, M * K);
    convertFP32ToFP16<<<blocksB, threadsPerBlock>>>(B, d_B_half, K * N);

    // Launch WMMA kernel
    dim3 blockDim(128, 4);
    dim3 gridDim(
        (M + (16 * blockDim.x / 32) - 1) / (16 * blockDim.x / 32),
        (N + 16 * blockDim.y - 1) / (16 * blockDim.y)
    );

    wmma_gemm<<<gridDim, blockDim>>>(d_A_half, d_B_half, d_C_float, M, N, K);

    // Copy result back to output matrix
    cudaMemcpy(C, d_C_float, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A_half);
    cudaFree(d_B_half);
    cudaFree(d_C_float);
}

// Modified checkResult function to calculate relative error
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
  // return inf norm of A - B
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

DataType checkResult(DataType *A, DataType *B, int rows, int cols) {
  return inftyNorm(A, B, rows, cols);
}

/**
 * @brief Returns the current time in seconds.
 */
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

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
  numARows = strtol(argv[1], NULL, 10);
  numAColumns = strtol(argv[2], NULL, 10);
  numBRows = strtol(argv[2], NULL, 10);
  numBColumns = strtol(argv[3], NULL, 10);
  numCRows = numARows;
  numCColumns = numBColumns;

  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  //@@ Insert code below to allocate Host memory for input and output
  hostA = (DataType*)malloc(numARows * numAColumns * sizeof(DataType));
  hostB = (DataType*)malloc(numBRows * numBColumns * sizeof(DataType));
  hostC = (DataType*)malloc(numCRows * numCColumns * sizeof(DataType));
  resultRef = (DataType*)malloc(numCRows * numCColumns * sizeof(DataType));

  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
  for (int i = 0; i < numARows * numAColumns; i++) {
    hostA[i] = 1.0 * rand() / RAND_MAX;
  }
  for (int i = 0; i < numBRows * numBColumns; ++i) {
    hostB[i] = 1.0 * rand() / RAND_MAX;
  }

  //@@ Insert code below to allocate GPU memory here
  double t0 = cpuSecond();
  cudaMalloc((void**)&deviceA, numARows * numAColumns * sizeof(DataType));
  cudaMalloc((void**)&deviceB, numBRows * numBColumns * sizeof(DataType));
  cudaMalloc((void**)&deviceC, numCRows * numCColumns * sizeof(DataType));
  double time_alloc = cpuSecond() - t0;

  //@@ Insert code to below to Copy memory to the GPU here
  t0 = cpuSecond();
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  double time_copy = cpuSecond() - t0;

  //@@ Insert code below to compare the output with the reference
  printf("\nCPU reference\n");
  t0 = cpuSecond();
  gemm_CPU(hostA, hostB, resultRef, numARows, numAColumns, numBRows, numBColumns);
  double time_cpu = cpuSecond() - t0;
  printf("Timing: %f\n", time_cpu);

  //@@ Initialize the grid and block dimensions here
  int tileSizes[] = {0, 32, 64};
  int numTileSizes = 3;

  for (int i = 0; i < numTileSizes; i++) {
    int tileSize = tileSizes[i];
    if (tileSize == 0) {
      //@@ Launch the normal GPU Kernel here
      printf("\nCUDA gemm\n");
      dim3 TPB(32, 32);
      dim3 BPG((numCRows + TPB.x - 1) / TPB.x, (numCColumns + TPB.y - 1) / TPB.y);
      t0 = cpuSecond();
      gemm<<<BPG, TPB>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
      cudaDeviceSynchronize();
      double time_kernel = cpuSecond() - t0;
      printf("Timing: %f\n", time_kernel);
    } else {
      //@@ Launch the tiled GPU Kernel here
      printf("\nCUDA tiled gemm with tile [%d, %d]\n", tileSize, tileSize);
      dim3 TPB(tileSize, tileSize);
      dim3 BPG((numCRows + TPB.x - 1) / TPB.x, (numCColumns + TPB.y - 1) / TPB.y);
      t0 = cpuSecond();
      tiled_gemm<<<BPG, TPB, 2 * tileSize * tileSize * sizeof(DataType)>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, tileSize);
      cudaDeviceSynchronize();
      double time_kernel = cpuSecond() - t0;
      printf("Timing: %f\n", time_kernel);
    }
    //@@ Copy the GPU memory back to the CPU here (for every kernel)
    cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    DataType error = checkResult(hostC, resultRef, numCRows, numCColumns);
    printf("Error: %f\n", error);
  }

  printf("\nCUDA WMMA gemm\n");
  t0 = cpuSecond();
  launchWMMAKernel(deviceA, deviceB, hostC, numARows, numAColumns, numBColumns);
  cudaDeviceSynchronize();
  double time_wmma = cpuSecond() - t0;
  printf("Timing: %f\n", time_wmma);
  float wmma_error = checkResult(hostC, resultRef, numCRows, numCColumns);
  printf("WMMA Error: %f\n", wmma_error);

  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  //@@ Free the CPU memory here
  free(hostA);
  free(hostB);
  free(hostC);
  free(resultRef);

  return 0;
}
