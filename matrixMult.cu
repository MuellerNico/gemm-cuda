#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define DataType float

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  //@@ Insert code to implement matrix multiplication here
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

__global__ void tiled_gemm(DataType *A, DataType *B, DataType *C, int numARows,
  int numAColumns, int numBRows, int numBColumns, int TILE_SIZE) 
{

  int blockRow = blockIdx.x;
  int blockCol = blockIdx.y;
  int row = threadIdx.x;
  int col = threadIdx.y;
  int globalRow = blockRow * TILE_SIZE + row;
  int globalCol = blockCol * TILE_SIZE + col;
  
  extern __shared__ DataType shared[];
  DataType* As = shared;
  DataType* Bs = shared + TILE_SIZE * TILE_SIZE;
  // tiles in shmem
 // __shared__ DataType As[TILE_SIZE][TILE_SIZE];
  //__shared__ DataType Bs[TILE_SIZE][TILE_SIZE];
  
  DataType sum = 0;
  
  // loop over tiles
  int numTiles = (numAColumns + TILE_SIZE - 1) / TILE_SIZE;
  
  for (int t = 0; t < numTiles; t++) {
      int tileIdx = t * TILE_SIZE;
      
      // load A tile into shared memory
      if (globalRow < numARows && (tileIdx + col) < numAColumns) {
          As[row][col] = A[globalRow * numAColumns + (tileIdx + col)];
      } 
      else {
          As[row][col] = 0;
      }
      
      // load B tile into shared memory
      if ((tileIdx + row) < numBRows && globalCol < numBColumns) {
          Bs[row][col] = B[(tileIdx + row) * numBColumns + globalCol];
      } 
      else {
          Bs[row][col] = 0;
      }
      
      __syncthreads(); // continue after all threads have loaded their tiles
      
      for (int k = 0; k < TILE_SIZE; k++) {
          sum += As[row][k] * Bs[k][col];
      }
      
      __syncthreads(); // finish computation before loading next tile
  }
  
  // Write result
  if (globalRow < numARows && globalCol < numBColumns) {
      C[globalRow * numBColumns + globalCol] = sum;
  }
  
}

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


DataType checkResult(DataType *A, DataType *B, int rows, int cols) {
  // return inf norm of A - B
  printf("Checking result\n");
  DataType diff = 0;
  DataType maxDiff = 0;
  for (int i = 0; i < rows*cols; ++i) {
    diff = abs(A[i] - B[i]);
    if (diff > maxDiff) {
      maxDiff = diff;
    }
  }
  return maxDiff;
}


//@@ Insert code to implement timer start
__host__ double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
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
  for (int i = 0; i < numARows*numAColumns; i++){
    hostA[i] = 1.0 * rand() / RAND_MAX;
  }
  for (int i = 0; i < numBRows*numBColumns; ++i) {
    hostB[i] = 1.0 * rand() / RAND_MAX;
  }

  //@@ Insert code below to allocate GPU memory here
  double t0 = cpuSecond();
  cudaMalloc((void**)&deviceA, numARows * numAColumns * sizeof(DataType));
  cudaMalloc((void**)&deviceB, numBRows * numBColumns * sizeof(DataType));
  cudaMalloc((void**)&deviceC, numCRows * numCColumns * sizeof(DataType));
  double time_alloc = cpuSecond() - t0;
  //printf("Time alloc is %f\n", time_alloc);

  //@@ Insert code to below to Copy memory to the GPU here
  t0 = cpuSecond();
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  double time_copy = cpuSecond() - t0;
  //printf("H2D copy is %f\n", time_copy);

  //@@ Insert code below to compare the output with the reference
  printf("\nCPU reference\n");
  t0 = cpuSecond();
  gemm_CPU(hostA, hostB, resultRef, numARows, numAColumns, numBRows, numBColumns);
  double time_cpu = cpuSecond() - t0;
  printf("Timing: %f ms\n", time_cpu);

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
      printf("Timing: %f ms\n", time_kernel);
    }
    else {
      //@@ Launch the tiled GPU Kernel here
      printf("\nCUDA tiled gemm with tile [%d, %d]\n", tileSize, tileSize);
      dim3 TPB(tileSize, tileSize);
      dim3 BPG((numCRows + TPB.x - 1) / TPB.x, (numCColumns + TPB.y - 1) / TPB.y);
      t0 = cpuSecond();
      tiled_gemm<<<BPG, TPB, 2 * tileSize * tileSize * sizeof(DataType)>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, TILE_SIZE);
      cudaDeviceSynchronize();
      double time_kernel = cpuSecond() - t0;
      printf("Timing: %f ms\n", time_kernel);
    }
    //@@ Copy the GPU memory back to the CPU here (for every kernel)
    cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    DataType error = checkResult(hostC, resultRef, numCRows, numCColumns);
    printf("Error: %f\n", error);
  }

  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);


  //@@ Free the CPU memory here
  free(hostA);
  free(hostB);
  free(hostC);
  free(resultRef);

  // N;H2D copy;kernel gpu;D2H copy;kernel cpu
  //int N = numCColumns;
  //printf("%d;%.6f;%.6f;%.6f;%.6f\n", N, time_copy, time_kernel, time_copyback, time_cpu);

  return 0;
}
