#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mma.h>
#include "cuda_fp16.h"

#define DataType float

    using namespace nvcuda;

// wmma array dimensions
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Macro for checking CUDA errors
#define CHECK_CUDA_ERROR(call)                                                             \
    {                                                                                      \
        cudaError_t err = call;                                                            \
        if (err != cudaSuccess)                                                            \
        {                                                                                  \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(2);                                                                       \
        }                                                                                  \
    }

// Macro for checking kernel launches
#define CHECK_KERNEL_ERROR()                                              \
    {                                                                     \
        cudaError_t err = cudaGetLastError();                             \
        if (err != cudaSuccess)                                           \
        {                                                                 \
            printf("Kernel launch error: %s\n", cudaGetErrorString(err)); \
            exit(2);                                                      \
        }                                                                 \
    }

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                     int numAColumns, int numBRows, int numBColumns)
{
    //@@ Insert code to implement matrix multiplication here
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= numARows || col >= numBColumns)
    {
        return;
    }

    DataType sum = 0;
    for (int i = 0; i < numAColumns; i++)
    {
        sum += A[row * numAColumns + i] * B[i * numBColumns + col];
    }
    C[row * numBColumns + col] = sum;
}

void gemm_CPU(DataType *A, DataType *B, DataType *C, int numARows,
              int numAColumns, int numBRows, int numBColumns)
{
    for (int i = 0; i < numARows; i++)
    {
        for (int j = 0; j < numBColumns; j++)
        {
            DataType sum = 0;
            for (int k = 0; k < numAColumns; k++)
            {
                sum += A[i * numAColumns + k] * B[k * numBColumns + j];
            }
            C[i * numBColumns + j] = sum;
        }
    }
}

template <int TILE_SIZE> // known at compile time
__global__ void tiled_gemm(DataType *A, DataType *B, DataType *C,
                           int numARows, int numAColumns,
                           int numBRows, int numBColumns)
{

    int blockRow = blockIdx.x;
    int blockCol = blockIdx.y;
    int row = threadIdx.x;
    int col = threadIdx.y;
    int globalRow = blockRow * TILE_SIZE + row;
    int globalCol = blockCol * TILE_SIZE + col;

    // tiles in shmem
    __shared__ DataType As[TILE_SIZE][TILE_SIZE];
    __shared__ DataType Bs[TILE_SIZE][TILE_SIZE];

    DataType sum = 0;

    // loop over tiles
    int numTiles = (numAColumns + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++)
    {
        int tileIdx = t * TILE_SIZE;

        // load A tile into shared memory
        if (globalRow < numARows && (tileIdx + col) < numAColumns)
        {
            As[row][col] = A[globalRow * numAColumns + (tileIdx + col)];
        }
        else
        {
            As[row][col] = 0;
        }

        // load B tile into shared memory
        if ((tileIdx + row) < numBRows && globalCol < numBColumns)
        {
            Bs[row][col] = B[(tileIdx + row) * numBColumns + globalCol];
        }
        else
        {
            Bs[row][col] = 0;
        }

        __syncthreads(); // continue after all threads have loaded their tiles

        for (int k = 0; k < TILE_SIZE; k++)
        {
            sum += As[row][k] * Bs[k][col];
        }

        __syncthreads(); // finish computation before loading next tile
    }

    // Write result
    if (globalRow < numARows && globalCol < numBColumns)
    {
        C[globalRow * numBColumns + globalCol] = sum;
    }
}

__global__ void convertSingleToHalf(DataType *in, half *out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        out[idx] = __float2half(in[idx]);
    }
}

__global__ void wmma_gemm(half *A, half *B, DataType *C, int numARows,
                          int numAColumns, int numBRows, int numBColumns)
{
    // global position
    int warp_row = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warp_col = (blockIdx.y * blockDim.y + threadIdx.y);

    // wmma fragment declaration
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;
    
    // init C fragment to 0
    wmma::fill_fragment(cFrag, 0.0f);

    // starting positions
    int globalRow = warp_row * WMMA_M;
    int globalCol = warp_col * WMMA_N;

    // loop over tiles
    for (int k = 0; k < numAColumns; k += WMMA_K)
    {
        if (globalRow < numARows && k < numAColumns)
        {
            wmma::load_matrix_sync(aFrag, A + globalRow * numAColumns + k, numAColumns);
            wmma::load_matrix_sync(bFrag, B + k * numBColumns + globalCol, numBColumns);
            wmma::mma_sync(cFrag, aFrag, bFrag, cFrag); // C <- A * B + C
        }
    }

    // store result
    if (globalRow < numARows && globalCol < numBColumns)
    {
        wmma::store_matrix_sync(C + globalRow * numBColumns + globalCol, cFrag, numBColumns, wmma::mem_row_major);
    }
}

DataType checkResult(DataType *A, DataType *B, int rows, int cols)
{
    // return inf norm of A - B
    printf("Checking result\n");
    DataType diff = 0;
    DataType maxDiff = 0;
    for (int i = 0; i < rows * cols; ++i)
    {
        diff = abs(A[i] - B[i]);
        if (diff > maxDiff)
        {
            maxDiff = diff;
        }
    }
    return maxDiff;
}

//@@ Insert code to implement timer start
__host__ double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

int main(int argc, char **argv)
{

    DataType *hostA;     // The A matrix
    DataType *hostB;     // The B matrix
    DataType *hostC;     // The output C matrix
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
    //   numARows = strtol(argv[1], NULL, 10);
    //   numAColumns = strtol(argv[2], NULL, 10);
    //   numBRows = strtol(argv[2], NULL, 10);
    //   numBColumns = strtol(argv[3], NULL, 10);
    numARows = 1024;
    numAColumns = 2048;
    numBRows = 2048;
    numBColumns = 1024;
    numCRows = numARows;
    numCColumns = numBColumns;

    printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    // Initialize host memory
    hostA = (DataType *)malloc(numARows * numAColumns * sizeof(DataType));
    hostB = (DataType *)malloc(numBRows * numBColumns * sizeof(DataType));
    hostC = (DataType *)malloc(numCRows * numCColumns * sizeof(DataType));
    resultRef = (DataType *)malloc(numCRows * numCColumns * sizeof(DataType));

    // Initialize input matrices
    for (int i = 0; i < numARows * numAColumns; i++)
    {
        hostA[i] = 1.0f * rand() / RAND_MAX;
    }
    for (int i = 0; i < numBRows * numBColumns; i++)
    {
        hostB[i] = 1.0f * rand() / RAND_MAX;
    }

    // GPU Memory allocation
    double t0 = cpuSecond();
    CHECK_CUDA_ERROR(cudaMalloc((void **)&deviceA, numARows * numAColumns * sizeof(DataType)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&deviceB, numBRows * numBColumns * sizeof(DataType)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&deviceC, numCRows * numCColumns * sizeof(DataType)));
    double time_alloc = cpuSecond() - t0;
    printf("Time alloc is %f\n", time_alloc);

    // Memory copies with error checking
    t0 = cpuSecond();
    CHECK_CUDA_ERROR(cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(DataType), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(DataType), cudaMemcpyHostToDevice));
    double time_copy = cpuSecond() - t0;
    printf("H2D copy is %f\n", time_copy);

    // Initialize grid and block dimensions
    dim3 TPB(32, 32);
    dim3 BPG((numCRows + TPB.x - 1) / TPB.x, (numCColumns + TPB.y - 1) / TPB.y);

    // Basic GEMM kernel launch
    printf("Launching kernel\n");
    t0 = cpuSecond();
    gemm<<<BPG, TPB>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
    CHECK_KERNEL_ERROR();
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    double time_kernel = cpuSecond() - t0;
    printf("Time kernel is %f\n", time_kernel);

    // Copy results back and check error
    t0 = cpuSecond();
    CHECK_CUDA_ERROR(cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyDeviceToHost));
    double time_copyback = cpuSecond() - t0;
    printf("D2H copy is %f\n", time_copyback);

    // Generate CPU reference result and check error
    t0 = cpuSecond();
    gemm_CPU(hostA, hostB, resultRef, numARows, numAColumns, numBRows, numBColumns);
    double time_cpu = cpuSecond() - t0;
    printf("Time cpu is %f\n", time_cpu);

    DataType basic_gemm_error = checkResult(hostC, resultRef, numCRows, numCColumns);
    printf("Basic GEMM Error: %f\n", basic_gemm_error);

    // ========================
    // Tiled GEMM
    // ========================

    // Tiled GEMM kernels with different tile sizes
    const int TS8 = 8;
    const int TS16 = 16;
    const int TS32 = 32;
    // const int TS64 = 64;
    DataType *tiled_result = (DataType *)malloc(numCRows * numCColumns * sizeof(DataType));

    // 8x8 tiles
    dim3 block8(TS8, TS8);
    dim3 grid8((numARows + TS8 - 1) / TS8, (numBColumns + TS8 - 1) / TS8);
    printf("Launching tiled kernel 8\n");
    double tiled_time8 = cpuSecond();
    tiled_gemm<TS8><<<grid8, block8>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
    CHECK_KERNEL_ERROR();
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    tiled_time8 = cpuSecond() - tiled_time8;
    printf("Tiled kernel time 8: %f\n", tiled_time8);

    // Check results for 8x8
    CHECK_CUDA_ERROR(cudaMemcpy(tiled_result, deviceC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyDeviceToHost));
    DataType tiled8_error = checkResult(tiled_result, resultRef, numCRows, numCColumns);
    printf("Tiled 8x8 Error: %f\n", tiled8_error);

    // 16x16 tiles
    dim3 block16(TS16, TS16);
    dim3 grid16((numARows + TS16 - 1) / TS16, (numBColumns + TS16 - 1) / TS16);
    printf("Launching tiled kernel 16\n");
    double tiled_time16 = cpuSecond();
    tiled_gemm<TS16><<<grid16, block16>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
    CHECK_KERNEL_ERROR();
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    tiled_time16 = cpuSecond() - tiled_time16;
    printf("Tiled kernel time 16: %f\n", tiled_time16);

    // Check results for 16x16
    CHECK_CUDA_ERROR(cudaMemcpy(tiled_result, deviceC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyDeviceToHost));
    DataType tiled16_error = checkResult(tiled_result, resultRef, numCRows, numCColumns);
    printf("Tiled 16x16 Error: %f\n", tiled16_error);

    // 32x32 tiles
    dim3 block32(TS32, TS32);
    dim3 grid32((numARows + TS32 - 1) / TS32, (numBColumns + TS32 - 1) / TS32);
    printf("Launching tiled kernel 32\n");
    double tiled_time32 = cpuSecond();
    tiled_gemm<TS32><<<grid32, block32>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
    CHECK_KERNEL_ERROR();
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    tiled_time32 = cpuSecond() - tiled_time32;
    printf("Tiled kernel time 32: %f\n", tiled_time32);

    // Check results for 32x32
    CHECK_CUDA_ERROR(cudaMemcpy(tiled_result, deviceC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyDeviceToHost));
    DataType tiled32_error = checkResult(tiled_result, resultRef, numCRows, numCColumns);
    printf("Tiled 32x32 Error: %f\n", tiled32_error);

    // ========================
    // WMMA
    // ========================

    // allocate half memory and convert&copy single to half array
    half *d_a_h, *d_b_h;
    CHECK_CUDA_ERROR(cudaMalloc(&d_a_h, numARows * numAColumns * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b_h, numBRows * numBColumns * sizeof(half)));

    dim3 s2hBlock(256);
    dim3 s2hGrid((numARows * numAColumns + 255) / 256);
    convertSingleToHalf<<<s2hGrid, s2hBlock>>>(deviceA, d_a_h, numARows * numAColumns);
    CHECK_KERNEL_ERROR();
    convertSingleToHalf<<<s2hGrid, s2hBlock>>>(deviceB, d_b_h, numBRows * numBColumns);
    CHECK_KERNEL_ERROR();

    dim3 wmmaBlock(128, 4);
    dim3 wmmaGrid(
        (numARows + (WMMA_M * wmmaBlock.x / 32) - 1) / (WMMA_M * wmmaBlock.x / 32),
        (numBColumns + WMMA_N * wmmaBlock.y - 1) / (WMMA_N * wmmaBlock.y));


    printf("Launching WMMA kernel\n");
    double wmma_time = cpuSecond();
    wmma_gemm<<<wmmaGrid, wmmaBlock>>>(d_a_h, d_b_h, deviceC, numARows, numAColumns, numBRows, numBColumns);
    CHECK_KERNEL_ERROR();
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    wmma_time = cpuSecond() - wmma_time;
    printf("WMMA kernel time: %f\n", wmma_time);

    // Free half memory
    CHECK_CUDA_ERROR(cudaFree(d_a_h));
    CHECK_CUDA_ERROR(cudaFree(d_b_h));

    // Copy results back and check error
    CHECK_CUDA_ERROR(cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyDeviceToHost));
    DataType wmma_error = checkResult(hostC, resultRef, numCRows, numCColumns);
    printf("WMMA Error: %f\n", wmma_error);

    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(deviceA));
    CHECK_CUDA_ERROR(cudaFree(deviceB));
    CHECK_CUDA_ERROR(cudaFree(deviceC));

    free(hostA);
    free(hostB);
    free(hostC);
    free(resultRef);
    free(tiled_result);

    // Print CSV format results
    int N = numCColumns;
    printf("%d;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f\n",
           N, time_copy, time_kernel, time_copyback, time_cpu, tiled_time8,
           tiled_time16, tiled_time32, wmma_time);

    return 0;
}
