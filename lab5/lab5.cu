#include <iostream>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>


#define CSC(call)                                  \
do {                                               \
    cudaError_t res = call;                        \
    if (res != cudaSuccess) {                      \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(res)); \
        exit(0);                                   \
    }                                              \
} while(0)

const int MAX_LIMIT = 1 << 24;
const int BLOCK_SIZE = 1024;

__host__ __device__ int index(int num) {
    return num + (num >> 5);
}

__global__ void addScan(int* data, int* scanSum, int size) {
    int blockId = blockIdx.x + 1;
    int offset = gridDim.x;

    for (int i = blockId; i < size; i += offset) {
        int idx = i * blockDim.x + threadIdx.x;

        data[idx] += (i ? scanSum[i] : 0);
    }
}

__global__ void prefixScan(int* data, int* scanSum, int size) {
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int offset = gridDim.x;
    
    __shared__ int temp[1024];

    for (int i = blockId; i < size; i += offset) {
        int curr;
        int prev;

        temp[index(threadId)] = data[i * blockDim.x + threadId];
        __syncthreads();

        for (int j = 1; j < BLOCK_SIZE; j <<= 1) {
            if ((threadId + 1) % (j << 1) == 0) {
                curr = index(threadId);
                prev = index(threadId - j);
                
                temp[curr] += temp[prev];
            }

            __syncthreads();
        }

        if (threadId == 0) {
            scanSum[i] = temp[index(BLOCK_SIZE - 1)];
            temp[index(BLOCK_SIZE - 1)] = 0;
        }

        __syncthreads();

        for (int j = 1 << 10 - 1; j >= 1; j >>= 1) {
            if ((threadId + 1) % (j << 1) == 0) {
                curr = index(threadId);
                prev = index(threadId - j);

                int temp2;

                temp2 = temp[curr];
                temp[curr] += temp[prev];
                temp[prev] = temp2;
            }

            __syncthreads();
        }

        data[i * blockDim.x + threadId] = temp[index(threadId)];
    }
}

void scan(int* data, int n) {
    n += (n % BLOCK_SIZE) ? (BLOCK_SIZE - n % BLOCK_SIZE) : 0;

    int* scanSum;
    CSC(cudaMalloc(&scanSum, (n / BLOCK_SIZE * sizeof(int))));

    prefixScan<<<1024, 1024, sizeof(int) * index(BLOCK_SIZE)>>>(data, scanSum, n / BLOCK_SIZE);
    CSC(cudaDeviceSynchronize());

    if (n <= BLOCK_SIZE) return;

    scan(scanSum, n / BLOCK_SIZE);

    addScan<<<1024, 1024>>>(data, scanSum, n / BLOCK_SIZE);
    CSC(cudaDeviceSynchronize());

    CSC(cudaFree(scanSum));
}

__global__ void hist(int *dev_res, int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += offsetX) {
        atomicAdd(dev_res + data[i], 1);
    }
}

__global__ void initRes(int *dev_res) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = gridDim.x * blockDim.x;

    for (int i = idx; i < MAX_LIMIT; i += offsetX) {
        dev_res[i] = 0;
    }
}

__global__ void finalRes(int *data, int *dev_res, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = gridDim.x * blockDim.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetY = gridDim.y * blockDim.y;

    for (int i = idx; i < MAX_LIMIT - 1; i += offsetX) {
        int k = dev_res[i];

        for (int j = idy + k; j < dev_res[i + 1]; j += offsetY) {
            data[j] = i;
        }
    }

    for (int i = idx + dev_res[MAX_LIMIT - 1]; i < size; i += offsetX) {
        data[i] = MAX_LIMIT - 1;
    }
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n;
    std::cin.read(reinterpret_cast<char*>(&n), sizeof(n));

    std::vector<int> data(n);
    std::cin.read(reinterpret_cast<char*>(data.data()), sizeof(int) * n);

    int *dev_data;
    int *dev_res;

    CSC(cudaMalloc(&dev_data, sizeof(int) * n));
    CSC(cudaMalloc(&dev_res, sizeof(int) * MAX_LIMIT));
    CSC(cudaMemcpy(dev_data, data.data(), sizeof(int) * n, cudaMemcpyHostToDevice));

    initRes<<<1024, 1024>>>(dev_res);
    CSC(cudaDeviceSynchronize());

    hist<<<1024, 1024>>>(dev_res, dev_data, n);
    CSC(cudaDeviceSynchronize());

    scan(dev_res, MAX_LIMIT - 1);

    finalRes<<<dim3(32, 32), dim3(32, 32)>>>(dev_data, dev_res, n);
    CSC(cudaDeviceSynchronize());

    CSC(cudaMemcpy(data.data(), dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost));

    std::cout.write(reinterpret_cast<const char*>(data.data()), sizeof(int) * n);

    CSC(cudaFree(dev_data));
    CSC(cudaFree(dev_res));

    return 0;
}
