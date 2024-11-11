#include <iostream>
#include <iomanip>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <cmath>


#define CSC(call)                                  \
do {                                               \
    cudaError_t res = call;                        \
    if (res != cudaSuccess) {                      \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(res)); \
        exit(0);                                   \
    }                                              \
} while(0)

__global__ void swap(double* matrix, double* identity, int n, int curr_row, int p_row) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += offset) {
        double temp = matrix[i * n + curr_row];
        matrix[i * n + curr_row] = matrix[i * n + p_row];
        matrix[i * n + p_row] = temp;

        temp = identity[i * n + curr_row];
        identity[i * n + curr_row] = identity[i * n + p_row];
        identity[i * n + p_row] = temp;
    }
}

__global__ void norm(double* matrix, double* identity, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = gridDim.x * blockDim.x;
    int offsety = gridDim.y * blockDim.y;

    for (int i = idx; i < n; i += offsetx) {
        for (int j = idy; j < n; j += offsety) {
            identity[j * n + i] /= matrix[i * n + i];
        }
    }
}

__global__ void update(double* matrix, double* identity, int n, int curr_row) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = gridDim.x * blockDim.x;
    int offsety = gridDim.y * blockDim.y;

    for (int i = curr_row + 1 + idx; i < n; i += offsetx) {
        double factor = matrix[curr_row * n + i] / matrix[curr_row * n + curr_row];

        for (int j = curr_row + 1 + idy; j < n; j += offsety) {
            matrix[j * n + i] -= factor * matrix[j * n + curr_row];
        }

        for (int j = idy; j < n; j += offsety) {
            identity[j * n + i] -= factor * identity[j * n + curr_row];
        }
    }
}

__global__ void back(double* matrix, double* identity, int n, int curr_row) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = gridDim.x * blockDim.x;
    int offsety = gridDim.y * blockDim.y;

    for (int i = curr_row - 1 - idx; i >= 0; i -= offsetx) {
        double factor = matrix[curr_row * n + i] / matrix[curr_row * n + curr_row];

        for (int j = curr_row + 1 + idy; j < n; j += offsety) {
            matrix[j * n + i] -= factor * matrix[j * n + curr_row];
        }

        for (int j = idy; j < n; j += offsety) {
            identity[j * n + i] -= factor * identity[j * n + curr_row];
        }
    }
}

struct Comparator {
    __host__ __device__ bool operator()(double a, double b) {
        return fabs(a) < fabs(b);
    }
};

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);

    int n;
    std::cin >> n;

    double* matrix = new double[n * n];
    double* identity = new double[n * n];

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cin >> matrix[j * n + i];
            identity[j * n + i] = (i == j) ? 1.0 : 0.0;
        }
    }

    double* dev_matrix;
    double* dev_identity;
    CSC(cudaMalloc(&dev_matrix, n * n * sizeof(double)));
    CSC(cudaMalloc(&dev_identity, n * n * sizeof(double)));
    CSC(cudaMemcpy(dev_matrix, matrix, n * n * sizeof(double), cudaMemcpyHostToDevice));
    CSC(cudaMemcpy(dev_identity, identity, n * n * sizeof(double), cudaMemcpyHostToDevice));

    thrust::device_ptr<double> p_matrix = thrust::device_pointer_cast(dev_matrix);
    Comparator comp;

    for (int i = 0; i < n - 1; i++) {
        thrust::device_ptr<double> max_el = thrust::max_element(p_matrix + i * n + i, p_matrix + (i + 1) * n, comp);
        int p_row = max_el - (p_matrix + i * n);

        if (p_row != i) {
            swap<<<256, 256>>>(dev_matrix, dev_identity, n, i, p_row);
            CSC(cudaDeviceSynchronize());
        }

        update<<<dim3(32, 32), dim3(32, 32)>>>(dev_matrix, dev_identity, n, i);
        CSC(cudaDeviceSynchronize());
    }

    for (int i = n - 1; i > 0; i--) {
        back<<<dim3(32, 32), dim3(32, 32)>>>(dev_matrix, dev_identity, n, i);
        CSC(cudaDeviceSynchronize());
    }

    norm<<<dim3(32, 32), dim3(32, 32)>>>(dev_matrix, dev_identity, n);
    CSC(cudaDeviceSynchronize());

    CSC(cudaMemcpy(identity, dev_identity, n * n * sizeof(double), cudaMemcpyDeviceToHost));

    std::cout << std::scientific << std::setprecision(10);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << identity[j * n + i];
            std::cout << " ";
        }
        std::cout << "\n";
    }

    free(matrix);
    free(identity);
    CSC(cudaFree(dev_matrix));
    CSC(cudaFree(dev_identity));

    return 0;
}
