#include <stdio.h>
#include <iostream>
#include <iomanip>


__global__ void reverseVector(double *arr, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = gridDim.x * blockDim.x;

  for (int i = idx; i < n / 2; i += offset) {
    int oppIdx = n - 1 - i;
    double temp = arr[i];
    arr[i] = arr[oppIdx];
    arr[oppIdx] = temp;
  }
}

int main() {
  int n;
  std::cin >> n;

  if (n <= 0 || n >= (1 << 25)) {
      fprintf(stderr, "ERROR: Неправильное n\n");

      return 0;
  }

  double* arr = (double*)malloc(sizeof(double) * n);

  if (arr == nullptr) {
    fprintf(stderr, "ERROR: Не удалось выделить память для массива\n");

    return 0;
  }

  for (int i = 0; i < n; i++) {
    if (!(std::cin >> arr[i])) {
      fprintf(stderr, "ERROR: Неверный ввод\n");
      free(arr);

      return 0;
    }
  }


  double *dev_arr;

  if (cudaMalloc(&dev_arr, sizeof(double) * n) != cudaSuccess) {
    fprintf(stderr, "ERROR: Не удалось выделить память CUDA\n");
    free(arr);

    return 0;
  }

  if (cudaMemcpy(dev_arr, arr, sizeof(double) * n, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "ERROR: Копирование памяти CUDA не удалось\n");
    free(arr);
    cudaFree(dev_arr);

    return 0;
  }

  reverseVector<<<1024, 1024>>>(dev_arr, n);

  if (cudaMemcpy(arr, dev_arr, sizeof(double) * n, cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "ERROR: Не удалось выполнить обратное копирование памяти CUDA\n");
    free(arr);
    cudaFree(dev_arr);

    return 0;
  }

  std::cout << std::scientific << std::setprecision(10);
    
  for (int i = 0; i < n - 1; i++) {
    std::cout << arr[i] << " ";
  }
  std::cout << arr[n - 1];

  std::cout << '\n';

  free(arr);
  cudaFree(dev_arr);

  return 0;
}
