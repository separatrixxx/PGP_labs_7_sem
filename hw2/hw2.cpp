#include <iostream>
#include <cmath>
#include <iomanip>


void bubbleSort(float* arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                float temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

int main() {
    int n;
    std::cin >> n;

    float* arr = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        std::cin >> arr[i];
    }

    bubbleSort(arr, n);

    std::cout << std::scientific << std::setprecision(6);

    for (int i = 0; i < n; i++) {
        std::cout << arr[i] << " ";
    }

    std::cout << '\n';

    free(arr);

    return 0;
}
