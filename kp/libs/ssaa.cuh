#ifndef __SSAA_CUH__
#define __SSAA_CUH__


__global__ void gpuSsaa(int* pic, const Vector3* image, int w, int h, int sqrtPerPix) {
    int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetX = blockDim.x * gridDim.x;
    int offsetY = blockDim.y * gridDim.y;

    for (int row = globalX; row < h; row += offsetX) {
        for (size_t col = globalY; col < w; col += offsetY) {
            int subImageStartRow = row * sqrtPerPix;
            int subImageStartCol = col * sqrtPerPix;

            Vector3 average = {0.0f, 0.0f, 0.0f};

            for (int y = subImageStartRow; y < subImageStartRow + sqrtPerPix; ++y) {
                for (size_t x = subImageStartCol; x < subImageStartCol + sqrtPerPix; ++x) {
                    average += image[y * w * sqrtPerPix + x];
                }
            }

            average /= static_cast<float>(sqrtPerPix * sqrtPerPix);

            pic[row * w + col] = floatToUint({
                fminf(average.x, 1.0f),
                fminf(average.y, 1.0f),
                fminf(average.z, 1.0f)
            });
        }
    }
}

__host__ void cpuSsaa(int* pic, const Vector3* image, int w, int h, int sqrtPerPix) {
    for (int row = 0; row < h; ++row) {
        for (size_t col = 0; col < w; ++col) {
            Vector3 avgColor = {0.0f, 0.0f, 0.0f};

            for (int y = row * sqrtPerPix; y < row * sqrtPerPix + sqrtPerPix; ++y) {
                for (int x = col * sqrtPerPix; x < col * sqrtPerPix + sqrtPerPix; ++x) {
                    avgColor += image[y * w * sqrtPerPix + x];
                }
            }

            avgColor /= static_cast<float>(sqrtPerPix * sqrtPerPix);

            pic[row * w + col] = floatToUint({
                fminf(avgColor.x, 1.0f),
                fminf(avgColor.y, 1.0f),
                fminf(avgColor.z, 1.0f)
            });
        }
    }
}

#endif
