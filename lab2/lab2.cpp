#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>


#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

__global__ void kernel(cudaTextureObject_t tex, uchar4 *out, int w, int h) {   
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
    
    int Mx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    int My[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

    for(int y = idy; y < h; y += offsety) {
		for(int x = idx; x < w; x += offsetx) {
            float gx = 0.0f;
            float gy = 0.0f;
            
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int ip = max(min(x + kx, w - 1), 0);
                    int jp = max(min(y + ky, h - 1), 0);

                    uchar4 p = tex2D<uchar4>(tex, ip, jp);

                    float intensity = 0.299f * p.x + 0.587f * p.y + 0.114f * p.z;

                    gx += intensity * Mx[ky + 1][kx + 1];
                    gy += intensity * My[ky + 1][kx + 1];
                }
            }

            float grad = sqrtf(gx * gx + gy * gy);
            grad = min(max(grad, 0.0f), 255.0f);

            out[y * w + x] = make_uchar4((unsigned char)grad, (unsigned char)grad, (unsigned char)grad, 255);
        }
    }
}

int main() {
    std::string in_file;
    std::string out_file;

    std::cin >> in_file;
    std::cin >> out_file;

    int w, h;
    FILE *fp = fopen(in_file.c_str(), "rb");

    if (fp == nullptr) {
        fprintf(stderr, "Failed to open input file\n");

        return 0;
    }

    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);
    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);

    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    cudaArray *arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));
    CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    cudaTextureObject_t tex = 0;
    CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    uchar4 *dev_out;
	CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

    kernel<<<dim3(16, 16), dim3(32, 32)>>>(tex, dev_out, w, h);

    CSC(cudaDeviceSynchronize());
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

	CSC(cudaDestroyTextureObject(tex));
	CSC(cudaFreeArray(arr));
	CSC(cudaFree(dev_out));

    fp = fopen(out_file.c_str(), "wb");

    if (fp == nullptr) {
        fprintf(stderr, "Failed to open output file\n");

        return 0;
    }

	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

    free(data);

    return 0;
}