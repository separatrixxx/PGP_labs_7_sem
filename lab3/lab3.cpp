#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <cfloat>


#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

__constant__ float3 avgj[32];

__global__ void kernel(uchar4* data, int w, int h, int nc) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offsetx = blockDim.x * gridDim.x;

    for (int i = idx; i < w * h; i += offsetx) {
        uchar4 ps = data[i];
        float3 p = make_float3(ps.x, ps.y, ps.z);

        float s, max_s = -FLT_MAX;
        int max_idx = 0;

        for (int j = 0; j < nc; j++) {
            float norm_avgj = sqrtf(avgj[j].x * avgj[j].x + avgj[j].y * avgj[j].y + avgj[j].z * avgj[j].z);
            float3 res = make_float3(avgj[j].x / norm_avgj, avgj[j].y / norm_avgj, avgj[j].z / norm_avgj);

            s = p.x * res.x + p.y * res.y + p.z * res.z;

            if (s > max_s) {
                max_s = s;
                max_idx = j;
            }
        }

        data[i].w = (unsigned char)max_idx;
    }
}


int main() {
    std::string in_file;
    std::string out_file;

    std::cin >> in_file;
    std::cin >> out_file;

    int w, h, nc, px, py;
    long long np;

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

    std::cin >> nc;

    float3 dev_avgj[nc];
    
    for (int i = 0; i < nc; i++) {
        std::cin >> np;

        dev_avgj[i] = make_float3(0.0f, 0.0f, 0.0f);

        for (long long j = 0; j < np; j++) {
            std::cin >> px >> py;
            uchar4 ps = data[py * w + px];

            dev_avgj[i].x += ps.x;
            dev_avgj[i].y += ps.y;
            dev_avgj[i].z += ps.z;
        }

        dev_avgj[i].x /= np;
        dev_avgj[i].y /= np;
        dev_avgj[i].z /= np;
    }

    CSC(cudaMemcpyToSymbol(avgj, dev_avgj, sizeof(float3) * nc, 0, cudaMemcpyHostToDevice));

    uchar4 *dev_out;
	CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));
    CSC(cudaMemcpy(dev_out, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

    kernel<<<1024, 1024>>>(dev_out, w, h, nc);

    CSC(cudaDeviceSynchronize());
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

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
