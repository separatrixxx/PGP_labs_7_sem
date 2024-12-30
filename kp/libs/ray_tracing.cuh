#ifndef __RAY_TRACING_CUH__
#define __RAY_TRACING_CUH__

#include "helpers.cuh"
#include "structures.cuh"
#include <vector>
#include <algorithm>


using namespace std;

__global__ void binCompute(int* bins, const Recursion* data, int dataSize, float threshold) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = globalIdx; idx < dataSize; idx += stride) {
        bins[idx] = (data[idx].power < threshold) ? 0 : 1;
    }
}

__device__ void scanBlock(int threadId, int* input, int* sharedTmp, int blockSize) {
    int idxA = threadId;
    int idxB = threadId + (blockSize >> 1);
    int offsetA = (idxA >> 16u) + (idxA >> 8u);
    int offsetB = (idxB >> 16u) + (idxB >> 8u);
    int stride = 1;

    sharedTmp[idxA + offsetA] = input[idxA];
    sharedTmp[idxB + offsetB] = input[idxB];

    for (int d = blockSize >> 1; d > 0; d >>= 1, stride <<= 1) {
        __syncthreads();
        if (threadId < d) {
            int idx1 = stride * ((threadId << 1) + 1) - 1;
            int idx2 = stride * ((threadId << 1) + 2) - 1;

            idx1 += (idx1 >> 16u) + (idx1 >> 8u);
            idx2 += (idx2 >> 16u) + (idx2 >> 8u);

            sharedTmp[idx2] += sharedTmp[idx1];
        }
    }

    if (threadId == 0) {
        sharedTmp[blockSize - 1 + ((blockSize - 1) >> 16u) + ((blockSize - 1) >> 8u)] = 0;
    }

    stride >>= 1;

    for (int d = 1; d < blockSize; d <<= 1, stride >>= 1) {
        __syncthreads();
        if (threadId < d) {
            int idx1 = stride * ((threadId << 1) + 1) - 1;
            int idx2 = stride * ((threadId << 1) + 2) - 1;

            idx1 += (idx1 >> 16u) + (idx1 >> 8u);
            idx2 += (idx2 >> 16u) + (idx2 >> 8u);

            int temp = sharedTmp[idx1];
            sharedTmp[idx1] = sharedTmp[idx2];
            sharedTmp[idx2] += temp;
        }
    }

    __syncthreads();

    input[idxA] += sharedTmp[idxA + offsetA];
    input[idxB] += sharedTmp[idxB + offsetB];
}

__global__ void scanStep(int* data, int dataSize) {
    __shared__ int sharedMem[1024];

    int threadIdxLocal = threadIdx.x;
    int offsetBase = blockIdx.x * 1024;
    int stride = gridDim.x * 1024;

    for (int offset = offsetBase; offset < dataSize; offset += stride) {
        scanBlock(threadIdxLocal, &data[offset], sharedMem, 1024);
    }
}

__global__ void addSums(int* data, const int* partialSums, int count) {
    int blockStart = blockIdx.x + 1;
    int blockStride = gridDim.x;
    int threadId = threadIdx.x;

    for (int i = blockStart; i < count; i += blockStride) {
        int addition = partialSums[i - 1];
        int baseIdx = i * 1024;

        data[baseIdx + (threadId << 1)] += addition;
        data[baseIdx + (threadId << 1) + 1] += addition;
    }
}

__global__ void binsSort(Recursion* rays, const Recursion* source, const int* bins, const int* partialSums, int size) {
    int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = globalIdx; idx < size; idx += stride) {
        if (bins[idx]) {
            rays[partialSums[idx] - 1] = source[idx];
        }
    }
}

int offsetCompute(int size, int base) {
    return ((size + base - 1) / base) * base;
}

void scan(int* data, int dataSize) {
    int blockCount = dataSize / 1024;
    int gridSize = blockCount > 1024 ? 1024 : blockCount;

    scanStep<<<gridSize, 512>>>(data, dataSize);
    CSC(cudaGetLastError());
    cudaDeviceSynchronize();

    if (blockCount == 1) {
        return;
    }

    int* partialSums = nullptr;
    int allocSize = offsetCompute(blockCount, 1024);

    CSC(cudaMalloc(&partialSums, allocSize * sizeof(int)));
    CSC(cudaMemcpy2D(partialSums, sizeof(int), data + 1023, 1024 * sizeof(int), sizeof(int), blockCount, cudaMemcpyDeviceToDevice));

    scan(partialSums, allocSize);

    addSums<<<gridSize, 512>>>(data, partialSums, blockCount);

    CSC(cudaGetLastError());
    CSC(cudaFree(partialSums));
}

int gpuCleanRays(Recursion* dRays, const int size) {
    if (!size) {
        return 0;
    }

    int counter = 0;
    int* bins;
    int* sums;
    int sizeHelper = offsetCompute(size, 1024u);

    Recursion* copy;
    
    CSC(cudaMalloc((void**) &bins, size * sizeof(int)));
    CSC(cudaMalloc((void**) &sums, sizeHelper * sizeof(int)));
    CSC(cudaMalloc((void**) &copy, size * sizeof(Recursion)));
    CSC(cudaMemset(bins, 0, size * sizeof(int))); 

    binCompute<<<1024u, 512u>>>(bins, dRays, size, 0.005);

    CSC(cudaGetLastError());
    cudaDeviceSynchronize();
    CSC(cudaMemcpy(sums, bins, sizeof(int) * size, cudaMemcpyDeviceToDevice));
    scan(sums, sizeHelper);
    CSC(cudaMemcpy(&counter, sums + (size - 1), sizeof(int), cudaMemcpyDeviceToHost));
    CSC(cudaMemcpy(copy, dRays, sizeof(Recursion) * size, cudaMemcpyDeviceToDevice));

    binsSort<<<1024u, 512u>>>(dRays, copy, bins, sums, size);

    CSC(cudaGetLastError());
    CSC(cudaFree(bins));
    CSC(cudaFree(sums));
    CSC(cudaFree(copy));

    return counter;
}

int cpuCleanRays(vector<Recursion>& rays, const int size) {
    std::sort(begin(rays), end(rays));

    for (int i = 0; i < size; ++i) {
        if (rays[i].power < 0.005) {
            return i + 1;
        }
    }

    return size;
}

__device__ __host__ float triangleIntersection(const Ray& ray, const Triangle& tri) {
    Vector3 edge1 = tri.vertex2 - tri.vertex1;
    Vector3 edge2 = tri.vertex3 - tri.vertex1;
    Vector3 pvec = cross(ray.direction, edge2);
    float det = dot(pvec, edge1);

    if (fabs(det) < 4e-6f) {
        return -1.0f;
    }

    Vector3 tvec = ray.origin - tri.vertex1;
    float u = dot(pvec, tvec) / det;

    if (u < 0.0f || u > 1.0f) {
        return -1.0f;
    }

    Vector3 qvec = cross(tvec, edge1);
    float v = dot(qvec, ray.direction) / det;

    if (v >= 0.0f && u + v <= 1.0f) {
        return dot(qvec, edge2) / det;
    }

    return -1.0f;
}

__device__ __host__ bool intersectionSearch(reqSearch& result, const Ray& ray, const Triangle* triangleArray, int triangleCount) {
    result.distance = 1e+32f;
    result.id = triangleCount;

    for (int idx = 0; idx < triangleCount; ++idx) {
        const Triangle& tri = triangleArray[idx];

        float t = triangleIntersection(ray, tri);

        if (t >= 4e-6f && t < result.distance) {
            result.distance = t;
            result.id = idx;
        }
    }

    return result.id < triangleCount;
}

__device__ __host__ Vector3 radioCompute(const Ray& ray, const Texture* textures, const Triangle* triangleArray, int triangleCount, int ignoreId, float maxDist) {
    Vector3 attenuation = {1.0f, 1.0f, 1.0f};

    for (int idx = 0; idx < triangleCount; ++idx) {
        const Triangle& tri = triangleArray[idx];

        float t = triangleIntersection(ray, tri);

        if (t >= 4e-6f && t < maxDist && idx != ignoreId) {
            const Texture& tex = textures[tri.textureId];
            attenuation *= tex.color;
            attenuation *= tex.refraction;
        }
    }

    return attenuation;
}

__device__ __host__ Vector3 reflect(const Vector3& direction, const Vector3& surfaceNormal) {
    Vector3 reflection = surfaceNormal * (2.0f * dot(surfaceNormal, direction));

    return direction - reflection;
}

__device__ __host__ Vector3 normal(const Vector3& direction, const Vector3& surfaceNormal) {
    return dot(direction, surfaceNormal) > 0.0f ? -surfaceNormal : surfaceNormal;
}

__device__ __host__ bool isTex(const RenderTexture& floorTexture, int triId) {
    return (triId == floorTexture.triangleId1) || (triId == floorTexture.triangleId2);
}

template<int Backend>
Vector3 intersectionTex(Ray& ray, const RenderTexture& floor, const Triangle* trianglesArr);

template<>
__device__ Vector3 intersectionTex<1>(Ray& ray, const RenderTexture& floorTexture, const Triangle* triangleArray) {
    const Triangle& tri = triangleArray[floorTexture.triangleId1];

    Vector3 edge1 = tri.vertex2 - tri.vertex1;
    Vector3 edge2 = tri.vertex3 - tri.vertex1;
    Vector3 pvec = cross(ray.direction, edge2);
    Vector3 tvec = ray.origin - tri.vertex1;
    Vector3 qvec = cross(tvec, edge1);

    float u = dot(pvec, tvec) / dot(pvec, edge1);
    float v = dot(qvec, ray.direction) / dot(pvec, edge1);

    float4 texel = tex2D<float4>(floorTexture.memWrapper, v, u);

    return {texel.x, texel.y, texel.z};
}

template<>
__host__ __device__ Vector3 intersectionTex<0>(Ray& ray, const RenderTexture& floorTexture, const Triangle* triangleArray) {
    const Triangle& tri = triangleArray[floorTexture.triangleId1];

    Vector3 edge1 = tri.vertex2 - tri.vertex1;
    Vector3 edge2 = tri.vertex3 - tri.vertex1;
    Vector3 pvec = cross(ray.direction, edge2);
    Vector3 tvec = ray.origin - tri.vertex1;
    Vector3 qvec = cross(tvec, edge1);

    float u = dot(pvec, tvec) / dot(pvec, edge1);
    float v = dot(qvec, ray.direction) / dot(pvec, edge1);

    int texCoordX = floorTexture.width * v;
    int texCoordY = floorTexture.height * u;

    return uintToFloat(floorTexture.memData[texCoordY * floorTexture.width + texCoordX]);
}

__device__ __host__ bool isLightDark(const Vector3& lightVector, const Vector3& surfaceNormal) {
    return dot(lightVector, surfaceNormal) < 4e-6f;
}

__device__ __host__ Vector3 phShading(const Vector3& surfaceNormal, const Vector3& viewDirection, const Vector3& lightDirection, const LightPoint& lightSource, const Texture& material, const Vector3& attenuation) {
    float diffusiveFactor = material.diffussion * dot(surfaceNormal, lightDirection);
    float specularFactor = material.reflection * fmaxf(0.0f, powf(dot(reflect(-lightDirection, surfaceNormal), viewDirection), 64));

    Vector3 shadingColor = lightSource.color * material.color * attenuation;

    return shadingColor * lightSource.power * (diffusiveFactor + specularFactor);
}

template<int Backend>
void atomicAdd(Vector3* adress, const Vector3& val);

template<> __device__ void atomicAdd<1>(Vector3* adress, const Vector3& val) {
    atomicAdd(&adress->x, val.x);
    atomicAdd(&adress->y, val.y);
    atomicAdd(&adress->z, val.z);
}

template<> __host__ __device__ void atomicAdd<0>(Vector3* adress, const Vector3& val) {
    *adress += val;
}

template<int Backend>
__device__ __host__ void rayTrace(int startIdx, int stepIdx, Recursion* raysArr, int count, Vector3* img, Texture* sceneTex, RenderTexture floor, Triangle* trianglesArr, int trianglesCount, LightPoint* lightsArr, int lightsCount, int w, int h) {
    for (int infoId = startIdx; infoId < count; infoId += stepIdx) {
        Recursion& rayInfoCurr = raysArr[infoId];

        reqSearch intersectedData;
    
        if (not intersectionSearch(intersectedData, rayInfoCurr.ray, trianglesArr, trianglesCount)) {
            raysArr[infoId].power = 0.0f;
            raysArr[count + infoId].power = 0.0f;
            
            continue;
        }

        Vector3 triangleNorm;
        Texture texProp;

        int triangleId = intersectedData.id;
        Triangle triangleFound = trianglesArr[intersectedData.id];
        triangleNorm = normal(rayInfoCurr.ray.direction, normTriangle(triangleFound));
        texProp = sceneTex[triangleFound.textureId];

        if (isTex(floor, triangleId)) {
            texProp.color *= intersectionTex<Backend>(rayInfoCurr.ray, floor, trianglesArr);
        }

        Vector3 intersectionPos = rayInfoCurr.ray.origin + rayInfoCurr.ray.direction * intersectedData.distance;
        Vector3 lightsShine = {0.0, 0.0, 0.0};

        for (int pid = 0; pid < lightsCount; ++pid) {
            LightPoint lightSrc = lightsArr[pid];
            Vector3 vectorSrc = lightSrc.pos - intersectionPos;
            Ray lightsRaySrc = Ray{intersectionPos, norm(vectorSrc)};

            if (isLightDark(lightsRaySrc.direction, triangleNorm)) {
                continue;
            }

            Vector3 shineLoss = radioCompute(lightsRaySrc, sceneTex, trianglesArr, trianglesCount, intersectedData.id, abs(vectorSrc));
            lightsShine += phShading(triangleNorm, -rayInfoCurr.ray.direction, lightsRaySrc.direction, lightSrc, texProp, shineLoss);
        }

        lightsShine *= rayInfoCurr.power;

        atomicAdd<Backend>(&img[rayInfoCurr.y * w + rayInfoCurr.x], lightsShine); 

        Recursion reflectedRayInfo = rayInfoCurr;
        reflectedRayInfo.power *= texProp.reflection;
        reflectedRayInfo.ray.origin = intersectionPos;
        reflectedRayInfo.ray.direction = norm(reflect(rayInfoCurr.ray.direction, triangleNorm));


        rayInfoCurr.ray.origin = intersectionPos;
        rayInfoCurr.power *= texProp.refraction;

        raysArr[infoId] = rayInfoCurr;
        raysArr[count + infoId] = reflectedRayInfo;
    }
}

__global__ void gpuRayTrace(Recursion* raysArr, int count, Vector3* img, Texture* sceneTex, RenderTexture floor, Triangle* trianglesArr, int trianglesCount, LightPoint* lightsArr, int lightsCount, int w, int h) {
    int step = blockDim.x * gridDim.x;
    int start = threadIdx.x + blockIdx.x * blockDim.x;

    rayTrace<1>(start, step, raysArr, count, img, sceneTex, floor, trianglesArr, trianglesCount, lightsArr, lightsCount, w, h);
}

__host__ void cpuRayTrace(Recursion* raysArr, int count, Vector3* img, Texture* sceneTex, RenderTexture floor, Triangle* trianglesArr, int trianglesCount, LightPoint* lightsArr, int lightsCount, int w, int h) {
    rayTrace<0>(0, 1, raysArr, count, img, sceneTex, floor, trianglesArr, trianglesCount, lightsArr,  lightsCount, w, h);
}

__device__ __host__ void initRays(int startIdx, int stepIdx, Recursion* raysArr, Vector3* img, Matrix matrixT, Vector3 pos, float z, int w, int h) {
    int count = w * h;
    
    float d_w = 2.0 / (w - 1.0);
    float d_h = 2.0 / (h - 1.0);
    float h_div_w = static_cast<float>(h) / static_cast<float>(w);

    for (int idx = startIdx; idx < count; idx += stepIdx) {
        int i = idx % w;
        int j = idx / w;

        Recursion activeRay;

        activeRay.x = i; activeRay.y = j;
        activeRay.power = 1.0;
        activeRay.ray.origin = pos;

        Vector3 coords = {-1.0f + d_w * i, (-1.0f + d_h * j) * h_div_w, z};

        activeRay.ray.direction = norm(mult(matrixT, coords));

        raysArr[idx] = activeRay;
        img[idx] = {0.0, 0.0, 0.0};
    }
}

__global__ void gpuInitRays(Recursion* raysArr, Vector3* img, Matrix matrixT, Vector3 pos, float z, int w, int h) {
    int step = blockDim.x * gridDim.x;
    int start = threadIdx.x + blockIdx.x * blockDim.x;
    
    initRays(start, step, raysArr, img, matrixT, pos, z, w, h);
}

__host__ void cpuInitRays(Recursion* raysArr, Vector3* img, Matrix matrixT, Vector3 pos, float z, int w, int h) {    
    initRays(0, 1, raysArr, img, matrixT, pos, z, w, h);
}

#endif
