#ifndef __HELPERS_CUH__
#define __HELPERS_CUH__
#include <stdexcept>
#include <string>

#include "structures.cuh"


using namespace std;

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

inline void fileWrite(int* dataD, int width, int height, const string& pathMod, int fileNum, bool isGpu) {
    int size = width * height;
    char buff[256];

    sprintf(buff, pathMod.c_str(), fileNum);

    ofstream ofile(buff, ios::binary | ios::out);

    if (!ofile) {
        throw runtime_error("File open error!");
    }

    int* dataH = nullptr;
    if (isGpu) {
        dataH = new int[size];

        cudaMemcpy(dataH, dataD, size * sizeof(int), cudaMemcpyDeviceToHost);
    } else {
        dataH = dataD;
    }

    ofile.write(reinterpret_cast<char*>(&width), sizeof(int));
    ofile.write(reinterpret_cast<char*>(&height), sizeof(int));
    ofile.write(reinterpret_cast<char*>(dataH), size * sizeof(int));
    ofile.close();

    if (isGpu) {
        delete[] dataH;
    }
}

int getTexId(const Texture& mat, unordered_map<Texture, int, TextureFunctor>& textureMap) {
    if (textureMap.count(mat)) {
        return textureMap[mat];
    }

    int size = textureMap.size();
    return textureMap[mat] = size;
}

void saveTex(std::vector<Texture>& vec, unordered_map<Texture, int, TextureFunctor>& textureMap) {
    vec.resize(textureMap.size());
    for (const auto& it : textureMap) {
        vec[it.second] = it.first;
    }
}

bool operator<(const Recursion& a, const Recursion& b) {
    return a.power > b.power;
}

bool operator==(const Texture& a, const Texture& b) {
    return a.color.x == b.color.x && a.color.y == b.color.y && a.color.z == b.color.z &&
        a.diffussion == b.diffussion && a.reflection == b.reflection && a.refraction == b.refraction;
}

__host__ __device__ Vector3 operator*(const Vector3& vec, float scalar) {
    Vector3 res = {vec.x * scalar, vec.y * scalar, vec.z * scalar};

    return res;
}

__host__ __device__ Vector3 operator*(const Vector3& vecA, const Vector3& vecB) {
    Vector3 res = {vecA.x * vecB.x, vecA.y * vecB.y, vecA.z * vecB.z};

    return res;
}

__host__ __device__ Vector3 operator/(const Vector3& vec, float scalar) {
    Vector3 res = {vec.x / scalar, vec.y / scalar, vec.z / scalar};

    return res;
}

__host__ __device__ Vector3 operator+(const Vector3& vecA, const Vector3& vecB) {
    Vector3 res = {vecA.x + vecB.x, vecA.y + vecB.y, vecA.z + vecB.z};

    return res;
}

__host__ __device__ Vector3 operator-(const Vector3& vecA, const Vector3& vecB) {
    Vector3 res = {vecA.x - vecB.x, vecA.y - vecB.y, vecA.z - vecB.z};

    return res;
}

__host__ __device__ Vector3 operator-(const Vector3& vec) {
    Vector3 res = {-vec.x, -vec.y, -vec.z};

    return res;
}

__host__ __device__ Vector3& operator+=(Vector3& vecA, const Vector3& vecB) {
    vecA.x += vecB.x;
    vecA.y += vecB.y;
    vecA.z += vecB.z;

    return vecA;
}

__host__ __device__ Vector3& operator-=(Vector3& vecA, const Vector3& vecB) {
    vecA.x -= vecB.x;
    vecA.y -= vecB.y;
    vecA.z -= vecB.z;

    return vecA;
}

__host__ __device__ Vector3& operator/=(Vector3& vec, float scalar) {
    vec.x /= scalar;
    vec.y /= scalar;
    vec.z /= scalar;

    return vec;
}

__host__ __device__ Vector3& operator*=(Vector3& vec, float scalar) {
    vec.x *= scalar;
    vec.y *= scalar;
    vec.z *= scalar;

    return vec;
}

__host__ __device__ Vector3& operator*=(Vector3& vecA, const Vector3& vecB) {
    vecA.x *= vecB.x;
    vecA.y *= vecB.y;
    vecA.z *= vecB.z;

    return vecA;
}

__host__ __device__ float dot(const Vector3& vecA, const Vector3& vecB) {
    float res = vecA.x * vecB.x + vecA.y * vecB.y + vecA.z * vecB.z;

    return res;
}

__host__ __device__ float abs(const Vector3& vec) {
    float magnitude = sqrt(dot(vec, vec));

    return magnitude;
}

__host__ __device__ float sign(const float value) {
    float res = (value > 0.0f) - (value < 0.0f);

    return res;
}

__host__ __device__ Vector3 norm(const Vector3& vec) {
    float magnitude = abs(vec);
    Vector3 normalizedVec = vec / magnitude;

    return normalizedVec;
}

__host__ __device__ Vector3 cross(const Vector3& vecA, const Vector3& vecB) {
    Vector3 res = {vecA.y * vecB.z - vecA.z * vecB.y, vecA.z * vecB.x - vecA.x * vecB.z, vecA.x * vecB.y - vecA.y * vecB.x};

    return res;
}

__host__ __device__ Vector3 mult(const Matrix& mat, const Vector3& vec) {
    Vector3 res = {mat.row1.x * vec.x + mat.row2.x * vec.y + mat.row3.x * vec.z,
        mat.row1.y * vec.x + mat.row2.y * vec.y + mat.row3.y * vec.z,
        mat.row1.z * vec.x + mat.row2.z * vec.y + mat.row3.z * vec.z};

    return res;
}

__host__ __device__ Vector3 normTriangle(const Triangle& tri) {
    Vector3 crossProd = cross(tri.vertex3 - tri.vertex1, tri.vertex2 - tri.vertex1);
    Vector3 normalizedVec = norm(crossProd);

    return normalizedVec;
}

__host__ __device__ int floatToUint(const Vector3& color) {
    int res = 0;
    res |= static_cast<int>(round(255.0 * color.x)) << 0;
    res |= static_cast<int>(round(255.0 * color.y)) << 8;
    res |= static_cast<int>(round(255.0 * color.z)) << 16;

    return res;
}

__host__ __device__ Vector3 uintToFloat(const int color) {
    Vector3 res;
    res.x = ((color >> 0) & 255) / 255.0f;
    res.y = ((color >> 8) & 255) / 255.0f;
    res.z = ((color >> 16) & 255) / 255.0f;

    return res;
}

__device__ void atomicAdd(Vector3* address, const Vector3& value) {
    atomicAdd(&address->x, value.x);
    atomicAdd(&address->y, value.y);
    atomicAdd(&address->z, value.z);
}

#endif
