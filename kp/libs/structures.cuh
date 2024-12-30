#ifndef __STRUCTURES_CUH__
#define __STRUCTURES_CUH__
#include <string>


using namespace std;

struct Vector3 {   
    float x;
    float y;
    float z;
};

struct Matrix {
    Vector3 row1;
    Vector3 row2;
    Vector3 row3;
};

struct Ray {
    Vector3 origin;
    Vector3 direction;
};

struct Texture {
    Vector3 color;
    float diffussion;
    float reflection;
    float refraction;
};

struct Triangle {
    Vector3 vertex1;
    Vector3 vertex2;
    Vector3 vertex3;
    int textureId;
};

struct LightPoint {
    Vector3 pos;
    Vector3 color;
    float power;
};

struct Recursion{
    Ray ray;
    float power;
    int x;
    int y;
};

struct RenderTexture {
    int triangleId1;
    int triangleId2;
    cudaTextureObject_t memWrapper;
    int* memData; 
    int width;
    int height;
};

struct Window{
    int width;
    int height;
    int sqrtScale;
    int* picture;
};

struct InputData {
    int frameCount;
    string modulePath;
    int imgWidth, imgHeight;
    float angleView;
    float rcInit, zcInit, fcInit, arcR, arcZ, wcR, wcZ, wcF, pcR, pcZ;
    float rnInit, znInit, fnInit, anR, anZ, wnR, wnZ, wnF, pnR, pnZ;
    Vector3 objCenter, objColor;
    float objRadius;
    float reflectCoeff, refractCoeff;
    int lightLineCount;
    Vector3 cornerA, cornerB, cornerC, cornerD;
    string textureFilePath;
    Vector3 textureColor;
    float textureReflect;
    int lightSources;
    int recursionDepth, pixelSampling;
};

struct TextureFunctor {
    int operator() (const Texture& tex) const {
        return 103245 * tex.color.x + 87599 * tex.color.y + 
            73245 * tex.color.z + 23412 * tex.diffussion + 
            3564653 * tex.refraction + 453452 * tex.reflection;
    }
};

struct reqSearch {
    float distance;
    int id;
};

#endif
