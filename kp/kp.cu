#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <math.h>
#include <cstdio>
#include <stdexcept>
#include <stdlib.h>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>
#include <unordered_map>

#include "libs/helpers.cuh"
#include "libs/structures.cuh"
#include "libs/ray_tracing.cuh"
#include "libs/ssaa.cuh"


using namespace std;
using namespace std::chrono;

unordered_map<Texture, int, TextureFunctor> textureTable;

class CommonFigure {
protected:
    using point = Vector3;
    using line = pair<int, int>;
    using edge = vector<int>;
    using color = Vector3;

public:
    CommonFigure(float radius, const point& centerPos, const Texture& glassTex, int lightsCount) 
    : radius(radius), centerCoords(centerPos), trigTex(glassTex), lightsCount(lightsCount) {
        lightsRadius *= radius;
    }

    void figureRender(vector<Triangle>& triangles, vector<LightPoint>& lightSrc) const{
        splitLine(triangles);
        splitEdge(triangles);
        lightsOn(lightSrc);
    }

protected:
    void figureGenerator() {
        vertexesGenerator();
        vertexesScale();
    }

    void vertexesScale() {
        for (auto& v : vertexes) {
            v *= radius;
        }
    }

    Triangle triangleEarth(const Triangle& triangleSingle) const{
        Triangle result = triangleSingle;
        
        result.vertex1 += centerCoords; 
        result.vertex2 += centerCoords; 
        result.vertex3 += centerCoords;
        
        return result;
    }

    virtual void vertexesGenerator() = 0;
    virtual void splitEdge(vector<Triangle>& triangles) const = 0;

    void lightsOn(vector<LightPoint>& lightSrc) const{
        lightSrc.push_back({centerCoords, lightsColor, lightsPower});
    }

    void splitLine(vector<Triangle>& triangles) const {
        int figTexId = getTexId(figureTex, textureTable);
        int lightsTexId = getTexId(lightsTex, textureTable);
    
        for (auto& pointsLine : lines) {
            Vector3 transY = norm(vertexes[pointsLine.first] + vertexes[pointsLine.second]);
            Vector3 transZ = norm(vertexes[pointsLine.second] - vertexes[pointsLine.first]);
            Vector3 transX = norm(cross(transY, transZ));
    
            float transformX = lightsRadius * figureCoeff;
            float len_of_y_transfrom = transformX / tan((M_PI - edgesAngle) / 2.0);
    
            transX *= transformX;
            transY *= len_of_y_transfrom;
    
            triangles.push_back(triangleEarth(Triangle{
                vertexes[pointsLine.first] + transY + transX,
                vertexes[pointsLine.first] + transY - transX,
                vertexes[pointsLine.second] + transY - transX,
                figTexId
            }));
    
            triangles.push_back(triangleEarth(Triangle{
                vertexes[pointsLine.second] + transY + transX,
                vertexes[pointsLine.second] + transY - transX,
                vertexes[pointsLine.first] + transY + transX,
                figTexId
            }));
    
            transY *= (1 - 5e-4);
            transX = norm(transX) * lightsRadius;
            transZ = norm(transZ) * lightsRadius;
    
            point startPos = vertexes[pointsLine.first];
            point endPos = vertexes[pointsLine.second];
    
            for (int i = 1; i <= lightsCount; ++i) {
                Vector3 startAdd = (endPos - startPos) * ((float)i) / ((float)(lightsCount + 1));
    
                point diodCenter = startPos + startAdd + transY;
    
                triangles.push_back(triangleEarth(Triangle{
                    diodCenter + transZ,
                    diodCenter - transX,
                    diodCenter + transX,
                    lightsTexId
                }));
            }
        }
    }    

    int lightsCount;
    float lightsRadius = 0.05f;
    float edgesAngle;
    const float radius;
    const float figureCoeff = 1.5f;
    const Texture figureTex = Texture{color{1.0, 0, 1.0}, 0.0f, 0, 0};
    const Texture lightsTex = Texture{color{1.0, 1.0, 1.0}, 1.0, 1.0, 0};
    const color lightsColor = color{1.0f, 1.0f, 1.0f};
    const float lightsPower = 0.7f;
    const point centerCoords;
    vector<line> lines;
    vector<edge> edges;
    vector<point> vertexes;
    Texture trigTex;
};

class Hexaeder : public CommonFigure {
public:
    Hexaeder(float figureRadius, const Vector3& centerPos, const Texture& glassTexerial, int lightsCount)
    : CommonFigure(figureRadius, centerPos, glassTexerial, lightsCount) {
        setFigure();
        figureGenerator();
    }

private:
    void vertexesGenerator() final {
        vertexes = {
            {-1.0f, -1.0f, -1.0f},
            {1.0f, -1.0f, -1.0f},
            {1.0f, 1.0f, -1.0f},
            {-1.0f, 1.0f, -1.0f},
            {-1.0f, -1.0f, 1.0f},
            {1.0f, -1.0f, 1.0f},
            {1.0f, 1.0f, 1.0f},
            {-1.0f, 1.0f, 1.0f}
        };
    }

    void splitEdge(vector<Triangle>& triangles) const final {
        int textId = getTexId(trigTex, textureTable);
    
        for (const auto& edgeIdx : edges) {
            Vector3 transform = {0.0f, 0.0f, 0.0f};
    
            for (auto v_idx : edgeIdx) {
                transform += vertexes[v_idx];
            }
    
            transform /= edgeIdx.size();
    
            float transformX = lightsRadius * figureCoeff;
            float transformLen = transformX / sin((M_PI - edgesAngle) / 2.0);
            transform = norm(transform) * transformLen;
    
            for (int i = 1; i < edgeIdx.size() - 1; i++) {
                triangles.push_back(triangleEarth(Triangle{
                    vertexes[edgeIdx[0]] + transform,
                    vertexes[edgeIdx[i]] + transform,
                    vertexes[edgeIdx[i + 1]] + transform,
                    textId
                }));
            }
        }
    }
    
    void setFigure() {
        lines = {
            {0, 1}, {1, 2}, {2, 3}, {3, 0},
            {4, 5}, {5, 6}, {6, 7}, {7, 4},
            {0, 4}, {1, 5}, {2, 6}, {3, 7}
        };

        edges = {
            {0, 1, 2, 3},
            {4, 5, 6, 7},
            {0, 1, 5, 4},
            {1, 2, 6, 5},
            {2, 3, 7, 6},
            {3, 0, 4, 7}
        };
    }
};

class Octaeder : public CommonFigure {
public:
    Octaeder(float figureRadius, const Vector3& centerPos, const Texture& glassTexerial, int lightsCount)
    : CommonFigure(figureRadius, centerPos, glassTexerial, lightsCount) {
        edgesAngle = acos(-1.0 / 3.0);
        setFigure();
        figureGenerator();
    }

private:
    void vertexesGenerator() final {
        vertexes = {
            {-sqrt(2.0f) / 2.0f, -sqrt(2.0f) / 2.0f, 0.0f},
            {sqrt(2.0f) / 2.0f, -sqrt(2.0f) / 2.0f, 0.0f},
            {sqrt(2.0f) / 2.0f, sqrt(2.0f) / 2.0f, 0.0f},
            {-sqrt(2.0f) / 2.0f, sqrt(2.0f) / 2.0f, 0.0f},
            {0.0f, 0.0f, -1.0f},
            {0.0f, 0.0f, 1.0f}
        };
    }

    void splitEdge(vector<Triangle>& triangles) const final {
        int textId = getTexId(trigTex, textureTable);
    
        for (const auto& edgeIdx : edges) {
            Vector3 transform = {0.0f, 0.0f, 0.0f};
    
            for (auto v_idx : edgeIdx) {
                transform += vertexes[v_idx];
            }
    
            transform /= edgeIdx.size();
    
            float transformX = lightsRadius * figureCoeff;
            float transformLen = transformX / sin((M_PI - edgesAngle) / 2.0);
            transform = norm(transform) * transformLen;
    
            triangles.push_back(triangleEarth(Triangle{
                vertexes[edgeIdx[0]] + transform,
                vertexes[edgeIdx[1]] + transform,
                vertexes[edgeIdx[2]] + transform,
                textId
            }));
        }
    }    

    void setFigure() {
        lines = {
            {0, 1}, {1, 2}, {2, 3}, {3, 0},
            {0, 4}, {1, 4}, {2, 4}, {3, 4},
            {0, 5}, {1, 5}, {2, 5}, {3, 5}
        };

        edges = {
            {0, 4, 1}, {1, 4, 2}, {2, 4, 3}, {3, 4, 0},
            {0, 5, 1}, {1, 5, 2}, {2, 5, 3}, {3, 5, 0}
        };
    }
};

class Icosaeder : public CommonFigure {
public:
    Icosaeder(float figureRadius, const Vector3& centerPos, const Texture& glassTexerial, int lightsCount)
    : CommonFigure(figureRadius, centerPos, glassTexerial, lightsCount) {
        setFigure();
        figureGenerator();
    }

private:
    void vertexesGenerator() final {
        vertexes.resize(12);

        const float phi = (1.0f + sqrt(5.0f)) / 2.0f;
        const float scale = 1.0f / sqrt(1.0f + phi * phi);
        const float a = scale;
        const float b = scale * phi;

        vertexes[0] = {-a, b, 0.0f};
        vertexes[1] = {a, b, 0.0f};
        vertexes[2] = {-a, -b, 0.0f};
        vertexes[3] = {a, -b, 0.0f};
        vertexes[4] = {0.0f, -a, b};
        vertexes[5] = {0.0f, a, b};
        vertexes[6] = {0.0f, -a, -b};
        vertexes[7] = {0.0f, a, -b};
        vertexes[8] = {b, 0.0f, -a};
        vertexes[9] = {b, 0.0f, a};
        vertexes[10] = {-b, 0.0f, -a};
        vertexes[11] = {-b, 0.0f, a};
    }

    void splitEdge(vector<Triangle>& triangles) const final {
        int textId = getTexId(trigTex, textureTable);
    
        for (const auto& edgeIdx : edges) {
            Vector3 transform = {0.0f, 0.0f, 0.0f};
            for (auto v_idx : edgeIdx) {
                transform += vertexes[v_idx];
            }
            transform /= edgeIdx.size();
    
            float transformX = lightsRadius * figureCoeff;
            float transformLen = transformX / sin((M_PI - edgesAngle) / 2.0);
            transform = norm(transform) * transformLen;
    
            triangles.push_back(triangleEarth(Triangle{
                vertexes[edgeIdx[0]] + transform,
                vertexes[edgeIdx[1]] + transform,
                vertexes[edgeIdx[2]] + transform,
                textId
            }));
        }
    }    

    void setFigure() {
        lines = {
            {0, 11}, {0, 5}, {0, 1}, {0, 7}, {0, 10},
            {1, 5}, {1, 9}, {1, 7},
            {2, 3}, {2, 4}, {2, 6}, {2, 10}, {2, 11},
            {3, 4}, {3, 9}, {3, 6},
            {4, 9}, {4, 11},
            {5, 9}, {5, 11},
            {6, 7}, {6, 10},
            {7, 9}, {7, 10}
        };

        edges = {
            {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
            {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 3},
            {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {6, 2, 10}, {6, 7, 3},
            {9, 5, 4}, {11, 2, 4}
        };
    }
};

class Camera{
public:
    Camera(
        int frames, float angle,
        float rc0, float zc0, float fc0, float Acr, float Acz,
        float wcr, float wcz, float wcf, float pcr, float pcz,
        float rn0, float zn0, float fn0, float Anr, float Anz,
        float wnr, float wnz, float wnf, float pnr, float pnz
    ):

    framesCount(frames), zView(1.0 / tan(angle * M_PI / 360.0)),
    timeStep(2.0*M_PI/frames), timeCurr(-timeStep),
    rc_0(rc0), zc_0(zc0), fc_0(fc0), Ac_r(Acr), Ac_z(Acz), 
    wc_r(wcr), wc_z(wcz), wc_f(wcf), pc_r(pcr), pc_z(pcz),
    rn_0(rn0), zn_0(zn0), fn_0(fn0), An_r(Anr), An_z(Anz), 
    wn_r(wnr), wn_z(wnz), wn_f(wnf), pn_r(pnr), pn_z(pnz) {}

    bool updatePos() {
        timeCurr += timeStep;
        getPos(timeCurr);

        return timeCurr < 2.0 * M_PI;
    }

    Matrix matrixView;
    Vector3 pc;
    float zView;
    int framesCount;

private:
    Vector3 decardCoordsConvert(float r, float z, float f) {
        return {r * cos(f), r * sin(f), z};
    }

    void getPos(float t) {
        pc = decardCoordsConvert(rc_0 + Ac_r * sin(wc_r * t + pc_r), zc_0 + Ac_z * sin(wc_z * t + pc_z), fc_0 + wc_f * t);
        Vector3 pn = decardCoordsConvert(rn_0 + An_r * sin(wn_r * t + pn_r), zn_0 + An_z * sin(wn_z * t + pn_z), fn_0 + wn_f * t);

        Vector3 vz = norm(pn - pc);
        Vector3 vx = norm(cross(vz, {0.0, 0.0, 1.0}));
        Vector3 vy = -norm(cross(vx, vz));

        matrixView = {vx, vy, vz};
    }

    float timeStep;
    float timeCurr;
    float rc_0;
    float zc_0;
    float fc_0;
    float Ac_r;
    float Ac_z;
    float wc_r;
    float wc_z;
    float wc_f;
    float pc_r;
    float pc_z;
    float rn_0;
    float zn_0;
    float fn_0;
    float An_r;
    float An_z;
    float wn_r;
    float wn_z;
    float wn_f;
    float pn_r;
    float pn_z;
};

class Scene{
public: 
    Scene(Camera& cam, bool isGpu, int width, int height, string pathMod)
    : viewer(cam), isGpu(isGpu), width(width), height(height), pathMod(pathMod) {
        window.width  = 0;
        window.height = 0;
        floor.memWrapper = 0;
        floor.memData = nullptr;
    }

    ~Scene() = default;

private:
    cudaArray* gpuFloorGen() {
        cudaArray* arrayDev;
        ifstream floorFin(flooorPath, ios::in | ios::binary);
    
        int width, height;
        floorFin.read(reinterpret_cast<char*>(&width), sizeof(int));
        floorFin.read(reinterpret_cast<char*>(&height), sizeof(int));

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
        CSC(cudaMallocArray(&arrayDev, &channelDesc, width, height));

        int* tmp = new int[width * height];
        floorFin.read(reinterpret_cast<char*>(tmp), width * height * sizeof(int));

        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr = make_cudaPitchedPtr(tmp, width * sizeof(int), width, height);
        copyParams.dstArray = arrayDev;
        copyParams.extent = make_cudaExtent(width, height, 1);
        copyParams.kind = cudaMemcpyHostToDevice;

        CSC(cudaMemcpy3D(&copyParams));

        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = arrayDev;

        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.readMode = cudaReadModeNormalizedFloat;
        texDesc.normalizedCoords = 1;

        floor.memWrapper = 0;
        CSC(cudaCreateTextureObject(&floor.memWrapper, &resDesc, &texDesc, 0));

        return arrayDev;
    }

    int* cpuFloorGen() {
        int* floorArr;
        
        ifstream floorFin(flooorPath, ios::in | ios::binary);

        int width, height;
        floorFin.read(reinterpret_cast<char*>(&width), sizeof(int));
        floorFin.read(reinterpret_cast<char*>(&height), sizeof(int));

        floorArr = new int[width * height * sizeof(int)];
        
        floorFin.read(reinterpret_cast<char*>(floorArr), width * height * sizeof(int));
        floor.width = width;
        floor.height = height;
        floor.memData = floorArr;
        floor.memWrapper = 0;

        return floorArr;
    }

public:
    void figureAdd(const CommonFigure& fig) {
        fig.figureRender(trianglesRender, lightsRender);
    }

    void lightsAdd(const LightPoint& point) {
        lightsRender.push_back(point);
    }

    void windowAdd(int w, int h, int pixRays) {
        window.width = w;
        window.height = h;
        window.sqrtScale = pixRays;
    }

    void floorAdd(const string& floorPath, Vector3 pA, Vector3 pB, Vector3 pC, Vector3 pD, const Texture& floorTex) {
        int textId = getTexId(floorTex, textureTable);
    
        floor.triangleId1 = trianglesRender.size();
        trianglesRender.push_back({pA, pB, pD, textId});
        floor.triangleId2 = trianglesRender.size();
        trianglesRender.push_back({pC, pB, pD, textId});
        floor.memWrapper = 1;
    
        flooorPath = floorPath;
    }    

    void sceneRender(const int recDepth = 1) {
        if (isGpu) {
            gpuSceneRender(recDepth);
        } else {
            cpuSceneRender(recDepth);
        }
    }

private:
    void cpuSceneRender(const int recDepth = 1) {
        int scaledW = window.width * window.sqrtScale;
        int scaledH = window.height * window.sqrtScale;
    
        Vector3* imgRender;
    
        saveTex(texRender, textureTable);
    
        int* floorArr = cpuFloorGen();
        int activeRaysCapacity = 2 * scaledW * scaledH;

        vector<Recursion> renderRays(activeRaysCapacity);
        imgRender = new Vector3[scaledH * scaledW];
    
        window.picture = new int[window.width * window.height];

        float totalTime = 0;
        float totalFrames = 0;
    
        for (int frameNum = 0; viewer.updatePos(); frameNum++) {
            uint64_t totalRays = 0;
            auto timeStart = steady_clock::now();    
            int activeRaysSize = scaledH * scaledW;
    
            cpuInitRays(renderRays.data(), imgRender, viewer.matrixView, viewer.pc, viewer.zView, scaledW, scaledH);
            totalRays += activeRaysSize;
    
            for (int _ = 0; _ < recDepth && activeRaysSize; ++_) {
                cpuRayTrace(renderRays.data(), activeRaysSize, imgRender, texRender.data(), floor, trianglesRender.data(), trianglesRender.size(), lightsRender.data(), lightsRender.size(), scaledW, scaledH);
    
                activeRaysSize <<= 1;
                activeRaysSize = cpuCleanRays(renderRays, activeRaysSize);
                totalRays += activeRaysSize;
    
                if (activeRaysCapacity < (activeRaysSize << 1)) {
                    activeRaysCapacity = activeRaysSize << 1;
                    renderRays.resize(activeRaysCapacity);
                }
            }
    
            cpuSsaa(window.picture, imgRender, window.width, window.height, window.sqrtScale);
            fileWrite(window.picture, width, height, pathMod, frameNum, isGpu);
    
            auto timeEnd = steady_clock::now();
            double frameTimeMs = duration_cast<microseconds>(timeEnd - timeStart).count() / 1000.0;

            totalTime += frameTimeMs;
            totalFrames++;
    
            cout << "Frame: " << frameNum << "\t" << "Time: " << frameTimeMs << " ms\t" << "Total rays: " << totalRays << "\n";
        }

        cout << "Average time: " << totalTime / totalFrames << "\n";
    
        delete[] floorArr;
        delete[] window.picture;
        delete[] imgRender;
    }

    void gpuSceneRender(const int recDepth = 1) {
        int scaledW = window.width * window.sqrtScale;
        int scaledH = window.height * window.sqrtScale;
    
        Vector3* devImg;
        Triangle* devTriangles;
        LightPoint* devLights;
        Recursion* devRays;
        Texture* devTex;
    
        saveTex(texRender, textureTable);
    
        CSC(cudaMalloc((void**) &devTriangles, trianglesRender.size() * sizeof(Triangle)));
        CSC(cudaMemcpy(devTriangles, trianglesRender.data(), sizeof(Triangle) * trianglesRender.size(), cudaMemcpyHostToDevice));
        CSC(cudaMalloc((void**) &devLights, lightsRender.size() * sizeof(LightPoint)));
        CSC(cudaMemcpy(devLights, lightsRender.data(),  sizeof(LightPoint) * lightsRender.size(), cudaMemcpyHostToDevice));
        CSC(cudaMalloc((void**) &devTex, texRender.size() * sizeof(Texture)));
        CSC(cudaMemcpy(devTex, texRender.data(), sizeof(Texture)*texRender.size(), cudaMemcpyHostToDevice));
    
        cudaArray* floorArr = gpuFloorGen();
    
        CSC(cudaMalloc((void**) &window.picture, window.height * window.width * sizeof(int)));
        CSC(cudaMalloc((void**) &devImg, scaledW * scaledH * sizeof(Vector3)));

        int activeRaysCapacity = 2 * scaledW * scaledH;

        CSC(cudaMalloc((void**) &devRays, activeRaysCapacity * sizeof(Recursion)));

        float totalTime = 0;
        float totalFrames = 0;
    
        for (int frameNum = 0; viewer.updatePos(); frameNum++) {
            uint64_t totalRays = 0;
            auto timeStart = steady_clock::now();
    
            Matrix matrixTransform = viewer.matrixView; 
            float viewerDist = viewer.zView;
            Vector3 cameraPos = viewer.pc;
    
            int activeRaysSize = scaledH * scaledW; 
    
            gpuInitRays<<<64, 64>>>(
                devRays, devImg, matrixTransform, 
                cameraPos, viewerDist, scaledW, scaledH
            );

            cudaDeviceSynchronize();
            CSC(cudaGetLastError());
            totalRays += activeRaysSize;
    
            for (int _ = 0; _ < recDepth && activeRaysSize; ++_) {
                gpuRayTrace<<<64, 64>>>(
                    devRays, activeRaysSize, devImg, devTex,
                    floor, devTriangles, trianglesRender.size(),
                    devLights, lightsRender.size(),
                    scaledW, scaledH
                );
                cudaDeviceSynchronize();
                CSC(cudaGetLastError());
    
                activeRaysSize <<= 1;
                activeRaysSize = gpuCleanRays(devRays, activeRaysSize);
                totalRays += activeRaysSize;
    
                if (activeRaysCapacity < (activeRaysSize << 1)) {
                    Recursion* tmpDevRays;
                    activeRaysCapacity = activeRaysSize << 1;
    
                    CSC(cudaMalloc((void**)&tmpDevRays, activeRaysCapacity * sizeof(Recursion)));
                    CSC(cudaMemcpy(tmpDevRays, devRays, sizeof(Recursion) * activeRaysSize, cudaMemcpyDeviceToDevice));
                    CSC(cudaFree(devRays));

                    devRays = tmpDevRays;
                }
            }
    
            gpuSsaa<<<dim3(32, 32), dim3(32, 32)>>>(
                window.picture, devImg, window.width, window.height, window.sqrtScale
            );
    
            cudaDeviceSynchronize();
            fileWrite(window.picture, width, height, pathMod, frameNum, isGpu);
    
            auto timeEnd =  steady_clock::now();
            double frameTimeMs = duration_cast<microseconds>(timeEnd - timeStart).count() / 1000.0;

            totalTime += frameTimeMs;
            totalFrames++;
    
            cout << "Frame: " << frameNum << "\t" << "Time: " << frameTimeMs << " ms\t" << "Total rays: " << totalRays << "\n";
        }

        cout << "Average time: " << totalTime / totalFrames << "\n";
    
        CSC(cudaFreeArray(floorArr));
        CSC(cudaFree(window.picture));
        CSC(cudaFree(devImg));
        CSC(cudaFree(devTriangles));
        CSC(cudaFree(devLights));
        CSC(cudaFree(devTex));
        CSC(cudaFree(devRays));
    }    

private:
    bool isGpu;
    int width, height;
    string pathMod;
    Camera& viewer;
    Window window;
    vector<Triangle> trianglesRender;
    vector<LightPoint> lightsRender;
    vector<Texture> texRender;
    RenderTexture floor;
    string path_to_dir;
    string flooorPath;
};

int main(int argc, const char** argv) {
    bool useGpu = true;

    if (argc == 2) {
        string parameter = argv[1];

        if (parameter == "--gpu") {
            useGpu = true;
        } else if (parameter == "--cpu") {
            useGpu = false;
        } else if (parameter == "--default") {
            ifstream configFile("best_config.txt");

            if (!configFile) {
                cerr << "Error: Unable to open best_config.txt" << endl;
                return 1;
            }

            string content;

            while (getline(configFile, content)) {
                cout << content << endl;
            }

            configFile.close();
            return 0;
        } else {
            cout << "Unknown parameter\n";
            return 1;
        }
    } else if (argc > 2) {
        cout << "Wrong parameters count\n";
        return 1;
    }

    InputData data;

    cin >> data.frameCount;
    cin >> data.modulePath;
    cin >> data.imgWidth >> data.imgHeight;
    cin >> data.angleView;
    cin >> data.rcInit >> data.zcInit >> data.fcInit >> data.arcR >> data.arcZ >> data.wcR >> data.wcZ >> data.wcF >> data.pcR >> data.pcZ;
    cin >> data.rnInit >> data.znInit >> data.fnInit >> data.anR >> data.anZ >> data.wnR >> data.wnZ >> data.wnF >> data.pnR >> data.pnZ;

    Camera cam(
        data.frameCount, data.angleView, 
        data.rcInit, data.zcInit, data.fcInit, data.arcR, data.arcZ, data.wcR, data.wcZ, data.wcF, data.pcR, data.pcZ,
        data.rnInit, data.znInit, data.fnInit, data.anR, data.anZ, data.wnR, data.wnZ, data.wnF, data.pnR, data.pnZ
    );

    Scene scene(cam, useGpu, data.imgWidth, data.imgHeight, data.modulePath);

    // Hexaeder
    cin >> data.objCenter.x >> data.objCenter.y >> data.objCenter.z;
    cin >> data.objColor.x >> data.objColor.y >> data.objColor.z;
    cin >> data.objRadius;
    cin >> data.reflectCoeff >> data.refractCoeff;
    cin >> data.lightLineCount;

    Texture hexaTexture = {data.objColor, 0.5, data.reflectCoeff, data.refractCoeff};
    scene.figureAdd(Hexaeder(data.objRadius, data.objCenter, hexaTexture, data.lightLineCount));

    // Octaeder
    cin >> data.objCenter.x >> data.objCenter.y >> data.objCenter.z;
    cin >> data.objColor.x >> data.objColor.y >> data.objColor.z;
    cin >> data.objRadius;
    cin >> data.reflectCoeff >> data.refractCoeff;
    cin >> data.lightLineCount;

    hexaTexture = {data.objColor, 0.5, data.reflectCoeff, data.refractCoeff};
    scene.figureAdd(Octaeder(data.objRadius, data.objCenter, hexaTexture, data.lightLineCount));

    // Icosaeder
    cin >> data.objCenter.x >> data.objCenter.y >> data.objCenter.z;
    cin >> data.objColor.x >> data.objColor.y >> data.objColor.z;
    cin >> data.objRadius;
    cin >> data.reflectCoeff >> data.refractCoeff;
    cin >> data.lightLineCount;

    hexaTexture = {data.objColor, 0.5, data.reflectCoeff, data.refractCoeff};
    scene.figureAdd(Icosaeder(data.objRadius, data.objCenter, hexaTexture, data.lightLineCount));

    cin >> data.cornerA.x >> data.cornerA.y >> data.cornerA.z;
    cin >> data.cornerB.x >> data.cornerB.y >> data.cornerB.z;
    cin >> data.cornerC.x >> data.cornerC.y >> data.cornerC.z;
    cin >> data.cornerD.x >> data.cornerD.y >> data.cornerD.z;
    cin >> data.textureFilePath;
    cin >> data.textureColor.x >> data.textureColor.y >> data.textureColor.z;
    cin >> data.textureReflect;

    Texture materialTexture = {data.textureColor, 0.5, data.textureReflect, 0.0};

    cin >> data.lightSources;

    for (int i = 0; i < data.lightSources; i++) {
        Vector3 lightPos, lightCol;
        cin >> lightPos.x >> lightPos.y >> lightPos.z;
        cin >> lightCol.x >> lightCol.y >> lightCol.z;

        scene.lightsAdd({lightPos, lightCol, 1.0});
    }

    cin >> data.recursionDepth >> data.pixelSampling;

    scene.windowAdd(data.imgWidth, data.imgHeight, data.pixelSampling);
    scene.floorAdd(data.textureFilePath, data.cornerA, data.cornerB, data.cornerC, data.cornerD, materialTexture);
    scene.sceneRender(data.recursionDepth);

    return 0;
}
