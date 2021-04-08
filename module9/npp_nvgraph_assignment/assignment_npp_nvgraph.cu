#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <ImageIO.h>
#include <Exceptions.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <npp.h>
#include "nvgraph.h"

inline int cudaDeviceInit(int argc, const char **argv)
{
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
        exit(EXIT_FAILURE);
    }

    int dev = findCudaDevice(argc, argv);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    std::cerr << "cudaSetDevice GPU" << dev << " = " << deviceProp.name << std::endl;

    checkCudaErrors(cudaSetDevice(dev));

    return dev;
}

void check_status(nvgraphStatus_t status)
{
    if ((int)status != 0)
    {
        printf("ERROR : %d\n",status);
        exit(0);
    }
}

void nvgraph_main() {
    const size_t  n = 6, nnz = 10, vertex_numsets = 1, edge_numsets = 1;
    float *widest_path_h;
    void** vertex_dim;
    // nvgraph variables
    nvgraphStatus_t status; 
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSCTopology32I_t CSC_input;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    cudaDataType_t* vertex_dimT;

    // Init host data
    widest_path_h = (float*)malloc(n*sizeof(float));
    vertex_dim  = (void**)malloc(vertex_numsets*sizeof(void*));
    vertex_dimT = (cudaDataType_t*)malloc(vertex_numsets*sizeof(cudaDataType_t));
    CSC_input = (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));
    vertex_dim[0]= (void*)widest_path_h;
    vertex_dimT[0] = CUDA_R_32F;
    float weights_h[] = {0.333333, 0.5, 0.333333, 0.5, 0.5, 1.0, 0.333333, 0.5, 0.5, 0.5};
    int destination_offsets_h[] = {0, 1, 3, 4, 6, 8, 10};
    int source_indices_h[] = {2, 0, 2, 0, 4, 5, 2, 3, 3, 4};
    check_status(nvgraphCreate(&handle));
    check_status(nvgraphCreateGraphDescr (handle, &graph));
    CSC_input->nvertices = n; 
    CSC_input->nedges = nnz;
    CSC_input->destination_offsets = destination_offsets_h;
    CSC_input->source_indices = source_indices_h;
    // Set graph connectivity and properties (tranfers)
    check_status(nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32));
    check_status(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
    check_status(nvgraphAllocateEdgeData  (handle, graph, edge_numsets, &edge_dimT));
    check_status(nvgraphSetEdgeData(handle, graph, (void*)weights_h, 0));
    // Solve
    int source_vert = 0;
    check_status(nvgraphWidestPath(handle, graph, 0, &source_vert, 0));
    // Get and print result
    check_status(nvgraphGetVertexData(handle, graph, (void*)widest_path_h, 0));
    for (int i = 0; i < n; i++) {
        printf("The largest min weight edge from vertex 0 to %d is: %f\n", i, widest_path_h[i]);
    }

    //Clean
    check_status(nvgraphDestroyGraphDescr(handle, graph));
    check_status(nvgraphDestroy(handle));
    free(widest_path_h);
    free(vertex_dim);
    free(vertex_dimT);
    free(CSC_input);
}

void npp_main(int argc, char **argv) {
    std::string sFilename;
    char *filepath;
    cudaDeviceInit(argc, (const char **)argv);

    filepath = sdkFindFilePath("Lena.pgm", argv[0]);
    if(filepath) sFilename = filepath;
    else sFilename = "Lena.pgm";

    std::string sResultFilename = sFilename;
    std::string::size_type dot = sResultFilename.rfind('.');
    if (dot != std::string::npos) sResultFilename = sResultFilename.substr(0, dot);

    sResultFilename += "_Sharpen.pgm";

    npp::ImageCPU_8u_C1 oHostSrc;
    npp::loadImage(sFilename, oHostSrc);
    npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

    NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    NppiSize oSizeROI = {(int)oDeviceSrc.width() , (int)oDeviceSrc.height() };
    npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);

    NPP_CHECK_NPP(
        nppiFilterSharpen_8u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(),
                                oDeviceDst.data(), oDeviceDst.pitch(),
                                oSizeROI)
    );

    npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

    saveImage(sResultFilename, oHostDst);
    std::cout << "Saved image: " << sResultFilename << std::endl;
}

int main(int argc, char **argv)
{
    auto start = std::chrono::high_resolution_clock::now();
    nvgraph_main();
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop - start;
    auto nvgraph_time = duration.count();

    start = std::chrono::high_resolution_clock::now();
    npp_main(argc, argv);
    stop = std::chrono::high_resolution_clock::now();
    duration = stop - start;
    auto npp_time = duration.count();

    std::cout << "Time taken for nvGRAPH: " << nvgraph_time << " ms" << std::endl;
    std::cout << "Time taken for NPP: " << npp_time << " ms" << std::endl;
    return EXIT_SUCCESS;
}
