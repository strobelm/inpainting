#ifndef SSD_CUH
#define SSD_CUH

#include <iostream>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <cuda_runtime.h>
#include "cutil5.cuh"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void gpu_ssd(uchar *d_image,
                        int im_width, int im_heigth,
                        int patch_width, int patch_heigth,
                        uchar *d_originalInpaintMask, uchar *d_updatedInpaintMask, uchar *d_invalid_mask,
                        int2 compare_point_fillfront_ul,
                        int2 flow_comparePt_ul,
                        float *d_patchErrorMat,
                        int nPics,
                        int pitch_char, int pitch_float, float beta);

cv::Point2i call_gpu_ssd(uchar *h_image,
                         int im_width, int im_height,
                         uchar *h_originalsourceRegion, uchar *h_sourceRegion, uchar *h_invalid_mask,
                         cv::Point2i compare_point_fill_front_ul,
                         int2 flow_comparePt_ul,
                         int patch_width, int patch_height, int nPics, float beta);


#endif // SSD_CUH
