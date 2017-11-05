//     This file is part of Scribble Video Inpainting.
// 
//     Scribble Video Inpainting is free software: you can redistribute it and/or modify
//     it under the terms of the GNU General Public License as published by
//     the Free Software Foundation, either version 3 of the License, or
//     (at your option) any later version.
// 
//     Scribble Video Inpainting is distributed in the hope that it will be useful,
//     but WITHOUT ANY WARRANTY; without even the implied warranty of
//     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//     GNU General Public License for more details.
// 
//     You should have received a copy of the GNU General Public License
//     along with Scribble Video Inpainting.  If not, see <http://www.gnu.org/licenses/>.

#include "ssd.cuh"
typedef unsigned char uchar;
using namespace std;

cv::Point2i call_gpu_ssd(uchar *h_image,
                         int im_width, int im_height,
                         uchar *h_originalsourceRegion,
                         uchar *h_sourceRegion,
                         uchar *h_invalid_mask,
                         cv::Point2i compare_point_fill_front_ul,
                         int2 flow_comparePt_ul,
                         int patch_width, int patch_height, int nPics, float beta){

#define BLOCKDIMX patch_width
#define BLOCKDIMY patch_height

dim3 dimBlock(BLOCKDIMX, BLOCKDIMY),dimGrid;
dimGrid.x = (im_width  % dimBlock.x) ? (im_width /dimBlock.x + 1) : (im_width /dimBlock.x);
dimGrid.y = (im_height % dimBlock.y) ? (im_height/dimBlock.y + 1) : (im_height/dimBlock.y);

// pitches
size_t pitch_char, pitch_float;

// define device pointers
float *d_patchErrorMat;
uchar *d_image, *d_originalSourceRegion, *d_updatedInpaintMask, *d_invalid_mask;


// allocate mem on cuda device and copy data

gpuErrchk(cudaMallocPitch((void**)&d_image,&pitch_char,im_width*sizeof(char),3*im_height));
gpuErrchk(cudaMemcpy2D(d_image,pitch_char,h_image,im_width*sizeof(char),im_width*sizeof(char),3*im_height,cudaMemcpyHostToDevice));

gpuErrchk(cudaMallocPitch((void**)&d_originalSourceRegion,&pitch_char,im_width*sizeof(char),im_height));
gpuErrchk(cudaMemcpy2D(d_originalSourceRegion,pitch_char,h_originalsourceRegion,im_width*sizeof(char),im_width*sizeof(char),im_height,cudaMemcpyHostToDevice));

gpuErrchk(cudaMallocPitch((void**)&d_updatedInpaintMask,&pitch_char,im_width*sizeof(char),im_height));
gpuErrchk(cudaMemcpy2D(d_updatedInpaintMask,pitch_char,h_sourceRegion,im_width*sizeof(char),im_width*sizeof(char),im_height,cudaMemcpyHostToDevice));

gpuErrchk(cudaMallocPitch((void**)&d_invalid_mask,&pitch_char,im_width*sizeof(char),im_height));
gpuErrchk(cudaMemcpy2D(d_invalid_mask,pitch_char,h_invalid_mask,im_width*sizeof(char),im_width*sizeof(char),im_height,cudaMemcpyHostToDevice));

gpuErrchk(cudaMallocPitch((void**)&d_patchErrorMat,&pitch_float,im_width*sizeof(float),im_height));

// alloc error mat for ssd values
float *h_patchErrorMat = new float[im_width*im_height];

// hand over current fill front point
int2 comp_pt_fill_front_ul;
comp_pt_fill_front_ul.x = compare_point_fill_front_ul.x;
comp_pt_fill_front_ul.y = compare_point_fill_front_ul.y;

gpu_ssd<<< dimGrid, dimBlock >>>(d_image,
                                 im_width,im_height,
                                 patch_width,patch_height,
                                 d_originalSourceRegion,
                                 d_updatedInpaintMask,
                                 d_invalid_mask,
                                 comp_pt_fill_front_ul,
                                 flow_comparePt_ul,
                                 d_patchErrorMat,
                                 nPics,
                                 pitch_char/sizeof(char), pitch_float/sizeof(float), beta);

// check if we are sane
gpuErrchk(cudaThreadSynchronize());
// get back error mat
gpuErrchk(cudaMemcpy2D((void*)h_patchErrorMat,im_width*sizeof(float),d_patchErrorMat,pitch_float,im_width*sizeof(float),im_height,cudaMemcpyDeviceToHost));

// free cuda memory
cudaFree(d_image);
cudaFree(d_originalSourceRegion);
cudaFree(d_invalid_mask);
cudaFree(d_patchErrorMat);
cudaFree(d_updatedInpaintMask);


// look for the patch patch - by default (0,0) with error infinity
cv::Point2i result_upper_left = cv::Point2i(0,0);
float minError = INFINITY;


for(int i = 0; i< im_width; i++)
    for(int j = 0; j < im_height; j++){
        //cout << " current error is:" << h_patchErrorMat[i+j*im_width] << endl;
        if(h_patchErrorMat[i+j*im_width] < minError){
            minError = h_patchErrorMat[i+j*im_width];
            result_upper_left = cv::Point2i(i,j);
        }
    }

cout << "the min error is:" << minError << endl;
cout << "x= " << result_upper_left.x << "y= " << result_upper_left.y << endl;

delete[] h_patchErrorMat;

return result_upper_left;

} // call_gpu_ssd


__global__ void gpu_ssd(uchar *d_image,
                        int im_width, int im_heigth,
                        int patch_width, int patch_heigth,
                        uchar *d_originalSourceRegion,
                        uchar *d_sourceRegion, uchar *d_invalid_mask,
                        int2 compare_point_fillfront_ul,
                        int2 flow_comparePt_ul,
                        float *d_patchErrorMat,
                        int nPics,
                        int pitch_char, int pitch_float,
                        float beta){

// hasHit = have we ever compared something?
bool hasHit = false;

uint patchError;
uint patch_diff;

const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;

// set default to infinity
if(x < im_width && y < im_heigth)
d_patchErrorMat[x+y*pitch_float] = INFINITY;

// out of bound with patch
if(x+patch_width > im_width|| y+patch_heigth > im_heigth){
    return;
}

// check if we hit the mask
for(int i = 0; i < patch_width; i++)
    for(int j = 0; j < patch_heigth; j++){
        if(d_originalSourceRegion[x+i+(y+j)*pitch_char] == 0 || d_invalid_mask[x+i+(y+j)*pitch_char] == 255){
        d_patchErrorMat[x+y*pitch_float]  = INFINITY;
        return;
        }

    }

patchError = 0;

// ssd for "normal distance"
for(int i = 0; i < patch_width; i++)
    for(int j = 0; j < patch_heigth; j++){

        // if we are 0 we have pixel information to compare
        if(d_sourceRegion[compare_point_fillfront_ul.x+i+(compare_point_fillfront_ul.y+j)*pitch_char] != 0){
        hasHit = true;

        patch_diff = d_image[compare_point_fillfront_ul.x+i + (compare_point_fillfront_ul.y+j)*pitch_char] - d_image[x+i + (j+y)*pitch_char];
        patchError += patch_diff*patch_diff;

        patch_diff = d_image[compare_point_fillfront_ul.x+i + (compare_point_fillfront_ul.y+j)*pitch_char + pitch_char*im_heigth] - d_image[x+i + (j+y)*pitch_char + pitch_char*im_heigth];
        patchError += patch_diff*patch_diff;

        patch_diff = d_image[compare_point_fillfront_ul.x+i + (compare_point_fillfront_ul.y+j)*pitch_char + 2*pitch_char*im_heigth] - d_image[x+i + (j+y)*pitch_char + 2*pitch_char*im_heigth];
        patchError += patch_diff*patch_diff;
        }
    }


// for flow distance
uint flowError = 0;
if(nPics > 1) 
{
uint flow_diff;
for(int i = 0; i < patch_width; i++)
    for(int j = 0; j < patch_heigth; j++){

        // if we are 0 we have pixel information to compare
        hasHit = true;

        flow_diff = d_image[flow_comparePt_ul.x+i + (flow_comparePt_ul.y+j)*pitch_char] - d_image[x+i + (j+y)*pitch_char];
        flowError += flow_diff*flow_diff;

        flow_diff = d_image[flow_comparePt_ul.x+i + (flow_comparePt_ul.y+j)*pitch_char + pitch_char*im_heigth] - d_image[x+i + (j+y)*pitch_char + pitch_char*im_heigth];
        flowError += flow_diff*flow_diff;

        flow_diff = d_image[flow_comparePt_ul.x+i + (flow_comparePt_ul.y+j)*pitch_char + 2*pitch_char*im_heigth] - d_image[x+i + (j+y)*pitch_char + 2*pitch_char*im_heigth];
        flowError += flow_diff*flow_diff;
    }
}

// check if we ever compared some pixels
if(hasHit)
d_patchErrorMat[x+y*pitch_float]  = (float)patchError + beta*(float)flowError;

return;

}
