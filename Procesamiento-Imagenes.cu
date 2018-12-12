/*
    To compile and run: nvcc Procesamiento-Imagenes.cu -o main `pkg-config --cflags --libs opencv`&& ./main
*/

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <iostream>
#include <math.h>


using namespace std;
using namespace cv;


// Image in color Mat format to char for color
unsigned char* mat2ucharColor (Mat image, int width, int height)
{

    unsigned char *imgOut=(unsigned char*)malloc(sizeof(unsigned char)*width*height*3);
    int pos = 0;

    for(int j=0; j<height; j++)
    {
        for(int i=0; i<width; i++)
        {
            imgOut[pos] = image.at<Vec3b>(j,i)[2];
            pos++;
            imgOut[pos] = image.at<Vec3b>(j,i)[1];
            pos++;
            imgOut[pos] = image.at<Vec3b>(j,i)[0];
            pos++;
        }
    }
    return imgOut;      
}

// Image in char format to Mat color format
Mat uchar2matColor (unsigned char* img, int ancho, int alto)
{
    Mat imgOut(alto, ancho, CV_8UC3);
    int pos = 0;

    for(int j=0; j<alto; j++)
    {
        for(int i=0; i<ancho; i++)
        {
            imgOut.at<Vec3b>(j,i)[2] = img[pos];
            pos++;
            imgOut.at<Vec3b>(j,i)[1] = img[pos];
            pos++;
            imgOut.at<Vec3b>(j,i)[0] = img[pos];
            pos++;
        }
    }
    return imgOut;
}

// Image in Mat format to char for one channel
unsigned char* mat2uchar (Mat image, int width, int height)
{
    // Generating a memory space - buffer
    unsigned char *imgOut=(unsigned char*)malloc(sizeof(unsigned char)*width*height);

    int pos = 0;

    for(int j=0; j<height; j++)
    {
        for(int i=0; i<width; i++)
        {
            imgOut[pos] = image.at<uchar>(j,i);
            pos++;
        }
    }
    return imgOut;
}

// Image in char format to Mat format
Mat uchar2mat (unsigned char* img, int ancho, int alto)
{
    Mat imgOut(alto, ancho, CV_8UC1);
    int pos = 0;

    for(int j=0; j<alto; j++)
    {
        for(int i=0; i<ancho; i++)
        {
            imgOut.at<uchar>(j,i) = img[pos];
            pos++;
        }
    }
    return imgOut;
}




//Gray Scale  kernel
__global__ void kernel_graySCale( unsigned char *color_image, unsigned char *gray_image)
{
    // Get the position in the input image and the position in the gpu vector (offset).
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    gray_image[offset] = (0.3*color_image[offset*3 + 0] + 0.59*color_image[offset*3 + 1] + 0.11*color_image[offset*3 + 2]);

}


/**
    Gray scale format:
        input : img             -> Mat RGB img
        output: result_image    -> Mat gray result_image
**/
Mat get_grayScale_gpu(Mat img)
{
    int width = img.cols;
    int height = img.rows;

    // Separte memmory in CPU 
    unsigned char *img_toGPU = mat2ucharColor(img, width, height);
    unsigned char *cpu_outputImage=(unsigned char*)malloc(sizeof(unsigned char)*width*height);    
    
    // Separate memory en GPU
    unsigned char *gpu_inputImage, *gpu_outputImage;
    cudaMalloc( (void**)&gpu_inputImage, width*height*3);
    cudaMalloc( (void**)&gpu_outputImage, width*height);

    // Assign the number of grids and threads to use
    dim3 grids(width,height);
    dim3 threads(1, 1);

    // CPU image to GPU Image
    cudaMemcpy(gpu_inputImage, img_toGPU, width*height*3, cudaMemcpyHostToDevice);

    // Function gray scale in GPU 
    kernel_graySCale<<<grids, threads>>>(gpu_inputImage, gpu_outputImage);

    // GPU Image to CPU image
    cudaMemcpy(cpu_outputImage, gpu_outputImage, width*height  , cudaMemcpyDeviceToHost );

    // Free mmemory of GPU
    cudaFree( gpu_inputImage);
    cudaFree( gpu_outputImage);

    // Get the result in mat format to show
    Mat result_image = uchar2mat (cpu_outputImage, width, height);
    return result_image;
}

//Sobel edge detector  kernel
__global__ void kernel_sobel( unsigned char *gray_image, unsigned char *out_image, int width, int height)
{
    // Get the position in the input image and the position in the gpu vector (offset).
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;


    if(x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1)
    {
        out_image[offset] = 0;
        return;
    }

    //Get neighborhood positions on the gpu vector
    int px_11 = offset - width - 1;
    int px_12 = offset - width;
    int px_13 = offset - width + 1;
    int px_21 = offset - 1;
    int px_22 = offset;
    int px_23 = offset + 1;
    int px_31 = offset + width - 1;
    int px_32 = offset + width;
    int px_33 = offset + width + 1;
    

    // Sobel kernel
    //int kx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    //int ky[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    

    int gx =    (-1) * gray_image[px_11] + 
                (0)  * gray_image[px_12] +
                (1)  * gray_image[px_13] + 
                (-2) * gray_image[px_21] + 
                (0)  * gray_image[px_22] +
                (2)  * gray_image[px_23] +
                (-1) * gray_image[px_31] + 
                (0)  * gray_image[px_32] +
                (1)  * gray_image[px_33] ; 

    int gy =    (-1) * gray_image[px_11] + 
                (-2) * gray_image[px_12] +
                (-1) * gray_image[px_13] + 
                (0)  * gray_image[px_21] + 
                (0)  * gray_image[px_22] +
                (0)  * gray_image[px_23] +
                (1)  * gray_image[px_31] + 
                (2)  * gray_image[px_32] +
                (1)  * gray_image[px_33] ;


    float gradient = sqrt(pow(gx, 2) + pow(gy, 2));

    out_image[offset] = gradient;
    
    
}



/**
    Sobel edge detector:
        input : img             -> Mat gray img
        output: result_image    -> Mat sobel edges result_image
**/
Mat get_sobel_gpu(Mat img)
{
    int width = img.cols;
    int height = img.rows;

    // Separte memmory in CPU 
    unsigned char *img_toGPU = mat2uchar(img, width, height);
    unsigned char *cpu_outputImage=(unsigned char*)malloc(sizeof(unsigned char)*width*height);    
    
    
    // Separate memory en GPU
    unsigned char *gpu_inputImage, *gpu_outputImage;
    cudaMalloc( (void**)&gpu_inputImage, width*height);
    cudaMalloc( (void**)&gpu_outputImage, width*height);

    // Assign the number of grids and threads to use
    dim3 grids(width,height);
    dim3 threads(1, 1);

    // CPU image to GPU Image
    cudaMemcpy(gpu_inputImage, img_toGPU, width*height, cudaMemcpyHostToDevice);

    // Function gray scale in GPU 
    kernel_sobel<<<grids, threads>>>(gpu_inputImage, gpu_outputImage, width, height);

    // GPU Image to CPU image
    cudaMemcpy(cpu_outputImage, gpu_outputImage, width*height  , cudaMemcpyDeviceToHost );

    // Free mmemory of GPU
    cudaFree( gpu_inputImage);
    cudaFree( gpu_outputImage);

    // Get the result in mat format to show
    Mat result_image = uchar2mat (cpu_outputImage, width, height);

    return result_image;
}

// Mean filter kernel
__global__ void kernel_mean_filter( unsigned char *gray_image, unsigned char *out_image, int width, int height)
{
    // Get the position in the input image and the position in the gpu vector (offset).
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;


    if(x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1)
    {
        out_image[offset] = 0;
        return;
    }

    //Get neighborhood positions on the gpu vector
    int px_11 = offset - width - 1;
    int px_12 = offset - width;
    int px_13 = offset - width + 1;
    int px_21 = offset - 1;
    int px_22 = offset;
    int px_23 = offset + 1;
    int px_31 = offset + width - 1;
    int px_32 = offset + width;
    int px_33 = offset + width + 1;


    out_image[offset] = (gray_image[px_11] + gray_image[px_12] + gray_image[px_13] +
                         gray_image[px_21] + gray_image[px_22] + gray_image[px_23] +
                         gray_image[px_31] + gray_image[px_32] + gray_image[px_33])/9;
    
    
}


/**
    Mean filter:
        input : img             -> Mat gray img
        output: result_image    -> Mat mean result_image
**/
Mat get_meanFilter_gpu(Mat img)
{
    int width = img.cols;
    int height = img.rows;

    // Separte memmory in CPU 
    unsigned char *img_toGPU = mat2uchar(img, width, height);
    unsigned char *cpu_outputImage=(unsigned char*)malloc(sizeof(unsigned char)*width*height);    
    
    
    // Separate memory en GPU
    unsigned char *gpu_inputImage, *gpu_outputImage;
    cudaMalloc( (void**)&gpu_inputImage, width*height);
    cudaMalloc( (void**)&gpu_outputImage, width*height);

    // Assign the number of grids and threads to use
    dim3 grids(width,height);
    dim3 threads(1, 1);

    // CPU image to GPU Image
    cudaMemcpy(gpu_inputImage, img_toGPU, width*height, cudaMemcpyHostToDevice);

    // Function gray scale in GPU 
    kernel_mean_filter<<<grids, threads>>>(gpu_inputImage, gpu_outputImage, width, height);

    // GPU Image to CPU image
    cudaMemcpy(cpu_outputImage, gpu_outputImage, width*height  , cudaMemcpyDeviceToHost );

    // Free mmemory of GPU
    cudaFree( gpu_inputImage);
    cudaFree( gpu_outputImage);

    // Get the result in mat format to show
    Mat result_image = uchar2mat (cpu_outputImage, width, height);

    return result_image;
}


//InsertionSort
__device__ void insertion_sort (unsigned char arr[], int length)
{
    int j;
    unsigned char temp;

    for (int i = 0; i < length; i++)
    {
        j = i;

        while (j > 0 && arr[j] < arr[j-1])
        {
            temp = arr[j];
            arr[j] = arr[j-1];
            arr[j-1] = temp;
            j--;
        }
    }
}

// Median filter kernel
__global__ void kernel_median_filter( unsigned char *gray_image, unsigned char *out_image, int width, int height)
{
    // Get the position in the input image and the position in the gpu vector (offset).
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;


    if(x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1)
    {
        out_image[offset] = 0;
        return;
    }

    //Get neighborhood positions on the gpu vector
    int px_11 = offset - width - 1;
    int px_12 = offset - width;
    int px_13 = offset - width + 1;
    int px_21 = offset - 1;
    int px_22 = offset;
    int px_23 = offset + 1;
    int px_31 = offset + width - 1;
    int px_32 = offset + width;
    int px_33 = offset + width + 1;


    unsigned char pixels[9] = {gray_image[px_11], gray_image[px_12], gray_image[px_13],
                               gray_image[px_21], gray_image[px_22], gray_image[px_23],
                               gray_image[px_31], gray_image[px_32], gray_image[px_33]};

    insertion_sort(pixels, 9);

    out_image[offset] = pixels[4];
    
}


/**
    Mean filter:
        input : img             -> Mat gray img
        output: result_image    -> Mat mean result_image
**/
Mat get_medianFilter_gpu(Mat img)
{
    int width = img.cols;
    int height = img.rows;

    // Separte memmory in CPU 
    unsigned char *img_toGPU = mat2uchar(img, width, height);
    unsigned char *cpu_outputImage=(unsigned char*)malloc(sizeof(unsigned char)*width*height);    
    
    
    // Separate memory en GPU
    unsigned char *gpu_inputImage, *gpu_outputImage;
    cudaMalloc( (void**)&gpu_inputImage, width*height);
    cudaMalloc( (void**)&gpu_outputImage, width*height);

    // Assign the number of grids and threads to use
    dim3 grids(width,height);
    dim3 threads(1, 1);

    // CPU image to GPU Image
    cudaMemcpy(gpu_inputImage, img_toGPU, width*height, cudaMemcpyHostToDevice);

    // Function gray scale in GPU 
    kernel_median_filter<<<grids, threads>>>(gpu_inputImage, gpu_outputImage, width, height);

    // GPU Image to CPU image
    cudaMemcpy(cpu_outputImage, gpu_outputImage, width*height  , cudaMemcpyDeviceToHost );

    // Free mmemory of GPU
    cudaFree( gpu_inputImage);
    cudaFree( gpu_outputImage);

    // Get the result in mat format to show
    Mat result_image = uchar2mat (cpu_outputImage, width, height);

    return result_image;
}


//===================================== COLOR ===========================================

//RGB to CMY  kernel
__global__ void kernel_RBG_to_CMY( unsigned char *color_image, unsigned char *out_image)
{
    // Get the position in the input image and the position in the gpu vector (offset).
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    // C = 1 - R
    // M = 1 - G
    // Y = 1 - B
    out_image[offset*3+1] = 255 - color_image[offset*3 + 0];
    out_image[offset*3+0] = 255 - color_image[offset*3 + 1];
    out_image[offset*3+2] = 255 - color_image[offset*3 + 2];
    
    //out_image[offset] = 255 - color_image[offset*3 + 0];
}


/**
    RGB to CMY format:
        input : img             -> Mat RGB img
        output: result_image    -> Mat CMY result_image
**/

Mat get_RGB_to_CMY(Mat img)
{
    int width = img.cols;
    int height = img.rows;

    // Separte memmory in CPU 
    unsigned char *img_toGPU = mat2ucharColor(img, width, height);
    unsigned char *cpu_outputImage=(unsigned char*)malloc(sizeof(unsigned char)*width*height*3);    
    //unsigned char *cpu_outputImage=(unsigned char*)malloc(sizeof(unsigned char)*width*height);    
    
    // Separate memory en GPU
    unsigned char *gpu_inputImage, *gpu_outputImage;
    cudaMalloc( (void**)&gpu_inputImage, width*height*3);
    cudaMalloc( (void**)&gpu_outputImage, width*height*3);
    //cudaMalloc( (void**)&gpu_outputImage, width*height);

    // Assign the number of grids and threads to use
    dim3 grids(width,height);
    dim3 threads(1, 1);

    // CPU image to GPU Image
    cudaMemcpy(gpu_inputImage, img_toGPU, width*height*3, cudaMemcpyHostToDevice);
    //cudaMemcpy(gpu_inputImage, img_toGPU, width*height*3, cudaMemcpyHostToDevice);

    // Function gray scale in GPU 
    kernel_RBG_to_CMY<<<grids, threads>>>(gpu_inputImage, gpu_outputImage);

    // GPU Image to CPU image
    cudaMemcpy(cpu_outputImage, gpu_outputImage, width*height*3 , cudaMemcpyDeviceToHost );
    //cudaMemcpy(cpu_outputImage, gpu_outputImage, width*height , cudaMemcpyDeviceToHost );

    // Free mmemory of GPU
    cudaFree( gpu_inputImage);
    cudaFree( gpu_outputImage);

    // Get the result in mat format to show
    Mat result_image = uchar2matColor(cpu_outputImage, width, height);
    //Mat result_image = uchar2mat(cpu_outputImage, width, height);
    return result_image;
}



//RGB to HSV  kernel
__global__ void kernel_RBG_to_HSV( unsigned char *color_image, unsigned char *out_image)
{
    // Get the position in the input image and the position in the gpu vector (offset).
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;


    /* R is max:
        H = (G-B)*60/(max-min)
        S = (max-min)/max
        V = max
    */

    if(color_image[offset*3 + 0] > color_image[offset*3 + 1] && color_image[offset*3 + 0] > color_image[offset*3 + 2])
    {
        if(color_image[offset*3 + 1] < color_image[offset*3 + 2])
        {
            out_image[offset*3+0] = (color_image[offset*3 + 1] - color_image[offset*3 + 2])*60/(color_image[offset*3 + 0]-color_image[offset*3 + 1])*1.0;
            out_image[offset*3+1] = (color_image[offset*3 + 0] - color_image[offset*3 + 1])/color_image[offset*3+0]*1.0;
            out_image[offset*3+2] = color_image[offset*3 + 0];
        }

        else
        {
            out_image[offset*3+0] = (color_image[offset*3 + 1] - color_image[offset*3 + 2])*60/(color_image[offset*3 + 0]-color_image[offset*3 + 2])*1.0;
            out_image[offset*3+1] = (color_image[offset*3 + 0] - color_image[offset*3 + 2])/color_image[offset*3+0]*1.0;
            out_image[offset*3+2] = color_image[offset*3 + 0];
        }

    }

    /* G is max:
        H = (B-R)*60/(max-min)+120
        S = (max-min)/max
        V = max
    */

    if(color_image[offset*3 + 1] > color_image[offset*3 + 0] && color_image[offset*3 + 1] > color_image[offset*3 + 2])
    {
        if(color_image[offset*3 + 0] < color_image[offset*3 + 2])
        {
            out_image[offset*3+0] = (color_image[offset*3 + 2] - color_image[offset*3 + 0])*60/(color_image[offset*3 + 1]-color_image[offset*3 + 0])*1.0+120;
            out_image[offset*3+1] = (color_image[offset*3 + 1] - color_image[offset*3 + 0])/color_image[offset*3+1]*1.0;
            out_image[offset*3+2] = color_image[offset*3 + 1];
        }

        else
        {
            out_image[offset*3+0] = (color_image[offset*3 + 2] - color_image[offset*3 + 0])*60/(color_image[offset*3 + 1]-color_image[offset*3 + 2])*1.0+120;
            out_image[offset*3+1] = (color_image[offset*3 + 1] - color_image[offset*3 + 2])/color_image[offset*3+1]*1.0;
            out_image[offset*3+2] = color_image[offset*3 + 1];
        }

    }

    /* B is max:
        H = (R-G)*60/(max-min)+240
        S = (max-min)/max
        V = max
    */

    else
    {
        if(color_image[offset*3 + 0] < color_image[offset*3 + 1])
        {
            out_image[offset*3+0] = (color_image[offset*3 + 0] - color_image[offset*3 + 1])*60/(color_image[offset*3 + 2]-color_image[offset*3 + 0])*1.0+240;
            out_image[offset*3+1] = (color_image[offset*3 + 2] - color_image[offset*3 + 0])/color_image[offset*3+2]*1.0;
            out_image[offset*3+2] = color_image[offset*3 + 2];
        }

        else
        {
            out_image[offset*3+0] = (color_image[offset*3 + 0] - color_image[offset*3 + 1])*60/(color_image[offset*3 + 2]-color_image[offset*3 + 1])*1.0+240;
            out_image[offset*3+1] = (color_image[offset*3 + 2] - color_image[offset*3 + 1])/color_image[offset*3+2]*1.0;
            out_image[offset*3+2] = color_image[offset*3 + 2];
        }

    }

    
}


/**
    RGB to HSV format:
        input : img             -> Mat RGB img
        output: result_image    -> Mat CMY result_image
**/

Mat get_RGB_to_HSV(Mat img)
{
    int width = img.cols;
    int height = img.rows;

    // Separte memmory in CPU 
    unsigned char *img_toGPU = mat2ucharColor(img, width, height);
    unsigned char *cpu_outputImage=(unsigned char*)malloc(sizeof(unsigned char)*width*height*3);    
    
    // Separate memory en GPU
    unsigned char *gpu_inputImage, *gpu_outputImage;
    cudaMalloc( (void**)&gpu_inputImage, width*height*3);
    cudaMalloc( (void**)&gpu_outputImage, width*height*3);

    // Assign the number of grids and threads to use
    dim3 grids(width,height);
    dim3 threads(1, 1);

    // CPU image to GPU Image
    cudaMemcpy(gpu_inputImage, img_toGPU, width*height*3, cudaMemcpyHostToDevice);

    // Function gray scale in GPU 
    kernel_RBG_to_HSV<<<grids, threads>>>(gpu_inputImage, gpu_outputImage);

    // GPU Image to CPU image
    cudaMemcpy(cpu_outputImage, gpu_outputImage, width*height*3 , cudaMemcpyDeviceToHost );

    // Free mmemory of GPU
    cudaFree( gpu_inputImage);
    cudaFree( gpu_outputImage);

    // Get the result in mat format to show
    Mat result_image = uchar2matColor(cpu_outputImage, width, height);
    return result_image;
}

//=============================== TRANSFORMATIONS ================================================

/*
    Zoom out
*/

__global__ void kernel_zoom_out_without_interpolation( unsigned char *color_image, unsigned char *out_image, int width, int scale_zoom)
{
    // Get the position in the input image and the position in the gpu vector (offset).
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    int offset_aux = (int)(x/scale_zoom) + (int)(y/scale_zoom)*width;


    out_image[offset*3 + 0] = color_image[offset_aux*3 + 0];
    out_image[offset*3 + 1] = color_image[offset_aux*3 + 1];
    out_image[offset*3 + 2] = color_image[offset_aux*3 + 2];
    
    
}


/**
    RGB to CMY format:
        input : img             -> Mat RGB img
        output: result_image    -> Mat CMY result_image
**/

Mat get_zoom_out(Mat img, int scale_zoom)
{
    int width = img.cols;
    int height = img.rows;

    // Separte memmory in CPU 
    unsigned char *img_toGPU = mat2ucharColor(img, width, height);
    unsigned char *cpu_outputImage=(unsigned char*)malloc(sizeof(unsigned char)*width*height*3*scale_zoom*scale_zoom);    
    
    // Separate memory en GPU
    unsigned char *gpu_inputImage, *gpu_outputImage;
    cudaMalloc( (void**)&gpu_inputImage, width*height*3);
    cudaMalloc( (void**)&gpu_outputImage, width*height*3*scale_zoom*scale_zoom);
    

    // Assign the number of grids and threads to use
    dim3 grids(width*scale_zoom, height*scale_zoom);
    dim3 threads(1, 1);

    // CPU image to GPU Image
    cudaMemcpy(gpu_inputImage, img_toGPU, width*height*3, cudaMemcpyHostToDevice);
    

    // Function gray scale in GPU 
    kernel_zoom_out_without_interpolation<<<grids, threads>>>(gpu_inputImage, gpu_outputImage, width, scale_zoom);

    // GPU Image to CPU image
    cudaMemcpy(cpu_outputImage, gpu_outputImage, width*height*3*scale_zoom*scale_zoom, cudaMemcpyDeviceToHost );
    

    // Free mmemory of GPU
    cudaFree( gpu_inputImage);
    cudaFree( gpu_outputImage);

    // Get the result in mat format to show
    Mat result_image = uchar2matColor(cpu_outputImage, scale_zoom*width, scale_zoom*height);
    
    return result_image;
}



__global__ void kernel_zoom_out_with_interpolation1( unsigned char *color_image, unsigned char *out_image, int width, int scale_zoom)
{
    // Get the position in the input image and the position in the gpu vector (offset).
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    if( x % scale_zoom == 0 && y % scale_zoom == 0)
    {
        int offset_aux = (int)(x/scale_zoom) + (int)(y/scale_zoom)*width;

        out_image[offset*3 + 0] = color_image[offset_aux*3 + 0];
        out_image[offset*3 + 1] = color_image[offset_aux*3 + 1];
        out_image[offset*3 + 2] = color_image[offset_aux*3 + 2];
    }

    else
    {
        out_image[offset*3 + 0] = 0;
        out_image[offset*3 + 1] = 0;
        out_image[offset*3 + 2] = 0;
    }
    
    
    
}


__global__ void kernel_zoom_out_with_interpolation2( unsigned char *color_image, int width, int scale_zoom)
{
    // Get the position in the input image and the position in the gpu vector (offset).
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    if( x % scale_zoom == 0 && y % scale_zoom == 0)
        return;

    else
    {
        
        int x_origin = x/scale_zoom;
        int y_origin = y/scale_zoom;

        int x1_zoom = x_origin * scale_zoom;
        int y1_zoom = y_origin * scale_zoom;
        int x2_zoom = x1_zoom + scale_zoom;
        int y2_zoom = y1_zoom + scale_zoom;

        int offset1 = y1_zoom * width*scale_zoom + x1_zoom;
        int offset2 = y1_zoom * width*scale_zoom + x2_zoom;
        int offset3 = y2_zoom * width*scale_zoom + x1_zoom;
        int offset4 = y2_zoom * width*scale_zoom + x2_zoom;

        float pixel1_red   =    ((1.0*x2_zoom-x)/(x2_zoom-x1_zoom))*color_image[offset1*3+0] + 
                                ((1.0*x-x1_zoom)/(x2_zoom-x1_zoom))*color_image[offset2*3+0];
        float pixel1_green =    ((1.0*x2_zoom-x)/(x2_zoom-x1_zoom))*color_image[offset1*3+1] + 
                                ((1.0*x-x1_zoom)/(x2_zoom-x1_zoom))*color_image[offset2*3+1];
        float pixel1_blue  =    ((1.0*x2_zoom-x)/(x2_zoom-x1_zoom))*color_image[offset1*3+2] + 
                                ((1.0*x-x1_zoom)/(x2_zoom-x1_zoom))*color_image[offset2*3+2];

        float pixel2_red   =    ((1.0*x2_zoom-x)/(x2_zoom-x1_zoom))*color_image[offset3*3+0] + 
                                ((1.0*x-x1_zoom)/(x2_zoom-x1_zoom))*color_image[offset4*3+0];
        float pixel2_green =    ((1.0*x2_zoom-x)/(x2_zoom-x1_zoom))*color_image[offset3*3+1] + 
                                ((1.0*x-x1_zoom)/(x2_zoom-x1_zoom))*color_image[offset4*3+1];
        float pixel2_blue  =    ((1.0*x2_zoom-x)/(x2_zoom-x1_zoom))*color_image[offset3*3+2] + 
                                ((1.0*x-x1_zoom)/(x2_zoom-x1_zoom))*color_image[offset4*3+2];

        float newpixel_red   =    ((1.0*y2_zoom-y)/(y2_zoom-y1_zoom))*pixel1_red + 
                                  ((1.0*y-y1_zoom)/(y2_zoom-y1_zoom))*pixel2_red;
        float newpixel_green =    ((1.0*y2_zoom-y)/(y2_zoom-y1_zoom))*pixel1_green + 
                                  ((1.0*y-y1_zoom)/(y2_zoom-y1_zoom))*pixel2_green; 
        float newpixel_blue  =    ((1.0*y2_zoom-y)/(y2_zoom-y1_zoom))*pixel1_blue + 
                                  ((1.0*y-y1_zoom)/(y2_zoom-y1_zoom))*pixel2_blue;


        color_image[offset*3 + 0] = newpixel_red;
        color_image[offset*3 + 1] = newpixel_green;
        color_image[offset*3 + 2] = newpixel_blue;
    }
    
}


/**
    RGB to CMY format:
        input : img             -> Mat RGB img
        output: result_image    -> Mat CMY result_image
**/

Mat get_zoom_out_interpolation(Mat img, int scale_zoom)
{
    int width = img.cols;
    int height = img.rows;

    // Separte memmory in CPU 
    unsigned char *img_toGPU = mat2ucharColor(img, width, height);
    unsigned char *cpu_outputImage=(unsigned char*)malloc(sizeof(unsigned char)*width*height*3*scale_zoom*scale_zoom);    
    
    // Separate memory en GPU
    unsigned char *gpu_inputImage, *gpu_outputImage;
    cudaMalloc( (void**)&gpu_inputImage, width*height*3);
    cudaMalloc( (void**)&gpu_outputImage, width*height*3*scale_zoom*scale_zoom);
    

    // Assign the number of grids and threads to use
    dim3 grids(width*scale_zoom, height*scale_zoom);
    dim3 threads(1, 1);

    // CPU image to GPU Image
    cudaMemcpy(gpu_inputImage, img_toGPU, width*height*3, cudaMemcpyHostToDevice);
    

    // Function gray scale in GPU
    kernel_zoom_out_with_interpolation1<<<grids, threads>>>(gpu_inputImage, gpu_outputImage, width, scale_zoom);
    kernel_zoom_out_with_interpolation2<<<grids, threads>>>(gpu_outputImage, width, scale_zoom);

    // GPU Image to CPU image
    cudaMemcpy(cpu_outputImage, gpu_outputImage, width*height*3*scale_zoom*scale_zoom, cudaMemcpyDeviceToHost );
    

    // Free mmemory of GPU
    cudaFree( gpu_inputImage);
    cudaFree( gpu_outputImage);

    // Get the result in mat format to show
    Mat result_image = uchar2matColor(cpu_outputImage, scale_zoom*width, scale_zoom*height);
    
    return result_image;
}

//==========================================================================================================



// =======================================LINEAR TRANSFORMATION=========================================


__global__ void kernel_linear_transformation( unsigned char *color_image, unsigned char *out_image, int width, int height, float c11, float c12, float c13, float c14, float c21, float c22, float c23, float c24)
{
    // Get the position in the input image and the position in the gpu vector (offset).
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    int new_x = int(c11*x + c12*y + c13*x*y + c14);
    int new_y = int(c21*x + c22*y + c23*x*y + c24);

    if (new_x<0 || new_x>=width || new_y<0 || new_y>=height)
    {
        return;
    }

    int offset_new = new_x + new_y*width;

    out_image[offset_new*3 + 0] = color_image[offset*3 + 0];
    out_image[offset_new*3 + 1] = color_image[offset*3 + 1];
    out_image[offset_new*3 + 2] = color_image[offset*3 + 2];
    
}


__global__ void kernel_linear_transformation_I( unsigned char *color_image, unsigned char *out_image, int width, int height, float c11, float c12, float c13, float c14, float c21, float c22, float c23, float c24)
{
    // Get the position in the input image and the position in the gpu vector (offset).
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    int new_x = c11*x + c12*y + c13*x*y + c14;
    int new_y = c21*x + c22*y + c23*x*y + c24;

    if (new_x<0 || new_x>=width || new_y<0 || new_y>=height)
    {
        return;
    }

    int offset_new = new_x + new_y*width;

    out_image[offset*3 + 0] = color_image[offset_new*3 + 0];
    out_image[offset*3 + 1] = color_image[offset_new*3 + 1];
    out_image[offset*3 + 2] = color_image[offset_new*3 + 2];
    
}

/**
    RGB to Lineal Transformation:
        input : img             -> Mat RGB img
        output: result_image    -> Mat CMY result_image
**/


void solve_Sistema(Mat &Rx, Mat &Ry,  int *F1, int *F2)
{
    float x1a = F1[0];
    float y1a = F1[1];
    float x2a = F1[2]; 
    float y2a = F1[3];
    float x3a = F1[4]; 
    float y3a = F1[5];
    float x4a = F1[6];
    float y4a = F1[7];

    float x1r = F2[0];
    float y1r = F2[1];
    float x2r = F2[2]; 
    float y2r = F2[3];
    float x3r = F2[4]; 
    float y3r = F2[5];
    float x4r = F2[6];
    float y4r = F2[7];

    


    float cx[4][4] =
    { 
     { x1a, y1a, x1a*y1a, 1},
     { x2a, y2a, x2a*y2a, 1},
     { x3a, y3a, x3a*y3a, 1},
     { x4a, y4a, x4a*y4a, 1}
    };

    float dx[4][1] = 
    { { x1r },
      { x2r },
      { x3r },
      { x4r }
    };

    Mat Cx = Mat(4, 4, CV_32FC1, cx);
    Mat Dx = Mat(4, 1, CV_32FC1, dx);
    solve(Cx, Dx, Rx);

    float cy[4][4] =
    { 
     { x1a, y1a, x1a*y1a, 1},
     { x2a, y2a, x2a*y2a, 1},
     { x3a, y3a, x3a*y3a, 1},
     { x4a, y4a, x4a*y4a, 1}
    };

    float dy[4][1] = 
    { { y1r },
      { y2r },
      { y3r },
      { y4r }
    };

    Mat Cy = Mat(4, 4, CV_32FC1, cy);
    Mat Dy = Mat(4, 1, CV_32FC1, dy);
    solve(Cy, Dy, Ry);
}

Mat get_linear_transformation(Mat img, int *F1, int *F2)
{
    Mat Rx, Ry; 
    solve_Sistema(Rx, Ry, F1, F2);
    
    float R[8] = 
    {   Rx.at<float>(0,0), Rx.at<float>(1,0), Rx.at<float>(2,0), Rx.at<float>(3,0),
        Ry.at<float>(0,0), Ry.at<float>(1,0), Ry.at<float>(2,0), Ry.at<float>(3,0) };


    int width = img.cols;
    int height = img.rows;


    // Separte memmory in CPU 
    unsigned char *img_toGPU = mat2ucharColor(img, width, height);
    unsigned char *cpu_outputImage=(unsigned char*)malloc(sizeof(unsigned char)*width*height*3);    
    
    // Separate memory en GPU
    unsigned char *gpu_inputImage, *gpu_outputImage;
    cudaMalloc( (void**)&gpu_inputImage, width*height*3);
    cudaMalloc( (void**)&gpu_outputImage, width*height*3);
    

    // Assign the number of grids and threads to use
    dim3 grids(width, height);
    dim3 threads(1, 1);

    // CPU image to GPU Image
    cudaMemcpy(gpu_inputImage, img_toGPU, width*height*3, cudaMemcpyHostToDevice);
    

    // Function gray scale in GPU
    kernel_linear_transformation<<<grids, threads>>>(gpu_inputImage, gpu_outputImage, width, height, Rx.at<float>(0,0), Rx.at<float>(1,0), Rx.at<float>(2,0), Rx.at<float>(3,0),
        Ry.at<float>(0,0), Ry.at<float>(1,0), Ry.at<float>(2,0), Ry.at<float>(3,0) );
    

    // GPU Image to CPU image
    cudaMemcpy(cpu_outputImage, gpu_outputImage, width*height*3, cudaMemcpyDeviceToHost );
    

    // Free mmemory of GPU
    cudaFree( gpu_inputImage);
    cudaFree( gpu_outputImage);

    // Get the result in mat format to show
    Mat result_image = uchar2matColor(cpu_outputImage, width, height);
    
    return result_image;
}


Mat get_linear_transformation_I(Mat img, int *F1, int *F2)
{
    Mat Rx, Ry; 
    solve_Sistema(Rx, Ry, F1, F2);
    
    float M[4][4] = 
    {   {Rx.at<float>(0,0), Rx.at<float>(1,0), Rx.at<float>(2,0), Rx.at<float>(3,0)},
        {Ry.at<float>(0,0), Ry.at<float>(1,0), Ry.at<float>(2,0), Ry.at<float>(3,0) },
        {0,0,1,0},
        {0,0,0,1}};
    Mat MR = Mat(4, 4, CV_32FC1, M);
    Mat MRI = MR.inv();
    cout << "Sitema de 3 Ecuaciones: " << endl << MR << endl << endl;
    cout << "Inversa A = "      << endl << MRI << endl << endl;


    float R[8] = 
    {   MRI.at<float>(0,0), MRI.at<float>(0,1), MRI.at<float>(0,2), MRI.at<float>(0,3),
        MRI.at<float>(1,0), MRI.at<float>(1,1), MRI.at<float>(1,2), MRI.at<float>(1,3) };
        

    int width = img.cols;
    int height = img.rows;


    // Separte memmory in CPU 
    unsigned char *img_toGPU = mat2ucharColor(img, width, height);
    unsigned char *cpu_outputImage=(unsigned char*)malloc(sizeof(unsigned char)*width*height*3);    
    
    // Separate memory en GPU
    unsigned char *gpu_inputImage, *gpu_outputImage;
    cudaMalloc( (void**)&gpu_inputImage, width*height*3);
    cudaMalloc( (void**)&gpu_outputImage, width*height*3);
    

    // Assign the number of grids and threads to use
    dim3 grids(width, height);
    dim3 threads(1, 1);

    // CPU image to GPU Image
    cudaMemcpy(gpu_inputImage, img_toGPU, width*height*3, cudaMemcpyHostToDevice);
    

    // Function gray scale in GPU
    kernel_linear_transformation_I<<<grids, threads>>>(gpu_inputImage, gpu_outputImage, width, height, Rx.at<float>(0,0), Rx.at<float>(1,0), Rx.at<float>(2,0), Rx.at<float>(3,0),
        Ry.at<float>(0,0), Ry.at<float>(1,0), Ry.at<float>(2,0), Ry.at<float>(3,0) );
    

    // GPU Image to CPU image
    cudaMemcpy(cpu_outputImage, gpu_outputImage, width*height*3, cudaMemcpyDeviceToHost );
    

    // Free mmemory of GPU
    cudaFree( gpu_inputImage);
    cudaFree( gpu_outputImage);

    // Get the result in mat format to show
    Mat result_image = uchar2matColor(cpu_outputImage, width, height);
    
    return result_image;
}


__global__ void kernel_image_combination( unsigned char *color_image, unsigned char *color_image1, unsigned char *out_image)
{
    // Get the position in the input image and the position in the gpu vector (offset).
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    
    out_image[offset*3 + 0] = (color_image[offset*3 + 0] + color_image1[offset*3 + 0])/2;
    out_image[offset*3 + 1] = (color_image[offset*3 + 1] + color_image1[offset*3 + 1])/2;
    out_image[offset*3 + 2] = (color_image[offset*3 + 2] + color_image1[offset*3 + 2])/2;
    
}


/**
    RGB to Perspective Transformation:
        input : img             -> Mat RGB img
        output: result_image    -> Mat CMY result_image
**/

Mat get_image_combination(Mat img, Mat img1)
{
    int width = img.cols;
    int height = img.rows;

    // Separte memmory in CPU 
    unsigned char *img_toGPU = mat2ucharColor(img, width, height);
    unsigned char *img1_toGPU = mat2ucharColor(img1, width, height);
    unsigned char *cpu_outputImage=(unsigned char*)malloc(sizeof(unsigned char)*width*height*3);    
    
    // Separate memory en GPU
    unsigned char *gpu_inputImage, *gpu_inputImage1, *gpu_outputImage;
    cudaMalloc( (void**)&gpu_inputImage, width*height*3);
    cudaMalloc( (void**)&gpu_inputImage1, width*height*3);
    cudaMalloc( (void**)&gpu_outputImage, width*height*3);
    

    // Assign the number of grids and threads to use
    dim3 grids(width, height);
    dim3 threads(1, 1);

    // CPU image to GPU Image
    cudaMemcpy(gpu_inputImage, img_toGPU, width*height*3, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_inputImage1, img1_toGPU, width*height*3, cudaMemcpyHostToDevice);
    

    // Function gray scale in GPU
    kernel_image_combination<<<grids, threads>>>(gpu_inputImage, gpu_inputImage1, gpu_outputImage);
    

    // GPU Image to CPU image
    cudaMemcpy(cpu_outputImage, gpu_outputImage, width*height*3, cudaMemcpyDeviceToHost );
    

    // Free mmemory of GPU
    cudaFree( gpu_inputImage);
    cudaFree( gpu_inputImage1);
    cudaFree( gpu_outputImage);

    // Get the result in mat format to show
    Mat result_image = uchar2matColor(cpu_outputImage, width, height);
    
    return result_image;
}


int main()
{
    printf("Hello \n");

    // Load Data
    char img_name [] = "pez.jpg";

    //Read data in color format
    Mat image;
    image = imread(img_name, CV_LOAD_IMAGE_COLOR);

    // Gray scale GPU
    Mat gray_image = get_grayScale_gpu(image);

    
    // Sobel edge detector
    Mat sobel_image = get_sobel_gpu(gray_image);

    // Mean filter
    Mat mean_filter_image = get_meanFilter_gpu(gray_image);

    // Median Filter
    Mat image_salt_pepper = imread("img/brain.jpg", CV_LOAD_IMAGE_COLOR);
    Mat g_image = get_grayScale_gpu(image_salt_pepper);
    Mat median_filter_image = get_medianFilter_gpu(g_image);
    median_filter_image = get_medianFilter_gpu(median_filter_image);
    median_filter_image = get_medianFilter_gpu(median_filter_image);
    
    // RGB to CMY
    Mat CMY_image = get_RGB_to_CMY(image);

    // RGB to HSV
    Mat HSV_image = get_RGB_to_HSV(image);

    // ZOOM OUT
    Mat image_fish = imread("img/yellow_fish.jpg", CV_LOAD_IMAGE_COLOR);
    Mat image_fish1 = imread("img/yellow_fish1.jpg", CV_LOAD_IMAGE_COLOR);
    Mat zoom_out_image = get_zoom_out(image_fish1, 4);
    Mat zoom_out_image_interpolation = get_zoom_out_interpolation(image_fish1, 4);

    // LINEAR TRANSFORMATION
    image = imread("img/foto.jpg", CV_LOAD_IMAGE_COLOR);

    int *F1 = new int[8];
    F1[0] = 0;          F1[1] = 0;//x, y
    F1[2] = image.cols; F1[3] = 0;
    F1[4] = image.cols; F1[5] = image.rows;
    F1[6] = 0;          F1[7] = image.rows;

    int *F2 = new int[8];
    F2[0] = 250; F2[1] = 200;//x, y
    F2[2] = 600; F2[3] = 100;
    F2[4] = 700; F2[5] = 700;
    F2[6] = 150; F2[7] = 600;
    
    Mat linear_image = get_linear_transformation(image, F1, F2);
    circle( linear_image, Point( F2[0], F2[1] ), 3, Scalar( 0, 0, 255 ), 1, 8 );
    circle( linear_image, Point( F2[2], F2[3] ), 3, Scalar( 0, 0, 255 ), 1, 8 );
    circle( linear_image, Point( F2[4], F2[5] ), 3, Scalar( 0, 0, 255 ), 1, 8 );
    circle( linear_image, Point( F2[6], F2[7] ), 3, Scalar( 0, 0, 255 ), 1, 8 );

    Mat linear_image_I = get_linear_transformation_I(linear_image, F1, F2);

    


    // IMAGES COMBINATION
    Mat photo = imread("img/foto.jpg", CV_LOAD_IMAGE_COLOR);
    resize(image, image, Size(photo.cols, photo.rows));
    Mat image_combination = get_image_combination(photo, image);

    // Show results
    //imshow("ORIGINAL", image);
    //imshow( "GRAY ", gray_image); 
    //imshow( "SOBEL ", sobel_image);                   
    //imshow( "MEAN FILTER", mean_filter_image); 
    //imshow( "MEDIAN FILTER", median_filter_image);
    //imshow( "CMY IMAGE", CMY_image);
    //imshow( "HSV IMAGE", HSV_image);
    //imshow( "ORIGINAL IMAGE", image_fish1);
    //imshow( "ZOOM OUT IMAGE WITHOUT INTERPOLATION", zoom_out_image);
    //imshow( "ZOOM OUT IMAGE WITH INTERPOLATION", zoom_out_image_interpolation);
    imshow( "LINEAR TRANSFORMATION", linear_image);
    imshow( "LINEAR TRANSFORMATION_I", linear_image_I);
    //imshow( "IMAGE COMBINATION", image_combination);

    waitKey(0);                                        // Wait for a keystroke in the window
    return 0;
}
