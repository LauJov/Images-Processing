// g++ -std=c++11 -o main main.c `pkg-config --cflags --libs opencv` && ./main

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <stdio.h>
#include <stdlib.h>

#include <iostream>

using namespace cv;
using namespace std;

typedef struct bmpFileHeader
{
	// 2 bytes de identificación
	uint32_t size;        // Tamaño del archivo
	uint16_t resv1;       // Reservado
	uint16_t resv2;       // Reservado
	uint32_t offset;      // Offset hasta hasta los datos de imagen
} bmpFileHeader;

typedef struct bmpInfoHeader
{
	uint32_t headersize;  // Tamaño de la cabecera
	uint32_t width;       // Ancho
	uint32_t height;      // Alto
	uint16_t planes;      // Planos de color (Siempre 1)
	uint16_t bpp;         // bits por pixel
	uint32_t compress;    // compresión
	uint32_t imgsize;     // tamaño de los datos de imagen
	uint32_t bpmx;        // Resolución X en bits por metro
	uint32_t bpmy;        // Resolución Y en bits por metro
	uint32_t colors;      // colors used en la paleta
	uint32_t imxtcolors;  // Colores importantes. 0 si son todos
} bmpInfoHeader;


void DisplayInfo(bmpInfoHeader *info)
{
	printf("Tamaño de la cabecera: %u\n", info->headersize);
	printf("Anchura: %d\n", info->width);
	printf("Altura: %d\n", info->height);
	printf("Planos (1): %d\n", info->planes);
	printf("Bits por pixel: %d\n", info->bpp);
	printf("Compresión: %d\n", info->compress);
	printf("Tamaño de datos de imagen: %u\n", info->imgsize);
	printf("Resolucón horizontal: %u\n", info->bpmx);
	printf("Resolucón vertical: %u\n", info->bpmy);
	printf("Colores en paleta: %d\n", info->colors);
	printf("Colores importantes: %d\n", info->imxtcolors);
}

unsigned char *LoadBMP(char *filename, bmpInfoHeader *bInfoHeader)
{
	FILE *f;
	bmpFileHeader header;     // cabecera
	unsigned char *imgdata;   // datos de imagen
	uint16_t type;        	  // 2 bytes identificativos

	f=fopen (filename, "r");
	if (!f)
		return NULL;        // Si no podemos leer, no hay imagen

	// Leemos los dos primeros bytes
	fread(&type, sizeof(uint16_t), 1, f);
	if (type !=0x4D42)        // Comprobamos el formato
	{
		fclose(f);
		return NULL;
	}
	// Leemos la cabecera de fichero completa
	fread(&header, sizeof(bmpFileHeader), 1, f);
	// Leemos la cabecera de información completa
	fread(bInfoHeader, sizeof(bmpInfoHeader), 1, f);

	if (int(bInfoHeader->bpp) == 24)
	{
		//bInfoHeader->width = bInfoHeader->width+1;
		bInfoHeader->imgsize = bInfoHeader->height*bInfoHeader->width*3;
	}
	else
	{
		bInfoHeader->imgsize = bInfoHeader->height*bInfoHeader->width;
	}
	// Reservamos memoria para la imagen, ¿cuánta?
	//Tanto como indique imgsize
	imgdata=(unsigned char*)malloc(sizeof(unsigned char)*bInfoHeader->imgsize);

	// Nos situamos en el sitio donde empiezan los datos de imagen,
	// nos lo indica el offset de la cabecera de fichero
	fseek(f, header.offset, SEEK_SET);

	// Leemos los datos de imagen, tantos bytes como imgsize
	fread(imgdata, bInfoHeader->imgsize,1, f);

	// Cerramos
	fclose(f);

	// Devolvemos la imagen
	return imgdata;
}

void mostrarImagenBuffer(unsigned char *bufImg, int ancho, int alto, int bpp, string nombre)
{
	int j=0;
	if(bpp==8)
	{
		Mat img = Mat::zeros(alto, ancho, CV_8UC1);
		for(int y=alto-1; y>=0; y--)
		{
			for(int x=0; x<ancho; x++)
			{
				img.at<uchar>(y,x) = bufImg[j];
				j++;
			}
		}
		imshow( nombre, img );
	}
	else
	{
		Mat img = Mat::zeros(alto, ancho, CV_8UC3);
		for(int y=alto-1; y>=0; y--)
		{
			for(int x=0; x<ancho; x++)
			{
				img.at<Vec3b>(y,x)[0] = bufImg[j];
				j++;
				img.at<Vec3b>(y,x)[1] = bufImg[j];
				j++;
				img.at<Vec3b>(y,x)[2] = bufImg[j];
				j++;
			}
		}
		imshow( nombre, img );
	}
}



int main()
{
	bmpInfoHeader infoImg;
	unsigned char *imgRead;

	//imgRead=LoadBMP((char *)"img/rio.bmp", &infoImg);
	//imgRead=LoadBMP((char *)"img/lena_gray.bmp", &infoImg);
	imgRead=LoadBMP((char *)"img/ima3.bmp", &infoImg);
	
	//DisplayInfo(&infoImg);


	//mostrarImagenBuffer(imgRead, infoImg.width, infoImg.height, 8, "Inicio");
	mostrarImagenBuffer(imgRead, infoImg.width, infoImg.height, int(infoImg.bpp), "Inicio");

	

	waitKey(0);
	return 0;
}




