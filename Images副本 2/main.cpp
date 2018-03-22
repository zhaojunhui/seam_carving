#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <cmath>
#include "opencv2/opencv.hpp"

#define MAX_INT 1000000000

using namespace std;
using namespace cv;

struct MouseArgs
{
	Mat *mat;
	int **mask;
	Vec3b color;
	MouseArgs(Mat *mat, int **mask, Vec3b color) : mat(mat), mask(mask), color(color) {}
};

//读入图片、操作类型、算子类型、以及目标图片的大小
int Input(Mat *&mat, int &rows, int &cols, int &new_rows, int &new_cols, int &flags, int &mode, char* &dir)
{
	printf("请输入图像文件名:");
	scanf("%s", dir);
	Mat pic = imread(dir, CV_LOAD_IMAGE_COLOR);
	rows = pic.rows;
	cols = pic.cols;
	mat = new Mat(rows, cols, CV_8UC3);
	Vec3b *p, *ansp;
	for(int i = 0; i < rows; ++i)
	{
		p = pic.ptr<Vec3b>(i);
		ansp = mat->ptr<Vec3b>(i);
		for(int j = 0; j < cols; ++j)
		{
			ansp[j][0] = p[j][0];
			ansp[j][1] = p[j][1];
			ansp[j][2] = p[j][2];
		}
	}
	if(!mat->data)
	{
		printf("load failed!\n");
		return 1;
	}
	if(rows == 0 || cols == 0) return 2;
	printf("the picture is %d * %d\n", rows, cols);
	printf("请选择想要的算子类型：(0：自实现梯度算子 1：Sobel 2：Laplacian 3：canny)：");
	scanf("%d", &mode);
	printf("请问您是想要缩放图片还是抠图：（0：缩放 1：抠图）：");
	scanf("%d", &flags);
	if(flags == 0)
	{
		printf("请选择想要修改为大小（格式：r c）：");
		scanf("%d%d", &new_rows, &new_cols);
	}
	else
	{
		new_rows = 0;
		new_cols = 0;
	}
	return 0;
}

//初始化操作，主要是建立用于后续算法所需的数组
void init(Mat *&mat, int **&mapX, int **&mapY, int **&image, int **&image_withlines, int **&gradX, int **&gradY, int **&energy, int **&mask_remove, int **&mask_highlight, int rows, int cols, int new_rows, int new_cols)
{
	int maxr = (rows > new_rows) ? rows : new_rows;
	int maxc = (cols > new_cols) ? cols : new_cols;
	mapX = new int*[maxr];
	mapY = new int*[maxr];
	image = new int*[maxr];
	image_withlines = new int*[maxr];
	gradX = new int*[maxr];
	gradY = new int*[maxr];
	energy = new int*[maxr];
	mask_remove = new int*[maxr];
	mask_highlight = new int*[maxr];
	for(int i = 0; i < maxr; ++i)
	{
		mapX[i] = new int[maxc];
		mapY[i] = new int[maxc];
		image[i] = new int[maxc];
		image_withlines[i] = new int[maxc];
		gradX[i] = new int[maxc];
		gradY[i] = new int[maxc];
		energy[i] = new int[maxc];
		mask_remove[i] = new int[maxc];
		mask_highlight[i] = new int[maxc];
	}
	for(int i = 0; i < maxr; ++i)
	{
		for(int j = 0; j < maxc; ++j)
		{
			mapX[i][j] = i;
			mapY[i][j] = j;
			mask_remove[i][j] = 0;
			mask_highlight[i][j] = 0;
		}
	}
	uchar *p;
	for(int i = 0; i < rows; ++i)
	{
		p = mat->ptr<uchar>(i);
		for(int j = 0; j < cols; ++j)
		{
				image[i][j] = (int)p[j];
				image_withlines[i][j] = (int)p[j];
		}
	}
}

//X方向梯度计算
void gradientX_define(int **image, int r, int c, int **grad)
{
	for(int i = 0; i < r; ++i)
	{
		grad[i][0] = abs(image[i][1] - image[i][0]);
		for(int j = 1; j < c - 1; ++j)
			grad[i][j] = abs(image[i][j + 1] - image[i][j - 1]) * 0.5;
		grad[i][c - 1] = abs(image[i][c - 1] - image[i][c - 2]);
	}
}

//Y方向梯度计算
void gradientY_define(int **image,int r, int c, int **grad)
{
	for(int j = 0; j < c; ++j)
		grad[0][j] = abs(image[1][j] - image[0][j]);
	for(int i = 1; i < r - 1; ++i)
	{
		for(int j = 0; j < c; ++j)
			grad[i][j] = abs(image[i + 1][j] - image[i - 1][j]) * 0.5;
	}
	for(int j = 0; j < c; ++j)
		grad[r - 1][j] = abs(image[r - 1][j] - image[r - 2][j]);
}

//根据算子和是否有对象删除和保护计算能量函数
void energy_define(Mat *mat, int **gradX, int **gradY, int **energy,int **mask_remove, int **mask_highlight, int r, int c, int flags, int mode)
{
	if(mode == 0)
		for(int i  = 0; i < r; ++i)
			for(int j = 0; j < c; ++j)
				energy[i][j] = gradX[i][j] + gradY[i][j];
	else if(mode == 1)
	{
		Mat kernel_Sobel_H = (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    	Mat kernel_Sobel_V = (cv::Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    	Mat H = Mat(r, c, CV_32F);
    	Mat V = Mat(r, c, CV_32F);
    	Mat mat_gray = Mat(r, c, CV_8U);
    	cvtColor(*mat, mat_gray, CV_BGR2GRAY);
    	filter2D(mat_gray, H, H.depth(), kernel_Sobel_H);
    	filter2D(mat_gray, V, V.depth(), kernel_Sobel_V);
    	for(int i = 0; i < r; ++i)
    	{
    		uchar *mp = mat_gray.ptr<uchar>(i);
    		float *hp = H.ptr<float>(i);
    		float *vp = V.ptr<float>(i);
    		for(int j = 0; j < c; ++j)
    		{
    			energy[i][j] = (int)(abs(hp[j]) + abs(vp[j]));
    			printf("%d %f %f\n", energy[i][j], hp[j], vp[j]);
    		}
    	}
	}
	else if(mode == 2)
	{
		Mat kernel_Laplace_H = (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    	Mat kernel_Laplace_V = (cv::Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    	Mat H = Mat(r, c, CV_32F);
    	Mat V = Mat(r, c, CV_32F);
    	Mat mat_gray = Mat(r, c, CV_32F);
    	cvtColor(*mat, mat_gray, CV_BGR2GRAY);
    	filter2D(mat_gray, H, H.depth(), kernel_Laplace_H);
    	filter2D(mat_gray, V, V.depth(), kernel_Laplace_V);
    	for(int i = 0; i < r; ++i)
    	{
    		float *hp = H.ptr<float>(i);
    		float *vp = V.ptr<float>(i);
    		for(int j = 0; j < c; ++j)
    		{
    			energy[i][j] = (int)(abs(hp[j]) + abs(vp[j]));
    		}
    	}
	}
	else if(mode == 3)
	{
		Mat kernel_Roberts_H = (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    	Mat kernel_Roberts_V = (cv::Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    	Mat H = Mat(r, c, CV_32F);
    	Mat V = Mat(r, c, CV_32F);
    	Mat mat_gray = Mat(r, c, CV_32F);
    	cvtColor(*mat, mat_gray, CV_BGR2GRAY);
    	filter2D(mat_gray, H, H.depth(), kernel_Roberts_H);
    	filter2D(mat_gray, V, V.depth(), kernel_Roberts_V);
    	for(int i = 0; i < r; ++i)
    	{
    		float *hp = H.ptr<float>(i);
    		float *vp = V.ptr<float>(i);
    		for(int j = 0; j < c; ++j)
    		{
    			energy[i][j] = (int)(abs(hp[j]) + abs(vp[j]));
    		}
    	}
	}
	if(flags == 1)
	{
		for(int i  = 0; i < r; ++i)
			for(int j = 0; j < c; ++j)
			{
				if(mask_highlight[i][j] == 1) energy[i][j] += 2500000;
				else if(mask_remove[i][j] == 1) energy[i][j] -= 2500000;
			}
	}
}

//找出X方向能量最小的seam
inline int min_energyX(int pre_r, int pre_c, int **M, int **image, int r, int c)
{
	int ans = M[pre_r][pre_c];
	if(pre_r != 0) ans = (ans < M[pre_r - 1][pre_c]) ? ans : M[pre_r - 1][pre_c];
	if(pre_r != r - 1) ans = (ans < M[pre_r + 1][pre_c]) ? ans : M[pre_r + 1][pre_c];
	return ans; 
}

//找出Y方向能量最小的seam
inline int min_energyY(int pre_r, int pre_c, int **M, int **image, int r, int c)
{
	int ans = M[pre_r][pre_c];
	if(pre_c != 0) ans = (ans < M[pre_r][pre_c - 1]) ? ans : M[pre_r][pre_c - 1];
	if(pre_c != c - 1) ans = (ans < M[pre_r][pre_c + 1]) ? ans : M[pre_r][pre_c + 1];
	return ans; 
}

//去除Xseam
void reduceX(int **mapX, int **mapY, int **image, int **image_withlines, int **energy, int **mask_remove, int **mask_highlight, int r, int c, int flags)
{
	int **M = new int*[r];
	for(int i = 0; i < r; ++i) M[i] = new int[c];

	for(int i = 0; i < r; ++i) M[i][0] = energy[i][0];
	for(int j = 1; j < c; ++j)
		for(int i = 0; i < r; ++i)
			M[i][j] = energy[i][j] + min_energyX(i, j - 1, M, image, r, c);

	int ans[c];
	int min = MAX_INT;
	for(int i = 0; i < r; ++i)
	{
		if(M[i][c - 1] < min)
		{
			min = M[i][c - 1];
			ans[c - 1] = i;
		}
	}
	for(int j = c - 2; j >= 0; --j)
	{
		if(ans[j + 1] == 0)
		{
			if(M[0][j] < M[1][j]) ans[j] = 0;
			else ans[j] = 1;
		}
		else if(ans[j + 1] == r - 1)
		{
			if(M[r - 1][j] < M[r - 2][j]) ans[j] = r - 1;
			else ans[j] = r - 2;
		}
		else
		{
			ans[j] = (M[ans[j + 1] - 1][j] < M[ans[j + 1]][j]) ? ans[j + 1] - 1 : ans[j + 1];
			ans[j] = (M[ans[j]][j] < M[ans[j + 1] + 1][j]) ? ans[j] : ans[j + 1] + 1;
		}
	}

	for(int j = 0; j < c; ++j)
	{
		image_withlines[mapX[ans[j]][j]][mapY[ans[j]][j]] = -1;
		for(int i = ans[j]; i < r - 1; ++i)
		{
			image[i][j] = image[i + 1][j];
			if(flags == 1)
			{
				mask_remove[i][j] = mask_remove[i + 1][j];
				mask_highlight[i][j] = mask_highlight[i + 1][j]; 
			}
		}
		for(int i = ans[j]; i < r - 1; ++i)
			mapX[i][j] = mapX[i + 1][j], mapY[i][j] = mapY[i + 1][j];
	}
}

//去除Yseam
void reduceY(int **mapX, int **mapY, int **image, int **image_withlines, int **energy, int **mask_remove, int **mask_highlight, int r, int c, int flags)
{
	int **M = new int*[r];
	for(int i = 0; i < r; ++i) M[i] = new int[c];

	for(int j = 0; j < c; ++j) M[0][j] = energy[0][j];
	for(int i = 1; i < r; ++i)
		for(int j = 0; j < c; ++j)
			M[i][j] = energy[i][j] + min_energyY(i - 1, j, M, image, r, c);

	int ans[r];
	int min = MAX_INT;
	for(int j = 0; j < c; ++j)
	{
		if(M[r - 1][j] < min)
		{
			min = M[r - 1][j];
			ans[r - 1] = j;
		}
	}
	for(int i = r - 2; i >= 0; --i)
	{
		if(ans[i + 1] == 0)
		{
			if(M[i][0] < M[i][1]) ans[i] = 0;
			else ans[i] = 1;
		}
		else if(ans[i + 1] == c - 1)
		{
			if(M[i][c - 1] < M[i][c - 2]) ans[i] = c - 1;
			else ans[i] = c - 2;
		}
		else
		{
			ans[i] = (M[i][ans[i + 1] - 1] < M[i][ans[i + 1]]) ? ans[i + 1] - 1 : ans[i + 1];
			ans[i] = (M[i][ans[i]] < M[i][ans[i + 1] + 1]) ? ans[i] : ans[i + 1] + 1;
		}
	}

	for(int i = 0; i < r; ++i)
	{
		image_withlines[mapX[i][ans[i]]][mapY[i][ans[i]]] = -2;
		for(int j = ans[i]; j < c - 1; ++j)
		{
			image[i][j] = image[i][j + 1];
			if(flags == 1)
			{
				mask_remove[i][j] = mask_remove[i][j + 1];
				mask_highlight[i][j] = mask_highlight[i][j + 1];
			}
		}
		for(int j = ans[i]; j < c - 1; ++j)
			mapX[i][j] = mapX[i][j + 1], mapY[i][j] = mapY[i][j + 1];
	}	
}

//输出删除后的结果
void ReduceOutput(Mat *mat, int **mapX, int **mapY, int **image, int **image_withlines, int rows, int cols, int new_rows, int new_cols, char* dir_output, char* dir_output_withlines)
{
	Mat *new_mat = new Mat(new_rows, new_cols, CV_8UC3);
	Vec3b *p;
	for(int i = 0; i < new_rows; ++i)
	{
		p = new_mat->ptr<Vec3b>(i);
		for(int j = 0; j < new_cols; ++j)
		{
			p[j][0] = mat->at<Vec3b>(mapX[i][j], mapY[i][j])[0];
			p[j][1] = mat->at<Vec3b>(mapX[i][j], mapY[i][j])[1];
			p[j][2] = mat->at<Vec3b>(mapX[i][j], mapY[i][j])[2];
		}
	}
	imwrite(dir_output, *new_mat);

	Mat *new_mat_withlines = new Mat(*mat);
	for(int i = 0; i < rows; ++i)
		for(int j = 0; j < cols; ++j)
		{
			if(image_withlines[i][j] == -1)
			{
				new_mat_withlines->at<Vec3b>(i, j)[0] = 0;
				new_mat_withlines->at<Vec3b>(i, j)[1] = 0;
				new_mat_withlines->at<Vec3b>(i, j)[2] = 255;
			}
			else if(image_withlines[i][j] == -2)
			{
				new_mat_withlines->at<Vec3b>(i, j)[0] = 255;
				new_mat_withlines->at<Vec3b>(i, j)[1] = 0;
				new_mat_withlines->at<Vec3b>(i, j)[2] = 0;
			}
		}
	imwrite(dir_output_withlines, *new_mat_withlines);
}

//缩小图片
void reduce(Mat *mat, int **mapX, int **mapY, int **gradX, int **gradY, int **image, int **image_withlines, int **energy, int **mask_remove, int **mask_highlight, int rows, int cols, int new_rows, int new_cols, int flags, int mode, char* dir)
{
	int r = rows, c = cols;
	int total = r - new_rows + c - new_cols;
	int num = 0;
	printf("正在删除seam...\n");
	while(r != new_rows)
	{
		gradientX_define(image, r, c, gradX);
		gradientY_define(image, r, c, gradY);
		energy_define(mat, gradX, gradY, energy, mask_remove, mask_highlight, r, c, flags, mode);
		reduceX(mapX, mapY, image, image_withlines, energy, mask_remove, mask_highlight, r, c, flags);
		r--;
		fprintf(stderr, "\r%d%c", (++num) * 100 / total, '%');	
	}
	while(c != new_cols)
	{
		gradientX_define(image, r, c, gradX);
		gradientY_define(image, r, c, gradY);
		energy_define(mat, gradX, gradY, energy, mask_remove, mask_highlight, r, c, flags, mode);
		reduceY(mapX, mapY, image, image_withlines, energy, mask_remove, mask_highlight, r, c, flags);
		c--;
		fprintf(stderr, "\r%d%c", (++num) * 100 / total, '%');
	}

	char* dir_output = new char[1000];
	char* dir_output_withlines = new char[1000];
	int len = strlen(dir);
	int tmplen;
	for(tmplen = 0; tmplen < len; ++tmplen)
	{
		if(dir[tmplen] == '.') break;
		dir_output[tmplen] = dir[tmplen];
	}
	dir_output[tmplen] = '\0';
	char* suffix = new char[50];
	switch(mode)
	{
		case 0 : suffix = (char*)"_reduced_userDefined"; break;
		case 1 : suffix = (char*)"_reduced_Sobel"; break;
		case 2 : suffix = (char*)"_reduced_Laplacian"; break;
		case 3 : suffix = (char*)"_reduced_Canny"; break;
	}
	dir_output = strcat(dir_output, suffix);
	strcpy(dir_output_withlines, dir_output);
	dir_output_withlines = strcat(dir_output_withlines, "_withlines");
	dir_output = strcat(dir_output, ".png");
	dir_output_withlines = strcat(dir_output_withlines, ".png");

	ReduceOutput(mat, mapX, mapY, image, image_withlines, rows, cols, new_rows, new_cols, dir_output, dir_output_withlines);
	printf("\n已成功生成图片:\n%s\n%s\n", dir_output, dir_output_withlines);
}

//放大图片
void enlarge(Mat *mat, int **mapX, int **mapY, int **gradX, int **gradY, int **image, int **image_withlines, int **energy, int **mask_remove, int **mask_highlight, int rows, int cols, int new_rows, int new_cols, int flags, int mode, char* dir)
{
	int dr = new_rows - rows;
	int dc = new_cols - cols;
	Mat *new_mat1 = new Mat(new_rows, new_cols, CV_8UC3);
	Mat *new_mat2 = new Mat(new_rows, new_cols, CV_8UC3);
	Mat *new_mat1_withlines = new Mat(new_rows, new_cols, CV_8UC3);
	Mat *new_mat2_withlines = new Mat(new_rows, new_cols, CV_8UC3);

	int r = rows, c = cols;
	int ideal_r = rows - dr;
	int ideal_c = cols - dc;
	int total = dr;
	int num = 0;
	printf("\n添加行seam中...\n");
	while(r > ideal_r)
	{	
		gradientX_define(image, r, c, gradX);
		gradientY_define(image, r, c, gradY);
		energy_define(mat, gradX, gradY, energy, mask_remove, mask_highlight, r, c, flags, mode);
		reduceX(mapX, mapY, image, image_withlines, energy, mask_remove, mask_highlight, r, c, flags);
		r--;
		fprintf(stderr, "\r%d%c", (++num) * 100 / total, '%');	
	}
	
	r = rows, c = cols;
	for(int i = 0; i < r; ++i)
	{
		for(int j = 0; j < c; ++j)
		{
			mapX[i][j] = i;
			mapY[i][j] = j;
		}
	}
	for(int j = 0; j < c; ++j)
	{
		if(image_withlines[0][j] == -1)
		{
			for(int ii = 1; ii < r; ++ii)
				mapX[ii][j]++;
		}
		for(int i = 1; i < r; ++i)
		{
			if(image_withlines[i][j] == -1)
			{
				for(int ii = i; ii < r; ++ii)
					mapX[ii][j]++;
			}
		}
	}
	Vec3b *p1, *p2, *p3, *p4;
	for(int j = 0; j < c; ++j)
		for(int i = 0; i < r - 1; ++i)
		{
			p1 = mat->ptr<Vec3b>(i);
			p2 = mat->ptr<Vec3b>(i + 1);
			new_mat1->at<Vec3b>(mapX[i][j], j)[0] = p1[j][0];
			new_mat1->at<Vec3b>(mapX[i][j], j)[1] = p1[j][1];
			new_mat1->at<Vec3b>(mapX[i][j], j)[2] = p1[j][2];
			new_mat1_withlines->at<Vec3b>(mapX[i][j], j)[0] = p1[j][0];
			new_mat1_withlines->at<Vec3b>(mapX[i][j], j)[1] = p1[j][1];
			new_mat1_withlines->at<Vec3b>(mapX[i][j], j)[2] = p1[j][2];				
			for(int ii = mapX[i][j] + 1; ii < mapX[i + 1][j]; ++ii)
			{
				new_mat1->at<Vec3b>(ii, j)[0] = (p1[j][0] + p2[j][0]) / 2;
				new_mat1->at<Vec3b>(ii, j)[1] = (p1[j][1] + p2[j][1]) / 2;
				new_mat1->at<Vec3b>(ii, j)[2] = (p1[j][2] + p2[j][2]) / 2;
				new_mat1_withlines->at<Vec3b>(ii, j)[0] = 0;
				new_mat1_withlines->at<Vec3b>(ii, j)[1] = 0;
				new_mat1_withlines->at<Vec3b>(ii, j)[2] = 255;
			}
			new_mat1->at<Vec3b>(mapX[i + 1][j], j)[0] = p2[j][0];
			new_mat1->at<Vec3b>(mapX[i + 1][j], j)[1] = p2[j][1];
			new_mat1->at<Vec3b>(mapX[i + 1][j], j)[2] = p2[j][2];
			new_mat1_withlines->at<Vec3b>(mapX[i + 1][j], j)[0] = p2[j][0];
			new_mat1_withlines->at<Vec3b>(mapX[i + 1][j], j)[1] = p2[j][1];
			new_mat1_withlines->at<Vec3b>(mapX[i + 1][j], j)[2] = p2[j][2];
		}


	Mat *new_mat_gray = new Mat;
	cvtColor(*new_mat1, *new_mat_gray, CV_BGR2GRAY);
	r = new_rows, c = cols;
	uchar *p;
	for(int i = 0; i < r; ++i)
	{
		p = new_mat_gray->ptr<uchar>(i);
		for(int j = 0; j < c; ++j)
		{
			image[i][j] = p[j];
			image_withlines[i][j] = p[j];
			mapX[i][j] = i;
			mapY[i][j] = j;
		}
	}

	total = dc;
	num = 0;
	printf("\n添加列seam中...\n");
	while(c > ideal_c)
	{
		gradientX_define(image, r, c, gradX);
		gradientY_define(image, r, c, gradY);
		energy_define(mat, gradX, gradY, energy, mask_remove, mask_highlight, r, c, flags, mode);
		reduceY(mapX, mapY, image, image_withlines, energy, mask_remove, mask_highlight, r, c, flags);
		c--;
		fprintf(stderr, "\r%d%c", (++num) * 100 / total, '%');
	}

	r = new_rows, c = cols;
	for(int i = 0; i < r; ++i)
	{
		p = new_mat_gray->ptr<uchar>(i);
		for(int j = 0; j < c; ++j)
		{
			mapX[i][j] = i;
			mapY[i][j] = j;
		}
	}

	for(int i = 0; i < r; ++i)
	{	
		if(image_withlines[i][0] == -2)
		{
			for(int jj = 1; jj < c; ++jj)
				mapY[i][jj]++;				
		}
		for(int j = 1; j < c; ++j)
		{
			if(image_withlines[i][j] == -2)
			{
				for(int jj = j; jj < c; ++jj)
					mapY[i][jj]++;
			}
		}
	}

	for(int i = 0; i < r; ++i)
		for(int j = 0; j < c - 1; ++j)
		{
			p1 = new_mat1->ptr<Vec3b>(i);
			p2 = new_mat2->ptr<Vec3b>(i);
			p3 = new_mat1_withlines->ptr<Vec3b>(i);
			p4 = new_mat2_withlines->ptr<Vec3b>(i);
			p2[mapY[i][j]][0] = p1[j][0];
			p2[mapY[i][j]][1] = p1[j][1];
			p2[mapY[i][j]][2] = p1[j][2];
			p4[mapY[i][j]][0] = p3[j][0];
			p4[mapY[i][j]][1] = p3[j][1];
			p4[mapY[i][j]][2] = p3[j][2];
			for(int jj = mapY[i][j] + 1; jj < mapY[i][j + 1]; ++jj)
			{
				p2[jj][0] = (p1[j][0] + p1[j + 1][0]) / 2;
				p2[jj][1] = (p1[j][1] + p1[j + 1][1]) / 2;
				p2[jj][2] = (p1[j][2] + p1[j + 1][2]) / 2;
				p4[jj][0] = 255;
				p4[jj][1] = 0;
				p4[jj][2] = 0;
			}
			p2[mapY[i][j + 1]][0] = p1[j + 1][0];
			p2[mapY[i][j + 1]][1] = p1[j + 1][1];
			p2[mapY[i][j + 1]][2] = p1[j + 1][2];
			p4[mapY[i][j + 1]][0] = p3[j + 1][0];
			p4[mapY[i][j + 1]][1] = p3[j + 1][1];
			p4[mapY[i][j + 1]][2] = p3[j + 1][2];
		}

	char* dir_output = new char[1000];
	char* dir_output_withlines = new char[1000];
	int len = strlen(dir);
	int tmplen;
	for(tmplen = 0; tmplen < len; ++tmplen)
	{
		if(dir[tmplen] == '.') break;
		dir_output[tmplen] = dir[tmplen];
	}
	dir_output[tmplen] = '\0';
	char* suffix = new char[50];
	switch(mode)
	{
		case 0 : suffix = (char*)"_enlarged_userDefined"; break;
		case 1 : suffix = (char*)"_enlarged_Sobel"; break;
		case 2 : suffix = (char*)"_enlarged_Laplacian"; break;
		case 3 : suffix = (char*)"_enlarged_Canny"; break;
	}
	dir_output = strcat(dir_output, suffix);
	strcpy(dir_output_withlines, dir_output);
	dir_output_withlines = strcat(dir_output_withlines, "_withlines");
	dir_output = strcat(dir_output, ".png");
	dir_output_withlines = strcat(dir_output_withlines, ".png");

	imwrite(dir_output, *new_mat2);
	imwrite(dir_output_withlines, *new_mat2_withlines);
	printf("\n已成功生成图片:\n%s\n%s\n", dir_output, dir_output_withlines);	
}

//UI界面鼠标行为
void onMouse(int event, int x, int y, int flags, void* param)
{
	MouseArgs *args = (MouseArgs *)param;
	if((event == CV_EVENT_MOUSEMOVE || event == CV_EVENT_LBUTTONDOWN) && (flags & CV_EVENT_FLAG_LBUTTON))
	{
		int radius = 10;
		int rows = args->mat->rows, cols = args->mat->cols;
		int starti = (y - radius > 0) ? y - radius : 0;
		int endi = (y + radius < rows) ? y + radius : rows;
		for(int i = starti; i < endi; ++i)
		{
			int halfChord = sqrt(radius*radius - (i - y)*(i - y));
			int startj = (x - halfChord > 0) ? x - halfChord : 0;
			int endj = (x + halfChord < cols) ? x + halfChord : cols;
			for(int j = startj; j < endj; ++j)
			{
				if(args->mask[i][j] == 0)
				{
					args->mat->at<Vec3b>(i, j) = args->mat->at<Vec3b>(i, j) * 0.7 + args->color * 0.3;
					args->mask[i][j] = 1;
				}
			} 
		}
	}	
}

//UI界面标记区域
void deleteObj(Mat *mat, int **mask_remove, int **mask_highlight)
{
	Mat *showImg = new Mat(mat->clone());
	MouseArgs *args = new MouseArgs(showImg, mask_remove, Vec3b(0, 0, 255));
	namedWindow("Please Draw Useless Part", CV_WINDOW_AUTOSIZE);
	setMouseCallback("Please Draw Useless Part", onMouse, (void*)args);
	while(1)
	{
		imshow("Please Draw Useless Part", *args->mat);
		if(waitKey(100) == 27) break;
	}
	setMouseCallback("Please Draw Useless Part", NULL, NULL);
	delete args; 

	args = new MouseArgs(showImg, mask_highlight, Vec3b(255, 0, 0));
	namedWindow("Please Draw ROI", CV_WINDOW_AUTOSIZE);
	setMouseCallback("Please Draw ROI", onMouse, (void*)args);
	while(1)
	{
		imshow("Please Draw ROI", *args->mat);
		if(waitKey(100) == 27) break;
	}
	setMouseCallback("Please Draw ROI", NULL, NULL);
	delete args;
}

//判断ROI是否被删除干净
bool empty(int **mask_remove, int &i, int &j, int r, int c)
{
	int starti = (i - 1 > 0) ? i - 1 : 0;
	for(int ii = starti; ii < r; ++ii)
		for(int jj = 0; jj < c; ++jj)
			if(mask_remove[ii][jj] != 0)
			{
				i = ii;
				j = jj;
				return false;
			}
	return true;
}

//对象保护和移除
void ROI(Mat *mat, int **mapX, int **mapY, int **gradX, int **gradY, int **image, int **image_withlines, int **energy, int **mask_remove, int **mask_highlight, int rows, int cols, int new_rows, int new_cols, int flags, int mode, char* dir)
{
	deleteObj(mat, mask_remove, mask_highlight);
	int i = 0, j = 0;
	int r = rows; int c = cols;
	bool cut_row = false;
	while(!empty(mask_remove, i, j, r, c))
	{
		if(cut_row)
		{
			gradientX_define(image, r, c, gradX);
			gradientY_define(image, r, c, gradY);
			energy_define(mat, gradX, gradY, energy, mask_remove, mask_highlight, r, c, flags, mode);
			reduceX(mapX, mapY, image, image_withlines, energy, mask_remove, mask_highlight, r, c, flags);
			r--;
		}
		else
		{
			gradientX_define(image, r, c, gradX);
			gradientY_define(image, r, c, gradY);
			energy_define(mat, gradX, gradY, energy, mask_remove, mask_highlight, r, c, flags, mode);
			reduceY(mapX, mapY, image, image_withlines, energy, mask_remove, mask_highlight, r, c, flags);
			c--;	
		}
	}

	char* dir_output = new char[1000];
	char* dir_output_withlines = new char[1000];
	int len = strlen(dir);
	int tmplen;
	for(tmplen = 0; tmplen < len; ++tmplen)
	{
		if(dir[tmplen] == '.') break;
		dir_output[tmplen] = dir[tmplen];
	}
	dir_output[tmplen] = '\0';
	char* suffix = new char[50];
	switch(mode)
	{
		case 0 : suffix = (char*)"_ROI_userDefined"; break;
		case 1 : suffix = (char*)"_ROI_Sobel"; break;
		case 2 : suffix = (char*)"_ROI_Laplacian"; break;
		case 3 : suffix = (char*)"_ROI_Canny"; break;
	}
	dir_output = strcat(dir_output, suffix);
	strcpy(dir_output_withlines, dir_output);
	dir_output_withlines = strcat(dir_output_withlines, "_withlines");
	dir_output = strcat(dir_output, ".png");
	dir_output_withlines = strcat(dir_output_withlines, ".png");

	ReduceOutput(mat, mapX, mapY, image, image_withlines, rows, cols, r, c, dir_output, dir_output_withlines);
	printf("已成功生成图片:\n%s\n%s\n", dir_output, dir_output_withlines);
}

int main()
{
	Mat *mat, *mat_gray;
	char* dir = new char[1000];
	int rows, cols;
	int new_rows, new_cols;
	int flags, mode;
	if(Input(mat, rows, cols, new_rows, new_cols, flags, mode, dir) != 0) return 1;

	mat_gray = new Mat;
	cvtColor(*mat, *mat_gray, CV_BGR2GRAY);

	int **mapX, **mapY, **image, **image_withlines, **gradX, **gradY, **energy, **mask_remove, **mask_highlight;
	init(mat_gray, mapX, mapY, image, image_withlines, gradX, gradY, energy, mask_remove, mask_highlight, rows, cols, new_rows, new_cols);

	if(flags == 0)
	{
		if(new_rows <= rows && new_cols <= cols)
			reduce(mat, mapX, mapY, gradX, gradY, image, image_withlines, energy, mask_remove, mask_highlight, rows, cols, new_rows, new_cols, flags, mode, dir);
		if(new_rows >= rows && new_cols >= cols)
			enlarge(mat, mapX, mapY, gradX, gradY, image, image_withlines, energy, mask_remove, mask_highlight, rows, cols, new_rows, new_cols, flags, mode, dir);
	}
	if(flags == 1)
		ROI(mat, mapX, mapY, gradX, gradY, image, image_withlines, energy, mask_remove, mask_highlight, rows, cols, new_rows, new_cols, flags, mode, dir);
	
	return 0;
}