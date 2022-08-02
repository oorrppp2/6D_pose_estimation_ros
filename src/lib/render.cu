#include "render.hpp"
#include <iostream>
#include <stdio.h>

__constant__ float c_threshold;
__constant__ int height;
__constant__ int width;
__constant__ int num_primitives;
__constant__ float cx, cy, fx, fy;
__constant__ glm::mat3 rotation;
__constant__ glm::vec3 translation;
__constant__ int render_primitives;
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
// #define Scores(i) data[Index + (i)]

__host__ __device__ static
double calculateSignedArea2(const glm::dvec3& a, const glm::dvec3& b, const glm::dvec3& c) {
	return ((c.x - a.x) * (b.y - a.y) - (b.x - a.x) * (c.y - a.y));
}

__host__ __device__ static
glm::dvec3 calculateBarycentricCoordinate(const glm::dvec3& a, const glm::dvec3& b, const glm::dvec3& c, const glm::dvec3& p) {
	double beta_tri = calculateSignedArea2(a, p, c);
	double gamma_tri = calculateSignedArea2(a, b, p);
	double tri_inv = 1.0f / calculateSignedArea2(a, b, c);
	double beta = beta_tri * tri_inv;
	double gamma = gamma_tri * tri_inv;
	double alpha = 1.0 - beta - gamma;
	return glm::vec3(alpha, beta, gamma);
}

__host__ __device__ static
bool isBarycentricCoordInBounds(const glm::dvec3 barycentricCoord) {
    return barycentricCoord.x >= 0.0 && barycentricCoord.x <= 1.0 &&
           barycentricCoord.y >= 0.0 && barycentricCoord.y <= 1.0 &&
           barycentricCoord.z >= 0.0 && barycentricCoord.z <= 1.0;
}

__host__ __device__ static
double getZAtCoordinate(const glm::dvec3 barycentricCoord, const glm::dvec3& a, const glm::dvec3& b, const glm::dvec3& c) {
	return (barycentricCoord.x * a.z
		+ barycentricCoord.y * b.z
		+ barycentricCoord.z * c.z);
}

__device__ int CompactRGBToInt(const glm::vec3& rgb) {
	return ((int)(rgb.x * 255) << 16)
	+ ((int)(rgb.y * 255) << 8)
	+ ((int)(rgb.z * 255))
	+ (255 << 24);
}

__device__ void atomicExchRGBZ(int* zbuffer, int* image, int z, int rgb) {
	while (true) {
		int expected_rgb = *image;
		int expected_z = *zbuffer;
		if (expected_z > z)
			break;

		int old_rgb = atomicCAS(image, expected_rgb, rgb);

		if (old_rgb == expected_rgb)
			break;
	}
}

__global__ void NaiveRender_gpu(int* color) {
	int pixel = blockIdx.x * blockDim.x + threadIdx.x;
	if (pixel >= height * width)
		return;
	int y = pixel / width;
	int x = pixel - width * y;

	color[pixel] = CompactRGBToInt(glm::vec3(y / (float)height, x / (float)width, 1));
}

__global__ void Render_gpu(glm::vec3* positions, glm::ivec3* indices, int* color, int* findices, int* zbuffer) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_primitives)
		return;

	glm::ivec3 face = indices[idx];
	glm::dvec3 p1 = glm::dvec3(rotation * positions[face[0]] + translation);
	glm::dvec3 p2 = glm::dvec3(rotation * positions[face[1]] + translation);
	glm::dvec3 p3 = glm::dvec3(rotation * positions[face[2]] + translation);

	if (p1.z < 0.02 || p2.z < 0.02 || p3.z < 0.02)
		return;

	p1.z = 1.0f / p1.z;
	p2.z = 1.0f / p2.z;
	p3.z = 1.0f / p3.z;

	p1.x = p1.x * p1.z;
	p1.y = p1.y * p1.z;
	p2.x = p2.x * p2.z;
	p2.y = p2.y * p2.z;
	p3.x = p3.x * p3.z;
	p3.y = p3.y * p3.z;

	int minX = (MIN(p1.x, MIN(p2.x, p3.x)) * fx + cx);
	int minY = (MIN(p1.y, MIN(p2.y, p3.y)) * fy + cy);
	int maxX = (MAX(p1.x, MAX(p2.x, p3.x)) * fx + cx) + 0.999999f;
	int maxY = (MAX(p1.y, MAX(p2.y, p3.y)) * fy + cy) + 0.999999f;

	minX = MAX(0, minX);
	minY = MAX(0, minY);
	maxX = MIN(width, maxX);
	maxY = MIN(height, maxY);

	for (int py = minY; py <= maxY; ++py) {
		for (int px = minX; px <= maxX; ++px) {
			if (px < 0 || px >= width || py < 0 || py >= height)
				continue;
			float x = (px - cx) / fx;
			float y = (py - cy) / fy;

			glm::dvec3 baryCentricCoordinate = calculateBarycentricCoordinate(p1, p2, p3, glm::dvec3(x, y, 0));
			/*
			if (idx == 2766 && px == 128 && py == 128) {
				printf("Step1\n");
				printf("========================\n");
				printf("%f %f %f; %f %f %f; %f %f %f\n", p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, p3.x, p3.y, p3.z);
				printf("%f %f %f %f %f\n", x, y, baryCentricCoordinate.x,baryCentricCoordinate.y,baryCentricCoordinate.z);
				printf("========================\n");
			}
			*/
			if (isBarycentricCoordInBounds(baryCentricCoordinate)) {
				int pixel = py * width + px;

				float z = getZAtCoordinate(baryCentricCoordinate, p1, p2, p3);
				int z_quantize = z * 100000;

				int original_z = atomicMax(&zbuffer[pixel], z_quantize);

				if (original_z < z_quantize) {
					/*
					if (px == 128 && py == 128) {
						printf("Update %d %d %d %d %d\n", z_quantize, face[0], face[1], face[2], idx);
						printf("Coordinate %f %f %f\n", baryCentricCoordinate.x, baryCentricCoordinate.y, baryCentricCoordinate.z);
					}
					*/
					glm::vec3 rgb = baryCentricCoordinate;
					if (render_primitives == 0) {
						atomicExchRGBZ(&zbuffer[pixel], &color[pixel], z_quantize, CompactRGBToInt(rgb));
					} else {
						atomicExchRGBZ(&zbuffer[pixel], &findices[pixel], z_quantize, idx);
					}
				}
			}
		}
	}
}

__global__ void FetchDepth_gpu(int* z, float* depth) {
	// __shared__ int temp[256];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// temp[threadIdx.x] = 1;
	// __syncthreads();
	// if(threadIdx.x == 0) {
	// 	int sum = 0;
	// 	for(int i = 0; i < 256; i++) {
	// 		sum += temp[i];
	// 	}
	// 	atomicAdd(scores, sum);
	// 	atomicSub(scores+1, sum);
	// }

	if (idx >= height * width)
		return;
	int px = idx % width;
	int py = idx / width;
	int z_value = z[py * width + px];
	float d = 0;
	if (z_value > 0)
		d = 100000.0 / z_value;
	depth[py * width + px] = d;
}

__global__ void FetchVMap_gpu(int* d_z, int* findices, glm::vec3* positions, glm::ivec3* faces, glm::ivec3* vindices, glm::vec3* vweights) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= height * width)
		return;
	if (d_z[idx] == 0) {
		vindices[idx] = glm::ivec3(0, 0, 0);
		vweights[idx] = glm::vec3(0, 0, 0);
		return;	
	}
	int px = idx % width;
	int py = idx / width;

	glm::ivec3 face = faces[findices[idx]];
	vindices[idx] = face;
	glm::dvec3 p1 = glm::dvec3(rotation * positions[face[0]] + translation);
	glm::dvec3 p2 = glm::dvec3(rotation * positions[face[1]] + translation);
	glm::dvec3 p3 = glm::dvec3(rotation * positions[face[2]] + translation);

	if (p1.z < 0.2 || p2.z < 0.2 || p3.z < 0.2) {
		vindices[idx] = glm::ivec3(0, 0, 0);
		vweights[idx] = glm::vec3(0, 0, 0);
		return;
	}

	p1.z = 1.0f / p1.z;
	p2.z = 1.0f / p2.z;
	p3.z = 1.0f / p3.z;

	p1.x = p1.x * p1.z;
	p1.y = p1.y * p1.z;
	p2.x = p2.x * p2.z;
	p2.y = p2.y * p2.z;
	p3.x = p3.x * p3.z;
	p3.y = p3.y * p3.z;


	glm::dvec3 barycentric = calculateBarycentricCoordinate(p1, p2, p3, glm::dvec3((px - cx) / fx, (py - cy) / fy, 0));
	double inv_z = 1.0f / getZAtCoordinate(barycentric, p1, p2, p3);
	/*
	if (px == 128 && py == 128) {
		printf("========================\n");
		printf("%f %f %f; %f %f %f; %f %f %f\n", p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, p3.x, p3.y, p3.z);
		printf("%f %f %f %f %f\n", (px - cx) / fx, (py - cy) / fy, barycentric.x,barycentric.y,barycentric.z);
		printf("%f %f %f", barycentric.x * p1.z * inv_z, barycentric.y * p2.z * inv_z, barycentric.z * p3.z * inv_z);
		printf("%f %f %f %f\n", p1.z, p2.z, p3.z, inv_z);
		printf("========================\n");
	}
	if (!(inv_z < 1e6)) {
		printf("Warning!\n");
	}
	*/
	vweights[idx] = glm::vec3(barycentric.x * p1.z * inv_z, barycentric.y * p2.z * inv_z, barycentric.z * p3.z * inv_z);
}

void NaiveRender(FrameBuffer& frameBuffer) {
	int num_pixels = frameBuffer.row * frameBuffer.col;
	cudaMemcpyToSymbol(height, &frameBuffer.row, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(width, &frameBuffer.col, sizeof(int), 0, cudaMemcpyHostToDevice);

	NaiveRender_gpu<<<(num_pixels + 255) / 256, 256>>>(frameBuffer.d_colors);
}

void Render(VertexBuffer& vertexBuffer, FrameBuffer& frameBuffer, int renderPrimitive) {
	cudaMemcpyToSymbol(height, &frameBuffer.row, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(width, &frameBuffer.col, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(cx, &frameBuffer.cx, sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(cy, &frameBuffer.cy, sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(fx, &frameBuffer.fx, sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(fy, &frameBuffer.fy, sizeof(float), 0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(render_primitives, &renderPrimitive, sizeof(int), 0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(rotation, &vertexBuffer.rotation, sizeof(float) * 9, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(translation, &vertexBuffer.translation, sizeof(float) * 3, 0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(num_primitives, &vertexBuffer.num_indices, sizeof(int), 0, cudaMemcpyHostToDevice);

	Render_gpu<<<(vertexBuffer.num_indices + 255) / 256, 256>>>(vertexBuffer.d_positions, vertexBuffer.d_indices, frameBuffer.d_colors, frameBuffer.d_findices, frameBuffer.d_z);

}

void FetchDepth(FrameBuffer& frameBuffer) {
	// int size_of_particles = 10;
	// int* scores = new int[size_of_particles];
	// memset(scores, 0, sizeof(int)*size_of_particles);

	// int* scores_ptr;
	// cudaMalloc( (void**)&scores_ptr, size_of_particles * sizeof(int));
	// cudaMemcpy(scores_ptr, scores, size_of_particles * sizeof(int), cudaMemcpyHostToDevice);

	int num_pixels = frameBuffer.row * frameBuffer.col;
	// printf("num_pixels : %d\n", num_pixels);
	cudaMemcpyToSymbol(height, &frameBuffer.row, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(width, &frameBuffer.col, sizeof(int), 0, cudaMemcpyHostToDevice);
	FetchDepth_gpu<<<(num_pixels+255)/256, 256>>>(frameBuffer.d_z, frameBuffer.d_depth);	

	// cudaMemcpy(scores, scores_ptr, size_of_particles * sizeof(int), cudaMemcpyDeviceToHost);
	// printf("score[0] : %d\n", scores[0]);
	// for(int i = 0; i < size_of_particles; i++){
	// 	printf("score[%d] : %d\n", i, scores[i]);
	// }
}

void FetchVMap(VertexBuffer& vertexBuffer, FrameBuffer& frameBuffer) {
	int num_pixels = frameBuffer.row * frameBuffer.col;
	cudaMemcpyToSymbol(height, &frameBuffer.row, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(width, &frameBuffer.col, sizeof(int), 0, cudaMemcpyHostToDevice);
	FetchVMap_gpu<<<(num_pixels+255)/256, 256>>>(frameBuffer.d_z, frameBuffer.d_findices, vertexBuffer.d_positions, vertexBuffer.d_indices, frameBuffer.d_vindices, frameBuffer.d_vweights);
}

__global__ void Rotate_gpu(glm::vec3* output, glm::vec3* input, glm::mat3 rot, int count) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= count)
		return;
	output[idx] = rot * input[idx];
}

void rotate_cuda(glm::vec3* output, glm::vec3* input, glm::mat3 rot, int count) {
	Rotate_gpu<<<(count + 255) / 256, 256>>>(output, input, rot, count);
}


__global__ void FetchMatchingScore(int* z, float* depth, float* depth_src, bool* other_objects_regions, unsigned int* scores) {
	__shared__ unsigned int score_temp[256];
	__shared__ unsigned int inter_temp[256];
	__shared__ unsigned int union_temp[256];
	__shared__ unsigned int temp[256];
	if (threadIdx.x == 0) {
		for(int i = 0; i < 256; i++) {
			score_temp[i] = 0;
			inter_temp[i] = 0;
			union_temp[i] = 0;
			temp[i] = 0;
		}
	}
	__syncthreads();
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= height * width)
		return;
	int px = idx % width;
	int py = idx / width;

	bool other_objects_regions_value = other_objects_regions[py * width + px];
	if (other_objects_regions_value == true) {
		// temp[threadIdx.x] += 1;	
		// return;
	}
	// __syncthreads();
	else {
		int z_value = z[py * width + px];
		float d = 0;
		if (z_value > 0)
			d = 100000.0 / z_value;
		// depth[py * width + px] = d;

		float src_depth_value = depth_src[py * width + px];


		// if (other_objects_regions_value == true)
		// 	temp[threadIdx.x] += 1;
		
		// if (other_objects_regions_value == false) {
		// 	temp[threadIdx.x] += 1;
		// 	// if (src_depth_value > 0) {
		// 	// }
		// 	// temp[threadIdx.x] += other_objects_regions[py * width + px];
		// 	if(src_depth_value != 0 && d != 0) {
		// 		inter_temp[threadIdx.x] += 1;
		// 		// if (threadIdx.x == 1) {
		// 		// 	printf("c_threshold(f) : %f\n", c_threshold);
		// 		// 	// printf("c_threshold(d) : %d\n", c_threshold);
		// 		// 	printf("d(f) : %f\n", d);
		// 		// 	// printf("d(d) : %d\n", d);
		// 		// 	printf("src_depth_value : %f\n", src_depth_value);
		// 		// 	printf("abs(src_depth_value - d) : %f\n", abs(src_depth_value - d));
		// 		// 	printf("abs(src_depth_value - d) - c_threshold : %f\n", abs(src_depth_value - d) - c_threshold);
		// 		// 	printf("c_threshold - abs(src_depth_value - d) : %f\n", c_threshold - abs(src_depth_value - d));
		// 		// }
		// 		if (abs(src_depth_value - d) > c_threshold) {
		// 			score_temp[threadIdx.x] += 1;
		// 		}
		// 	}
		// 	if(src_depth_value > 0 || d > 0)
		// 		union_temp[threadIdx.x] += 1;
		// }
		// if(py * width + px >= 420*width+124 && py * width + px <= 420*width+150) {
		// 	printf("depth value[%d] : %f\n", py * width + px, depth_src[py * width + px]);
		// 	printf("d value[%d] : %f\n", py * width + px, d);
		// }

		// if(src_depth_value != 0 )
		// 	inter_temp[threadIdx.x] += 1;
		// if(d != 0 )
		// union_temp[threadIdx.x] += 1;
		// if(src_depth_value > 0 || d > 0)
		// 	temp[threadIdx.x] += 1;

		// if(src_depth_value != 0) {
		// 	score_temp[threadIdx.x] += 1;
		// }

		if(src_depth_value != 0 && d != 0) {
			inter_temp[threadIdx.x] += 1;
			if (abs(src_depth_value - d) > c_threshold) {
				score_temp[threadIdx.x] += 1;
			}
		}
		if(src_depth_value != 0 || d != 0 )
			union_temp[threadIdx.x] += 1;

		// if(d != 0)
		// 	temp[threadIdx.x] += 1;
	}



	__syncthreads();

	if(threadIdx.x == 0) {
		unsigned int score_sum = 0;
		unsigned int inter_sum = 0;
		unsigned int union_sum = 0;
		unsigned int sum = 0;
		for(int i = 0; i < 256; i++) {
			score_sum += score_temp[i];
			inter_sum += inter_temp[i];
			union_sum += union_temp[i];
			sum += temp[i];
		}
		atomicAdd(scores++, score_sum);
		atomicAdd(scores++, inter_sum);
		atomicAdd(scores++, union_sum);
		atomicAdd(scores++, sum);

		// free(score_temp);
		// free(inter_temp);
		// free(union_temp);
		// free(temp);
	}
	__syncthreads();
}

void CalcMatchingScore_GPU(FrameBuffer& frameBuffer) {
	// int row = 480, col = 640;
	int num_pixels = frameBuffer.row * frameBuffer.col;

	int score_vector_size = 10;
	unsigned int* scores = new unsigned int[score_vector_size];
	memset(scores, 0, sizeof(unsigned int)*score_vector_size);

	unsigned int* scores_ptr;
	cudaMalloc( (void**)&scores_ptr, score_vector_size * sizeof(unsigned int));
	cudaMemcpy(scores_ptr, scores, score_vector_size * sizeof(unsigned int), cudaMemcpyHostToDevice);


	// int num_pixels = frameBuffer.row * frameBuffer.col;
	cudaMemcpyToSymbol(height, &frameBuffer.row, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(width, &frameBuffer.col, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_threshold, &frameBuffer.c_threshold, sizeof(float), 0, cudaMemcpyHostToDevice);
	FetchMatchingScore<<<(num_pixels+255)/256, 256>>>(frameBuffer.d_z, frameBuffer.d_depth, frameBuffer.depth_src_ptr, frameBuffer.other_objects_regions_ptr, scores_ptr);	
	// FetchMatchingScore<<<(num_pixels+255)/256, 256>>>(frameBuffer.d_z, frameBuffer.d_depth, depth_src_ptr, scores_ptr);	

	cudaMemcpy(scores, scores_ptr, score_vector_size * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	/*
		scores[0] : number of good matching pixels (distance < threshold)
		scores[1] : number of intersection pixels
		scores[2] : number of union pixels
	*/
	// printf("scores[0] : %d\n", scores[0]);
	// printf("scores[1] : %d\n", scores[1]);
	// printf("scores[2] : %d\n", scores[2]);
	// printf("scores[3] : %d\n", scores[3]);
	
	float IOU = float(scores[1]) / float(scores[2]);
	//costs = dists >= tau
	float eVSD = (float(scores[0]) + float(scores[2] - scores[1])) / float(scores[2]);

	// printf("%f\n", 1.-eVSD);
	// if (IOU > 1.0) {
	// 	printf("error\n");
	// }
	float matching_scores = 0.8 * (1.-eVSD) + 0.2 * IOU;
	// printf("IOU : %f\n", IOU);
	// printf("matching_score : %f\n", matching_scores);
	frameBuffer.SetMatchingScore(matching_scores);
	cudaFree((void*)scores_ptr);
	free(scores);
	// printf("score[0] : %d\n", scores[0]);
	// printf("score[1] : %d\n", scores[1]);
	// printf("score[2] : %d\n", scores[2]);
	// printf("sum : %d\n", sum);

	// for(int i = 0; i < size_of_particles; i++){
	// 	printf("score[%d] : %d\n", i, scores[i]);
	// }
	// return scores;
}
