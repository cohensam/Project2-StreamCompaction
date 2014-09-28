#include "CPU_stream_compaction.h"
#include <iostream>;
using namespace std;
float* CPU_stream_compaction::CPU_prefix_sum_inclusive(float arr[], int n) {
	float* res = new float[n];
	res[0] = arr[0];
	for (int i = 1; i < n; i++) {
		res[i] = arr[i] + res[i-1];
	}
	return res;
}

float* CPU_stream_compaction::CPU_prefix_sum_exclusive(float arr[], int n) {
	//Scatter
	float* barr = new float[n];
	for (int i = 0; i < n; i++) {
		if (arr[i] > 0) {
			barr[i] = 1.0f;
		} else {
			barr[i] = 0.0f;
		}
	}

	float index = 0;
	float* iarr = new float[n];
	for (int i = 0; i < n; i++) {
		iarr[i] = index;
		if (barr[i] != 0) {
			index++;
		}
	}

	int curr = 0;
	int farrIndex = 0;
	float* farr = new float[index];
	for (int i = 0; i < n; i++) {
		if (iarr[i] != curr) {
			farr[curr] = arr[i-1];
			curr++;
			farrIndex++;
		}
	}

	float* res = new float[n];
	res[0] = 0;
	for (int i = 1; i < n; i++) {
		res[i] = farr[i-1] + res[i-1];
	}

	return res;
}