#include "CPU_stream_compaction.h"
#include <iostream>;
using namespace std;
float* CPU_stream_compaction::prefix_sum_inclusive(float arr[], int n) {
	float* res = new float[n-1];
	res[0] = arr[0];
	for (int i = 1; i < n; i++) {
		res[i] = arr[i] + res[i-1];
	}
	return res;
}

float* CPU_stream_compaction::prefix_sum_exclusive(float arr[], int n) {
	float* res = new float[n-1];
	res[0] = 0;
	for (int i = 1; i < n; i++) {
		res[i] = arr[i-1] + res[i-1];
	}
	return res;
}