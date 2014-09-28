#include "CPU_stream_compaction.h"
#include <iostream>;
using namespace std;

int main1() {
	CPU_stream_compaction sc;
	float arr[6];
	arr[0] = 3.0f;
	arr[1] = 4.0f;
	arr[2] = 6.0f;
	arr[3] = 7.0f;
	arr[4] = 9.0f;
	arr[5] = 10.0f;
	float* in = sc.CPU_prefix_sum_inclusive(arr, 6);
	cout << "Prefix Sum Inclusive:" << endl;
	cout << "Starting Array: [" << arr[0] << " " << arr[1] << " " << arr[2] << " " << arr[3] << " " << arr[4] << " " << arr[5] << "]" << endl;
	cout << "Resulting Array: [" << in[0] << " " << in[1] << " " << in[2] << " " << in[3] << " " << in[4] << " " << in[5] << "]" << endl;

	float* ex = sc.CPU_prefix_sum_exclusive(arr, 6);
	cout << "Prefix Sum Inclusive:" << endl;
	cout << "Starting Array: [" << arr[0] << " " << arr[1] << " " << arr[2] << " " << arr[3] << " " << arr[4] << " " << arr[5] << "]" << endl;
	cout << "Resulting Array: [" << ex[0] << " " << ex[1] << " " << ex[2] << " " << ex[3] << " " << ex[4] << " " << ex[5] << "]" << endl;
	cin.get();
	return 0;
}