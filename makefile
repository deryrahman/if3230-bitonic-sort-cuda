make: src/bitonic_sort.cu
	nvcc -o bitonic_sort src/bitonic_sort.cu -w
run:make
	./bitonic_sort 512
	./bitonic_sort 1024
	./bitonic_sort 4096
	./bitonic_sort 65536
	./bitonic_sort 262144
	./bitonic_sort 1048576
	./bitonic_sort 8388608
