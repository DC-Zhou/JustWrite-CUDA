//
// Created by Zhou on 2023/3/28.
//
// this is a simple example of Check if OpenMP is available
#include <iostream>
#include <omp.h>
#include <windows.h>

#define THREAD_NUM 4
int main()
{
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    #pragma omp parallel
    {
        Sleep(500 * omp_get_thread_num()); // do this to avoid race condition while printing
        std::cout << "Number of available threads: " << omp_get_thread_num() << std::endl;
        // each thread can also get its own number
        std::cout << "Current thread number: " << omp_get_thread_num() << std::endl;
        std::cout << "Hello, World!" << std::endl;
    }
    return 0;
}