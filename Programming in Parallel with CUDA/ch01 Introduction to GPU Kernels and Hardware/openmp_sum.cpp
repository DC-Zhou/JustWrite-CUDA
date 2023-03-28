//
// Created by Zhou on 2023/3/31.
//
//omp_sum OMP CPU calculation of a sin integral
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <chrono>

inline float sinsum(float x, int terms)
{
    float term = x;
    float sum = term;
    float x2 = x*x;
    for(int n = 1; n < terms; n++) {
        term *= -x2 / ((2*n)*(2*n+1));
        sum += term;
    }
    return sum;
}


int main(int argc, char *argv[]) {
    int steps = (argc > 1) ? atoi(argv[1]) : 10000000;
    int terms = (argc > 2) ? atoi(argv[2]) : 1000;
    int threads= (argc > 3) ? atoi(argv[3]) : 4;

    double pi = 3.141592653;
    double step_size = pi / (steps - 1);

    auto start = std::chrono::steady_clock::now();
    double omp_sum = 0.0;

    omp_set_num_threads(threads);
    #pragma omp parallel for reduction(+:omp_sum)
    for(int step = 0; step < steps; step++) {
        float x = step * step_size;
        omp_sum += sinsum(x, terms);

    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;

    // Trapezoidal Rule correction
    omp_sum -= 0.5*(sinsum(0.0,terms)+sinsum(pi, terms));
    omp_sum *= step_size;
    printf("omp sum = %.10f,steps %d terms %d time %.3f ms\n",
           omp_sum, steps, terms, diff);
}



