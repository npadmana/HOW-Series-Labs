#include <cstdio>
#include <cstdlib>

#pragma offload_attribute(push, target(mic))
const int n=100000;
double b[n];

void gemv(int m, int n, double* A, double c[]) {
#ifdef __MIC__
	printf("Running on a MIC!\n");
#else 
	printf("Running on host!\n");
#endif
	b[:] = b[:] + 1.0;
	for ( int i = 0 ; i < m ; i++)
		for ( int j = 0 ; j < n ; j++)
			c[i] += A[i*n+j] * b[j];
}
#pragma offload_attribute(pop)


int main(){

    const int m=10;
    const int maxIter = 5;
    double c[m];
    double * A = (double*) malloc(sizeof(double)*n*m);

    // Cilk Plus array notation
    A[0:n*m]=1.0/(double)n;
    b[:]=0.0;
    c[:]=0;

// Transfer A once
#pragma offload target(mic) in(A : length(n*m) alloc_if(1) free_if(0))
{}
    printf("Running the matrix-vector multiplication\n");
    for ( int iter = 0; iter < maxIter ; iter++) {
// No data transfer
#pragma offload target(mic) in(A: length(0) alloc_if(0) free_if(0)) optional
	{
            gemv(m, n, A, c);	
	// End offload
	}
    }
// clean up
#pragma offload target(mic) in(A : length(0) alloc_if(0) free_if(1))
{}

    printf("Checking the results...\n");
    double norm = 0.0;
    for ( int i = 0 ; i < m ; i++)
        norm += (c[i]-(maxIter+1)*maxIter/2.0)*(c[i]-(maxIter+1)*maxIter/2.0);

    if (norm > 1e-10)
        printf("Something is wrong... Norm is equal to %f\n", norm);
    else
        printf("Yup, we're good!\n");

}
