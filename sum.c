#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <emmintrin.h> /* where intrinsics are defined */

#define CLOCK_RATE_GHZ 2.26e9

/* Time stamp counter from Lecture 2/17 */
static __inline__ unsigned long long RDTSC(void) {
    unsigned hi,lo;
    __asm__ volatile("rdtsc" : "=a"(lo),"=d"(hi));
    return ((unsigned long long) lo)| (((unsigned long long)hi) << 32);
}

int sum_naive( int n, int *a )
{
    int sum = 0;
    for( int i = 0; i < n; i++ )
        sum += a[i];
    return sum;
}

int sum_unrolled( int n, int *a )
{
    int sum = 0;

    /* do the body of the work in a faster unrolled loop */
    for( int i = 0; i < n/4*4; i += 4 )
    {
        sum += a[i+0];
        sum += a[i+1];
        sum += a[i+2];
        sum += a[i+3];
    }

    /* handle the small tail in a usual way */
    for( int i = n/4*4; i < n; i++ )   
        sum += a[i];

    return sum;
}

int sum_vectorized( int n, int *a )
{
    /* WRITE YOUR VECTORIZED CODE HERE */
    int sum = 0;
    __m128i a_vec = _mm_setzero_si128();
    int i,j;
    int b[4] = {0,0,0,0};
    for (i = 0; i < (n/4)*4; i+=4 ) {
        __m128i b_vec = _mm_loadu_si128( (__m128i*)(a + i));
        a_vec = _mm_add_epi32(a_vec,b_vec);
    }
    for(j = n/4*4; j < n; j++ )   
        sum += a[j];
    _mm_storeu_si128( b , a_vec );
    sum = sum + b[0] + b[1] + b[2] + b[3];
    return sum;
    /*
    int sum = 0;
    int b[4] = {0,0,0,0};
    int j = 0;
    __m128i a_vec = _mm_setzero_si128( );
    for (int i = 0; i < (n/4); i += 1) {
        printf("i is %d\n", i);
        a_vec = _mm_add_epi32( a_vec, _mm_loadu_si128((__m128i*)(a + (4 * i)) ));
    }
    printf("done\n");
    for( int j = n/4*4; j < n; j++ )  
        printf("handle tails\n"); 
        sum += a[j];
    printf("done with tails\n");
    _mm_storeu_si128( b , a_vec );
    printf("done with assign\n");
    sum = sum + b[0]+b[1]+b[2]+b[3];
    return sum;
    */
}

int sum_vectorized_unrolled( int n, int *a )
{
    /* UNROLL YOUR VECTORIZED CODE HERE*/
    int sum = 0;
    __m128i a_vec = _mm_setzero_si128();
    int i,j;
    int b[4] = {0,0,0,0};
    for (i = 0; i < (n/16)*16; i+=16 )
    {
        __m128i b_vec = _mm_loadu_si128( (__m128i*)(a + i + 0));
        a_vec = _mm_add_epi32(a_vec,b_vec);
        __m128i c_vec = _mm_loadu_si128( (__m128i*)(a + i + 4));
        a_vec = _mm_add_epi32(a_vec,c_vec);
        __m128i d_vec = _mm_loadu_si128( (__m128i*)(a + i + 8));
        a_vec = _mm_add_epi32(a_vec,d_vec);
        __m128i e_vec = _mm_loadu_si128( (__m128i*)(a + i + 12));
        a_vec = _mm_add_epi32(a_vec,e_vec);
    }
    for(j = n/16*16; j < n; j++ )  
        sum += a[j];
    _mm_storeu_si128( b , a_vec );
    sum = sum + b[0] + b[1] + b[2] + b[3];
    return sum;
}

void benchmark( int n, int *a, int (*computeSum)(int,int*), char *name )
{
    /* warm up */
    int sum = computeSum( n, a );

    /* measure */
    unsigned long long cycles = RDTSC();
    sum += computeSum( n, a );
    cycles = RDTSC()-cycles;
    
    double microseconds = cycles/CLOCK_RATE_GHZ*1e6;
    
    /* report */
    printf( "%20s: ", name );
    if( sum == 2*sum_naive(n,a) ) printf( "%.2f microseconds\n", microseconds );
    else	                  printf( "ERROR!\n" );
}

int main( int argc, char **argv )
{
    const int n = //2000;
	7777; /* small enough to fit in cache */
    
    /* init the array */
    srand48( time( NULL ) );
    int a[n] __attribute__ ((aligned (16))); /* align the array in memory by 16 bytes */
    for( int i = 0; i < n; i++ ) a[i] = lrand48( );
    
    /* benchmark series of codes */
    benchmark( n, a, sum_naive, "naive" );
    benchmark( n, a, sum_unrolled, "unrolled" );
    benchmark( n, a, sum_vectorized, "vectorized" );
    benchmark( n, a, sum_vectorized_unrolled, "vectorized unrolled" );

    return 0;
}

