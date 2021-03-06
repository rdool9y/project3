#include <emmintrin.h>
#include <stdio.h>
#include <nmmintrin.h>

#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.

/*
  SSE                         : yes
  cache blocking              : yes
  loop unrolling              : yes
  zero padding                : yes
  
  optimized padded_in creation: no
*/
 
int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
	   float* kernel)
{
    // the x coordinate of the kernel's center
    int kern_cent_X = (KERNX - 1)/2;
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;

    int blocksize = 256; // must be multiple of 4 (or possibly 8/12/16? if loop unrolling)

    // Declare Intrinsics Registers :
    __m128 kernel_vector = _mm_setzero_ps(); // "shared" by 1-4

    __m128 input_vector1 = _mm_setzero_ps();
    __m128 output_vector1 = _mm_setzero_ps();
    __m128 product_vector1 = _mm_setzero_ps();

    // Intrinsics Registers for unrolling:
    __m128 input_vector2 = _mm_setzero_ps();
    __m128 output_vector2 = _mm_setzero_ps();
    __m128 product_vector2 = _mm_setzero_ps();
    
    // build zero-padded copy 
    int padding_x = (KERNX / 2);  // can we assume that kernel is a square matrix??
    int padding_y = (KERNY /2);
    int padded_size = (data_size_X + 2*padding_x) * (data_size_Y + 2*padding_y); // not initialized to zero?
    float* padded_in = malloc(padded_size * sizeof(float));
    
    int x,y;
    
    for(x = 0; x < padded_size; x++) {                                       // manually fill with zero
        padded_in[x] = 0.0f;                                                 // optimize: zero edges only
	} 
   
    for(y = 0; y < data_size_Y; y++) {
        for(x = 0; x < data_size_X; x++) {
	    padded_in[(x+padding_x) + (y+padding_y)*(data_size_X + 2*padding_y)] = in[x+y*data_size_X];
	}
	}      
    
//    memset(&padded_in, 0.0f, (data_size_X+(2*padding_x))*(data_size_Y+(2*padding_y)));
//    for(y =0; y < data_size_Y; y++) {
//	memset(&padded_in, 0.0f, padding_x);
//	memcpy(&padded_in, &in, data_size_X);
//	memset(&padded_in, 0.0f, padding_x);
//    }
//    memset(&padded_in, 0.0f, (data_size_X+(2*padding_x))*(data_size_Y+(2*padding_y)));
    
    int a, b, i, j;
    
    // main convolution loop
    for(y = 0; y < data_size_Y; y+=blocksize){ 
        for(x = 0; x < data_size_X; x+=blocksize){ 
            for(b = y; b < y + blocksize && b < data_size_Y; b++) {        // no leftovers for y
       	        for(a = x; a < x + blocksize && a <= data_size_X-8; a+=8) {   // leftovers for x b/c increment by 8
                    // set output vector to 0
                    output_vector1 = _mm_setzero_ps();
		    output_vector2 = _mm_setzero_ps();
		    
		    for(i = -kern_cent_X; i <= kern_cent_X; i++){          // inner loop; after all iterations, write 4 output sums
		        for(j = -kern_cent_Y; j <= kern_cent_Y; j++){ 

			    kernel_vector = _mm_load1_ps(kernel + ((kern_cent_X-i) + (kern_cent_Y-j)*KERNX));
			    input_vector1 = _mm_loadu_ps(padded_in + ((a+i+padding_x) + (b+j+padding_y)*(data_size_X+2*padding_y)));
			    // above must be loadu; can't use aligned load
			    
                            product_vector1 = _mm_mul_ps(kernel_vector, input_vector1);
                            output_vector1 = _mm_add_ps(output_vector1, product_vector1);
			    
			    input_vector2 = _mm_loadu_ps(padded_in + 4 + ((a+i+padding_x) + (b+j+padding_y)*(data_size_X+2*padding_y)));
			    product_vector2 = _mm_mul_ps(kernel_vector, input_vector2);
			    output_vector2 = _mm_add_ps(output_vector2, product_vector2);
			    
			}
                    }
 		    // After inner loop completes, write output vector to output matrix
		    // must be storeu; can't use store aligned
                    _mm_storeu_ps(out + (a + b*data_size_X), output_vector1);
		    _mm_storeu_ps(out + 4 + (a + b*data_size_X), output_vector2);
		}
	    }
	}
    }
    
    float output_float, kernel_float, input_float, product_float;
    
    // Deal with leftovers (0-3 columns on far right of out)
    for(b = 0; b < data_size_Y; b++) {        
        for(a = (data_size_X/8)*8; a < data_size_X; a++) {   
            // set output to 0
            output_float = 0.0f;
		    
 	    for(i = -kern_cent_X; i <= kern_cent_X; i++){          // inner loop : all kernel elements
	        for(j = -kern_cent_Y; j <= kern_cent_Y; j++){ 
		    
                    product_float = kernel[(kern_cent_X - i) + (kern_cent_Y-j)*KERNX] * padded_in[(a+i+padding_x)+(b+j+padding_y)*(data_size_X+2*padding_y)];
                    output_float += product_float;
                }
            }
                    out[a + b*data_size_X] = output_float;
	}
    }
 
    free(padded_in);
    
    return 1;
}




