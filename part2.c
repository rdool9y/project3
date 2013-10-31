#include <emmintrin.h>
#include <omp.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.

/*

blocksize       : 256 (seems like fastest vs. 128, 512)
SIMD            : yes
loop unrolling  : 16, followed by leftovers as SIMD, then leftovers as naive
fill w/ zero    : padding only
openMP          : yes

openMP design:
-Divide into blocks of rows so each thread deals with (total rows)/num_threads rows.
-Use floordiv to divide up rows, so highest numbered thread must deal with remainder:
 (set y_max variable before loop by testing if highest numbered thread)

Writing to shared output array is too slow!
-malloc() private array
-Write to a private array for each thread
    -There will be 1 loop per row instead of 3 loops (16, 4, 1). This makes it easier
     to write to the private array in the right order and improves cache blocking, because
     the matrix is row-major.
-After thread fills private array, write to shared memory in critical directive.
-free() private array

*/
int robert_conv2D(float* in, float* out, int data_size_X, int data_size_Y,
	   float* kernel)
{
    // the x coordinate of the kernel's center
    int kern_cent_X = (KERNX - 1)/2;
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;

    int blocksize = 256; // must be multiple of 16

    // Declare Intrinsics Registers :
    __m128 kernel_vector; // "shared" by 1-4
    __m128 input_vector1, output_vector1, product_vector1;
    __m128 input_vector2, output_vector2, product_vector2;
    __m128 input_vector3, output_vector3, product_vector3;
    __m128 input_vector4, output_vector4, product_vector4;

    // build zero-padded copy 
    int padding_x = (KERNX / 2);  // can we assume that kernel is a square matrix??
    int padding_y = (KERNY /2);
    int padded_size = (data_size_X + 2*padding_x) * (data_size_Y + 2*padding_y); // not initialized to zero?
    float* padded_in = malloc(padded_size * sizeof(float));
    
    int x,y;

// left col padding
   for(y = 0; y < data_size_Y+2*padding_y; y++) 
        for(x = 0; x < padding_x; x++)
            padded_in[x + y*(data_size_X+2*padding_x)] = 0.0f;

// right col padding
   for(y = 0; y < data_size_Y+2*padding_y; y++) 
        for(x = data_size_X+2*padding_x-padding_x; x < data_size_X; x++)
            padded_in[x + y*(data_size_X+2*padding_x)] = 0.0f;

// top row padding
   for(y = 0; y < padding_y; y++) 
        for(x = 0; x < data_size_X+2*padding_x; x++)
            padded_in[x + y*(data_size_X+2*padding_x)] = 0.0f;

// bottom row padding
   for(y = data_size_Y+2*padding_y-padding_y; y < data_size_Y+2*padding_y; y++) 
        for(x = 0; x < data_size_X+2*padding_x; x++)
            padded_in[x + y*(data_size_X+2*padding_x)] = 0.0f;

// fill padded_in with src matrix 
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
    
    int a, b, i, j, y_max;
    //(float *) *all_private_outputs[16];
    float *private_output;

    # pragma omp parallel private(a, b, i, j, x, y, y_max, kernel_vector, input_vector1, input_vector2, input_vector3, input_vector4, output_vector1, output_vector2, output_vector3, output_vector4, product_vector1, product_vector2, product_vector3, product_vector4, private_output)
    {
    int num_threads = omp_get_num_threads();
    int thread_ID = omp_get_thread_num();

    if (thread_ID == num_threads-1) {     // adjust loop test for leftovers if highest thread_ID
        y_max = data_size_Y;
    } else {
        y_max = (thread_ID+1)*(data_size_Y/num_threads);
    }

    // printf("ymax : %d\n", y_max);
    
//    float* private_output = malloc(data_size_x * y_max * sizeof(float));     // allocate space for each private_output array
    
//  # pragma omp critical
//    all_private_outputs[thread_ID] = private_output;  

    // main convolution loop
    
    for(y = thread_ID*(data_size_Y/num_threads); y < y_max; y+=blocksize){ 
        for(x = 0; x < data_size_X; x+=blocksize){ 
            for(b = y; b < y + blocksize && b < data_size_Y; b++) {        // no leftovers for y
       	        for(a = x; a < x + blocksize && a <= data_size_X-16; a+=16) {   // leftovers for x b/c increment by 8
                    // set output vector to 0
                    output_vector1 = _mm_setzero_ps();
		    output_vector2 = _mm_setzero_ps();
		    output_vector3 = _mm_setzero_ps();
		    output_vector4 = _mm_setzero_ps();
		    
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

                            input_vector3 = _mm_loadu_ps(padded_in + 8 + ((a+i+padding_x) + (b+j+padding_y)*(data_size_X+2*padding_y)));
			    product_vector3 = _mm_mul_ps(kernel_vector, input_vector3);
			    output_vector3 = _mm_add_ps(output_vector3, product_vector3);
			    
			    input_vector4 = _mm_loadu_ps(padded_in + 12  + ((a+i+padding_x) + (b+j+padding_y)*(data_size_X+2*padding_y)));
			    product_vector4 = _mm_mul_ps(kernel_vector, input_vector4);
			    output_vector4 = _mm_add_ps(output_vector4, product_vector4);
			    
			}
                    }
 		    // After inner loop completes, write output vector to output matrix
		    // must be storeu; can't use store aligned
                    # pragma omp critical
                    {
                    _mm_storeu_ps(out + (a + b*data_size_X), output_vector1);
		    _mm_storeu_ps(out + 4 + (a + b*data_size_X), output_vector2);
		    _mm_storeu_ps(out + 8 + (a + b*data_size_X), output_vector3);
		    _mm_storeu_ps(out + 12 + (a + b*data_size_X), output_vector4);
                    }
		}
	    }
	}
    }

    float output_float, kernel_float, input_float, product_float;
    
    // Deal with leftovers (0-15 columns on far right of out)
    for(b = 0; b < data_size_Y; b++) {        
        // Four at a time (One SIMD vector per loop)
       	for(a = (data_size_X/16)*16 ; a <= data_size_X-4; a+=4) {   // leftovers for x b/c increment by 8
            // set output vector to 0
            output_vector1 = _mm_setzero_ps();
  
            for(i = -kern_cent_X; i <= kern_cent_X; i++){          // inner loop; after all iterations, write 4 output sums
            for(j = -kern_cent_Y; j <= kern_cent_Y; j++){ 

	    kernel_vector = _mm_load1_ps(kernel + ((kern_cent_X-i) + (kern_cent_Y-j)*KERNX));
	    input_vector1 = _mm_loadu_ps(padded_in + ((a+i+padding_x) + (b+j+padding_y)*(data_size_X+2*padding_y)));
	    // above must be loadu; can't use aligned load
			    
            product_vector1 = _mm_mul_ps(kernel_vector, input_vector1);
            output_vector1 = _mm_add_ps(output_vector1, product_vector1);
            }
            }
 	    // After inner loop completes, write output vector to output matrix
	    // must be storeu; can't use store aligned
            # pragma omp critical
            _mm_storeu_ps(out + (a + b*data_size_X), output_vector1);
	}

        // One at a time 
        for(; a<data_size_X ; a++) {		    
            // set output to 0
            output_float = 0.0f;
 	    
            for(i = -kern_cent_X; i <= kern_cent_X; i++){          // inner loop : all kernel elements
	        for(j = -kern_cent_Y; j <= kern_cent_Y; j++){ 
		    
                    product_float = kernel[(kern_cent_X - i) + (kern_cent_Y-j)*KERNX] * padded_in[(a+i+padding_x)+(b+j+padding_y)*(data_size_X+2*padding_y)];
                    output_float += product_float;
                }
            }
                    #pragma omp critical
                    out[a + b*data_size_X] = output_float;
         }
    }

    } // closes omp parallel
 
    free(padded_in);
    
    return 1;
}


int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
       float* kernel)
{
    // the x coordinate of the kernel's center
    int kern_cent_X = (KERNX - 1)/2;
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;

    int blocksize = 256; // must be multiple of 16

    // Declare Intrinsics Registers :
    __m128 kernel_vector; // "shared" by 1-4
    __m128 input_vector1, output_vector1, product_vector1;
    __m128 input_vector2, output_vector2, product_vector2;
    __m128 input_vector3, output_vector3, product_vector3;
    __m128 input_vector4, output_vector4, product_vector4;

    // build zero-padded copy 
    int padding_x = (KERNX / 2);  // can we assume that kernel is a square matrix??
    int padding_y = (KERNY /2);
    int padded_size = (data_size_X + 2*padding_x) * (data_size_Y + 2*padding_y); // not initialized to zero?
    float* padded_in = malloc(padded_size * sizeof(float));
    
    int x,y;
/*
// left col padding
   for(y = 0; y < data_size_Y+2*padding_y; y++) 
        for(x = 0; x < padding_x; x++)
            padded_in[x + y*(data_size_X+2*padding_x)] = 0.0f;

// right col padding
   for(y = 0; y < data_size_Y+2*padding_y; y++) 
        for(x = data_size_X+2*padding_x-padding_x; x < data_size_X; x++)
            padded_in[x + y*(data_size_X+2*padding_x)] = 0.0f;

// top row padding
   for(y = 0; y < padding_y; y++) 
        for(x = 0; x < data_size_X+2*padding_x; x++)
            padded_in[x + y*(data_size_X+2*padding_x)] = 0.0f;

// bottom row padding
   for(y = data_size_Y+2*padding_y-padding_y; y < data_size_Y+2*padding_y; y++) 
        for(x = 0; x < data_size_X+2*padding_x; x++)
            padded_in[x + y*(data_size_X+2*padding_x)] = 0.0f;

    for(x = 0; x < padded_size; x++) {                                       // manually fill with zero
        padded_in[x] = 0.0f;                                                 // optimize: zero edges only
    } 
   
    for(y = 0; y < data_size_Y; y++) {
        for(x = 0; x < data_size_X; x++) {
        padded_in[(x+padding_x) + (y+padding_y)*(data_size_X + 2*padding_y)] = in[x+y*data_size_X];
    }
    } 

// fill padded_in with src matrix 
    for(y = 0; y < data_size_Y; y++) {
        for(x = 0; x < data_size_X; x++) {
        padded_in[(x+padding_x) + (y+padding_y)*(data_size_X + 2*padding_y)] = in[x+y*data_size_X];
    }
    } */
    printf("survive1\n");
    printf("padded_size=%u\n",padded_size);
    memset(padded_in, 0.0f, (data_size_X+(2*padding_x))*(data_size_Y+(2*padding_y)));
    printf("now fill in 0 for %u\n", (data_size_X+(2*padding_x))*(data_size_Y+(2*padding_y)));
    printf("data_size_Y is %u\n", data_size_Y);
    for(y =0; y < data_size_Y; y++) {
        printf("row %u", y);
        memset(&padded_in, 0.0f, padding_x);
        memcpy(&padded_in, &in + y, data_size_X);
        memset(&padded_in, 0.0f, padding_x);
        printf(" done!\n");
    }
    printf("done with loop\n");
    //memset(&padded_in +, 0.0f, (data_size_X+(2*padding_x))*(data_size_Y+(2*padding_y)));
    printf("survive2\n");
    int a, b, i, j, y_max;
    //(float *) *all_private_outputs[16];
    float *private_output;

    # pragma omp parallel private(a, b, i, j, x, y, y_max, kernel_vector, input_vector1, input_vector2, input_vector3, input_vector4, output_vector1, output_vector2, output_vector3, output_vector4, product_vector1, product_vector2, product_vector3, product_vector4, private_output)
    {
    int num_threads = omp_get_num_threads();
    int thread_ID = omp_get_thread_num();

    if (thread_ID == num_threads-1) {     // adjust loop test for leftovers if highest thread_ID
        y_max = data_size_Y;
    } else {
        y_max = (thread_ID+1)*(data_size_Y/num_threads);
    }

    // printf("ymax : %d\n", y_max);
    
//    float* private_output = malloc(data_size_x * y_max * sizeof(float));     // allocate space for each private_output array
    
//  # pragma omp critical
//    all_private_outputs[thread_ID] = private_output;  

    // main convolution loop
    
    for(y = thread_ID*(data_size_Y/num_threads); y < y_max; y+=blocksize){ 
        for(x = 0; x < data_size_X; x+=blocksize){ 
            for(b = y; b < y + blocksize && b < data_size_Y; b++) {        // no leftovers for y
                for(a = x; a < x + blocksize && a <= data_size_X-16; a+=16) {   // leftovers for x b/c increment by 8
                    // set output vector to 0
                    output_vector1 = _mm_setzero_ps();
            output_vector2 = _mm_setzero_ps();
            output_vector3 = _mm_setzero_ps();
            output_vector4 = _mm_setzero_ps();
            
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

                            input_vector3 = _mm_loadu_ps(padded_in + 8 + ((a+i+padding_x) + (b+j+padding_y)*(data_size_X+2*padding_y)));
                product_vector3 = _mm_mul_ps(kernel_vector, input_vector3);
                output_vector3 = _mm_add_ps(output_vector3, product_vector3);
                
                input_vector4 = _mm_loadu_ps(padded_in + 12  + ((a+i+padding_x) + (b+j+padding_y)*(data_size_X+2*padding_y)));
                product_vector4 = _mm_mul_ps(kernel_vector, input_vector4);
                output_vector4 = _mm_add_ps(output_vector4, product_vector4);
                
            }
                    }
            // After inner loop completes, write output vector to output matrix
            // must be storeu; can't use store aligned
                    # pragma omp critical
                    {
                    _mm_storeu_ps(out + (a + b*data_size_X), output_vector1);
            _mm_storeu_ps(out + 4 + (a + b*data_size_X), output_vector2);
            _mm_storeu_ps(out + 8 + (a + b*data_size_X), output_vector3);
            _mm_storeu_ps(out + 12 + (a + b*data_size_X), output_vector4);
                    }
        }
        }
    }
    }

    float output_float, kernel_float, input_float, product_float;
    
    // Deal with leftovers (0-15 columns on far right of out)
    for(b = 0; b < data_size_Y; b++) {        
        // Four at a time (One SIMD vector per loop)
        for(a = (data_size_X/16)*16 ; a <= data_size_X-4; a+=4) {   // leftovers for x b/c increment by 8
            // set output vector to 0
            output_vector1 = _mm_setzero_ps();
  
            for(i = -kern_cent_X; i <= kern_cent_X; i++){          // inner loop; after all iterations, write 4 output sums
            for(j = -kern_cent_Y; j <= kern_cent_Y; j++){ 

        kernel_vector = _mm_load1_ps(kernel + ((kern_cent_X-i) + (kern_cent_Y-j)*KERNX));
        input_vector1 = _mm_loadu_ps(padded_in + ((a+i+padding_x) + (b+j+padding_y)*(data_size_X+2*padding_y)));
        // above must be loadu; can't use aligned load
                
            product_vector1 = _mm_mul_ps(kernel_vector, input_vector1);
            output_vector1 = _mm_add_ps(output_vector1, product_vector1);
            }
            }
        // After inner loop completes, write output vector to output matrix
        // must be storeu; can't use store aligned
            # pragma omp critical
            _mm_storeu_ps(out + (a + b*data_size_X), output_vector1);
    }

        // One at a time 
        for(; a<data_size_X ; a++) {            
            // set output to 0
            output_float = 0.0f;
        
            for(i = -kern_cent_X; i <= kern_cent_X; i++){          // inner loop : all kernel elements
            for(j = -kern_cent_Y; j <= kern_cent_Y; j++){ 
            
                    product_float = kernel[(kern_cent_X - i) + (kern_cent_Y-j)*KERNX] * padded_in[(a+i+padding_x)+(b+j+padding_y)*(data_size_X+2*padding_y)];
                    output_float += product_float;
                }
            }
                    #pragma omp critical
                    out[a + b*data_size_X] = output_float;
         }
    }

    } // closes omp parallel
 
    free(padded_in);
    
    return 1;
}


/*

blocksize       : 256 (seems like fastest vs. 128, 512)
SIMD            : yes
loop unrolling  : 16, followed by leftovers as SIMD, then leftovers as naive
fill w/ zero    : padding only

openMP          : no

*/


int NO_OPEN_MP_conv2D(float* in, float* out, int data_size_X, int data_size_Y,
	   float* kernel)
{
    // the x coordinate of the kernel's center
    int kern_cent_X = (KERNX - 1)/2;
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;

    int blocksize = 256; // must be multiple of 16

    // Declare Intrinsics Registers :
    __m128 kernel_vector; // "shared" by 1-4
    __m128 input_vector1, output_vector1, product_vector1;
    __m128 input_vector2, output_vector2, product_vector2;
    __m128 input_vector3, output_vector3, product_vector3;
    __m128 input_vector4, output_vector4, product_vector4;

    // build zero-padded copy 
    int padding_x = (KERNX / 2);  // can we assume that kernel is a square matrix??
    int padding_y = (KERNY /2);
    int padded_size = (data_size_X + 2*padding_x) * (data_size_Y + 2*padding_y); // not initialized to zero?
    float* padded_in = malloc(padded_size * sizeof(float));
    
    int x,y;

// left col padding
   for(y = 0; y < data_size_Y+2*padding_y; y++) 
        for(x = 0; x < padding_x; x++)
            padded_in[x + y*(data_size_X+2*padding_x)] = 0.0f;

// right col padding
   for(y = 0; y < data_size_Y+2*padding_y; y++) 
        for(x = data_size_X+2*padding_x-padding_x; x < data_size_X; x++)
            padded_in[x + y*(data_size_X+2*padding_x)] = 0.0f;

// top row padding
   for(y = 0; y < padding_y; y++) 
        for(x = 0; x < data_size_X+2*padding_x; x++)
            padded_in[x + y*(data_size_X+2*padding_x)] = 0.0f;

// bottom row padding
   for(y = data_size_Y+2*padding_y-padding_y; y < data_size_Y+2*padding_y; y++) 
        for(x = 0; x < data_size_X+2*padding_x; x++)
            padded_in[x + y*(data_size_X+2*padding_x)] = 0.0f;

// fill padded_in with src matrix 
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
       	        for(a = x; a < x + blocksize && a <= data_size_X-16; a+=16) {   // leftovers for x b/c increment by 8
                    // set output vector to 0
                    output_vector1 = _mm_setzero_ps();
		    output_vector2 = _mm_setzero_ps();
		    output_vector3 = _mm_setzero_ps();
		    output_vector4 = _mm_setzero_ps();
		    
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

                            input_vector3 = _mm_loadu_ps(padded_in + 8 + ((a+i+padding_x) + (b+j+padding_y)*(data_size_X+2*padding_y)));
			    product_vector3 = _mm_mul_ps(kernel_vector, input_vector3);
			    output_vector3 = _mm_add_ps(output_vector3, product_vector3);
			    
			    input_vector4 = _mm_loadu_ps(padded_in + 12  + ((a+i+padding_x) + (b+j+padding_y)*(data_size_X+2*padding_y)));
			    product_vector4 = _mm_mul_ps(kernel_vector, input_vector4);
			    output_vector4 = _mm_add_ps(output_vector4, product_vector4);
			    
			}
                    }
 		    // After inner loop completes, write output vector to output matrix
		    // must be storeu; can't use store aligned
                    _mm_storeu_ps(out + (a + b*data_size_X), output_vector1);
		    _mm_storeu_ps(out + 4 + (a + b*data_size_X), output_vector2);
		    _mm_storeu_ps(out + 8 + (a + b*data_size_X), output_vector3);
		    _mm_storeu_ps(out + 12 + (a + b*data_size_X), output_vector4);
		}
	    }
	}
    }

    float output_float, kernel_float, input_float, product_float;
    
    // Deal with leftovers (0-15 columns on far right of out)
    for(b = 0; b < data_size_Y; b++) {        
        // Four at a time (One SIMD vector per loop)
       	for(a = (data_size_X/16)*16 ; a <= data_size_X-4; a+=4) {   // leftovers for x b/c increment by 8
            // set output vector to 0
            output_vector1 = _mm_setzero_ps();
  
            for(i = -kern_cent_X; i <= kern_cent_X; i++){          // inner loop; after all iterations, write 4 output sums
            for(j = -kern_cent_Y; j <= kern_cent_Y; j++){ 

	    kernel_vector = _mm_load1_ps(kernel + ((kern_cent_X-i) + (kern_cent_Y-j)*KERNX));
	    input_vector1 = _mm_loadu_ps(padded_in + ((a+i+padding_x) + (b+j+padding_y)*(data_size_X+2*padding_y)));
	    // above must be loadu; can't use aligned load
			    
            product_vector1 = _mm_mul_ps(kernel_vector, input_vector1);
            output_vector1 = _mm_add_ps(output_vector1, product_vector1);
            }
            }
 	    // After inner loop completes, write output vector to output matrix
	    // must be storeu; can't use store aligned
            _mm_storeu_ps(out + (a + b*data_size_X), output_vector1);
	}

        // One at a time 
        for(; a<data_size_X ; a++) {		    
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




int OLD_conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{
    // the x coordinate of the kernel's center
    int kern_cent_X = (KERNX - 1)/2;
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;
    
    // main convolution loop
	for(int x = 0; x < data_size_X; x++){ // the x coordinate of the output location we're focusing on
		for(int y = 0; y < data_size_Y; y++){ // the y coordinate of theoutput location we're focusing on
			for(int i = -kern_cent_X; i <= kern_cent_X; i++){ // kernel unflipped x coordinate
				for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){ // kernel unflipped y coordinate
					// only do the operation if not out of bounds
					if(x+i>-1 && x+i<data_size_X && y+j>-1 && y+j<data_size_Y){
						//Note that the kernel is flipped
						out[x+y*data_size_X] += 
								kernel[(kern_cent_X-i)+(kern_cent_Y-j)*KERNX] * in[(x+i) + (y+j)*data_size_X];
					}
				}
			}
		}
	}
	return 1;
}
