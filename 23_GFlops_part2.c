#include <emmintrin.h>
#include <omp.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.


int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
       float* kernel)
{

    // the x coordinate of the kernel's center
    int kern_cent_X = (KERNX - 1)/2;
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;

    int blocksize = 72; // must be multiple of 4 (or possibly 8/12/16? if loop unrolling)
    int blocksize_Y = 50;

    __m128 kernel_vector;
    __m128 input_vector1, output_vector1, product_vector1, input_vector2, output_vector2, product_vector2, input_vector3, output_vector3, product_vector3;
    __m128 input_vector4, output_vector4, product_vector4, input_vector5, output_vector5, product_vector5, input_vector6, output_vector6, product_vector6;
    __m128 input_vector7, output_vector7, product_vector7, input_vector8, output_vector8, product_vector8, input_vector9, output_vector9, product_vector9;
    
    int padding_x = (KERNX / 2);  // can we assume that kernel is a square matrix??
    int padding_y = (KERNY /2);
    int padded_size = (data_size_X + 2*padding_x) * (data_size_Y + 2*padding_y); // not initialized to zero?
    float* padded_in = malloc(padded_size * sizeof(float));
    
    int x,y;

    int row_index;
/*
    memset(padded_in, 0.0f, padded_size);
    
    for(row_index = 0; row_index < data_size_Y; row_index++) 
        memcpy(padded_in+(padding_x)+(row_index+padding_y)*(data_size_X+2*padding_y), in + (row_index*data_size_X), sizeof(float)*data_size_X);
*/
/*
    #pragma omp parallel for
	for(x = 0; x < padded_size; x++) {                                       // manually fill with zero
	    padded_in[x] = 0.0f;                                                 // optimize: zero edges only
	} 
    #pragma omp parallel for
	for(y = 0; y < data_size_Y; y++) {
	    for(x = 0; x < data_size_X; x++) {
		padded_in[(x+padding_x) + (y+padding_y)*(data_size_X + 2*padding_y)] = in[x+y*data_size_X];
	    }
	}

*/

    if (data_size_X > 500 && data_size_X < 900 && data_size_Y > 500 && data_size_Y < 900) {
      
        memset(padded_in, 0.0f, padded_size);
    
        for(row_index = 0; row_index < data_size_Y; row_index++) 
	    memcpy(padded_in+(padding_x)+(row_index+padding_y)*(data_size_X+2*padding_y), in + (row_index*data_size_X), sizeof(float)*data_size_X);
    } else {
    #pragma omp parallel for num_threads(8)
	for(x = 0; x < padded_size; x++) {                                       // manually fill with zero
	    padded_in[x] = 0.0f;                                                 // optimize: zero edges only
	} 
    #pragma omp parallel for num_threads(8)
	for(y = 0; y < data_size_Y; y++) {
	    for(x = 0; x < data_size_X; x++) {
		padded_in[(x+padding_x) + (y+padding_y)*(data_size_X + 2*padding_y)] = in[x+y*data_size_X];
	    }
	}

    }

    int a, b, i, j;

    int k;
    float local_kern[KERNX*KERNY];

    for(k = 0; k < KERNX*KERNY; k++) {
	local_kern[k] = kernel[k];
    } 


    omp_set_num_threads(8);
    # pragma omp parallel
    {

//    printf("There are %d threads running\n",omp_get_num_threads());

# pragma omp for private(a, b, i, j, x, y, kernel_vector, input_vector1, output_vector1, product_vector1, input_vector2, output_vector2, product_vector2, input_vector3, output_vector3,product_vector3,input_vector4,output_vector4,product_vector4,input_vector5,output_vector5,product_vector5,input_vector6,output_vector6,product_vector6,input_vector7,output_vector7,product_vector7,input_vector8,output_vector8,product_vector8,input_vector9,output_vector9,product_vector9) firstprivate(local_kern) schedule(static,1)
        for(y = 0; y < data_size_Y; y+=blocksize_Y) {
          for(x = 0; x < data_size_X; x+=blocksize){ 
            for(a = x; a < x + blocksize && a <= data_size_X-36; a+=36) {   
              for(b = y; b < y + blocksize_Y && b < data_size_Y; b++){ 
                    // set output vector to 0
                    output_vector1 = _mm_setzero_ps();
                    output_vector2 = _mm_setzero_ps();
		    output_vector3 = _mm_setzero_ps();
		    output_vector4 = _mm_setzero_ps();
		    output_vector5 = _mm_setzero_ps();
		    output_vector6 = _mm_setzero_ps();
		    output_vector7 = _mm_setzero_ps();
		    output_vector8 = _mm_setzero_ps();
		    output_vector9 = _mm_setzero_ps();
            
                    for(i = -kern_cent_X; i <= kern_cent_X; i++){          // inner loop; after all iterations, write 4 output sums
                        for(j = -kern_cent_Y; j <= kern_cent_Y; j++){ 

                            kernel_vector = _mm_load1_ps(local_kern + ((kern_cent_X-i) + (kern_cent_Y-j)*KERNX));
			    
                            input_vector1 = _mm_loadu_ps(padded_in + ((a+i+padding_x) + (b+j+padding_y)*(data_size_X+2*padding_y)));
                            product_vector1 = _mm_mul_ps(kernel_vector, input_vector1);
                            output_vector1 = _mm_add_ps(output_vector1, product_vector1);
                
                            input_vector2 = _mm_loadu_ps(padded_in + 4 + ((a+i+padding_x) + (b+j+padding_y)*(data_size_X+2*padding_y)));
                            product_vector2 = _mm_mul_ps(kernel_vector, input_vector2);
                            output_vector2 = _mm_add_ps(output_vector2, product_vector2);

			    input_vector3 = _mm_loadu_ps(padded_in + 8 + ((a+i+padding_x) + (b+j+padding_y)*(data_size_X+2*padding_y)));
                            product_vector3 = _mm_mul_ps(kernel_vector, input_vector3);
                            output_vector3 = _mm_add_ps(output_vector3, product_vector3);

			    input_vector4 = _mm_loadu_ps(padded_in + 12 + ((a+i+padding_x) + (b+j+padding_y)*(data_size_X+2*padding_y)));
                            product_vector4 = _mm_mul_ps(kernel_vector, input_vector4);
                            output_vector4 = _mm_add_ps(output_vector4, product_vector4);

			    input_vector5 = _mm_loadu_ps(padded_in + 16 + ((a+i+padding_x) + (b+j+padding_y)*(data_size_X+2*padding_y)));
                            product_vector5 = _mm_mul_ps(kernel_vector, input_vector5);
                            output_vector5 = _mm_add_ps(output_vector5, product_vector5);

			    input_vector6 = _mm_loadu_ps(padded_in + 20 + ((a+i+padding_x) + (b+j+padding_y)*(data_size_X+2*padding_y)));
                            product_vector6 = _mm_mul_ps(kernel_vector, input_vector6);
                            output_vector6 = _mm_add_ps(output_vector6, product_vector6);

			    input_vector7 = _mm_loadu_ps(padded_in + 24 + ((a+i+padding_x) + (b+j+padding_y)*(data_size_X+2*padding_y)));
                            product_vector7 = _mm_mul_ps(kernel_vector, input_vector7);
                            output_vector7 = _mm_add_ps(output_vector7, product_vector7);

			    input_vector8 = _mm_loadu_ps(padded_in + 28 + ((a+i+padding_x) + (b+j+padding_y)*(data_size_X+2*padding_y)));
                            product_vector8 = _mm_mul_ps(kernel_vector, input_vector8);
                            output_vector8 = _mm_add_ps(output_vector8, product_vector8);

			    input_vector9 = _mm_loadu_ps(padded_in + 32 + ((a+i+padding_x) + (b+j+padding_y)*(data_size_X+2*padding_y)));
                            product_vector9 = _mm_mul_ps(kernel_vector, input_vector9);
                            output_vector9 = _mm_add_ps(output_vector9, product_vector9);
                        }
                    }
		    
                    // After inner loop completes, write output vector to output matrix
                    _mm_storeu_ps(out + (a + b*data_size_X), output_vector1);
                    _mm_storeu_ps(out + 4 + (a + b*data_size_X), output_vector2);
		    _mm_storeu_ps(out + 8 + (a + b*data_size_X), output_vector3);
		    _mm_storeu_ps(out + 12 + (a + b*data_size_X), output_vector4);
		    _mm_storeu_ps(out + 16 + (a + b*data_size_X), output_vector5);
		    _mm_storeu_ps(out + 20 + (a + b*data_size_X), output_vector6);
		    _mm_storeu_ps(out + 24 + (a + b*data_size_X), output_vector7);
		    _mm_storeu_ps(out + 28 + (a + b*data_size_X), output_vector8);
		    _mm_storeu_ps(out + 32 + (a + b*data_size_X), output_vector9);

                }
            }
        }
     }
    

    } // end parallel
    //printf("done with parallel\n");
    float output_float, kernel_float, input_float, product_float;
        for(b = 0; b < data_size_Y; b++) {        
            for(a = (data_size_X/36)*36; a < data_size_X; a++) {   
                // set output to 0
                output_float = 0.0f;
            
                for(i = -kern_cent_X; i <= kern_cent_X; i++){          // inner loop : all kernel elements
                    for(j = -kern_cent_Y; j <= kern_cent_Y; j++){ 
            
                        product_float = local_kern[(kern_cent_X - i) + (kern_cent_Y-j)*KERNX] * padded_in[(a+i+padding_x)+(b+j+padding_y)*(data_size_X+2*padding_y)];
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
