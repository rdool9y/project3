#include <emmintrin.h>
#include <omp.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.

int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
       float* kernel)
{
    int kern_cent_X = (KERNX - 1)/2;
    int kern_cent_Y = (KERNY - 1)/2;
    
    int blocksize_X = 16; // must be multiple of 8
    int blocksize_Y = 16; 

    int padding_x = (KERNX / 2);  
    int padding_y = (KERNY / 2);
    int padded_size = (data_size_X + 2*padding_x) * (data_size_Y + 2*padding_y); 
    float* padded_in = malloc(padded_size * sizeof(float));

//_______________________________________________________________________
    // CREATE PADDED IN
 
    int row_index;

    
    memset(padded_in, 0.0f, padded_size);
    
    for(row_index = 0; row_index < data_size_Y; row_index++) 
	memcpy(padded_in+(padding_x)+(row_index+padding_y)*(data_size_X+2*padding_y), in + (row_index*data_size_X), sizeof(float)*data_size_X);

/*
    for(int k = 0; k < padded_size; k++) {
        if(k % (data_size_X + 2*padding_x) == 0)
            printf("\n");
        printf("%.3f ",padded_in[k]);
     }
*/

//_______________________________________________________________________
    // PUT KERNEL INTO REGISTERS   
    // The kernel is now flipped!

    __m128 kernel_TOP_LEFT = _mm_load1_ps(kernel + 8);
    __m128 kernel_TOP_CENTER = _mm_load1_ps(kernel + 7);
    __m128 kernel_TOP_RIGHT = _mm_load1_ps(kernel + 6);

    __m128 kernel_MIDDLE_LEFT = _mm_load1_ps(kernel + 5);
    __m128 kernel_MIDDLE_CENTER = _mm_load1_ps(kernel + 4);
    __m128 kernel_MIDDLE_RIGHT = _mm_load1_ps(kernel + 3);

    __m128 kernel_BOTTOM_LEFT = _mm_load1_ps(kernel + 2);
    __m128 kernel_BOTTOM_CENTER = _mm_load1_ps(kernel + 1);
    __m128 kernel_BOTTOM_RIGHT = _mm_load1_ps(kernel + 0);

//_______________________________________________________________________
    // DECLARE VECTORS

    __m128 padded_TOP_LEFT, padded_TOP_CENTER, padded_TOP_RIGHT;
    __m128 padded_MIDDLE_LEFT, padded_MIDDLE_CENTER, padded_MIDDLE_RIGHT;
    __m128 padded_BOTTOM_LEFT, padded_BOTTOM_CENTER, padded_BOTTOM_RIGHT; 

    __m128 output_vector;

//_______________________________________________________________________
    // FOR LOOP (Loops over Y-blocks, including leftovers) 

    int x, y, a, b, i, j;
    
    for(y = 0; y < data_size_Y; y+=blocksize_Y) {
        for(x = 0; x < data_size_X; x+=blocksize_X){ 
            
            for(a = x; a < x + blocksize_X && a <= data_size_X-4; a+=4) {   

                padded_MIDDLE_LEFT = _mm_loadu_ps(padded_in + ((a-1+padding_x) + (y-1+padding_y)*(data_size_X+2*padding_y)));
                padded_MIDDLE_CENTER = _mm_loadu_ps(padded_in + ((a+i+padding_x) + (y-1+padding_y)*(data_size_X+2*padding_y)));
                padded_MIDDLE_RIGHT = _mm_loadu_ps(padded_in + ((a+1+padding_x) + (y-1+padding_y)*(data_size_X+2*padding_y)));

                padded_BOTTOM_LEFT = _mm_loadu_ps(padded_in + ((a-1+padding_x) + (y+padding_y)*(data_size_X+2*padding_y)));
                padded_BOTTOM_CENTER = _mm_loadu_ps(padded_in + ((a+padding_x) + (y+padding_y)*(data_size_X+2*padding_y))); // actual first middle center
                padded_BOTTOM_RIGHT = _mm_loadu_ps(padded_in + ((a+1+padding_x) + (y+padding_y)*(data_size_X+2*padding_y)));
                
                for(b = y; b < y + blocksize_Y && b < data_size_Y; b++){ 
                
                    output_vector = _mm_setzero_ps();

                    padded_TOP_LEFT = padded_MIDDLE_LEFT;
                    padded_TOP_CENTER = padded_MIDDLE_CENTER;
                    padded_TOP_RIGHT = padded_MIDDLE_RIGHT; 

                    padded_MIDDLE_LEFT = padded_BOTTOM_LEFT;
                    padded_MIDDLE_CENTER = padded_BOTTOM_CENTER;
                    padded_MIDDLE_RIGHT = padded_BOTTOM_RIGHT;

                    padded_BOTTOM_LEFT = _mm_loadu_ps(padded_in + ((a-1+padding_x) + (b+1+padding_y)*(data_size_X+2*padding_y)));
                    padded_BOTTOM_CENTER = _mm_loadu_ps(padded_in + ((a+i+padding_x) + (b+1+padding_y)*(data_size_X+2*padding_y)));
                    padded_BOTTOM_RIGHT = _mm_loadu_ps(padded_in + ((a+1+padding_x) + (b+1+padding_y)*(data_size_X+2*padding_y)));

                    output_vector = _mm_add_ps(output_vector, _mm_mul_ps(kernel_TOP_LEFT, padded_TOP_LEFT));
                    output_vector = _mm_add_ps(output_vector, _mm_mul_ps(kernel_TOP_CENTER, padded_TOP_CENTER));
                    output_vector = _mm_add_ps(output_vector, _mm_mul_ps(kernel_TOP_RIGHT, padded_TOP_RIGHT));

                    output_vector = _mm_add_ps(output_vector, _mm_mul_ps(kernel_MIDDLE_LEFT, padded_MIDDLE_LEFT));
                    output_vector = _mm_add_ps(output_vector, _mm_mul_ps(kernel_MIDDLE_CENTER, padded_MIDDLE_CENTER));
                    output_vector = _mm_add_ps(output_vector, _mm_mul_ps(kernel_MIDDLE_RIGHT, padded_MIDDLE_RIGHT));

                    output_vector = _mm_add_ps(output_vector, _mm_mul_ps(kernel_BOTTOM_LEFT, padded_BOTTOM_LEFT));
                    output_vector = _mm_add_ps(output_vector, _mm_mul_ps(kernel_BOTTOM_CENTER, padded_BOTTOM_CENTER));
                    output_vector = _mm_add_ps(output_vector, _mm_mul_ps(kernel_BOTTOM_RIGHT, padded_BOTTOM_RIGHT));

                    _mm_storeu_ps(out + a + b*data_size_X, output_vector);

                } // end b loop
            } // end a loop

        } // end x loop (blocksize_X)
     } // end y loop (blocksize_Y)
    
   
    float output_float, kernel_float, input_float, product_float;
    for(b = 0; b < data_size_Y; b++) {        
        for(a = (data_size_X/4)*4; a < data_size_X; a++) {   
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
