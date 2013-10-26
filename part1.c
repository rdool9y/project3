#include <emmintrin.h>
#include <stdio.h>
#include <nmmintrin.h>

#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.

/*
int fah_conv2D(float* in, float* out, int data_size_X, int data_size_Y,
	   float* kernel)
{
    // the x coordinate of the kernel's center
    int kern_cent_X = (KERNX - 1)/2;
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;

    int blocksize = 150;
    int ker_x[4] = {KERNX,KERNX,KERNX,KERNX};
    __m128i ker_x_size = __mm_loadu_si128((__m128i*)(ker_x));
    int dt_size_x[4] = {data_size_X,data_size_X,data_size_X,data_size_X};
    __m128i dt_x = __mm_loadu_si128((__m128i*)(dt_size_x));

    // main convolution loop
    for(int x = 0; x < data_size_X; x+=blocksize){ // the x coordinate of the output location we're focusing on
	for(int y = 0; y < data_size_Y; y+=blocksize){ // the y coordinate of theoutput location we're focusing on
	    for (int a = x; a < x + blocksize && a < data_size_X; a++) {
		for (int b = y; b < y + blocksize && b < data_size_Y; b++) {
		    for(int i = -kern_cent_X; i <= kern_cent_X; i++){ // kernel unflipped x coordinate
			for(int j = -kern_cent_Y; j <= kern_cent_Y && a+i>-1 && a+i<data_size_X && b+j>-1 && b+j<data_size_Y; j++){ // kernel unflipped y coordinate
			    // only do the operation if not out of bounds
			    // int sum = 0;
			       __m128i a_vec = _mm_setzero_sil123();
			    if(a+i>-1 && a+i<data_size_X && b+j>-1 && b+j<data_size_Y){
			    //
			    //Note that the kernel is flipped
			    for(int c = 0; c < 4; c++) {
				__m128i ker_x_vec = __mm_setzero_si128();
				ker_x_vec = __mm_loadu_si128((__m128i*)(kern_cent_X-i));
				__m128i ker_y_vec = __mm_setzero_si128();
				ker_y_vec = __mm_loadu_si128((__m128i*)(kern_cent_Y-j));
				ker_y_vec = __mm_mul_ps(ker_y_vec, ker_x_size);
				ker_x_vec = __mm_add_ps(ker_x_vec, ker_y_vec);

				__m128i in_a_i = __mm_setzero_si128();
				in_a_i = __mm_loadu_si128((__m128i*)(a+i));
				__m128i in_b_j = __mm_setzero_si128();
				in_b_j = __mm_loadu_si128((__m128i*)(b+j));
				in_b_j = __mm_mul_ps(in_b_j, dt_x);
				in_a_i = __mm_add_ps(in_a_i, in_b_j);

				int knl[4] = {0,0,0,0};
				int input[4] = {0,0,0,0};
				__mm_storeu_si128(knl, ker_x_vec);
				__mm_storeu_si128(input, in_a_i);

				__mm_storeu_si128( knl , ker_x_vec );
				__mm_storeu_si128(input, in_a_i);

				out[a+b*data_size_X] += knl[0];

                                out[a+b*data_size_X] +=
    			        kernel[(kern_cent_X-i)+(kern_cent_Y-j)*KERNX] * in[(a+i) + (b+j)*data_size_X];
			    }
			}
		    }
		}
	    }
	}
    }
    return 1;
}
*/


/* zero padding   : yes
   SSE            : yes
   cache blocking : yes
   loop unrolling : no

   key idea for this function : Roger's comment on Piazza at
   https://piazza.com/class/hhdehma91q41zv?cid=670

   We can accumulate 4 outputs at once if the inner loop runs KERNX*KERNY times.
   (Each inner loop multiplies and accumulates for one kernel element for four output elements.)

   This solves accumulation issue, and should make vectorization easier (for example, an input vector's
   "top left" vector is the same vector shifted up by 1 and left by 1).

   Possible problem : leftovers? (basically between 0 and 3 bottom rows)
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
    __m128 input_vector = _mm_setzero_ps();
    __m128 output_vector = _mm_setzero_ps();
    __m128 kernel_vector = _mm_setzero_ps();
    __m128 product_vector = _mm_setzero_ps();
    
    // build zero-padded copy 
    int padding = (KERNX / 2);
    int padded_size = (data_size_X + 2*padding) * (data_size_Y + 2*padding); // not initialized to zero?
    float padded_in[padded_size];
    
    int x,y;

    for(x = 0; x < padded_size; x++) {                                       // manually fill with zero
        padded_in[x] = 0.0f;                                                 // optimize: zero edges only
    }

    for(y = 0; y < data_size_Y; y++) {
        for(x = 0; x < data_size_X; x++) {
	    padded_in[(x+padding) + (y+padding)*(data_size_X + 2*padding)] = in[x+y*data_size_X];
	}
    }	

    
    //    for(int z = 0; z < padded_size; z++)
    //	printf("padded %d : %.4f\n",z,padded_in[z]);
    

/* out[a+b*data_size_X] 
    kernel[(kern_cent_X-i)+(kern_cent_Y-j)*KERNX] * in[(a+i) + (b+j)*data_size_X];
 */

    int a, b, i, j;
    
    // main convolution loop
    for(int y = 0; y < data_size_Y; y+=blocksize){ 
        for(int x = 0; x < data_size_X; x+=blocksize){ 
            for(int b = y; b < y + blocksize && b < data_size_Y; b++) {        // no leftovers for y
       	        for(int a = x; a < x + blocksize && a < data_size_X; a+=4) {   // leftovers for x b/c increment by 4
                    // set output vector to 0
                    output_vector = _mm_setzero_ps();
		    
		    for(int i = -kern_cent_X; i <= kern_cent_X; i++){          // inner loop; after all iterations, write 4 output sums
		        for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){ 

			    kernel_vector = _mm_load1_ps(kernel + ((kern_cent_X-i) + (kern_cent_Y-j)*KERNX));
			    input_vector = _mm_loadu_ps(padded_in + ((a+i+padding) + (b+j+padding)*(data_size_X+2*padding)));
			    // above must be loadu; can't use aligned load
			    
                            product_vector = _mm_mul_ps(kernel_vector, input_vector);
                            output_vector = _mm_add_ps(output_vector, product_vector);
                         }
                    }
 		    // After inner loop completes, write output vector to output matrix
		    // must be storeu; can't use store aligned
                    _mm_storeu_ps(out + (a + b*data_size_X), output_vector);
		}
	    }
	}
    }
    
    // Deal with leftovers (0-3 columns on far right of out)
    // Perhaps use vertical vectors? (would have to load manually because row-major)
    
    return 1;
}


int fah_lab_conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{
    // the x coordinate of the kernel's center
    int kern_cent_X = (KERNX - 1)/2;
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;

    int blocksize = 16;
    

    // main convolution loop
	for(int x = 0; x < data_size_X; x+=blocksize){ // the x coordinate of the output location we're focusing on
		for(int y = 0; y < data_size_Y; y+=blocksize){ // the y coordinate of theoutput location we're focusing on
			for (int a = x; a < x + blocksize && a < data_size_X; a++) {
				for (int b = y; b < y + blocksize && b < data_size_Y; b++) {
					for(int i = -kern_cent_X; i <= kern_cent_X; i++){ // kernel unflipped x coordinate
						for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){ // kernel unflipped y coordinate
							// only do the operation if not out of bounds
							if(a+i>-1 && a+i<data_size_X && b+j>-1 && b+j<data_size_Y){
							//Note that the kernel is flipped
								out[a+b*data_size_X] += 
									kernel[(kern_cent_X-i)+(kern_cent_Y-j)*KERNX] * in[(a+i) + (b+j)*data_size_X];
							}
						}
					}
				}
			}
		}
	}
	return 1;
}

int original_conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{
    int kern_cent_X = (KERNX - 1)/2;
    int kern_cent_Y = (KERNY - 1)/2;
    
    for(int x = 0; x < data_size_X; x++){
       for(int y = 0; y < data_size_Y; y++){
  	   for(int i = -kern_cent_X; i <= kern_cent_X; i++){
	       for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){
	  	   if(x+i>-1 && x+i<data_size_X && y+j>-1 && y+j<data_size_Y){
	       	       out[x+y*data_size_X] += kernel[(kern_cent_X-i)+(kern_cent_Y-j)*KERNX] * in[(x+i) + (y+j)*data_size_X];
                   }
        	}
	    }
	}
    }
    return 1;
}



