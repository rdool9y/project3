#include <emmintrin.h>
#include <stdio.h>
#include <nmmintrin.h>

#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.

int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
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
							/*
						    int sum = 0;
						    __m128i a_vec = _mm_setzero_sil123();
							if(a+i>-1 && a+i<data_size_X && b+j>-1 && b+j<data_size_Y){
								*/
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


int conv2D2(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{
    int kern_cent_X = (KERNX - 1)/2;
    int kern_cent_Y = (KERNY - 1)/2;

    int blocksize = 110;
    int x_block, y_block, x, y, i, j, k, r;

    // declare holder of 4 identical kernel elements
    __m128 kernel_element = _mm_setzero_ps();

    // declare holder of next 4 to multiply
    __m128 photo = _mm_setzero_ps();
    float photos_as_floats[4];//__attribute__((aligned(16)));

    // declare results holder (and corresponding float array);
    __m128 result = _mm_setzero_ps();
    float result_as_array[4];//__attribute__((aligned(16)));


    for(i = -kern_cent_X; i <= kern_cent_X; i++) {
        for(j = -kern_cent_Y; j <= kern_cent_Y; j++) {
            // fill kernel_element (all four floats are same kernel element)
  	    kernel_element = _mm_load1_ps(kernel+(kern_cent_X-i)+(kern_cent_Y-j)*KERNX);
	    
            for(x_block = 0; x_block <= data_size_X-blocksize ; x_block+=blocksize) {
                for(y_block = 0; y_block <= data_size_Y-blocksize ; y_block+=blocksize) {
                    for(x = 0; x < blocksize; x++) {
                        for(y = 0; y < blocksize; y+=4) {
                            // fill photo_to_multiply USING if to set to 0.
			    // possible optimization later if find out row vs. column - major?
			    // Because can just zero-pad and load directly
			    
                            for(k = 0; k < 4; k++) {
			    
	  	                if((x_block+x+i)>-1 && (x_block+x+i)<data_size_X && (y_block+y+j+k)>-1 && (y_block+y+j+k)<data_size_Y) {
                                    photos_as_floats[k] = in[(x_block+x+i) + (y_block+y+j+k)*data_size_X]; }
                                else {
				    photos_as_floats[k] = 0; }
			    }
				
      			        // multiply and put results in output
	                        photo = _mm_load_ps(photos_as_floats);	    
	                        result = _mm_mul_ps (kernel_element, photo);
			        _mm_store_ps(result_as_array, result);
                               
			        for (r = 0; r < 4 ; r++) {                            			    
			            out[(x_block+x)+((y_block+y+r)*data_size_X)] += result_as_array[r]; }
			    	
			}
                    }
	        }		    
	    }		    		    
	}		    		    
    }		    			                                    


    // leftover (not covered by blocking)

    int y_max = (data_size_Y/blocksize)*blocksize;

   for(i = -kern_cent_X; i <= kern_cent_X; i++){
       for(j = -kern_cent_Y; j <= kern_cent_Y; j++){
         for(x = 0; x < data_size_X; x++){
     	     for(y = y_max; y < data_size_Y; y++){
	  	   if(x+i>-1 && x+i<data_size_X && y+j>-1 && y+j<data_size_Y){
	       	       out[x+y*data_size_X] += kernel[(kern_cent_X-i)+(kern_cent_Y-j)*KERNX] * in[(x+i) + (y+j)*data_size_X];
                   }
        	}
	    }
	}
    }

   for(i = -kern_cent_X; i <= kern_cent_X; i++){
       for(j = -kern_cent_Y; j <= kern_cent_Y; j++){
           for(x = ((data_size_X/blocksize)*blocksize); x < data_size_X; x++){
               for(y = 0; y < y_max; y++){
	  	   if(x+i>-1 && x+i<data_size_X && y+j>-1 && y+j<data_size_Y){
	       	       out[x+y*data_size_X] += kernel[(kern_cent_X-i)+(kern_cent_Y-j)*KERNX] * in[(x+i) + (y+j)*data_size_X];
                   }
        	}
	    }
	}
    }
	

    return 1;
}

int conv2D_CLEAN(float* in, float* out, int data_size_X, int data_size_Y,
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



