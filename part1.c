#include <emmintrin.h>
#include <stdio.h>

#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.


int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{
    int kern_cent_X = (KERNX - 1)/2;
    int kern_cent_Y = (KERNY - 1)/2;

    int blocksize = 32;
    int x_block, y_block, x, y, i, j, k, r;

    float get_kernel;
    float *kernel_pointer = &get_kernel;

    // declare holder of 4 identical kernel elements
    __m128 kernel_element = _mm_setzero_ps();

    // declare holder of next 4 to multiply
    __m128 photo = _mm_setzero_ps();
    float photos_as_floats[4];

    // declare holder of results
    float products_to_add[4];

    // declare pointer to results holder (then cast into float array);
    __m128 result = _mm_setzero_ps();
    float result_as_array[4];


    for(i = -kern_cent_X; i <= kern_cent_X; i++)
        for(j = -kern_cent_Y; j <= kern_cent_Y; j++)
            // fill kernel_element (all four floats are same kernel element)
	    get_kernel = kernel[(kern_cent_X-i)+(kern_cent_Y-j)*KERNX];
            kernel_element = _mm_load1_ps(kernel_pointer);
	    
            for(x_block = 0; x_block <= data_size_X-blocksize ; x_block+=blocksize)
                for(y_block = 0; y_block <= data_size_Y-blocksize ; y_block+=blocksize)

                    for(x = 0; x < blocksize; x++)
			
                        for(y = 0; y < blocksize; y+=4)
                            // fill photo_to_multiply USING if to set to 0.
			    // possible optimization later if find out row vs. column - major?
			    
                            for(k = 0; k < 4; k++)
			    
	  	            if((x_block+x+i)>-1 && (x_block+x+i)<data_size_X && (y_block+y+j+k)>-1 && (y_block+y+j+k)<data_size_Y)
                                photos_as_floats[k] = 0;
                            else
			        photos_as_floats[k] = in[(x_block+x+i) + (y_block+y+j+k)*data_size_X];
				
			    // multiply and put results in output
	                    photo = _mm_load_ps(photos_as_floats);	    
	                    result = _mm_mul_ps (kernel_element, photo);
			    _mm_store_ps(result_as_array, result);
                               
			    for (r = 0; r < 4 ; r++)                            			    
			        out[(x_block+x)+((y_block+y+r)*data_size_X)] += result_as_array[r];
				                                    


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



