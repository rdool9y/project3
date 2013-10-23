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
    int x_block, y_block, x, y, i, j;

    for(x_block = 0; x_block <= data_size_X-blocksize ; x_block+=blocksize)
        for(y_block = 0; y_block <= data_size_Y-blocksize ; y_block+=blocksize)
            for(x = 0; x < blocksize; x++)
                for(y = 0; y < blocksize; y++)
  	            for(i = -kern_cent_X; i <= kern_cent_X; i++)
	                for(j = -kern_cent_Y; j <= kern_cent_Y; j++)
	  	            if((x_block+x+i)>-1 && (x_block+x+i)<data_size_X && (y_block+y+j)>-1 && (y_block+y+j)<data_size_Y)
	       	                out[(x_block+x)+((y_block+y)*data_size_X)] += kernel[(kern_cent_X-i)+(kern_cent_Y-j)*KERNX] *
				                                    in[(x_block+x+i) + (y_block+y+j)*data_size_X];

    
     for(x = 0; x < data_size_X; x++){
	 for(y = ((data_size_Y/blocksize)*blocksize); y < data_size_Y; y++){
  	   for(i = -kern_cent_X; i <= kern_cent_X; i++){
	       for(j = -kern_cent_Y; j <= kern_cent_Y; j++){
	  	   if(x+i>-1 && x+i<data_size_X && y+j>-1 && y+j<data_size_Y){
	       	       out[x+y*data_size_X] += kernel[(kern_cent_X-i)+(kern_cent_Y-j)*KERNX] * in[(x+i) + (y+j)*data_size_X];
                   }
        	}
	    }
	}
    }

    int y_max = (data_size_Y/blocksize)*blocksize;
     
    for(x = ((data_size_X/blocksize)*blocksize); x < data_size_X; x++){
       for(y = 0; y < y_max; y++){
  	   for(i = -kern_cent_X; i <= kern_cent_X; i++){
	       for(j = -kern_cent_Y; j <= kern_cent_Y; j++){
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



