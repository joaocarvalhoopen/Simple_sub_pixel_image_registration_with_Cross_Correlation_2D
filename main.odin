// Name        : Simple sub pixel image registration with Cross_Correlation_2D.
// Description : This is a lib and an example test for doing registration
//               of one image B in relation to another image A, that have
//               to have the same size.
//               The registration that is done is only with regard to
//               translational movements in XX and in YY, but it
//               registers with sub pixel accuracy.
//               And it registers in a fast way the two images.
//               It also provides a function to shift the image with
//               sub pixel precision.
//               There is also in this code a small lib that is a wrapper
//               over the Odin stb image bindings to load and save images
//               in several image formats.
//
// To compile this code you will need to compile stb image
// in the Odin diretory.
//
// To compile stb :
//
// $ cd Odin/vendor/stb/src
// $ make
//
// To compile and run the project :
//
// $ make clean
// $ make opti
// $ make run
//
//
// Date        : 2025.05.17
// License     : MIT Open Source License
// Author      : Joao Carvalho
//

// main.odin

package main

import "core:fmt"
import "core:math/cmplx"
import "core:os"

import img     "./image_load_save"
import corr_2d "./cross_correlation_2d"

Shift_Type :: enum {

    Rolling,
    Normal,
}

main :: proc( ) {

	fmt.println( "\nBegin fast sub pixel image registration ...\n" )

    // corr_2d.test_cross_correlate_2d( )

    // shift_type := Shift_Type.Normal
    shift_type := Shift_Type.Rolling

    align_images_cross_correlation_2D( shift_type )


	fmt.println( "\n... End fast sub pixel image registration.\n" )
}

coord_xy_to_index :: #force_inline proc ( x     : int,
                                          y     : int,
                                          len_x : int  ) ->
                                          int {

	return y * len_x + x
} 

align_images_cross_correlation_2D :: proc( shift_type : Shift_Type ) {

    // Load the image.
    image_source_path  := "./images_source/lena.png"
    image_shifted_path := "./images_target/lena_shifted_output.jpg"
    image_aligned_path := "./images_target/lena_aligned_output.jpg"

    // Load the image.
    image_input, ok := img.image_load( image_source_path, image_aligned_path )
    if !ok {

        fmt.printfln( "Error loading image_input..." )
        return
    }
    fmt.printfln( "Image loaded: %s, %d x %d, components: %d",
                  image_input.path_name_source,
                  image_input.size_x,
                  image_input.size_y,
                  image_input.components )

    // Free the image.
    defer img.image_free( & image_input )

    image_input.file_type = img.Img_Type.PNG

    img.image_info_print( & image_input )


    // Convert the image to gray scale.
    image_gray := make( [ ]f32, image_input.size_x * image_input.size_y )
    if image_gray == nil {

    	fmt.printfln( "ERROR: while allocating memory for image_gray slice." )
    	os.exit( -1 )
    }

    
    // Copy from image_input to image_source_gray.
    for y in 0 ..< int( image_input.size_y ) {
    	
    	for x in 0 ..< int( image_input.size_x ) {
    	
    	    r, b, g := img.image_get_pixel( & image_input, i32( x ), i32( y ) )
    		gray_val_f32 : f32 = ( f32( r ) + f32( b ) + f32( g ) ) / 3.0
    		
    		index := coord_xy_to_index( x, y, int( image_input.size_x ) ) 
    		image_gray[ index ] = gray_val_f32 
    	}
	}


    // Creating reference image and fill image buffer.

    // Non-POT ( Non Power of Two ) to test padding
    img_rows, img_cols := int( image_input.size_y ), int( image_input.size_x )

	fmt.printf( "Creating reference image ( %d x %d )...\n", img_rows, img_cols )


//    img_width  := 5000 // Non-power of two
//    img_height := 7000 // Non-power of two


    // True shift to apply.
//    true_shift_y :=  3.25
//    true_shift_x := -2.75

    true_shift_y := 100.253333
    true_shift_x := -50.753333

//    true_shift_y :=  220.253333
//    true_shift_x := -200.753333

    if shift_type == Shift_Type.Normal { 

        true_shift_y =  100.0
        true_shift_x =  -50.0
    }

    fmt.printf( "True shift ( Y, X ): ( %.8f, %.8f )\n", true_shift_y, true_shift_x )

    // 1. Generate original image.

    // Centering the Gaussian slightly off-center of the pixel grid can make subpixel effects more apparent
    
//    img_orig := corr_2d.generate_gaussian_image(img_width, img_height, sigma, sigma, 0.1, 0.1)

    img_orig := corr_2d.create_image( img_rows, img_cols ) 

    for y in 0 ..< img_rows {
        
        for x in 0 ..< img_cols {

    		index := coord_xy_to_index( x, y, int( img_cols ) ) 
    		val := image_gray[ index ] 

            img_orig.data[ y * img_cols + x ] = f64( val )
        }
    }

    // defer corr_2d.destroy_image( & img_orig )


    img_shifted : corr_2d.Image

    switch shift_type {

        case Shift_Type.Rolling : { 

            // 2. Apply true shift to create a second image.
            
            // CORRECT
            img_shifted = corr_2d.apply_subpixel_shift( img_orig, true_shift_y, true_shift_x )
            
            // defer destroy_image(&img_shifted)

            }

        case Shift_Type.Normal : {

            // Sloppy image shift without subpixel and without rolling shift, it shifts and places dark regions in the image.

            img_shifted = corr_2d.create_image( img_rows, img_cols ) 

            for y in 0 ..< img_rows {
                
                for x in 0 ..< img_cols {

                    s_x := x - int( true_shift_x ) 
                    s_y := y - int( true_shift_y )

                    if s_x > 0 && s_x < img_cols && s_y > 0 && s_y < img_rows {

                        index := coord_xy_to_index( x, y, int( img_cols ) ) 
                        val := img_orig.data[ index ] 

                        img_shifted.data[ s_y * img_cols + s_x ] = f64( val )

                    }
                }
            }

        }

    }


// img_shifted

    image_to_save_1   := img_shifted
    image_file_path_1 := image_shifted_path  
    save_image_jpef_from_corr_2d( image_to_save_1, image_file_path_1 )


    // 3. Detect shift using cross-correlation.
    fmt.println( "Detecting shift..." )

    detected_shift_y, detected_shift_x := corr_2d.cross_correlate_2d( img_orig, img_shifted )


    fmt.printf( "Detected shift ( Y, X ): ( %.8f, %.8f )\n", detected_shift_y, detected_shift_x )

    // Error in detected shift
    err_shift_y := detected_shift_y - true_shift_y
    err_shift_x := detected_shift_x - true_shift_x

    fmt.printf( "Error in detected shift ( Y, X ): ( %.8f, %.8f )\n", err_shift_y, err_shift_x )
    
    // 4. Realign the shifted image using the "negative" of the detected shift.
    // We want to shift img_shifted by (-detected_sy, -detected_sx) to bring it back to img_orig
    fmt.println( "Realigning image..." )


    img_realigned := corr_2d.apply_subpixel_shift( img_shifted, -detected_shift_y, -detected_shift_x )
    // defer destroy_image( & img_realigned )
    

    // 5. Calculate Mean Squared Error between original and realigned image.

    mse := corr_2d.calculate_mse( img_orig, img_realigned )

    fmt.printf( "MSE between original and realigned image: %.6e\n", mse )

    // Cleanup
    defer corr_2d.destroy_image( & img_orig )
    defer corr_2d.destroy_image( & img_shifted )
    defer corr_2d.destroy_image( & img_realigned )

// img_realigned

    image_to_save_2   := img_realigned
    image_file_path_2 := image_aligned_path  
    save_image_jpef_from_corr_2d( image_to_save_2, image_file_path_2 )


    fmt.println( "\n Image saved.\n" )
}



save_image_jpef_from_corr_2d :: proc ( image_to_save   : corr_2d.Image,
                                       image_file_path : string ) {

    img_rows := image_to_save.height
    img_cols := image_to_save.width

    // Save to disc the registered image or realigned image.
    // Convert the image to gray scale.

    image_aligned_out := new( img.Image )
    if image_aligned_out == nil {

        fmt.printfln( "ERROR: while allocating memory for image_aligned_out to save." )
        os.exit( -1 )
    }
 
    image_aligned_out.components       = 3
    image_aligned_out.file_type        = img.Img_Type.JPG
    image_aligned_out.path_name_source = image_file_path    // ""
    image_aligned_out.path_name_target = image_file_path
    image_aligned_out.size_x           = i32( img_cols ) 
    image_aligned_out.size_y           = i32( img_rows )

    image_aligned_out.img_buffer = make( [ ]u8, img_cols * img_rows * int( image_aligned_out.components ) )

    if image_aligned_out == nil {

        fmt.printfln( "ERROR: while allocating memory for image_aligned_out output image." )
        os.exit( -1 )
    }


// img_realigned
    
    // Copy from image_input to image_source_gray
    for y in 0 ..< int( img_rows ) {
        
        for x in 0 ..< int( img_cols ) {
 
            f_pixel : f64 = image_to_save.data[ y * img_cols + x ]

            // fmt.printfln( "%f", f_pixel )

            img.image_set_pixel( image_aligned_out,
                                 i32( x ),
                                 i32( y ),
                                 u8( f_pixel ),
                                 u8( f_pixel ),
                                 u8( f_pixel )  )
            }
    }

    image_target_path_with_name : string = ""
    img.image_save( image_aligned_out, & image_target_path_with_name )
}


