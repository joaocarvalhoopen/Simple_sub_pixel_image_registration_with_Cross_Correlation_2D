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
// To compile the stb :
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

package image_load_save

import "core:fmt"
import "core:strings"
import "core:math"
import "core:os"
import "core:slice"
import "core:math/rand"

import img "vendor:stb/image"

NUM_CHANNELS : i32 = 3

Img_Type :: enum {

    None,
    PNG,
    JPG,
    BMP,
    TGA,
    GIF,
    PSD,
    HDR,
    PIC,
    PNM,
}

// Internal color
RGBA :: struct #packed {

    r : u8,
    g : u8,
    b : u8,
    a : u8,
}

// Represents one plot.
Image :: struct {

    path_name_source  : string,
    path_name_target  : string,
    file_type         : Img_Type,    
    size_x            : i32,
    size_y            : i32,
    // Number of components per pixel.
    components        : i32,              // 3 RGB, 4 RGBA
    // This is the RGBA image data buffer.
    img_buffer        : []u8,
}

Coord :: struct {

    x : i32,
    y : i32,
}

get_image_type :: proc ( path_name : string ) ->
                         Img_Type {

    // Get the file extension.
    lower := strings.to_lower( path_name )

    // Check the extension.
    switch {

        case strings.has_suffix( lower, ".png" ):
            return Img_Type.PNG
        case strings.has_suffix( lower, ".jpg" ):
            return Img_Type.JPG
        case strings.has_suffix( lower, ".jpeg" ):
            return Img_Type.JPG
        case strings.has_suffix( lower, ".bmp" ):
            return Img_Type.BMP
        case strings.has_suffix( lower, ".tga" ):
            return Img_Type.TGA
        case strings.has_suffix( lower, ".gif" ):
            return Img_Type.GIF
        case strings.has_suffix( lower, ".psd" ):
            return Img_Type.PSD
        case strings.has_suffix( lower, ".hdr" ):
            return Img_Type.HDR
        case strings.has_suffix( lower, ".pic" ):
            return Img_Type.PIC
        case strings.has_suffix( lower, ".pnm" ):
            return Img_Type.PNM
    }

    return Img_Type.None
}

// Create a copy of the path_name string with the "_copy_" sufix
// added before last dot, exemple .jpeg extension.
image_path_name_inject :: proc ( path_name    : string,
                                 injected_str : string  ) ->
                                 string {
 
    // Find the last dot.
    dot_pos := strings.last_index( path_name, "." )

    // Check if the dot was found.
    if dot_pos == -1 {
        
        return fmt.aprintf( "%s%s", path_name, injected_str )
    } else {

        return fmt.aprintf( "%s%s%s",
                            path_name[ : dot_pos ],
                            injected_str,
                            path_name[ dot_pos : ]  )
    }
}

image_load :: proc ( path_name_source : string,
                     path_name_target : string ) ->
                   ( res_image : Image,
                     ok        : bool ) {

    size_x     : i32 = 0
    size_y     : i32 = 0
    components : i32 = 0
    
    // Load the image from the file.
    data : [ ^ ]u8 = img.load( strings.clone_to_cstring( path_name_source ), 
                               & size_x,
                               & size_y,
                               & components,
                               3 )

    // Check if the image was loaded.
    if data == nil {

        fmt.printfln( "Error loading image: %s", path_name_source )
        ok := false
        res_image := Image{}
        return res_image, ok
    }

    // Create the image object.
    res_image = Image {

        path_name_source = strings.clone( path_name_source ),
        path_name_target = image_path_name_inject( path_name_target, "__copy" ),
        file_type        = get_image_type( path_name_source ),
        size_x           = size_x,
        size_y           = size_y,
        components       = components,
        // img_buffer     = ( transmute( [ ^ ]Pixel ) data )[ 0 : size_x * size_y ],
        img_buffer       = data[ 0 : size_x * size_y * NUM_CHANNELS ],
    }

    return res_image, true
}

image_save :: proc ( image     : ^Image,
                     with_name : ^string = nil ) {

    file_name : string
    if with_name == nil {

        file_name = image^.path_name_target 
    } else {

        file_name = image_path_name_inject( image^.path_name_target,
                                            with_name^ )
    }
    
    // stride is in bytes.
    stride : i32 = image.size_x * NUM_CHANNELS

    ret : i32

    switch image^.file_type {

        case Img_Type.PNG:
            ret = img.write_png( 
                        strings.clone_to_cstring( file_name ),
                        image.size_x,
                        image.size_y,
                        image.components,                        // 4 components: RGBA
                        rawptr( & ( image^.img_buffer[ 0 ] ) ),
                        stride )  // in bytes

        case Img_Type.JPG:
            ret = img.write_jpg( 
                        strings.clone_to_cstring( file_name ),
                        image.size_x,
                        image.size_y,
                        image.components,                        // 4 components: RGBA
                        rawptr( & ( image^.img_buffer[ 0 ] ) ),
                        0 )   // No compression

        case Img_Type.BMP:
            ret = img.write_bmp( 
                        strings.clone_to_cstring( file_name ),
                        image.size_x,
                        image.size_y,
                        image.components,                        // 4 components: RGBA
                        rawptr( & ( image^.img_buffer[ 0 ] ) ),
                        )

        case Img_Type.TGA:
            ret = img.write_tga( 
                        strings.clone_to_cstring( file_name ),
                        image.size_x,
                        image.size_y,
                        image.components,                   // 4 components: RGBA
                        rawptr( & ( image^.img_buffer[ 0 ] ) ),
                        )

        case Img_Type.GIF:
            fmt.printfln( "Error: Writing GIF format, Unsupported image type: %d",
                          image.file_type )
            os.exit( 1 )

        case Img_Type.PSD:
            fmt.printfln( "Error: Writing PSD format, Unsupported image type: %d",
                          image.file_type )
            os.exit( 1 )

        case Img_Type.HDR:
            img.write_tga( 
                strings.clone_to_cstring( file_name ),
                image.size_x,
                image.size_y,
                image.components,                        // 4 components: RGBA
                rawptr( & ( image^.img_buffer[ 0 ] ) ),  // &data[0],
                // stride
                )

        case Img_Type.PIC:
            fmt.printfln( "Error: Writing PIC format, Unsupported image type: %d",
                          image.file_type )
            os.exit( 1 )

        case Img_Type.PNM:
            fmt.printfln( "Error: Writing PIC format, Unsupported image type: %d",
                          image.file_type )
            os.exit( 1 )

        case Img_Type.None:
            fmt.printfln( "Error: Unsupported image type: %v, %v",
                          image.file_type, file_name )
            os.exit( 1 )

        case:
            fmt.printfln( "Error: Unsupported image type: %v, %v",
                          image.file_type, file_name )
            os.exit( 1 )
    }

    if ret != 1 {
        fmt.printfln( "Error saving image: %s, ret: %v", file_name, ret )
    }

}

image_free :: proc ( image : ^Image ) {

    delete( image^.path_name_source )
    delete( image^.path_name_target )
    
    // Free the data.
    img.image_free( & image^.img_buffer[ 0 ] )
    
    //delete( image^.img_buffer )
    image^.img_buffer = nil
}

image_info_print :: proc ( image : ^Image ) {

    fmt.printfln( "Image: \n" + 
                        "  path_name_orig: %s,\n" +
                        "  path_name_des:  %s,\n" +
                        "              x:  %d,\n" +
                        "              y:  %d,\n" +
                        "     components:  %d\n\n",
                  image.path_name_source,
                  image.path_name_target,
                  image.size_x,
                  image.size_y,
                  image.components )
}   

image_get_pixel :: #force_inline proc ( image : ^Image,
                                        x     : i32,
                                        y     : i32  ) -> 
                                      ( r : u8,
                                        g : u8,
                                        b : u8  ) {

    // img_buffer := image^.img_buffer
    index := 3 * ( y * image.size_x + x ) 
    r = image^.img_buffer[ index ]
    g = image^.img_buffer[ index + 1 ]
    b = image^.img_buffer[ index + 2 ]    
    return r, g, b
}

image_set_pixel :: #force_inline proc ( image : ^Image,
                                        x     : i32,
                                        y     : i32,
                                        r     : u8,
                                        g     : u8,
                                        b     : u8  ) {

    // img_buffer := image^.img_buffer
    index := 3 * ( y * image.size_x + x )
    image^.img_buffer[ index ]     = r
    image^.img_buffer[ index + 1 ] = g
    image^.img_buffer[ index + 2 ] = b
}


