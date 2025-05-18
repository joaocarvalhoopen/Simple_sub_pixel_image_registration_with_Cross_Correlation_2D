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


package cross_corrletation_2d

import "core:fmt"
import "core:math"
import "core:math/cmplx"
import "core:mem"
import "core:slice"
import "core:strings"

Image :: struct {

    data   : [ ]f64,
    width  : int,
    height : int,
}

next_power_of_two :: #force_inline proc( n : int ) ->
                                         int {

    if n <= 0 {

        return 1
    }
    p := 1
    for p < n {

        p <<= 1
    }
    return p
}

bit_reverse_permutation :: #force_inline proc( data : [ ]complex128 ) {

    n := len( data )
    j := 0
    for i in 1 ..< n {

        bit := n >> 1
        for j >= bit {

            j -= bit
            bit >>= 1
        }
        j += bit
        if i < j {

            data[ i ], data[ j ] = data[ j ], data[ i ]
        }
    }
}


// In place 1D FFT Implementation ( Cooley-Tukey, Radix-2, Iterative )
// "inverse" flag determines if FFT or IFFT
// Assumes len(data) is a power of two
_fft1d :: proc( data : [ ]complex128,
                inverse : bool ) {

    n := len( data )
    if n <= 1 {

        return
    }

    // Bit-reversal permutation
    bit_reverse_permutation( data )

    // Iterative FFT
    for len_ := 2; len_ <= n; len_ <<= 1 {

        half_len := len_ >> 1
        angle_step_val := math.PI / f64( half_len )
        if inverse {

            angle_step_val = -angle_step_val
        }
        
        w_len := cmplx.exp( complex128( complex( 0, angle_step_val ) ) )

        for i := 0; i < n; i += len_ {

            w: complex128 = complex( 1, 0 )
            for j := 0; j < half_len; j += 1 {

                u := data[ i + j ]
                v := data[ i + j + half_len ] * w
                data[ i + j ] = u + v
                data[ i + j + half_len ] = u - v
                w *= w_len
            }
        }
    }

    if inverse {

        inv_n := 1.0 / f64( n )
        for i := 0; i < n; i += 1 {

            data[ i ] *= complex( inv_n, 0 )
        }
    }
}

fft1d :: #force_inline proc( data : [ ]complex128 ) {

    _fft1d(data, false)
}

ifft1d :: #force_inline proc( data : [ ]complex128 ) {

    _fft1d( data, true )
}

// 2D FFT/IFFT
// Assumes data is in row-major order and dimensions are powers of two
_fft2d :: proc( data          : [ ]complex128,
                width, height : int,
                inverse       : bool ) {

    // Row-wise FFTs
    row_buffer := make( [ ]complex128, width )
    defer delete( row_buffer )
    for r in 0 ..< height {

        offset := r * width
    
    //    copy( row_buffer, data[ offset : offset + width ] )
    //    _fft1d( row_buffer, inverse )
    //    copy( data[ offset : offset + width ], row_buffer )
    
        _fft1d( data[ offset : offset + width ], inverse )
    }

    // Column-wise FFTs ( requires transpose-like access or a column buffer )
    col_buffer := make( [ ]complex128, height )
    defer delete( col_buffer )
    for c in 0 ..< width {

        // Extract column
        for r in 0 ..< height {

            col_buffer[ r ] = data[ r * width + c ]
        }

        _fft1d( col_buffer, inverse )
        
        // Place back column
        for r in 0 ..< height {

            data[ r * width + c] = col_buffer[ r ]
        }
    }
}

fft2d :: #force_inline proc( data          : [ ]complex128,
                             width, height : int ) {

    _fft2d( data, width, height, false )
}

ifft2d :: #force_inline proc( data          : [ ]complex128,
                              width, height : int ) {

    _fft2d( data, width, height, true )
}

create_image :: proc( width, height : int,
                      initial_val   : f64 = 0.0 ) ->
                      Image {

    img_data := make( [ ]f64, width * height )
    if initial_val != 0.0 {

        slice.fill( img_data, initial_val )
    }
    return Image{ img_data, width, height }
}

destroy_image :: proc( img : ^Image ) {

    delete( img.data )
    img.data   = nil
    img.width  = 0
    img.height = 0
}

// Pads image with zeros to new_width, new_height ( must be >= original ).
pad_image :: proc( img                   : Image,
                   new_width, new_height : int ) ->
                   Image {

    assert( new_width >= img.width && new_height >= img.height )
    padded_data := make( [ ]f64, new_width * new_height ) 
   
    for r in 0 ..< img.height {

        src_offset := r * img.width
        dst_offset := r * new_width
        copy( padded_data[ dst_offset : dst_offset + img.width ],
              img.data[ src_offset : src_offset + img.width ] )
    }
    return Image{ padded_data, new_width, new_height }
}

// Crops image. Offset is from top-left of original.
crop_image :: proc( img                                         : Image,
                    crop_width, crop_height, offset_x, offset_y : int ) ->
                    Image {

    assert( crop_width > 0 && crop_height > 0 )
    assert( offset_x >= 0 && offset_y >= 0 )
    assert( offset_x + crop_width <= img.width )
    assert( offset_y + crop_height <= img.height )

    cropped_data := make( [ ]f64, crop_width * crop_height )
    
    for r in 0 ..< crop_height {

        src_row_start := ( offset_y + r ) * img.width + offset_x
        dst_row_start := r * crop_width
        copy( cropped_data[ dst_row_start : dst_row_start + crop_width ],
              img.data[ src_row_start : src_row_start + crop_width ] )
    }
    
    return Image{ cropped_data, crop_width, crop_height }
}

// Convert f64 image data to complex128.
image_to_complex :: proc( img_data : [ ]f64 ) ->
                          [ ]complex128 {

    c_data := make( [ ]complex128, len( img_data ) )
    for val, i  in img_data {

        c_data[ i ] = complex( val, 0 )
    }

    return c_data
}

// Convert complex128 image data ( real part ) to f64.
complex_to_real_image :: proc( c_data : [ ]complex128 ) ->
                               [ ]f64 {

    r_data := make( [ ]f64, len( c_data ) )
    for c_val, i in c_data {

        r_data[ i ] = real( c_val )
    }
    return r_data
}

// FFTShift for 1D complex data ( in-place )
fftshift1d_complex :: proc( data : [ ]complex128 ) {

    n := len( data )
    if n <= 1 { return }
    half_n := n / 2 // Assumes n is even, which it will be for power-of-two FFTs.
    
    temp := make( [ ]complex128, half_n )
    defer delete( temp )

    // Copy first half to temp.
    copy( temp, data[ 0 : half_n ] )
    // Move second half to first half.
    copy( data[ 0 : n - half_n ], data[ half_n : n ])
    // Copy temp (original first half) to second half.
    copy( data[ n - half_n : n ], temp )
}


// FFTShift for 2D complex data ( in-place ), width / height are of the data.
fftshift2d_complex :: proc( data          : [ ]complex128,
                            width, height : int ) {

    // Shift rows
    row_buffer := make( [ ]complex128, width )
    defer delete( row_buffer )
    for r in 0 ..< height {

        offset := r * width
        copy( row_buffer, data[ offset : offset + width ] )
        fftshift1d_complex( row_buffer )
        copy( data[ offset : offset + width ], row_buffer )
    }

    // Shift columns
    col_buffer := make( [ ]complex128, height )
    defer delete( col_buffer )
    for c in 0 ..< width {

        for r in 0 ..< height {

            col_buffer[ r ] = data[ r * width + c ]
        }
        fftshift1d_complex( col_buffer )
        for r in 0 ..< height {

            data[ r * width + c ] = col_buffer[ r ]
        }
    }
}

// FFTShift for 2D real data ( returns new slice ).
fftshift2d_real :: proc( data          : [ ]f64,
                         width, height : int ) -> 
                         [ ]f64 {

    shifted_data := make( [ ]f64, len( data ) )
    // Convert to complex, shift, convert back to real.
    // This is inefficient but reuses complex shift. A direct real shift is better.
    // Direct real shift:
    half_w := width / 2
    half_h := height / 2

    for r_orig in 0 ..< height {

        for c_orig in 0 ..< width {

            r_new := ( r_orig + half_h ) % height
            c_new := ( c_orig + half_w ) % width
            shifted_data[ r_new * width + c_new] = data[ r_orig * width + c_orig ]
        }
    }

    return shifted_data
}


// Cross-Correlate Function.
// Returns ( shift_y, shift_x )
// This shift indicates how much img2 is shifted relative to img1.
// E.g., if img2 is img1 shifted right by 2.5px, sx = 2.5.
cross_correlate_2d :: proc( img1, img2 : Image ) -> 
                          ( shift_y : f64,
                            shift_x : f64  ) {

    // For aligning two images of ( potentially ) same scene, pad to common NPoT - Non Power Of Two size.
    // If dimensions differ, use max. Here, assume img1 and img2 are same logical size for shift detection.
    assert( img1.width == img2.width &&
            img1.height == img2.height,
            "Images must have same dimensions for this simplified cross-correlation." )
    
    orig_w, orig_h := img1.width, img1.height
    fft_w := next_power_of_two( orig_w )
    fft_h := next_power_of_two( orig_h )

    // 1. Pad images and convert to complex.
    padded_img1 := pad_image( img1, fft_w, fft_h )
    defer destroy_image( & padded_img1 )
    c_img1_data := image_to_complex( padded_img1.data )
    defer delete( c_img1_data )

    padded_img2 := pad_image( img2, fft_w, fft_h )
    defer destroy_image( & padded_img2 )
    c_img2_data := image_to_complex( padded_img2.data )
    defer delete( c_img2_data )

    // 2. Perform 2D FFT.
    fft2d( c_img1_data, fft_w, fft_h )
    fft2d( c_img2_data, fft_w, fft_h )

    // 3. Compute cross-power spectrum: FFT( img1 ) * conj( FFT( img2 ) ).
    cross_power_spectrum := make( [ ]complex128, len( c_img1_data ) )
    defer delete( cross_power_spectrum )
    for i in 0 ..< len( c_img1_data ) {

        cross_power_spectrum[ i ] = c_img1_data[ i ] * cmplx.conj( c_img2_data[ i ] )
        // One can also do it with : 
        //
        // Phase correlation ( normalize magnitude )
        // abs_val := complex.abs( cross_power_spectrum[ i ] )
        // if abs_val > 1e-9 { // Avoid division by zero
        //  cross_power_spectrum[ i ] /= complex( abs_val, 0 )
        // }
    }
    
    // 4. Perform 2D IFFT.
    correlation_surface_complex := cross_power_spectrum // Reuse memory
    ifft2d( correlation_surface_complex, fft_w, fft_h )

    // 5. Take real part and fftshift.
    correlation_surface_real_unshifted := complex_to_real_image( correlation_surface_complex )
    defer delete( correlation_surface_real_unshifted )
    
    // fftshift the real correlation surface for peak finding.
    correlation_surface_real_shifted := fftshift2d_real( correlation_surface_real_unshifted, fft_w, fft_h )
    defer delete( correlation_surface_real_shifted )


    // 6. Find integer peak in the SHIFTED correlation surface.
    max_val : f64 = -1.0e30 // Very small number.
    peak_r_shifted, peak_c_shifted : int = 0, 0
    for r in 0 ..< fft_h {

        for c in 0 ..< fft_w {

            val := correlation_surface_real_shifted[ r * fft_w + c ]
            if val > max_val {

                max_val = val
                peak_r_shifted = r
                peak_c_shifted = c
            }
        }
    }
    
    // 7. Subpixel refinement using quadratic interpolation around the peak
    //    Uses values from the SHIFTED correlation surface.
    sub_dr, sub_dc : f64
    
    // Y direction ( column of peak )
    // y_m1, y_0, y_p1
    // Indices need to wrap around for correlation surface
    y_m1_r := ( peak_r_shifted - 1 + fft_h ) % fft_h
    y_0_r  := peak_r_shifted
    y_p1_r := ( peak_r_shifted + 1 ) % fft_h

    val_ym1 := correlation_surface_real_shifted[ y_m1_r * fft_w + peak_c_shifted ]
    val_y0  := correlation_surface_real_shifted[ y_0_r  * fft_w + peak_c_shifted ] // This is max_val
    val_yp1 := correlation_surface_real_shifted[ y_p1_r * fft_w + peak_c_shifted ]

    denominator_y := 2 * ( val_ym1 + val_yp1 - 2 * val_y0 )
    if math.abs( denominator_y ) < 1e-9 { // Avoid division by zero / flat peak
        
        sub_dr = 0.0
    } else {
        
        sub_dr = ( val_ym1 - val_yp1 ) / denominator_y
    }

    // X direction (row of peak)
    x_m1_c := ( peak_c_shifted - 1 + fft_w ) % fft_w
    x_0_c  := peak_c_shifted
    x_p1_c := ( peak_c_shifted + 1 ) % fft_w

    val_xm1 := correlation_surface_real_shifted[ peak_r_shifted * fft_w + x_m1_c ]
    val_x0  := correlation_surface_real_shifted[ peak_r_shifted * fft_w + x_0_c ] // This is max_val
    val_xp1 := correlation_surface_real_shifted[ peak_r_shifted * fft_w + x_p1_c ]

    denominator_x := 2 * ( val_xm1 + val_xp1 - 2 * val_x0 )
    if math.abs( denominator_x ) < 1e-9 {

        sub_dc = 0.0
    } else {

        sub_dc = ( val_xm1 - val_xp1 ) / denominator_x
    }
    
    // The peak in the fftshifted correlation map corresponds to the displacement.
    // A peak at ( H / 2, W / 2 ) in the shifted map means zero shift.
    // A peak at ( H / 2 + dy, W / 2 + dx ) means a shift of ( dy, dx ).
    // So, actual shift = peak_coord_in_shifted_map - center_coord_of_shifted_map.
    // And the subpixel refinement is an offset from the integer peak.
    
    // Integer part of shift from peak_r_shifted, peak_c_shifted
    // These are coordinates in the SHIFTED map.
    // Center of shifted map is ( fft_h / 2, fft_w / 2 )
    // The shift is peak_location - center_location
    // So, if peak is at ( r, c ) in shifted map, dy = r - fft_h / 2, dx = c - fft_w / 2
    // This dy, dx is how much img2 is shifted "from" img1 if the peak is tau.
    // Cross-correlation R_fg( tau ) = int f( t ) g * ( t - tau) dt. Peaks when tau = shift of g relative to f.
    // No, the convention is : R_fg( tau ) = int f( t ) g * ( t + tau ) dt. Peak when tau = -( shift of g relative to f ).
    // OR R_fg( tau ) = int f * ( t ) g( t + tau ) dt. Peak when tau = ( shift of g relative to f ).
    // Let's stick to common Fourier definition :
    // if g( x ) = f( x - delta ), then XCORR( f, g ) peaks at -delta ( when using IFFT( F * G_conj ) ).
    // Our peak_r_shifted, peak_c_shifted are indices.
    // Shift represented by peak_r_shifted is:
    //   peak_r_shifted if peak_r_shifted < fft_h / 2
    //   peak_r_shifted - fft_h if peak_r_shifted >= fft_h / 2
    // This is the "tau_y" value.
    // The true shift is "delta_y = -tau_y".

    // Calculate effective integer peak coordinates in unshifted domain for interpretation
    // ( This is tau_y_int, tau_x_int )
    final_peak_r := f64( peak_r_shifted ) + sub_dr
    final_peak_c := f64( peak_c_shifted ) + sub_dc
    
    // Convert peak location in shifted map to actual shift value ( tau )
    // If peak_r_shifted is at H / 2, means 0 shift. If at H / 2 + 1, means +1 shift. If H / 2 - 1, means -1 shift.
    tau_y := final_peak_r - f64( fft_h ) / 2.0
    tau_x := final_peak_c - f64( fft_w ) / 2.0

    // The detected shift ( delta_y, delta_x ) is how much img2 is displaced relative to img1.
    // If img2 = img1_shifted_by_D, then XCORR( img1, img2 ) peaks at D when using the definition
    // "IFFT( F1 * conj( F2 ) ) / | ... |" and then interpreting the peak location directly.
    // However, the standard "IFFT( F1 * conj( F2 ) )" method has the peak at "D" for "f( t )"" vs "f( t - D )".
    // If `g(x) = f(x-D)`, then `G(k) = F(k)exp(-j2pikD/N)`.
    // " F( k ) conj( G ( k ) ) = F( k ) conj( F( k ) ) exp( j2pikD / N ) = | F( k ) | ^ 2 exp( j2pikD / N ) ".
    // IFFT of this will peak at "D".
    // So, tau_y and tau_x are the shifts.

    return tau_y, tau_x
}


// Apply Subpixel Shift
apply_subpixel_shift :: proc( img              : Image,
                              shift_y, shift_x : f64 ) -> 
                              Image {

    orig_w, orig_h := img.width, img.height
    fft_w := next_power_of_two( orig_w )
    fft_h := next_power_of_two( orig_h )

    // 1. Pad image and convert to complex.
    padded_img := pad_image( img, fft_w, fft_h )
    // "destroy_image" will be called by caller on "img" if needed. "padded_img" is temporary.
    c_data := image_to_complex( padded_img.data )
    delete( padded_img.data ) // Padded image data no longer needed after conversion.

    // 2. Perform 2D FFT.
    fft2d( c_data, fft_w, fft_h )

    // 3. Create phase ramp and multiply.
    // Phase shift : exp( -j * 2 * pi * (u * dx / N_cols + v * dy / N_rows ) )
    // u is freq_x, v is freq_y
    // Frequencies need to be handled correctly ( fftshifted conceptually ).
    for r in 0 ..< fft_h {

        v_freq : f64
        if r < fft_h / 2 {

            v_freq = f64( r )
        } else {

            v_freq = f64( r - fft_h )
        }

        for c in 0 ..< fft_w {

            u_freq : f64
            if c < fft_w / 2 {

                u_freq = f64( c )
            } else {

                u_freq = f64( c - fft_w )
            }
            
            phase_angle := -2.0 * math.PI * ( u_freq * shift_x / f64( fft_w ) + v_freq * shift_y / f64( fft_h ) )
            phase_multiplier := cmplx.exp( complex( 0, phase_angle ) )
            
            c_data[ r * fft_w + c ] *= phase_multiplier
        }
    }

    // 4. Perform 2D IFFT.
    ifft2d( c_data, fft_w, fft_h )

    // 5. Convert back to real and crop.
    shifted_padded_real_data := complex_to_real_image( c_data )
    delete( c_data ) // Complex data no longer needed.

    shifted_padded_img := Image{ shifted_padded_real_data, fft_w, fft_h }
    shifted_img := crop_image( shifted_padded_img, orig_w, orig_h, 0, 0 )
    
    destroy_image( & shifted_padded_img ) // Free data of the large padded image.

    return shifted_img
}


// Test image.
generate_gaussian_image :: proc( width, height                                      : int,
                                 sigma_x, sigma_y, center_x_offset, center_y_offset : f64 ) -> 
                                 Image {

    img := create_image( width, height )
    center_x := f64( width - 1 ) / 2.0 + center_x_offset
    center_y := f64( height - 1 ) / 2.0 + center_y_offset

    for r in 0 ..< height {

        for c in 0 ..< width {

            dx := f64( c ) - center_x
            dy := f64( r ) - center_y
            val := math.exp( -( dx * dx / ( 2 * sigma_x * sigma_x ) + dy * dy / ( 2 * sigma_y * sigma_y ) ) )
            img.data[r*width + c] = val
        }
    }

    return img
}

calculate_mse :: proc( img1, img2 : Image ) ->
                       f64 {

    assert( img1.width == img2.width &&
            img1.height == img2.height )

    sum_sq_err : f64 = 0
    num_pixels := img1.width * img1.height
    for i in 0 ..< num_pixels {

        err := img1.data[ i ] - img2.data[ i ]
        sum_sq_err += err * err
    }

    return sum_sq_err / f64( num_pixels )
}

test_cross_correlate_2d_f64 :: proc( ) {

    fmt.println( "\nTest Subpixel Image Alignment of a Gaussian image...\n" )

    // Image parameters
//    img_width  := 50 // Non-power of two
//    img_height := 70 // Non-power of two
    
//    img_width  := 500 // Non-power of two
//    img_height := 700 // Non-power of two

    img_width  := 500 // Non-power of two
    img_height := 700 // Non-power of two

    sigma      := 10.0

    // True shift to apply
//    true_shift_y :=  3.25
//    true_shift_x := -2.75

    true_shift_y :=  300.253333
    true_shift_x := -200.753333

    fmt.printf( "True shift ( Y, X ): ( %.4f, %.4f )\n", true_shift_y, true_shift_x )

    // 1. Generate original image.
    // Centering the Gaussian slightly off-center of the pixel grid can make subpixel effects more apparent.
    img_orig := generate_gaussian_image( img_width, img_height, sigma, sigma, 0.1, 0.1 )
    // defer destroy_image( & img_orig )

    // 2. Apply true shift to create a second image.
    img_shifted := apply_subpixel_shift( img_orig, true_shift_y, true_shift_x )
    // defer destroy_image( & img_shifted )

    // 3. Detect shift using cross-correlation.
    fmt.println( "Detecting shift..." )
    detected_shift_y, detected_shift_x := cross_correlate_2d( img_orig, img_shifted )
    fmt.printf( "Detected shift ( Y, X ): ( %.8f, %.8f )\n", detected_shift_y, detected_shift_x )

    // Error in detected shift.
    err_shift_y := detected_shift_y - true_shift_y
    err_shift_x := detected_shift_x - true_shift_x
    fmt.printf( "Error in detected shift ( Y, X ): ( %.8f, %.8f )\n", err_shift_y, err_shift_x )
    
    // 4. Realign the shifted image using the "negative" of the detected shift.
    // We want to shift img_shifted by ( -detected_sy, -detected_sx ) to bring it back to img_orig.
    fmt.println( "Realigning image..." )
    img_realigned := apply_subpixel_shift( img_shifted, -detected_shift_y, -detected_shift_x )
    // defer destroy_image( & img_realigned )

    // 5. Calculate Mean Squared Error between original and realigned image.
    mse := calculate_mse( img_orig, img_realigned )
    fmt.printf( "MSE between original and realigned image: %.6e\n", mse )

    destroy_image( & img_orig )
    destroy_image( & img_shifted )
    destroy_image( & img_realigned )

    fmt.println("\n...end of test of Subpixel Image Alignment of a Gaussian image.\n")
}


