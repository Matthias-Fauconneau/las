pub use ::image::{Image, bgra, bgra8, sRGB};

use std::lazy::SyncLazy;
#[allow(non_upper_case_globals)] const sRGB_reverse : SyncLazy<[f32; 256]> = SyncLazy::new(|| iter::eval(|i| {
    let linear = i as f32 / 255.;
    if linear > 0.04045 { f32::powf((linear+0.055)/1.055, 2.4) } else { linear / 12.92 }
}));

pub fn sRGB_to_linear([b,g,r]: &[Image<&[u8]>; 3]) -> Image<Box<[u8]>> {
    let size = b.size;
    let /*mut*/ target = Image::uninitialized(size);
    {
        let stride = b.stride;
        assert!(stride%16 == 0 && size.x%16==0 && target.stride == size.x);
        use std::arch::x86_64::*;
        use rayon::prelude::*;
        (0..size.y).into_par_iter().for_each(|y| unsafe {
            let lookup = sRGB_reverse.as_ptr();
            let target : *const u8 = target.as_ptr();
            let [b,g,r] = [b,g,r].map(|plane| plane.data.as_ptr());
            let target_row = target.offset((y*size.x) as isize);
            let [b,g,r] = [b,g,r].map(|plane| plane.offset((y*stride) as isize));
            let [B,G,R] = [_mm512_set1_ps(255.*0.0722),_mm512_set1_ps(255.*0.7152),_mm512_set1_ps(255.*0.2126)];
            for x in (0..size.x).step_by(16) {
                _mm_store_si128(target_row.offset(x as isize) as *mut __m128i,
                _mm512_cvtepi32_epi8(
                    _mm512_cvtps_epu32(
                    _mm512_add_ps(
                    _mm512_mul_ps(B,
                        _mm512_i32gather_ps(
                            _mm512_cvtepu8_epi32(
                                _mm_load_si128(b.offset(x as isize) as *const __m128i)), lookup as *const u8, 4)),
                                _mm512_add_ps(
                                    _mm512_mul_ps(G,
                                        _mm512_i32gather_ps(
                                            _mm512_cvtepu8_epi32(
                                                _mm_load_si128(g.offset(x as isize) as *const __m128i)), lookup as *const u8, 4)),
                                    _mm512_mul_ps(R,
                                        _mm512_i32gather_ps(
                                            _mm512_cvtepu8_epi32(
                                                _mm_load_si128(r.offset(x as isize) as *const __m128i)), lookup as *const u8, 4)))))));
            }
        });
    }
    target
}

use vector::{xy, vec2};

pub fn blit(target:&mut Image<&mut[bgra8]>, source: &Image<&[u8]>) {
    let size = target.size;
    for y in 0..size.y { for x in 0..size.x {
        let v = source[(y*source.size.y/size.y*source.stride+x*source.size.x/size.x) as usize];
        let v = image::sRGB(&(v as f32/255.));
        target[xy{x, y: size.y-1-y}] = bgra{b:v, g:v, r:v, a: 0xFF};
    }}
}
pub fn blit_sRGB(target:&mut Image<&mut[bgra8]>, [b,g,r]: &[Image<&[u8]>; 3]) {
    let size = target.size;
    for y in 0..size.y { for x in 0..size.x {
        let [b,g,r] = {
            let xy{x,y} = ((size/2*b.size).signed()+(xy{x,y}.signed()-size.signed()/2)*b.size.signed()/5).unsigned();
            let i = (y/size.y*b.stride+x/size.x) as usize;
            [b.data[i],g.data[i],r.data[i]]
        };
        target[xy{x, y: size.y-1-y}] = bgra{b, g, r, a: 0xFF};
    }}
}

pub fn affine_blit(target:&mut Image<&mut[bgra8]>, source: Image<&[u8]>, A: mat3) {
    let size = target.size;
    for y in 0..size.y { for x in 0..size.x {
        let v = {
            let p = {let size=vec2::from(size); size/2.+(vec2::from(xy{x,y})-size/2.)/5.};
            let p = apply(A, p).map(|&c| c as u32);
            if p.x >= source.size.x || p.y >= source.size.y { continue; }
            sRGB(&(source[p] as f32/255.))
        };
        target[xy{x, y: size.y-1-y}] = bgra{b: v, g: v, r: v, a: 0xFF};
    }}
}

use crate::matrix::{mat3, apply};

pub fn affine_blit_sRGB(target:&mut Image<&mut[bgra8]>, [b,g,r]: &[Image<&[u8]>; 3], A: mat3) {
    let size = target.size;
    for y in 0..size.y { for x in 0..size.x {
        let [b,g,r] = {
            let p = {let size=vec2::from(size); size/2.+(vec2::from(xy{x,y})-size/2.)/5.};
            let xy{x,y} = apply(A, p).map(|&c| c as u32);
            if x >= b.size.x || y >= b.size.y { continue; }
            let i = (y*b.stride+x) as usize;
            [b.data[i],g.data[i],r.data[i]]
        };
        target[xy{x, y: size.y-1-y}] = bgra{b, g, r, a: 0xFF};
    }}
}
