#![feature(type_alias_impl_trait, portable_simd, stdsimd, let_else, new_uninit)]
#![allow(non_snake_case)]

/// T should be a basic type (i.e valid when casted from any data)
pub unsafe fn from_bytes<T>(slice: &[u8]) -> &[T] {
    std::slice::from_raw_parts(slice.as_ptr() as *const T, slice.len() / std::mem::size_of::<T>())
}

use {ui::{Error, Result}, fehler::throws};
type Bounds = vector::MinMax<vec3>;
use owning_ref::OwningRef;

type Array<T=f32> = OwningRef<Box<memmap::Mmap>, [T]>;

#[throws] fn map<T>(field: &str, name: &str) -> Array<T> {
    OwningRef::new(Box::new(unsafe{memmap::Mmap::map(&std::fs::File::open(format!("cache/{name}.{field}"))?)}?)).map(|data| unsafe{from_bytes(&*data)})
}

use image::{Image, bgra8};

struct View {
    x: Array,
    y: Array,
    z: Array,

    image: [Image<Array<u8>>; 3],

    start_position: vec2,
    position: vec2,
    vertical_scroll: f32,
}

use {vector::{xy, size, vec2, xyz, vec3}, ui::{Widget, RenderContext as Target, widget::{EventContext, Event}}, vector::num::{Zero, IsZero}};

fn xy(xyz{x,y,..}: vec3) -> vec2 { xy{x, y} }
use std::f32::consts::PI;
fn rotate(xy{x,y,..}: vec2, angle: f32) -> vec2 { xy{x: f32::cos(angle)*x+f32::sin(angle)*y, y: f32::cos(angle)*y-f32::sin(angle)*x} }

impl Widget for View {
#[throws] fn event(&mut self, _size: size, _event_context: &EventContext, event: &Event) -> bool {
    match event {
        &Event::Motion{position, ..} => {
            if self.start_position.is_zero() { self.start_position = position; }
            self.position = position;
            true
        },
        &Event::Scroll(vertical_scroll) => {
            self.vertical_scroll = f32::clamp(1., self.vertical_scroll + vertical_scroll, 64.);
            true
        },
        _ => false,
    }
}

fn paint(&mut self, target: &mut Target, _size: size) -> Result {
    /*for y in 0..target.size.y {
        for x in 0..target.size.x {
            let [b,g,r] = &self.image;
            let i = (y*b.stride+x) as usize;
            let [b,g,r] = [b.data[i],g.data[i],r.data[i]];
            target.data[(y*target.stride+x) as usize] = image::bgra{b, g, r, a: 0xFF};
        }
    }*/
    target.fill(bgra8{b:0,g:0,r:0,a:0xFF});
    let size = vec2::from(target.size);
    let transform = |p:vec3| {
        let yaw = (self.position.x-self.start_position.x)/size.x*2.*PI;
        let pitch = f32::clamp(0., (1.-self.position.y/size.y)*PI/2., PI/2.);
        let scale = size.x.min(size.y)*self.vertical_scroll;
        let z = p.z;
        let p = scale*rotate(xy(p), yaw);
        let p = size/2.+xy{x: p.x, y: f32::cos(pitch)*p.y+f32::sin(pitch)*scale*z};
        let p = xy{x: p.x, y: (size.y-1.) - p.y};
        p
    };
    let O = transform(vec3{x:0., y:0., z:0.});
    use std::simd::{Simd, f32x16, u32x16};
    let [[e0_x,e0_y],[e1_x,e1_y],[e2_x,e2_y]] = [vec3{x:1., y:0., z:0.}, vec3{x:0., y:1., z:0.}, vec3{x:0., y:0., z:1.}].map(|e| { let e = transform(e)-O; [e.x,e.y].map(Simd::splat) });
    let O = O.map(|&x| Simd::splat(x));
    let [x,y,z] = [&self.x, &self.y, &self.z].map(|array| unsafe { let ([], array, _) = array.align_to::<f32x16>() else { unreachable!() }; array});
    let size = target.size.map(|&x| Simd::splat(x));
    use rayon::prelude::*;
    (x, y, z).into_par_iter().for_each(|(x, y, z)| { unsafe {
    	let [b,g,r] = &self.image;
        let u : u32x16 = _mm512_cvttps_epu32(((x+Simd::splat(1./2.))*Simd::splat(b.size.x as f32)).into()).into();
        let v : u32x16 = _mm512_cvttps_epu32(((y+Simd::splat(1./2.))*Simd::splat(b.size.y as f32)).into()).into();
        let indices = v * Simd::splat(b.stride) + u;
        let b : u32x16 = _mm512_i32gather_epi32(indices.into(), (b.as_ptr() as *const u8).offset(-0), 1).into();
        let g : u32x16 = _mm512_i32gather_epi32(indices.into(), (g.as_ptr() as *const u8).offset(-1), 1).into();
        let r : u32x16 = _mm512_i32gather_epi32(indices.into(), (r.as_ptr() as *const u8).offset(-2), 1).into();
        let bgra = Simd::splat(0xFF_00_00_00) | r & Simd::splat(0x00_FF_00_00) | g & Simd::splat(0x00_00_FF_00) | b & Simd::splat(0x00_00_00_FF);
        let p_y : u32x16 = _mm512_cvttps_epu32((x * e0_y + y * e1_y + z * e2_y + O.y).into()).into();
        let p_x : u32x16 = _mm512_cvttps_epu32((x * e0_x + y * e1_x + z * e2_x + O.x).into()).into();
        let indices = p_y * size.x + p_x;
        //unsafe{bgra.scatter_select_unchecked(target, indices.lanes_lt(Simd::splat(target.len())), indices)};
        use std::arch::x86_64::*;
        _mm512_mask_i32scatter_epi32(target.as_ptr() as *mut u8, _mm512_cmplt_epu32_mask(p_x.into(), size.x.into()) & _mm512_cmplt_epu32_mask(p_y.into(), size.y.into()), indices.into(), bgra.into(), 4);
    }});
    Ok(())
}
}

#[throws] fn size(name: &str) -> size {
    let tiff = unsafe{memmap::Mmap::map(&std::fs::File::open(format!("{name}.tif"))?)?};
    let mut tiff = tiff::decoder::Decoder::new(std::io::Cursor::new(&*tiff))?;
    let (size_x, size_y) = tiff.dimensions()?;
    size{x: size_x, y: size_y}
}

#[throws] fn image_bounds(name: &str) -> Bounds {
    let size = size(name)?;
    let tiff = unsafe{memmap::Mmap::map(&std::fs::File::open(format!("{name}.tif"))?)?};
    let mut tiff = tiff::decoder::Decoder::new(std::io::Cursor::new(&*tiff))?;
    let [_, _, _, E, N, _] = tiff.get_tag_f64_vec(tiff::tags::Tag::ModelTiepointTag)?[..] else { panic!() };
    let [scale_E, scale_N, _] = tiff.get_tag_f64_vec(tiff::tags::Tag::ModelPixelScaleTag)?[..] else { panic!() };
    let min = vec3{x: E as f32, y: (N-scale_N*size.y as f64) as f32, z: 0.};
    let max = vec3{x: (E+scale_E*size.x as f64) as f32, y: N as f32, z: f32::MAX};
    vector::MinMax{min, max}
}

#[throws] fn points_bounds(name: &str) -> Bounds {
    let reader = las::Reader::from_path(format!("{name}.las"))?;
    let las::Bounds{min, max} = las::Read::header(&reader).bounds();
    vector::MinMax{min: vec3{x: min.x as f32, y: min.y as f32, z: min.z as f32}, max: vec3{x: max.x as f32, y: max.y as f32, z: max.z as f32}}
}

impl View {
#[throws] fn new(image: &str, points: &str) -> Self {
    /*let vector::MinMax{min, max} = image_bounds(image).clip(points_bounds(points))
    println!("{min:?} {max:?}");

    let center = (1./2.)*(min+max);
    let extent = max-min;
    let extent = extent.x.min(extent.y);

    let mut X = Vec::with_capacity(las::Read::header(&reader).number_of_points() as usize);
    let mut Y = Vec::with_capacity(las::Read::header(&reader).number_of_points() as usize);
    let mut Z = Vec::with_capacity(las::Read::header(&reader).number_of_points() as usize);
    for point in las::Read::points(&mut reader) {
        let las::Point{x: E,y: N, z, ..} = point.unwrap();
        let x = (E as f32-center.x)/extent;
        let y = (N as f32-center.y)/extent;
        let z = (z as f32-center.z)/extent;
        if x >= -1./2. && x <= 1./2. && y >= -1./2. && y <= 1./2. {
            X.push(x);
            Y.push(y);
            Z.push(z);
        }
    }
    #[throws] fn write<T:bytemuck::Pod>(name: &str, field: &str, points: &[T]) { std::fs::write(format!("cache/{name}.{field}"), bytemuck::cast_slice(points))? }
    write(points, "x", &X)?;
    write(points, "y", &Y)?;
    write(points, "z", &Z)?;*/

    let size = size(image)?;
    let image_bounds = image_bounds(image)?;
    let vector::MinMax{min, max} = {let mut x = image_bounds.clip(points_bounds(points)?); x.translate(-image_bounds.min); x};
    let scale = size.x as f32 / (image_bounds.max - image_bounds.min).x;
    assert!(scale == 20.);
    let min = scale*min;
    assert_eq!(min.x, f32::trunc(min.x));
    assert_eq!(min.y, f32::trunc(min.y));
    let max = scale*max;

    /*let tiff = unsafe{memmap::Mmap::map(&std::fs::File::open(format!("{image}.tif"))?)?};
    let mut tiff = tiff::decoder::Decoder::new(std::io::Cursor::new(&*tiff))?.with_limits(tiff::decoder::Limits::unlimited());
    let tiff::decoder::DecodingResult::U8(mut rgba) = tiff.read_image()? else { panic!() };
    println!("OK");
    for i in 0..rgba.len()/4 {
        let i = i*4;
        rgba[i+0] = rgba[i+2]; // B
        rgba[i+1] = rgba[i+1]; // G
        rgba[i+2] = rgba[i+0]; // R
        rgba[i+3] = rgba[i+3]; // A
    }
    let bgra = rgba;*/    
     
    let stride = size.x;
    let plane_stride = (size.y*stride) as usize;
  
    /*let flip = {
		let image = map("bil", image).unwrap();
		let mut flip = unsafe{Box::new_uninit_slice(image.len()).assume_init()};
		for i in 0..3 {
			for y in 0..size.y {
				for x in 0..size.x {
					flip[i*plane_stride+(y*stride+x) as usize] = image[i*plane_stride+((size.y-1-y)*stride+x) as usize];
				}
			}
		}
		flip
	};
	std::fs::write(format!("cache/{image}.rgb"), flip)?;*/
    
    let size = xy{x: max.x as u32 - min.x as u32, y: max.y as u32 - min.y as u32};
    let [r,g,b] = iter::eval(|plane| {
        let image = map("rgb", image).unwrap();
        Image::strided(size, image.map(|data| &data[plane*plane_stride + ((min.y as u32)*stride+(min.x as u32)) as usize..][..(size.y*stride) as usize]), stride)
    });

    let map = |field| map(field, points).unwrap();
    Self{
        x: map("x"), y: map("y"), z: map("z"),
        image: [b,g,r],
        start_position: vec2::ZERO, position: vec2::ZERO, vertical_scroll: 1.
    }
}
}
#[throws] fn main() { ui::run(Box::new(View::new("2408", "2684_1248")?))? }
