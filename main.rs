#![feature(type_alias_impl_trait, portable_simd, stdsimd, let_else, vec_into_raw_parts)]
#![allow(non_snake_case)]

/// T should be a basic type (i.e valid when casted from any data)
pub unsafe fn from_bytes<T>(slice: &[u8]) -> &[T] {
    std::slice::from_raw_parts(slice.as_ptr() as *const T, slice.len() / std::mem::size_of::<T>())
}

use {ui::{Error, Result}, fehler::throws};
type Bounds = vector::MinMax<vec3>;
use owning_ref::OwningRef;

type Array<T=f32> = OwningRef<Box<memmap::Mmap>, [T]>;

mod rgba { vector::vector!(4 rgba T T T T, r g b a, Red Green Blue Alpha); }
#[allow(non_camel_case_types)] pub type rgba8 = rgba::rgba<u8>;
use image::Image;

struct View {
    x: Array,
    y: Array,
    z: Array,

    image_bounds: Bounds,
    image: Image<Array<rgba8>>,

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
    use image::bgra8;
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
    let [O_x,O_y] = [O.x, O.y].map(Simd::splat);
    let [x,y,z] = [&self.x, &self.y, &self.z].map(|array| unsafe { let ([], array, _) = array.align_to::<f32x16>() else { unreachable!() }; array});
    let stride = Simd::splat(target.stride);
    let white : u32x16 = Simd::splat(bytemuck::cast::<_,u32>(bgra8{b: 0xFF, g: 0xFF, r: 0xFF, a: 0xFF}));
    use rayon::prelude::*;
    (x, y, z).into_par_iter().for_each(|(x, y, z)| { unsafe {
        //let p_y : u32x16 = (x * e0_y + y * e1_y + z * e2_y + O_y).cast::<u32>();
        //let p_x : u32x16 = (x * e0_x + y * e1_x + z * e2_x + O_x).cast::<u32>();
        let p_y : u32x16 = _mm512_cvttps_epu32((x * e0_y + y * e1_y + z * e2_y + O_y).into()).into();
        let p_x : u32x16 = _mm512_cvttps_epu32((x * e0_x + y * e1_x + z * e2_x + O_x).into()).into();
        let indices = p_y * stride + p_x;
        //unsafe{white.scatter_select_unchecked(target, indices.lanes_lt(Simd::splat(target.len())), indices)};
        use std::arch::x86_64::*;
        _mm512_mask_i32scatter_epi32(target.as_ptr() as *mut u8, _mm512_cmplt_epu32_mask(indices.into(), Simd::splat(target.len()).into()), indices.into(), white.into(), 4);
    }});
    Ok(())
}
}

#[throws] fn size(name: &str) -> size {
    let tiff = unsafe{memmap::Mmap::map(&std::fs::File::open("2408.tif")?)?};
    let mut tiff = tiff::decoder::Decoder::new(std::io::Cursor::new(&*tiff))?;
    let (size_x, size_y) = tiff.dimensions()?;
    size{x: size_x, y: size_y}
}

#[throws] fn bounds(name: &str) -> Bounds {
    let size = size(name)?;
    let tiff = unsafe{memmap::Mmap::map(&std::fs::File::open("2408.tif")?)?};
    let mut tiff = tiff::decoder::Decoder::new(std::io::Cursor::new(&*tiff))?;
    let [0., 0., 0., E, N, 0.] = tiff.get_tag_f64_vec(tiff::tags::Tag::ModelTiepointTag)?[..] else { panic!() };
    let [scale_E, scale_N, 0.] = tiff.get_tag_f64_vec(tiff::tags::Tag::ModelPixelScaleTag)?[..] else { panic!() };
    let min = vec3{x: E as f32, y: (N-scale_N*size.y as f64) as f32, z: 0.};
    let max = vec3{x: (E+scale_E*size.x as f64) as f32, y: N as f32, z: f32::MAX};
    vector::MinMax{min, max}
}

impl View {
    #[throws] fn new(raster: &str, points: &str) -> Self {
        #[throws] fn map<T>(field: &str, name: &str) -> Array<T> {
            OwningRef::new(Box::new(unsafe{memmap::Mmap::map(&std::fs::File::open(format!("{name}.{field}"))?)}?)).map(|data| unsafe{from_bytes(&*data)})
        }
        let size = size(raster)?;
        let data = map("rgba", raster)?;
        let image = Image::new(size, data);
        let map = |field| map(field, points).unwrap();
        Self{
            x: map("x"), y: map("y"), z: map("z"),
            image_bounds: bounds(raster)?,
            image,
            start_position: vec2::ZERO, position: vec2::ZERO, vertical_scroll: 1.
        }
    }
}
#[throws] fn main() {
    /*let tiff = unsafe{memmap::Mmap::map(&std::fs::File::open("2408.tif")?)?};
    let mut tiff = tiff::decoder::Decoder::new(std::io::Cursor::new(&*tiff))?.with_limits(tiff::decoder::Limits::unlimited());
    let tiff::decoder::DecodingResult::U8(rgba) = tiff.read_image()? else { panic!() };
    println!("{}", rgba.len());
    std::fs::write("2408.rgba", &rgba)?;
    println!("2408.rgba");*/

    /*let mut reader = las::Reader::from_path("2684_1248.las")?;
    let points_bounds = {
        let las::Bounds{min, max} = las::Read::header(&reader).bounds();
        vector::MinMax{min: vec3{x: min.x as f32, y: min.y as f32, z: min.z as f32}, max: vec3{x: max.x as f32, y: max.y as f32, z: max.z as f32}}
    };
    let raster_bounds = bounds("2408.tif");
    let min = vector::component_wise_max(points_bounds.min, raster_bounds.min);
    let max = vector::component_wise_min(points_bounds.max, raster_bounds.max);
    //println!("{raster_bounds:?}\n{points_bounds:?}\n{min:?} {max:?}");

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
    #[throws] fn write<T:bytemuck::Pod>(field: &str, points: &[T]) { std::fs::write(format!("2684_1248.{field}"), bytemuck::cast_slice(points))? }
    write("x", &X)?;
    write("y", &Y)?;
    write("z", &Z)?;*/

    ui::run(Box::new(View::new("2408", "2684_1248")?))?
}
