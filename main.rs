#![feature(type_alias_impl_trait, portable_simd, stdsimd, let_else)]
#![allow(non_snake_case)]

/// T should be a basic type (i.e valid when casted from any data)
pub unsafe fn from_bytes<T>(slice: &[u8]) -> &[T] {
    std::slice::from_raw_parts(slice.as_ptr() as *const T, slice.len() / std::mem::size_of::<T>())
}

use {ui::{Error, Result}, fehler::throws};
use owning_ref::OwningRef;

vector::vector!(3 xyz T T T, x y z, X Y Z);
#[allow(non_camel_case_types)] pub type vec3 = xyz<f32>;

type Array<T=f32> = OwningRef<Box<memmap::Mmap>, [T]>;

struct LAS {
    x: Array,
    y: Array,
    z: Array,

    start_position: vec2,
    position: vec2,
    vertical_scroll: f32,
}

use {std::iter::zip, ::xy::{xy, size, vec2}, ui::{Widget, RenderContext as Image, widget::{EventContext, Event}}, vector::num::{Zero, IsZero}};

fn xy(xyz{x,y,..}: vec3) -> vec2 { xy{x, y} }
use std::f32::consts::PI;
fn rotate(xy{x,y,..}: vec2, angle: f32) -> vec2 { xy{x: f32::cos(angle)*x+f32::sin(angle)*y, y: f32::cos(angle)*y-f32::sin(angle)*x} }

impl Widget for LAS {
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

fn paint(&mut self, target: &mut Image, _size: size) -> Result {
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

impl LAS {
    fn new(name: &str) -> Self {
        #[throws] fn map<T>(field: &str, name: &str) -> Array<T> {
            OwningRef::new(Box::new(unsafe{memmap::Mmap::map(&std::fs::File::open(format!("{name}.{field}"))?)}?)).map(|data| unsafe{from_bytes(&*data)})
        }
        let map = |field| map(field, name).unwrap();
        Self{x: map("x"), y: map("y"), z: map("z"), start_position: vec2::ZERO, position: vec2::ZERO, vertical_scroll: 1.}
    }
}
#[throws] fn main() {
    /*let points = Box::new(unsafe{memmap::Mmap::map(&std::fs::File::open("2684_1248.points")?)}?);
    #[derive(Clone, Copy)] struct Point { position: vec3, #[allow(dead_code)] intensity: u16 }
    let ref points = OwningRef::new(points).map(|points| unsafe{from_bytes(&*points)});
    //println!("{:?}", vector::minmax(points.iter().map(|Point{intensity,..}| *intensity)).unwrap());
    let vector::MinMax{min,max} = vector::minmax(points.iter().map(|Point{position,..}| *position)).unwrap();
    let center = (1./2.)*(min+max);
    let extent = max-min;
    let extent = extent.x.max(extent.y.max(extent.z));

    pub fn to_byte_slice<T>(slice: &[T]) -> &[u8] { unsafe{std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * std::mem::size_of::<T>())} }
    #[throws] fn write<T>(points: &[Point], field: &str, f: impl Fn(&Point)->T) { std::fs::write(format!("2684_1248.{field}"), to_byte_slice(&points.iter().map(f).collect::<Box<_>>()))? }
    write(points, "x", |&Point{position,..}| (position.x-center.x)/extent)?;
    write(points, "y", |&Point{position,..}| (position.y-center.y)/extent)?;
    write(points, "z", |&Point{position,..}| (position.z-center.z)/extent)?;*/

    ui::run(Box::new(LAS::new("2684_1248")))?
}
