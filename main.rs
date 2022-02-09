#![feature(type_alias_impl_trait)]

/// T should be a basic type (i.e valid when casted from any data)
pub unsafe fn from_bytes<T>(slice: &[u8]) -> &[T] {
    std::slice::from_raw_parts(slice.as_ptr() as *const T, slice.len() / std::mem::size_of::<T>())
}

use {ui::Error, fehler::throws};
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

#[throws] fn paint(&mut self, target: &mut Image, _size: size) {
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
    #[allow(non_snake_case)] let O = transform(vec3{x:0., y:0., z:0.});
    let e = [vec3{x:1., y:0., z:0.}, vec3{x:0., y:1., z:0.}, vec3{x:0., y:0., z:1.}].map(|e| transform(e)-O);
    for ((&x, &y), &z) in zip(zip(&*self.x, &*self.y), &*self.z) { //.step_by(4) {
        let p = x * e[0] + y * e[1] + z * e[2] + O;
        fn from(f: xy<f32>) -> xy<u32> { xy{x: f.x as u32, y: f.y as u32} }
        let p = from(p);
        if p.x < target.size.x && p.y < target.size.y {
            target[p] = bgra8{b: 0xFF, g: 0xFF, r: 0xFF, a: 0xFF};
            /*let v = intensity as f32/u16::MAX as f32;
            //let v = (p.z-min.z)/(max.z-min.z);
            use image::{rgb::rgb, sRGB};
            target[p] = bgra8::from(rgb::from([sRGB(&v); 3]));*/
        }
    }
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
