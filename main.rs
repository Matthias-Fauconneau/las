#![feature(type_alias_impl_trait)]

/// T should be a basic type (i.e valid when casted from any data)
pub unsafe fn from_bytes<T>(slice: &[u8]) -> &[T] {
    std::slice::from_raw_parts(slice.as_ptr() as *const T, slice.len() / std::mem::size_of::<T>())
}

use {ui::{Error, Result}, fehler::throws};
use owning_ref::OwningRef;

vector::vector!(3 xyz T T T, x y z, X Y Z);
#[allow(non_camel_case_types)] pub type vec3 = xyz<f32>;

#[derive(Clone, Copy)] struct Point { position: vec3, #[allow(dead_code)] intensity: u16 }

struct LAS {
    points: OwningRef<Box<memmap::Mmap>, [Point]>,
    min: vec3,
    max: vec3,
    start_position: vec2,
    position: vec2,
    vertical_scroll: f32,
}

use {::xy::{xy, size, vec2}, ui::{Widget, RenderContext as Image, widget::{EventContext, Event}}, vector::num::{Zero, IsZero}};
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
    use image::{bgra8, rgb::rgb, sRGB};
    target.fill(bgra8{b:0,g:0,r:0,a:0xFF});
    let &mut Self{min,max,..} = self;
    for &Point{position,intensity:_} in self.points.iter().step_by(64) {
        fn xy(xyz{x,y,..}: vec3) -> vec2 { xy{x, y} }
        let v = (position.z-min.z)/(max.z-min.z);
        let center = (1./2.)*(min+max);
        fn from(f: xy<f32>) -> xy<u32> { xy{x: f.x as u32, y: f.y as u32} }
        //let v = intensity as f32/u16::MAX as f32;
        fn rotate(xy{x,y,..}: vec2, angle: f32) -> vec2 { xy{x: f32::cos(angle)*x+f32::sin(angle)*y, y: f32::cos(angle)*y-f32::sin(angle)*x} }
        let size = std::cmp::min(target.size.x, target.size.y) as f32;
        use std::f32::consts::PI;
        let yaw = (self.position.x-self.start_position.x)/(target.size.x as f32)*2.*PI;
        let extent = max-min;
        let scale = size*self.vertical_scroll/extent.x.max(extent.y.max(extent.z));
        let p = scale*rotate(xy(position)-xy(center), yaw);
        let pitch = f32::clamp(0., (1.-self.position.y/target.size.y as f32)*PI/2., PI/2.);
        let p = vec2::from(target.size)/2.+xy{x: p.x, y: f32::cos(pitch)*p.y+f32::sin(pitch)*scale*(position.z-center.z)};
        let p = from(p);
        let p = xy{x:p.x, y:target.size.y-1-p.y};
        if p.x < target.size.x && p.y < target.size.y { target[p] = bgra8::from(rgb::from([sRGB(&v); 3])); }
    }
}
}

fn main() -> Result {
    let points = Box::new(unsafe{memmap::Mmap::map(&std::fs::File::open("2684_1248.points")?)}?);
    let points = OwningRef::new(points).map(|points| unsafe{from_bytes(&*points)});
    //println!("{:?}", vector::minmax(points.iter().map(|Point{intensity,..}| *intensity)).unwrap());
    let vector::MinMax{min,max} = vector::minmax(points.iter().map(|Point{position,..}| *position)).unwrap();
    ui::run(Box::new(LAS{points, min, max, start_position: vec2::ZERO, position: vec2::ZERO, vertical_scroll: 1.}))
}
