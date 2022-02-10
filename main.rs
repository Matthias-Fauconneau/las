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

type Mesh = (Box<[vec3]>, Box<[[u16; 3]]>);
struct View {
    x: Array,
    y: Array,
    z: Array,

    image: [Image<Array<u8>>; 3],
    meshes: Box<[Mesh]>,

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

fn paint(&mut self, target: &mut Target, size: size) -> Result {
    /*for y in 0..target.size.y {
        for x in 0..target.size.x {
            let [b,g,r] = &self.image;
            let i = (y*b.stride+x) as usize;
            let [b,g,r] = [b.data[i],g.data[i],r.data[i]];
            target.data[(y*target.stride+x) as usize] = image::bgra{b, g, r, a: 0xFF};
        }
    }*/
    target.fill(bgra8{b:0,g:0,r:0,a:0xFF});
    let size = vec2::from(size);
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
    let e = [vec3{x:1., y:0., z:0.}, vec3{x:0., y:1., z:0.}, vec3{x:0., y:0., z:1.}].map(|e| transform(e)-O);
    /*use std::simd::{Simd, f32x16, u32x16};
    let [x,y,z] = [&self.x, &self.y, &self.z].map(|array| unsafe { let ([], array, _) = array.align_to::<f32x16>() else { unreachable!() }; array});
    let size = size.map(|&x| Simd::splat(x));
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
        let e = e.map(|e| e.map(Simd::splat));
        let O = O.map(|&x| Simd::splat(x));
        let p = x * e[0] + y * e[1] + z * e[2] + O;
        let p : xy<u32x16> = p.map(|p| _mm512_cvttps_epu32(p.into()).into());
        //unsafe{bgra.scatter_select_unchecked(target, indices.lanes_lt(Simd::splat(target.len())), p.y * size.x + p.x)};
        use std::arch::x86_64::*;
        _mm512_mask_i32scatter_epi32(target.as_ptr() as *mut u8, _mm512_cmplt_epu32_mask(p.x.into(), size.x.into()) & _mm512_cmplt_epu32_mask(p.y.into(), size.y.into()), (p.y * size.x + p.x).into(), bgra.into(), 4);
    }});*/

    for (vertices, triangles) in &*self.meshes {
        for triangle in triangles.iter() {
            let vertices = triangle.map(|i| { let xyz{x,y,z} = vertices[i as usize]; x * e[0] + y * e[1] + z * e[2] + O });
            let vector::MinMax{min,max} = vector::minmax(vertices.into_iter()).unwrap();
            let min = xy{x: f32::max(0., min.x), y: f32::max(0., min.y)};
            let max = xy{x: f32::min(max.x+1., size.x), y: f32::min(max.y+1., size.y)};
            for y in min.y as u32 .. max.y as u32 {
                for x in min.x as u32 .. max.x as u32 {
                    let p = xy{x,y};
                    if {
                        let p = p.map(|&c| c as f32);
                        let [v0,v1,v2] = vertices;
                        use vector::cross;
                        cross(v1-v0, p-v0) > 0. && cross(v2-v1, p-v1) > 0. && cross(v0-v2, p-v2) > 0.
                    } {
                        target[p] = bgra8{b:0xFF,g:0xFF,r:0xFF,a:0xFF};
                    }
                }
            }
        }
    }

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
fn new(image: &str, points: &str) -> Result<Self> {
    let image_bounds = image_bounds(image)?;
    let vector::MinMax{min, max} = image_bounds.clip(points_bounds(points)?);
    //println!("{min:?} {max:?}");

    let center = (1./2.)*(min+max);
    let extent = max-min;
    let extent = extent.x.min(extent.y);

    /*let mut X = Vec::with_capacity(las::Read::header(&reader).number_of_points() as usize);
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

    let [r,g,b] = {
        let size = size(image)?;
        let scale = size.x as f32 / (image_bounds.max - image_bounds.min).x;
        assert!(scale == 20.);
        let min = scale*(min-image_bounds.min);
        assert_eq!(min.x, f32::trunc(min.x));
        assert_eq!(min.y, f32::trunc(min.y));
        let max = scale*(max-image_bounds.min);

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
        iter::eval(|plane| {
            let image = map("rgb", image).unwrap();
            Image::strided(size, image.map(|data| &data[plane*plane_stride + ((min.y as u32)*stride+(min.x as u32)) as usize..][..(size.y*stride) as usize]), stride)
        })
    };

    /*let meshes = dxf::Drawing::load_file("cache/SWISSBUILDINGS3D_2_0_CHLV95LN02_1091-23.dxf")?.entities();
    std::fs::write("cache/meshes", bincode::serialize(&meshes.collect::<Vec<_>>())?)?;*/
    let meshes : Vec<dxf::entities::Entity> = bincode::deserialize(&std::fs::read("cache/mesh")?)?;
    let meshes = meshes.into_iter().filter_map(|mesh| if let dxf::entities::EntityType::Polyline(mesh) = mesh.specific {
        let mut vertices_and_indices = mesh.__vertices_and_handles.iter().peekable();
        let mut vertices = Vec::new();
        while let Some((vertex,_)) = vertices_and_indices.next_if(|(v,_)|
            [v.polyface_mesh_vertex_index1, v.polyface_mesh_vertex_index2, v.polyface_mesh_vertex_index3, v.polyface_mesh_vertex_index4] == [0,0,0,0]
        ) {
            let dxf::Point{x,y,z} = vertex.location;
            vertices.push((vec3{x: x as f32,y: y as f32, z: z as f32}-center)/extent)
        }
        let triangles = vertices_and_indices.map(|(v,_)| {
            let face = [v.polyface_mesh_vertex_index1, v.polyface_mesh_vertex_index2, v.polyface_mesh_vertex_index3, v.polyface_mesh_vertex_index4];
            assert!(face[3] == 0);
            let face : [i32; 3] = face[..3].try_into().unwrap();
            let face : [u16; 3] = face.map(|i| (i-1).try_into().unwrap());
            for i in face { assert!((i as usize) < vertices.len(), "{face:?} {}", vertices.len()); }
            face
        }).collect::<Box<_>>();
        Some((vertices.into_boxed_slice(),triangles))
    } else { None }).collect::<Box<_>>();
    //println!("{} {}", meshes.iter().map(|(v,t)| v.len()*4+t.len()*3*2).sum::<usize>(), meshes.iter().map(|(_,t)| t.len()*3*4).sum::<usize>());

    let map = |field| map(field, points).unwrap();
    Ok(Self{
        x: map("x"), y: map("y"), z: map("z"),
        image: [b,g,r],
        meshes,
        start_position: vec2::ZERO, position: vec2::ZERO, vertical_scroll: 1.
    })
}
}
#[throws] fn main() { ui::run(Box::new(View::new("2408", "2684_1248")?))? }
