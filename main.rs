#![feature(type_alias_impl_trait, portable_simd, stdsimd, let_else, new_uninit, array_methods, array_windows, once_cell)]
#![allow(non_snake_case, non_upper_case_globals, dead_code)]
use {ui::{Error, Result}, fehler::throws};

use owning_ref::OwningRef;
type Array<T=f32> = OwningRef<Box<memmap::Mmap>, [T]>;
#[throws] fn map<T:bytemuck::Pod>(path: &str) -> Array<T> {
    OwningRef::new(Box::new(unsafe{memmap::Mmap::map(&std::fs::File::open(path)?)}?)).map(|data| bytemuck::cast_slice(&*data))
}

mod matrix; use matrix::*;
mod image; use crate::image::*;
mod layout; use layout::*;
use vector::{xy, uint2, size, int2, vec2};

fn draw_cross(target: &mut Image<&mut [image::bgra8]>, xy{x,y}: uint2) {
    const R: u32 = 16;
    for x in i32::max(0, x as i32-R as i32) as u32..u32::min(x+R+1, target.size.x) { let size = target.size; target[xy{x,y: size.y-1-y}] = bgra8{b:0,g:0,r:0xFF,a:0xFF}; }
    for y in i32::max(0, y as i32-R as i32) as u32..u32::min(y+R+1, target.size.y) { let size = target.size; target[xy{x,y: size.y-1-y}] = bgra8{b:0,g:0,r:0xFF,a:0xFF}; }
}

fn correspondences() -> Box<[[uint2; 5]]> {
    (0..1)
    .map(|point| -> [uint2; 5] {
        (&*std::str::from_utf8(&std::fs::read(format!("../data/{point}")).unwrap()).unwrap().lines().map(|line| {
            let [x, y] : [u32; 2] = (&*line.split(' ').map(|c| str::parse(c).unwrap()).collect::<Box<_>>()).try_into().unwrap();
            uint2{x,y}
        }).collect::<Box<_>>()).try_into().unwrap()
    }).collect::<Box<_>>()
}

fn affine_transform(index: usize) -> mat3 {
    let correspondences : [[uint2; 5]; 3] = (&*correspondences()).try_into().unwrap();
    let X = correspondences.map(|p| p[4]);
    let Y = correspondences.map(|p| p[index]);
    let X = [
        X.map(|p| p.x as f32),
        X.map(|p| p.y as f32),
        X.map(|_| 1.)
    ];
    let Y = [
        Y.map(|p| p.x as f32),
        Y.map(|p| p.y as f32),
        Y.map(|_| 1.)
    ];
    mul(mul(X, transpose(Y)), inverse(mul(Y, transpose(Y))))
}

type Mesh = (Box<[vec3]>, Box<[[u16; 3]]>);
struct View {
    oblique: [[Image<Array<u8>>; 3]; 4],
    //oblique: [Image<Array<u8>>; 4],
	//ground: Image<Array<f32>>,
    ortho: [Image<Array<u8>>; 3],
    buildings: Box<[Mesh]>,
    x: Array,
    y: Array,
    z: Array,

    start_position: f32,
    position: vec2,
    vertical_scroll: f32,
    index: usize,
}

#[throws] fn size(name: &str) -> size {
    let tiff = unsafe{memmap::Mmap::map(&std::fs::File::open(format!("../data/{name}.tif"))?)?};
    let mut tiff = tiff::decoder::Decoder::new(std::io::Cursor::new(&*tiff))?;
    let (size_x, size_y) = tiff.dimensions()?;
    size{x: size_x, y: size_y}
}

type Bounds = MinMax<vec3>;

#[throws] fn raster_bounds(name: &str) -> Bounds {
    let size = size(name)?;
    let tiff = unsafe{memmap::Mmap::map(&std::fs::File::open(format!("../data/{name}.tif"))?)?};
    let mut tiff = tiff::decoder::Decoder::new(std::io::Cursor::new(&*tiff))?;
    let [_, _, _, E, N, _] = tiff.get_tag_f64_vec(tiff::tags::Tag::ModelTiepointTag)?[..] else { panic!() };
    let [scale_E, scale_N, _] = tiff.get_tag_f64_vec(tiff::tags::Tag::ModelPixelScaleTag)?[..] else { panic!() };
    let min = vec3{x: E as f32, y: (N-scale_N*size.y as f64) as f32, z: 0.};
    let max = vec3{x: (E+scale_E*size.x as f64) as f32, y: N as f32, z: f32::MAX};
    MinMax{min, max}
}

#[throws] fn points_bounds(name: &str) -> Bounds {
    let reader = las::Reader::from_path(format!("../data/{name}.las"))?;
    let las::Bounds{min, max} = las::Read::header(&reader).bounds();
    MinMax{min: vec3{x: min.x as f32, y: min.y as f32, z: min.z as f32}, max: vec3{x: max.x as f32, y: max.y as f32, z: max.z as f32}}
}

const neighbours: [int2; 8] =
[   xy{x:-1,y:-1}, xy{x:0,y:-1}, xy{x:1,y:-1},
    xy{x:-1,y: 0},                       xy{x:1,y: 0},
    xy{x:-1,y: 1}, xy{x:0,y: 1}, xy{x:1,y: 1} ];

impl View {
fn new(image: &str, points: &str) -> Result<Self> {
    let image_bounds = raster_bounds(image)?;
    let MinMax{min, max} = image_bounds.clip(points_bounds(points)?);
    //println!("{min:?} {max:?}");
    let center = (1./2.)*(min+max);
    let extent = max-min;
    let extent = extent.x.min(extent.y);

    fn map<T:bytemuck::Pod>(field: &str, name: &str) -> Result<Array<T>> { self::map(&format!("../.cache/las/{name}.{field}")) }

	#[cfg(o)] let ground = {
		let name = "swissalti3d_2020_2684-1248_0.5_2056_5728";
		let tiff = unsafe{memmap::Mmap::map(&std::fs::File::open(format!("../data/{name}.tif"))?)?};
		let mut tiff = tiff::decoder::Decoder::new(std::io::Cursor::new(&*tiff))?;
		let tiff::decoder::DecodingResult::F32(ground) = tiff.read_image()? else { panic!() };
	    let size = size(name)?;
        let stride = size.x;
        let ground = { // Flip Y and normalizes height
            let mut target = unsafe{Box::new_uninit_slice(ground.len()).assume_init()};
            for y in 0..size.y {
                for x in 0..size.x {
                    target[(y*stride+x) as usize] = (ground[((size.y-1-y)*stride+x) as usize]-center.z)/extent;
                }
            }
            target
        };
        std::fs::write(format!("../.cache/las/{points}.ground"), bytemuck::cast_slice(&ground))?;

		let bounds = raster_bounds(name)?;
        let scale = size.x as f32 / (bounds.max - bounds.min).x;
        assert!(scale == 2., "{scale}");
        let min = scale*(min-bounds.min);
        assert_eq!(min.x, f32::trunc(min.x));
        assert_eq!(min.y, f32::trunc(min.y));
        let max = scale*(max-bounds.min);
        let size = xy{x: max.x as u32 - min.x as u32, y: max.y as u32 - min.y as u32};
        let ground = map("ground", points).unwrap();
        Image::strided(size, ground.map(|data| &data[((min.y as u32)*stride+(min.x as u32)) as usize..][..(size.y*stride) as usize]), stride)
    };

    let [r,g,b] = {
        let size = size(image)?;
        let scale = size.x as f32 / (image_bounds.max - image_bounds.min).x;
        assert!(scale == 20.);
        let min = scale*(min-image_bounds.min);
        assert_eq!(min.x, f32::trunc(min.x));
        assert_eq!(min.y, f32::trunc(min.y));
        let max = scale*(max-image_bounds.min);

        #[cfg(o)] let bgra = {
            let tiff = unsafe{memmap::Mmap::map(&std::fs::File::open(format!("../data/{image}.tif"))?)?};
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
            rgba
        };

        let stride = size.x;
        let plane_stride = (size.y*stride) as usize;

        if false {
            let flip = {
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
            std::fs::write(format!("../.cache/las/{image}.rgb"), flip)?;
        }

        let size = xy{x: max.x as u32 - min.x as u32, y: max.y as u32 - min.y as u32};
        iter::eval(|plane| {
            let image = map("rgb", image).unwrap();
            Image::strided(size, image.map(|data| &data[plane*plane_stride + ((min.y as u32)*stride+(min.x as u32)) as usize..][..(size.y*stride) as usize]), stride)
        })
    };

    /*let buildings = dxf::Drawing::load_file("../data/SWISSBUILDINGS3D_2_0_CHLV95LN02_1091-23.dxf")?.entities();
    std::fs::write("../.cache/las/1091-23.buildings", bincode::serialize(&buildings.collect::<Vec<_>>())?)?;*/
    let buildings : Vec<dxf::entities::Entity> = bincode::deserialize(&std::fs::read("../.cache/las/1091-23.buildings")?)?;
    let buildings = buildings.into_iter().filter_map(|mesh| if let dxf::entities::EntityType::Polyline(mesh) = mesh.specific {
        let mut vertices_and_indices = mesh.__vertices_and_handles.iter().peekable();
        let mut vertices = Vec::new();
        while let Some((vertex,_)) = vertices_and_indices.next_if(|(v,_)|
            [v.polyface_mesh_vertex_index1, v.polyface_mesh_vertex_index2, v.polyface_mesh_vertex_index3, v.polyface_mesh_vertex_index4] == [0,0,0,0]
        ) {
            let dxf::Point{x,y,z} = vertex.location;
            let v = (vec3{x: x as f32,y: y as f32, z: z as f32}-center)/extent;
            if !(v.x > -1./2. && v.x < 1./2. && v.y > -1./2. && v.y < 1./2.) { return None; }
            vertices.push(v);
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
    //println!("{} {}", buildings.iter().map(|(v,t)| v.len()*4+t.len()*3*2).sum::<usize>(), buildings.iter().map(|(_,t)| t.len()*3*4).sum::<usize>());

    let oblique = iter::eval(|index| {
        let orientation = index*90;
        let decoder = png::Decoder::new(std::fs::File::open(format!("../data/47.38380,8.55692,{orientation}.png")).unwrap());
        let mut reader = decoder.read_info().unwrap();
        let size = xy{x:reader.info().width, y:reader.info().height};
        let stride = size.x;
        let plane_stride = (size.y*stride) as usize;
        if false {
            //println!("{orientation}");
            let mut buffer = vec![0; reader.output_buffer_size()];
            let info = reader.next_frame(&mut buffer).unwrap();
            let image = &buffer[..info.buffer_size()];
            let mut planar = unsafe{Box::new_uninit_slice(image.len()).assume_init()};
            for i in 0..3 {
                for y in 0..size.y {
                    for x in 0..size.x {
                        planar[i*plane_stride+(y*stride+x) as usize] = image[(y*stride+x) as usize * 3 + i];
                    }
                }
            }
            std::fs::write(format!("../.cache/las/{orientation}.rgb"), planar).unwrap();
        }
        let [r,g,b] = iter::eval(|plane| Image::new(size, map("rgb", &format!("{orientation}")).unwrap().map(|data:&[u8]| &data[plane*plane_stride..][..plane_stride])));
        [b,g,r]
        //std::fs::write(format!("../.cache/las/{orientation}.L"), &sRGB_to_linear(&([b,g,r].each_ref().map(|image| image.as_ref()))).data).unwrap();
        //let L = Image::<Array<u8>>::new(size, map("L", &format!("{orientation}")).unwrap());
        //L
        /*if false {
            let L = Image::<Array<u8>>::new(size, map("L", &format!("{orientation}")).unwrap());
            let size = L.size;
            let mut target = Image::zero(size);
            for y in 1..size.y-1 { for x in 1..size.x-1 {
                let p = xy{x,y}.signed();
                let Lx = (-1.) * L[(p+xy{x:-1,y:-1}).unsigned()] as f32 + (-2.) * L[(p+xy{x:-1,y:0}).unsigned()] as f32 + (-1.) * L[(p+xy{x:-1,y:1}).unsigned()] as f32 +
                            (1.) * L[(p+xy{x:1,y:-1}).unsigned()] as f32 + (2.) * L[(p+xy{x:1,y:0}).unsigned()] as f32 + (1.) * L[(p+xy{x:1,y:1}).unsigned()] as f32;
                let Ly = (-1.) * L[(p+xy{x:-1,y:-1}).unsigned()] as f32 + (-2.) * L[(p+xy{x:0,y:-1}).unsigned()] as f32 + (-1.) * L[(p+xy{x:1,y:-1}).unsigned()] as f32 +
                            (1.) * L[(p+xy{x:-1,y:1}).unsigned()] as f32 + (2.) * L[(p+xy{x:0,y:1}).unsigned()] as f32 + (1.) * L[(p+xy{x:1,y:1}).unsigned()] as f32;
                target[p.unsigned()] = f32::min(255., f32::sqrt(Lx*Lx+Ly*Ly)) as u8;
            }}
            std::fs::write(format!("../.cache/las/{orientation}.Δ"), &target.data).unwrap();
        }
        if false {
            let Δ = Image::<Array<u8>>::new(size, map("Δ", &format!("{orientation}")).unwrap());
            let size = Δ.size;
            let mut target = Image::zero(size);
            for y in 1..size.y-1 { for x in 1..size.x-1 {
                let p = xy{x,y};
                target[p] = if neighbours.iter().all(|dp| Δ[(p.signed()+dp).unsigned()] <= Δ[p]) { Δ[p] } else { 0 }; // FIXME: keep non max along edge
            }}
            std::fs::write(format!("../.cache/las/{orientation}.max"), &target.data).unwrap();
        }
        if false {
            let max = Image::<Array<u8>>::new(size, map("max", &format!("{orientation}")).unwrap());
            let size = max.size;
            let mut target = Image::zero(size);
            for y in 1..size.y-1 { for x in 1..size.x-1 {
                let p = xy{x,y};
                target[p] = if max[p] > 0xFF/2 && neighbours.iter().any(|dp| max[(p.signed()+dp).unsigned()] > 0xFF/2) { max[p] } else { 0 };
            }}
            std::fs::write(format!("../.cache/las/{orientation}.edge"), &target.data).unwrap();
        }
        Image::new(size, map("edge", &format!("{orientation}")).unwrap())*/
    });

    if false {
        let mut reader = las::Reader::from_path(format!("../data/{points}.las"))?;
        let mut X = Vec::with_capacity(las::Read::header(&reader).number_of_points() as usize);
        let mut Y = Vec::with_capacity(las::Read::header(&reader).number_of_points() as usize);
        let mut Z = Vec::with_capacity(las::Read::header(&reader).number_of_points() as usize);
        for point in las::Read::points(&mut reader) {
            let las::Point{x: E,y: N, z, ..} = point.unwrap();
            let x = (E as f32-center.x)/extent;
            let y = (N as f32-center.y)/extent;
            let z = (z as f32-center.z)/extent;
            if x > -1./2. && x < 1./2. && y > -1./2. && y < 1./2. {
                X.push(x);
                Y.push(y);
                Z.push(z);
            }
        }
        #[throws] fn write<T:bytemuck::Pod>(name: &str, field: &str, points: &[T]) { std::fs::write(format!("../.cache/las/{name}.{field}"), bytemuck::cast_slice(points))? }
        write(points, "x", &X)?;
        write(points, "y", &Y)?;
        write(points, "z", &Z)?;
    }

    let map = |field| map(field, points).unwrap();
    Ok(Self{
        oblique,
    	//ground,
        ortho: [b,g,r],
        buildings,
        x: map("x"), y: map("y"), z: map("z"),
        start_position: 0., position: xy{x:0.,y:f32::MAX}, vertical_scroll: 1., index: 0,
    })
}
}

use vector::{vec3, MinMax};
use ui::{Widget, RenderContext as Target, widget::{EventContext, Event}};

impl Widget for View {
#[throws] fn event(&mut self, size: size, _event_context: &EventContext, event: &Event) -> bool {
    match event {
        Event::Key(' ') => { self.index = (self.index+1)%5; true }
        /*&Event::Motion{position, ..} => {
            if self.start_position.is_zero() { self.start_position = position.x; }
            self.position = position;
            true
        },
        &Event::Scroll(vertical_scroll) => {
            self.vertical_scroll = f32::clamp(1., self.vertical_scroll + vertical_scroll, 64.);
            true
        },*/
        &Event::Button{position, state: ui::widget::ButtonState::Pressed, ..} => {
            let position = position.map(|&c| c as u32);
            /*let f = |index: usize, (offset, size): (uint2, size), source: size| if position > offset && position < offset+size {
                let xy{x,y} = position-offset;
                let y = size.y-1-y;
                let xy{x,y} = xy{x,y}*source/size;
                println!("{index} {x} {y}");
            };
            //for (index, image) in self.oblique.iter().enumerate() { f(index, fit(sub(size, index), image[0].size), image[0].size); }
            //f(4, fit(sub(size, 4), self.ortho[0].size), self.ortho[0].size);
            f(0, (xy{x:0,y:0}, fit(size, self.ortho[0].size)), self.ortho[0].size));*/
            let xy{x,y} = position;
            let p = xy{x, y: size.y-1-y};
            let source = if self.index < 4 { &self.oblique[self.index] } else { &self.ortho };
            let p = p*source[0].size/size;
            let xy{x,y} = p;
            println!("{} {x} {y}", self.index);
            /*{
                let source = self.ortho[0].size;
                let xy{x,y} = ((size/2*source).signed()+(p.signed()-size.signed()/2)*source.signed()/5).unsigned()/size;
                print!("{x} {y} ");
            }
            {
                let p = vec2::from(p);
                let p = {let size=vec2::from(size); size/2.+(1./5.)*(p-size/2.)};
                let p = apply(inverse(affine_transform(0)), p).map(|&c| c as u32);
                let xy{x,y} = p;
                println!("{x} {y}");
            }*/
            false
        },
        _ => false,
    }
}

fn paint(&mut self, target: &mut Target, _size: size) -> Result {
    let _start = std::time::Instant::now();
    let source = if self.index < 4 { &self.oblique[self.index] } else { &self.ortho };
    let target = &mut fit_image(target, source[0].size);
    blit_sRGB(target, &source.each_ref().map(|image| image.as_ref()));
    //affine_blit(target, self.oblique[0].as_ref(), inverse(affine_transform(0)));

    for correspondence in &*correspondences() {
        let p = correspondence[self.index]*target.size/source[0].size;
        draw_cross(target, p);
    }

    /*let edges = self.oblique.iter().enumerate().map(|(_index, image)| {
        if _index > 0 { return Vec::new(); }
        let size = image.size;
        let mut image = Image::new(size, Vec::from(&*image.data));
        let mut edges = Vec::new();
        for y in 0..image.size.y { for x in 0..image.size.x {
            let p = xy{x,y};
            if image[p] == 0 { continue; }
            image[p] = 0;
            let mut walk = |edge:&mut Vec<uint2>, mut p:uint2| {
                'walk: loop {
                    for dp in neighbours { // FIXME: first do proper thinning to guarantee only one edge (otherwise => top-left bias)
                        let next = (p.signed()+dp).unsigned();
                        if image[next] > 0 {
                            edge.push(next);
                            image[next] = 0;
                            p = next;
                            continue 'walk;
                        }
                    }
                    break;
                }
            };
            let mut edge = Vec::new();
            walk(&mut edge, p);
            edge.reverse();
            walk(&mut edge, p);
            if edge.len() > 2 && vector::sq(edge[0].signed()-edge.last().unwrap().signed())>num::sq(128) {
                //println!("{:?}", edge.len());
                edges.push(edge);
            }
            //break;
        }}
        //use itertools::Itertools; println!("{}", edges.iter().format_with(" ",|e,f| f(&format_args!("{}", e.len()))));
        println!("{}", edges.len());
        edges
    }).collect::<Box<_>>();
    //for (index, image) in self.oblique.iter().enumerate() { blit(&mut fit_image(&mut sub_image(target, index), image[0].size), &image.each_ref().map(|image| image.as_ref())); }
    //blit_sRGB(&mut fit_image(&mut sub_image(target, 4), self.ortho[0].size), &self.ortho.each_ref().map(|image| image.as_ref()));

    use std::iter::zip;
    for ((index, image), edges) in zip(self.oblique.iter().enumerate(), &*edges) {
        /*use vector::{xyz, cross, minmax};
        let render = |target:&mut Image<&mut[bgra8]>| self.buildings.iter().for_each(|(vertices, triangles)| {
            for triangle in triangles.iter() {
                let vertices = triangle.map(|i| vertices[i as usize]);
                fn from_normalized(size: size, xy{x,y,..}: vec2) -> vec2 { let size = vec2::from(size); size/2. + size * xy{x,y}}
                let view = vertices.map(|point| from_normalized(target.size, /*5.**/point.xy()));
                let [v0,v1,v2] = view;
                let w = cross(v1-v0, v2-v0);
                //println!("{v0:?} {v1:?} {v2:?} {w}");
                if w <= 0. { continue; }
                let MinMax{min,max} : MinMax<vec2> = minmax(view.into_iter()).unwrap();
                let min = xy{x: f32::max(0., min.x), y: f32::max(0., min.y)};
                let max = xy{x: u32::min(max.x as u32+1, target.size.x), y: u32::min(max.y as u32+1, target.size.y)};
                for y in min.y as u32 .. max.y {
                    for x in min.x as u32 .. max.x {
                        let p = xy{x: x as f32, y: y as f32};
                        let (w2,w0,w1) = (cross(v1-v0, p-v0), cross(v2-v1, p-v1), cross(v0-v2, p-v2));
                        if w2 > 0. && w0 > 0. && w1 > 0. {
                            let size = target.size;
                            target[xy{x,y: size.y-1-y}] = {
                                let xyz{x,y,..} = (w0*vertices[0] + w1*vertices[1] + w2*vertices[2])/w;
                                let [b,g,r] = &self.ortho;
                                let u = ((x+1./2.)*(b.size.x as f32)) as u32;
                                let v = ((y+1./2.)*(b.size.y as f32)) as u32;
                                let index = v * b.stride + u;
                                let b = b[index as usize];
                                let g = g[index as usize];
                                let r = r[index as usize];
                                bgra8{b,g,r,a:0xFF}
                            };
                        }
                    }
                }
            }
        });
        render(target);*/
        /*for edge in edges {
            //println!("{edge:?}");
            for &p in edge {
                let p = apply(map, vec2::from(p));//.map(|&c| c as u32);
                let size = target.size;
                let p = {let size=vec2::from(size); size/2.+/*5.**/(/*vec2::from(p)*/p-size/2.)};//.map(|&c| c as u32);
                if p.x < 0. || p.x >= target.size.x as f32 || p.y < 0. || p.y >= target.size.y as f32 { continue; }
                let p = p.map(|&c| c as u32);
                //if p.x >= size.x || p.y >= size.y { continue; }
                target[xy{x:p.x,y:size.y-1-p.y}] = bgra{b:0, g:0, r:0xFF, a:0xFF};
            }
        }*/
        break;
    }*/
    //println!("{:?}", std::time::Instant::now().duration_since(start));
    Ok(())
}
}

#[throws] fn main() { ui::run(Box::new(View::new("2408", "2684_1248")?))? }
