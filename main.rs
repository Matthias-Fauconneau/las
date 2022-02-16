#![feature(type_alias_impl_trait, portable_simd, stdsimd, let_else, new_uninit, array_methods)]
#![allow(non_snake_case)]

use {ui::{Error, Result}, fehler::throws};

use owning_ref::OwningRef;

type Array<T=f32> = OwningRef<Box<memmap::Mmap>, [T]>;

#[throws] fn map<T:bytemuck::Pod>(field: &str, name: &str) -> Array<T> {
    OwningRef::new(Box::new(unsafe{memmap::Mmap::map(&std::fs::File::open(format!("../.cache/las/{name}.{field}"))?)}?)).map(|data| bytemuck::cast_slice(&*data))
}

use image::{Image, bgra, bgra8};

type Mesh = (Box<[vec3]>, Box<[[u16; 3]]>);
struct View {
    oblique: [[Image<Array<u8>>; 3]; 4],
	//ground: Image<Array<f32>>,
    ortho: [Image<Array<u8>>; 3],
    buildings: Box<[Mesh]>,
    x: Array,
    y: Array,
    z: Array,

    start_position: f32,
    position: vec2,
    vertical_scroll: f32,
}

use {vector::{xy, size, vec2, xyz, vec3}, ui::{Widget, RenderContext as Target, widget::{EventContext, Event}}, vector::num::IsZero};

fn rotate(xy{x,y,..}: vec2, angle: f32) -> vec2 { xy{x: f32::cos(angle)*x+f32::sin(angle)*y, y: f32::cos(angle)*y-f32::sin(angle)*x} }

use vector::MinMax;


/*fn sub<T, D:std::ops::DerefMut<Target=[T]>>(target: &mut Image<D>, index: usize) {
    let size = target.size/xy{x: 3, y: 2};
    target.slice_mut(xy{x: index as u32%3, y:index as u32/3}*size, size)
}*/

fn fit<'m, T>(target: &'m mut Image<&mut[T]>, source: size) -> Image<&'m mut [T]> {
    let size = target.size;
    let fit = if size.x*source.y < size.y*source.x
    { xy{x: size.x, y: source.y*size.x/source.x} } else
    { xy{x: source.x*size.y/source.y, y: size.y} };
    target.slice_mut((size-fit)/2, fit)
}

fn blit(target:&mut Image<&mut[bgra8]>, [b,g,r]: &[Image<&[u8]>; 3]) {
    let size = target.size;
    for y in 0..size.y {
        for x in 0..size.x {
            let i = (y*b.size.y/size.y*b.stride+x*b.size.x/size.x) as usize;
            let [b,g,r] = [b.data[i],g.data[i],r.data[i]];
            target[xy{x, y: size.y-1-y}] = bgra{b:b/2, g:g/2, r:r/2, a: 0xFF};
        }
    }
}

impl Widget for View {
#[throws] fn event(&mut self, _size: size, _event_context: &EventContext, event: &Event) -> bool {
    match event {
        &Event::Motion{position, ..} => {
            if self.start_position.is_zero() { self.start_position = position.x; }
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
    let ord = |x:f32| -> ordered_float::NotNan<f32> { ordered_float::NotNan::new(x).unwrap() };
    let from_normalized = |size: size, xyz{x,y,..}: xyz<f32>| { let size = vec2::from(size); size/2. + size.y * xy{x,y}};

    let building = self.buildings.iter().max_by_key(|(vertices,_)| vertices.iter().map(|&xyz{z,..}| ord(z)).max()).unwrap();
    let MinMax{min,max} = vector::minmax(building.0.iter().copied()).unwrap();
    let MinMax{min,max} = {let size = max-min; MinMax{min: min-size, max: max+size}};
    let crop = |p:vec3| (p-(min+max)/2.)/(max-min);
    let ortho = {
        let MinMax{min,max} = MinMax{min: from_normalized(self.ortho[0].size, min).map(|&c| c as u32), max: from_normalized(self.ortho[0].size, max).map(|&c| c as u32 + 1)};
        self.ortho.each_ref().map(|c| c.slice(min, max-min))
    };

    //for (index, image) in self.oblique.iter().enumerate() { blit(fit(sub(index), image[0].size), index, image); }
    //blit(fit(sub(4), self.ortho[0].size), &self.ortho);
    let mut target = fit(target, ortho[0].size);
    blit(&mut target, &ortho);

    let roof = {
        let (vertices, triangles) = building;
        let mut set = Vec::new();
        use vector::{cross, normalize};
        for triangle in triangles.iter() {
            let triangle = triangle.map(|i| vertices[i as usize]);
            let cross = |a:vec3,b:vec3| vec3{x: cross(a.yz(), b.yz()), y: cross(a.zx(), b.zx()), z: cross(a.xy(), b.xy())};
            let [v0,v1,v2] = triangle;
            let n = normalize(cross(v1-v0, v2-v0));
            if n.z > 1./2. { for v in triangle { if !set.contains(&v) { set.push(v); } } }
        }
        let mut hull = vec![*set.iter().min_by_key(|&&xyz{x,..}| ord(x)).unwrap()];
        loop {
            let &last = hull.last().unwrap();
            let mut candidate = set[0];
            for &p in &set {
                let angle = cross((candidate-last).xy(), (p-last).xy());
                if candidate == last || angle < 0. {
                    candidate = p;
                    //println!("{candidate:?} => {p:?}");
                }
            }
            if candidate == hull[0] { break; }
            hull.push(candidate);
        }
        hull
    };
    assert!(roof.len() == 4);
    //println!("{roof:?}");
    let mut disk = |point, r| {
        if point < min || point > max { return; }
        let p = from_normalized(target.size, crop(point)).map(|&c| c as i32);
        let min = xy{x: i32::max(0, p.x - r as i32) as u32, y: i32::max(0, p.y - r as i32) as u32};
        let max = xy{x: u32::min(p.x as u32+r+1, target.size.x), y: u32::min(p.y as u32+r+1, target.size.y)};
        for y in min.y .. max.y {
            for x in min.x .. max.x {
                if vector::sq(xy{x: x as i32,y: y as i32}-p.map(|&c| c as i32)) as u32 > r*r { continue; }
                let size = target.size;
                target[xy{x,y: size.y-1-y}] = {
                    let [b,g,r] = &ortho;
                    let xy{x,y} = (from_normalized(b.size, crop(point)).map(|&c| c as i32) + xy{
                        x: (x as i32 - p.x) * b.size.x as i32 / size.x as i32,
                        y: (y as i32 - p.y) * b.size.y as i32 / size.y as i32
                    }).map(|&c| c as u32);
                    let index = y * b.stride + x;
                    if (index as usize) < b.len() {
                        let b = b[index as usize];
                        //let g = image::sRGB(&y);
                        //let r = image::sRGB(&x);
                        let g = g[index as usize];
                        let r = r[index as usize];
                        bgra8{b,g,r,a:0xFF}
                    } else {
                        bgra8{b:0,g:0,r:0xFF,a:0xFF}
                    }
                };
            }
        }
    };
    if true { for point in roof { disk(point, 64); }}

    if true {
        let [x,y,z] = [&self.x, &self.y, &self.z];
        use std::iter::zip;
        println!("Points");
        for ((&x, &y), &z) in zip(zip(&**x,&**y),&**z) { disk(xyz{x,y,z}, 0); }
        println!("OK");
    }

    if false { self.buildings.iter().for_each(|(vertices, triangles)| {
        for triangle in triangles.iter() {
            let vertices = triangle.map(|i| vertices[i as usize]);
            let view = vertices.map(|point| from_normalized(target.size, crop(point)));
            let [v0,v1,v2] = view;
            use vector::cross;
            let w = cross(v1-v0, v2-v0);
            //println!("{v0:?} {v1:?} {v2:?} {w}");
			if w <= 0. { continue; }
            let MinMax{min,max} : MinMax<vec2> = vector::minmax(view.into_iter()).unwrap();
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
                            //let g = image::sRGB(&y);
                            //let r = image::sRGB(&x);
                            let g = g[index as usize];
                            let r = r[index as usize];
                            bgra8{b,g,r,a:0xFF}
                        };
                    }
                }
            }
        }
    });}

    //target.fill(bgra8{b:0,g:0,r:0,a:0xFF});
    /*let transform = |p:vec3| {
        let size = vec2::from(size);
        use std::f32::consts::PI;
        let yaw = 0.;//(self.position.x-self.start_position)/size.x*2.*PI;
        let pitch = PI/4.;//f32::clamp(0., (1.-self.position.y/size.y)*PI/2., PI/2.);
        let scale = size.x.min(size.y)*self.vertical_scroll;
        let z = p.z;
        let p = scale*rotate(p.xy(), yaw);
        let p = size/2.+xy{x: p.x, y: f32::cos(pitch)*p.y+f32::sin(pitch)*scale*z};
        //let p = xy{x: p.x, y: (size.y-1.) - p.y};
        p
    };
    let O = transform(vec3{x:0., y:0., z:0.});
    let e = [vec3{x:1., y:0., z:0.}, vec3{x:0., y:1., z:0.}, vec3{x:0., y:0., z:1.}].map(|e| transform(e)-O);*/

	/*let mut Z = Image::fill(size, -1./2.);
    self.buildings.into_par_iter().for_each(|(vertices, triangles)| {
        for triangle in triangles.iter() {
            let vertices = triangle.map(|i| vertices[i as usize]);
            let view = vertices.map(|xyz{x,y,z}| x * e[0] + y * e[1] + z * e[2] + O );
            let [v0,v1,v2] = view;
            use vector::cross;
            let w = cross(v1-v0, v2-v0);
            //println!("{v0:?} {v1:?} {v2:?} {w}");
			if w <= 0. { continue; }
            let MinMax{min,max} : MinMax<vec2> = vector::minmax(view.into_iter()).unwrap();
            let min = xy{x: f32::max(0., min.x), y: f32::max(0., min.y)};
            let max = xy{x: u32::min(max.x as u32+1, size.x), y: u32::min(max.y as u32+1, size.y)};
            for y in min.y as u32 .. max.y {
                for x in min.x as u32 .. max.x {
                    let xy = xy{x,y};
                    let p = xy.map(|&c| c as f32);
                    let (w2,w0,w1) = (cross(v1-v0, p-v0), cross(v2-v1, p-v1), cross(v0-v2, p-v2));
                    if w2 > 0. && w0 > 0. && w1 > 0. {
                        let xyz{x,y,z} = (w0*vertices[0] + w1*vertices[1] + w2*vertices[2])/w;
                        let i = (size.y-1-xy.y)*size.x+xy.x;
                        if !(z > Z[i as usize]) { continue; }
                        unsafe{(Z.as_ptr() as *mut f32).offset(i as isize).write(z)};
						/*let [b,g,r] = &self.ortho;
                        let u = ((x+1./2.)*(b.size.x as f32)) as u32;
                        let v = ((y+1./2.)*(b.size.y as f32)) as u32;
                        let index = v * b.stride + u;
                        let b = b[index as usize];
                        let g = g[index as usize];
                        let r = r[index as usize];
                        unsafe{(target.as_ptr() as *mut bgra8).offset(i as isize).write(bgra8{b,g,r,a:0xFF})};*/
                        unsafe{(target.as_ptr() as *mut bgra8).offset(i as isize).write(bgra8{b:0xFF,g:0xFF,r:0xFF,a:0xFF})};
                    }
                }
            }
        }
    });*/

    /*let ground = &self.ground;
    (0..ground.size.y-1).into_par_iter().for_each(|y| {
        for x in 0..ground.size.x-1 {
            let map = |x,y| xyz{x: (x as f32)/(ground.size.x as f32)-1./2., y: (y as f32)/(ground.size.y as f32)-1./2., z: ground[xy{x,y}]};
            for vertices in [[map(x,y),map(x+1,y),map(x+1,y+1)], [map(x,y),map(x+1,y+1),map(x,y+1)]] {
                let view = vertices.map(|xyz{x,y,z}| x * e[0] + y * e[1] + z * e[2] + O );
                let [v0,v1,v2] = view;
                use vector::cross;
                let w = cross(v1-v0, v2-v0);
                //println!("{v0:?} {v1:?} {v2:?} {w}");
                if w <= 0. { continue; }
                let MinMax{min,max} : MinMax<vec2> = vector::minmax(view.into_iter()).unwrap();
                let min = xy{x: f32::max(0., min.x), y: f32::max(0., min.y)};
                let max = xy{x: u32::min(max.x as u32+1, size.x), y: u32::min(max.y as u32+1, size.y)};
                for y in min.y as u32 .. max.y {
                    for x in min.x as u32 .. max.x {
                        let xy = xy{x,y};
                        let p = xy.map(|&c| c as f32);
                        let (w2,w0,w1) = (cross(v1-v0, p-v0), cross(v2-v1, p-v1), cross(v0-v2, p-v2));
                        if w2 > 0. && w0 > 0. && w1 > 0. {
                            let xyz{x,y,z} = (w0*vertices[0] + w1*vertices[1] + w2*vertices[2])/w;
                            let i = (size.y-1-xy.y)*size.x+xy.x;
                            if !(z > Z[i as usize]) { continue; }
                            unsafe{(Z.as_ptr() as *mut f32).offset(i as isize).write(z)};
                            let [b,g,r] = &self.image;
                            let u = ((x+1./2.)*(b.size.x as f32)) as u32;
                            let v = ((y+1./2.)*(b.size.y as f32)) as u32;
                            let index = v * b.stride + u;
                            let b = b[index as usize];
                            let g = g[index as usize];
                            let r = r[index as usize];
                            unsafe{(target.as_ptr() as *mut bgra8).offset(i as isize).write(bgra8{b,g,r,a:0xFF})};
                        }
                    }
                }
            }
        }
    });*/

    /*use std::simd::{Simd, f32x16, u32x16};
    use std::arch::x86_64::*;
    let [x,y,z] = [&self.x, &self.y, &self.z].map(|array| unsafe { let ([], array, _) = array.align_to::<f32x16>() else { unreachable!() }; array});
    (x, y, z).into_par_iter().for_each(|(&x, &y, &z)| { unsafe {
    	let [b,g,r] = &self.image;
        let u : u32x16 = _mm512_cvttps_epu32(((x+Simd::splat(1./2.))*Simd::splat(b.size.x as f32)).into()).into();
        let v : u32x16 = _mm512_cvttps_epu32(((y+Simd::splat(1./2.))*Simd::splat(b.size.y as f32)).into()).into();
        let indices = v * Simd::splat(b.stride) + u;
        let b : u32x16 = _mm512_i32gather_epi32(indices.into(), (b.as_ptr() as *const u8).offset(-0), 1).into();
        let g : u32x16 = _mm512_i32gather_epi32(indices.into(), (g.as_ptr() as *const u8).offset(-1), 1).into();
        let r : u32x16 = _mm512_i32gather_epi32(indices.into(), (r.as_ptr() as *const u8).offset(-2), 1).into();
        let bgra = Simd::splat(0xFF_00_00_00) | r & Simd::splat(0x00_FF_00_00) | g & Simd::splat(0x00_00_FF_00) | b & Simd::splat(0x00_00_00_FF);
        let e = e.map(|e| e.map(|&x| Simd::splat(x)));
        let O = O.map(|&x| Simd::splat(x));
        let p = xy{x: x * e[0].x + y * e[1].x + z * e[2].x + O.x, y: x * e[0].y + y * e[1].y + z * e[2].y + O.y};
        let p : xy<u32x16> = p.map(|&p| _mm512_cvttps_epu32(p.into()).into());
        //unsafe{bgra.scatter_select_unchecked(target, indices.lanes_lt(Simd::splat(target.len())), p.y * size.x + p.x)};
        _mm512_mask_i32scatter_epi32(target.as_ptr() as *mut u8, _mm512_cmplt_epu32_mask(p.x.into(), Simd::splat(size.x).into()) & _mm512_cmplt_epu32_mask(p.y.into(), Simd::splat(size.y).into()), ((Simd::splat(size.y-1)-p.y) * Simd::splat(size.x) + p.x).into(), bgra.into(), 4);
    }});*/

    Ok(())
}
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

impl View {
fn new(image: &str, points: &str) -> Result<Self> {
    let image_bounds = raster_bounds(image)?;
    let MinMax{min, max} = image_bounds.clip(points_bounds(points)?);
    //println!("{min:?} {max:?}");
    let center = (1./2.)*(min+max);
    let extent = max-min;
    let extent = extent.x.min(extent.y);

	/*let ground = {
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
    };*/

    let [r,g,b] = {
        let size = size(image)?;
        let scale = size.x as f32 / (image_bounds.max - image_bounds.min).x;
        assert!(scale == 20.);
        let min = scale*(min-image_bounds.min);
        assert_eq!(min.x, f32::trunc(min.x));
        assert_eq!(min.y, f32::trunc(min.y));
        let max = scale*(max-image_bounds.min);

        /*let tiff = unsafe{memmap::Mmap::map(&std::fs::File::open(format!("../data/{image}.tif"))?)?};
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
        std::fs::write(format!("../.cache/las/{image}.rgb"), flip)?;*/

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
    });

    /*let mut X = Vec::with_capacity(las::Read::header(&reader).number_of_points() as usize);
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
    write(points, "z", &Z)?;*/

    let map = |field| map(field, points).unwrap();
    Ok(Self{
        oblique,
    	//ground,
        ortho: [b,g,r],
        buildings,
        x: map("x"), y: map("y"), z: map("z"),
        start_position: 0., position: xy{x:0.,y:f32::MAX}, vertical_scroll: 1.
    })
}
}
#[throws] fn main() { ui::run(Box::new(View::new("2408", "2684_1248")?))? }
