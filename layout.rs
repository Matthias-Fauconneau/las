use {vector::{xy, uint2, size}, image::Image};
pub fn sub(size: size, index: usize) -> (uint2, size) {
    let size = size/xy{x: 3, y: 2};
    (xy{x: index as u32%3, y:index as u32/3}*size, size)
}
pub fn sub_image<T, D:std::ops::DerefMut<Target=[T]>>(target: &mut Image<D>, index: usize) -> Image<&mut[T]> {
    let (offset, size) = sub(target.size, index);
    target.slice_mut(offset, size)
}

pub fn fit((offset, size): (uint2, size), source: size) -> (uint2, size) {
    /*let fit = if size.x*source.y < size.y*source.x
    { xy{x: size.x/source.x*source.x, y: size.x/source.x*source.y} } else
    { xy{x: size.y/source.y*source.x, y: size.y/source.y*source.y} };*/
    let fit = if size.x*source.y < size.y*source.x
    { xy{x: size.x, y: size.x*source.y/source.x} } else
    { xy{x: size.y*source.x/source.y, y: size.y} };
    (offset+(size-fit)/2, fit)
}
pub fn fit_image<'m, T>(target: &'m mut Image<&mut[T]>, source: size) -> Image<&'m mut [T]> {
    let (offset, size) = fit((num::zero(), target.size), source);
    target.slice_mut(offset, size)
}