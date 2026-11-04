use crate::{
    ReducePrecision,
    components::{
        instructions::Item,
        readers::{Reader, ReaderExpand},
    },
};
use cubecl::prelude::*;

#[derive(CubeType)]
pub struct CubeReader<'a, P: ReducePrecision> {
    reader: Reader<'a, P>,
}

#[cube]
#[allow(clippy::len_without_is_empty)]
impl<'a, P: ReducePrecision> CubeReader<'a, P> {
    pub fn new(reader: Reader<'a, P>) -> CubeReader<'a, P> {
        CubeReader::<'a, P> { reader }
    }

    pub fn read(&self, vector_index: usize) -> Item<P> {
        match &self.reader {
            Reader::Parallel(reader) => reader.read_cube(vector_index),
            Reader::Perpendicular(reader) => reader.read_cube(vector_index),
        }
    }

    /// [`Self::read`] with `group` consecutive vectors per unit at each step (parallel
    /// vectorization only, unless `group` is 1).
    pub fn read_grouped(
        &self,
        vector_index: usize,
        #[comptime] group: usize,
        #[comptime] member: usize,
    ) -> Item<P> {
        match &self.reader {
            Reader::Parallel(reader) => reader.read_cube_grouped(vector_index, group, member),
            Reader::Perpendicular(reader) => reader.read_cube(vector_index),
        }
    }

    pub fn length_grouped(&self, #[comptime] group: usize) -> usize {
        match &self.reader {
            Reader::Parallel(reader) => reader.length_cube_grouped(group),
            Reader::Perpendicular(reader) => reader.length_cube(),
        }
    }

    pub fn length(&self) -> usize {
        match &self.reader {
            Reader::Parallel(reader) => reader.length_cube(),
            Reader::Perpendicular(reader) => reader.length_cube(),
        }
    }
}
