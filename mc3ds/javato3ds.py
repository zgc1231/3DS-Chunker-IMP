from __future__ import annotations

import json
import logging
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from anvil import Chunk, ChunkNotFound, Region
from nbt import nbt

from .classes import World, parse_position

logger = logging.getLogger(__name__)

OVERWORLD = 0
NETHER = 1
END = 2

MAGIC_CDB = 0xABCDEF99
FILE_HEADER_STRUCT = struct.Struct("<HHIIII")


def _dimension_path(world: Path, dimension: int) -> Path:
    if dimension == OVERWORLD:
        return world
    if dimension == NETHER:
        return world / "DIM-1"
    if dimension == END:
        return world / "DIM1"
    raise ValueError(f"invalid dimension {dimension!r}")


def _canonicalise_properties(properties: dict | None) -> dict[str, str]:
    if not properties:
        return {}
    normalised: dict[str, str] = {}
    for key, value in properties.items():
        if isinstance(value, bool):
            value_str = str(value).lower()
        else:
            value_str = str(value)
        normalised[str(key)] = value_str
    return normalised


def _format_block_state(name: str, properties: dict[str, str]) -> str:
    if not properties:
        return name
    props = ",".join(f"{key}={value}" for key, value in sorted(properties.items()))
    return f"{name}[{props}]"


def _load_inverse_block_map() -> dict[str, tuple[int, int]]:
    mapping_path = Path(__file__).parent / "data" / "blocks.json"
    with mapping_path.open("r", encoding="utf-8") as blocks_file:
        raw_blocks = json.load(blocks_file)



def _load_inverse_block_map() -> dict[str, tuple[int, int]]:
    mapping_path = Path(__file__).parent / "data" / "blocks.json"
    with mapping_path.open("r", encoding="utf-8") as blocks_file:
        raw_blocks = json.load(blocks_file)

    inverse: dict[str, tuple[int, int]] = {}
    for block_key, state in raw_blocks["blocks"].items():
        block_id, meta_id = (int(part) for part in block_key.split(":"))
        if isinstance(state, str):
            name, _, raw_props = state.partition("[")
            props: dict[str, str]
            if raw_props:
                raw_props = raw_props.rstrip("]")
                props = {}
                for prop in raw_props.split(","):
                    if not prop:
                        continue
                    if "=" in prop:
                        prop_key, prop_value = prop.split("=", 1)
                        props[prop_key] = prop_value
                    else:
                        props[prop] = ""
            else:
                props = {}
        else:
            name = state.get("Name", "minecraft:air")
            props = {
                str(k): str(v)
                for k, v in (state.get("Properties") or {}).items()
            }
        canonical = _format_block_state(name, props)
        inverse[canonical] = (block_id, meta_id)
    return inverse


def _tag_get(tag: nbt.TAG_Compound | nbt.NBTFile | None, key: str):
    if tag is None:
        return None
    try:
        return tag[key]
    except KeyError:
        return None


def _tag_value(tag, default=None):
    if tag is None:
        return default
    return getattr(tag, "value", tag)


def _compound_to_dict(tag) -> dict[str, object]:
    if not isinstance(tag, nbt.TAG_Compound):
        return {}
    props: dict[str, object] = {}
    for child in tag.tags:
        props[child.name] = _tag_value(child)
    return props


def _decode_block_state_indices(data_tag, palette_size: int, value_count: int = 4096) -> list[int]:
    if palette_size <= 1:
        return [0] * value_count
    if data_tag is None:
        return [0] * value_count

    raw_values = getattr(data_tag, "value", [])
    if not raw_values:
        return [0] * value_count

    bits_per_block = max(4, (palette_size - 1).bit_length())
    mask = (1 << bits_per_block) - 1

    results: list[int] = []
    bit_buffer = 0
    bits_in_buffer = 0
    iterator = iter(raw_values)

    for _ in range(value_count):
        while bits_in_buffer < bits_per_block:
            next_long = next(iterator, 0)
            bit_buffer |= (next_long & 0xFFFFFFFFFFFFFFFF) << bits_in_buffer
            bits_in_buffer += 64
        results.append(bit_buffer & mask)
        bit_buffer >>= bits_per_block
        bits_in_buffer -= bits_per_block

    return results


def _build_flat_map_from_nbt(
    chunk: nbt.NBTFile | nbt.TAG_Compound,
    chunk_x: int,
    chunk_z: int,
    inverse_block_map: dict[str, tuple[int, int]],
    max_height: int,
) -> tuple[dict[tuple[int, int, int], tuple[int, int]], set[str]]:
    flat_map: dict[tuple[int, int, int], tuple[int, int]] = {}
    missing: set[str] = set()

    level = chunk
    if isinstance(level, (nbt.NBTFile, nbt.TAG_Compound)) and "Level" in level:
        level = level["Level"]

    sections = _tag_get(level, "sections") or _tag_get(level, "Sections")
    if sections is None:
        return flat_map, missing

    base_x = chunk_x * 16
    base_z = chunk_z * 16

    for section in sections:
        section_y_tag = _tag_get(section, "Y") or _tag_get(section, "y")
        if section_y_tag is None:
            continue
        section_y = int(_tag_value(section_y_tag, 0))
        section_min_y = section_y * 16
        section_max_y = section_min_y + 16
        if section_max_y <= 0 or section_min_y >= max_height:
            continue

        block_states_container = _tag_get(section, "block_states")
        palette_tag = _tag_get(block_states_container, "palette")
        data_tag = _tag_get(block_states_container, "data")

        if palette_tag is None:
            palette_tag = _tag_get(section, "Palette")
        if data_tag is None:
            data_tag = _tag_get(section, "BlockStates")

        if palette_tag is None:
            continue

        palette: list[tuple[int, int] | None] = []
        for entry in palette_tag:
            name = str(_tag_value(_tag_get(entry, "Name"), "minecraft:air"))
            props = _compound_to_dict(_tag_get(entry, "Properties"))
            canonical = _format_block_state(name, _canonicalise_properties(props))
            block_id = inverse_block_map.get(canonical)
            if block_id is None:
                missing.add(canonical)
            palette.append(block_id)

        palette_size = len(palette)
        if palette_size == 0:
            continue

        indices = _decode_block_state_indices(data_tag, palette_size)
        for idx, palette_index in enumerate(indices):
            if palette_index >= palette_size:
                continue
            block_id = palette[palette_index]
            if block_id is None or block_id == (0, 0):
                continue

            local_x = idx & 0xF
            local_z = (idx >> 4) & 0xF
            local_y = (idx >> 8) & 0xF
            world_y = section_min_y + local_y
            if world_y < 0 or world_y >= max_height:
                continue
            world_x = base_x + local_x
            world_z = base_z + local_z
            flat_map[(world_x, world_y, world_z)] = block_id

    return flat_map, missing


class RegionCache:
    def __init__(self, world: Path) -> None:
        self.world = Path(world)
        self._cache: dict[tuple[int, int, int], Region] = {}

    def get_chunk(
        self, chunk_x: int, chunk_z: int, dimension: int
    ) -> Chunk | nbt.NBTFile | nbt.TAG_Compound | None:
    def get_chunk(self, chunk_x: int, chunk_z: int, dimension: int) -> Chunk | None:
        region_x = chunk_x >> 5
        region_z = chunk_z >> 5
        cache_key = (region_x, region_z, dimension)
        try:
            region = self._cache[cache_key]
        except KeyError:
            region_path = (
                _dimension_path(self.world, dimension)
                / "region"
                / f"r.{region_x}.{region_z}.mca"
            )
            if not region_path.is_file():
                return None
            with region_path.open("rb") as region_handle:
                region = Region.from_file(region_handle)
            self._cache[cache_key] = region

        try:
            return region.get_chunk(chunk_x, chunk_z)
        except ChunkNotFound:
            return None
        except KeyError:
            try:
                chunk_nbt = region.chunk_data(chunk_x, chunk_z)
            except ChunkNotFound:
                return None
            if chunk_nbt is None:
                return None
            if isinstance(chunk_nbt, nbt.NBTFile) and "Level" in chunk_nbt:
                return chunk_nbt["Level"]
            return chunk_nbt


def _build_flat_map(
    chunk: Chunk | nbt.NBTFile | nbt.TAG_Compound,


def _build_flat_map(
    chunk: Chunk,
    chunk_x: int,
    chunk_z: int,
    inverse_block_map: dict[str, tuple[int, int]],
    max_height: int = 128,
) -> tuple[dict[tuple[int, int, int], tuple[int, int]], set[str]]:
    if not isinstance(chunk, Chunk):
        return _build_flat_map_from_nbt(chunk, chunk_x, chunk_z, inverse_block_map, max_height)

    flat_map: dict[tuple[int, int, int], tuple[int, int]] = {}
    missing: set[str] = set()
    base_x = chunk_x * 16
    base_z = chunk_z * 16

    for local_y in range(max_height):
        for local_z in range(16):
            for local_x in range(16):
                block = chunk.get_block(local_x, local_y, local_z)
                if block.name() == "minecraft:air" and not block.properties:
                    continue
                canonical = _format_block_state(
                    block.name(), _canonicalise_properties(block.properties)
                )
                try:
                    block_id = inverse_block_map[canonical]
                except KeyError:
                    missing.add(canonical)
                    continue
                if block_id == (0, 0):
                    continue
                world_pos = (base_x + local_x, local_y, base_z + local_z)
                flat_map[world_pos] = block_id
    return flat_map, missing


def _encode_position(x: int, z: int, dimension: int) -> int:
    if x < 0:
        x += 1 << 14
    if z < 0:
        z += 1 << 14
    return ((dimension & 0xF) << 28) | ((z & 0x3FFF) << 14) | (x & 0x3FFF)


def _build_block_data(
    flat_map: dict[tuple[int, int, int], tuple[int, int]],
    chunk_x: int,
    chunk_z: int,
    height: int,
) -> bytes:
    subchunk_count = (height + 15) // 16
    if subchunk_count > 8:
        subchunk_count = 8

    raw = bytearray()
    raw.append(subchunk_count)

    for subchunk in range(subchunk_count):
        raw.append(0)
        for local_x in range(16):
            for local_z in range(16):
                for local_y in range(16):
                    absolute_y = subchunk * 16 + local_y
                    if absolute_y >= height:
                        raw.append(0)
                        continue
                    world_x = chunk_x * 16 + local_x
                    world_z = chunk_z * 16 + local_z
                    block_id = flat_map.get((world_x, absolute_y, world_z), (0, 0))
                    raw.append(block_id[0] & 0xFF)

        nibble_array = bytearray(0x800)
        for local_y in range(16):
            for local_z in range(16):
                for local_x in range(16):
                    absolute_y = subchunk * 16 + local_y
                    world_x = chunk_x * 16 + local_x
                    world_z = chunk_z * 16 + local_z
                    if absolute_y >= height:
                        meta = 0
                    else:
                        meta = flat_map.get((world_x, absolute_y, world_z), (0, 0))[1]
                    index = local_y * 256 + local_z * 16 + local_x
                    byte_index = index >> 1
                    if index & 1:
                        nibble_array[byte_index] = (
                            nibble_array[byte_index] & 0x0F
                        ) | ((meta & 0x0F) << 4)
                    else:
                        nibble_array[byte_index] = (
                            nibble_array[byte_index] & 0xF0
                        ) | (meta & 0x0F)
        raw.extend(nibble_array)
        raw.extend(b"\x00" * 0x1000)

    raw.extend(b"\x00" * 0x200)
    raw.extend(b"\x00" * 256)
    return bytes(raw)


def _build_chunk_bytes(
    flat_map: dict[tuple[int, int, int], tuple[int, int]],
    chunk_x: int,
    chunk_z: int,
    dimension: int,
    parameters: tuple[int, int],
    unknown_fields: tuple[int, int, int],
    height: int,
) -> bytes:
    block_data = _build_block_data(flat_map, chunk_x, chunk_z, height)
    compressed = zlib.compress(block_data)

    header = bytearray(0x6C)
    struct.pack_into("<I", header, 0, _encode_position(chunk_x, chunk_z, dimension))
    struct.pack_into("<bb", header, 4, *parameters)
    struct.pack_into("<HHH", header, 6, *unknown_fields)
    struct.pack_into("<iiii", header, 12, 0, 4 + len(header), len(compressed), len(block_data))
    for index in range(1, 6):
        struct.pack_into("<iiii", header, 12 + index * 16, -1, -1, 0, 0)

    subfile = bytearray()
    subfile.extend(struct.pack("<I", MAGIC_CDB))
    subfile.extend(header)
    subfile.extend(compressed)
    return bytes(subfile)


@dataclass
class CDBWriter:
    path: Path

    def __post_init__(self) -> None:
        self._handle = self.path.open("r+b")
        file_header = self._handle.read(FILE_HEADER_STRUCT.size)
        if len(file_header) != FILE_HEADER_STRUCT.size:
            raise ValueError(f"{self.path} is not a valid CDB file")
        (_s0, _s1, self.subfile_count, _footer_size, self.subfile_size, _unk0) = FILE_HEADER_STRUCT.unpack(
            file_header
        )

    def write_subfile(self, index: int, data: bytes) -> None:
        if index < 0 or index >= self.subfile_count:
            raise IndexError("subfile index out of range")
        if len(data) > self.subfile_size:
            raise ValueError("chunk data exceeds slot capacity")
        offset = FILE_HEADER_STRUCT.size + index * self.subfile_size
        self._handle.seek(offset)
        self._handle.write(data)
        remaining = self.subfile_size - len(data)
        if remaining > 0:
            self._handle.write(b"\x00" * remaining)

    def close(self) -> None:
        self._handle.close()


def _iter_index_entries(world: World) -> Iterator[tuple[tuple[int, int, int], int, int, object]]:
    for entry in world.index.entries:
        position = parse_position(entry.position)
        yield position, entry.slot, entry.subfile, entry


def convert_java(world_3ds: Path, java_world: Path, delete_out: bool) -> None:
    world_3ds = Path(world_3ds)
    java_world = Path(java_world)

    if not world_3ds.exists():
        raise FileNotFoundError(f"3DS world path {world_3ds} does not exist")
    if not (world_3ds / "db" / "cdb").exists():
        raise FileNotFoundError("3DS world does not contain db/cdb directory")
    if not java_world.exists():
        raise FileNotFoundError(f"Java world path {java_world} does not exist")

    inverse_block_map = _load_inverse_block_map()
    region_cache = RegionCache(java_world)
    world = World(world_3ds)

    writers: dict[int, CDBWriter] = {}
    missing_blocks: set[str] = set()

    try:
        for position, slot, subfile, index_entry in _iter_index_entries(world):
            entry = world.entries.get(position)
            if entry is None:
                continue

            chunk_x, chunk_z, dimension = position
            chunk = region_cache.get_chunk(chunk_x, chunk_z, dimension)
            if chunk is None:
                flat_map: dict[tuple[int, int, int], tuple[int, int]] = {}
                chunk_missing: set[str] = set()
            else:
                flat_map, chunk_missing = _build_flat_map(
                    chunk, chunk_x, chunk_z, inverse_block_map
                )
            missing_blocks.update(chunk_missing)

            parameters = (
                index_entry.parameters.unknown0,
                index_entry.parameters.unknown1,
            )
            chunk_header = entry.chunk._header
            unknown_fields = (chunk_header.unknown0, chunk_header.unknown1, chunk_header.unknown2)

            chunk_bytes = _build_chunk_bytes(
                flat_map,
                chunk_x,
                chunk_z,
                dimension,
                parameters,
                unknown_fields,
                height=128,
            )

            writer = writers.get(slot)
            if writer is None:
                cdb_path = world.cdb.get_file(slot)
                writer = CDBWriter(Path(cdb_path))
                writers[slot] = writer
            writer.write_subfile(subfile, chunk_bytes)
    finally:
        for writer in writers.values():
            writer.close()

    if missing_blocks:
        logger.warning(
            "Missing block mappings for %d states; these were written as air.",
            len(missing_blocks),
        )
