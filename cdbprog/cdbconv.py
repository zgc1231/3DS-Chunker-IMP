#!/usr/bin/env python3
import os
import json
import zlib
import struct
import logging
import traceback
import re
from pathlib import Path
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
import nbtlib

# --- Logging ---
LOG_PATH = Path(__file__).with_name("cdb_inserter.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# --- Constants ---
# templates now use the same magic as VDB subfiles: 0xABCDEF99
MAGIC_CDB = 0xABCDEF99

# maximum allowed cdb file size before duplicating (as you specified)
MAX_CDB_SIZE = 1310740  # 1,310,740 bytes

# --- Globals ---
block_map = {}
inverse_block_map = {}

# --- Utilities ---
def format_block_name(name, props):
    """Canonical formatting used by the NBT palette code."""
    if props:
        return f"{name}[{','.join(f'{k}={v}' for k, v in sorted(props.items()))}]"
    return name

def normalize_block_value(v):
    """
    Accepts:
      - "minecraft:stone"
      - "minecraft:grass_block[snowy=false]"
      - {"Name":"minecraft:stone", "Properties": {"foo":"bar"}}
    Returns: (name_str, props_dict)
    """
    # dict form
    if isinstance(v, dict):
        name = str(v.get("Name", "minecraft:air"))
        props = v.get("Properties", {}) or {}
        props = {str(k): str(v) for k, v in props.items()}
        return name, props

    # string form
    if isinstance(v, str):
        s = v.strip()
        if "[" in s and s.endswith("]"):
            name, rest = s.split("[", 1)
            props_str = rest[:-1]  # drop trailing ]
            props = {}
            if props_str:
                for part in props_str.split(","):
                    if "=" in part:
                        k, vv = part.split("=", 1)
                        props[k.strip()] = vv.strip()
                    else:
                        props[part.strip()] = ""
            return name.strip(), props
        else:
            return s, {}

    # fallback
    return (str(v), {})

def get_block_id(name):
    """returns (rid, meta) tuple, default to air (0,0)"""
    return inverse_block_map.get(name, (0, 0))

def encode_position(x: int, z: int, dim: int) -> int:
    # encode signed 14-bit x,z with 4-bit dim into u32
    if x < 0:
        x += 1 << 14
    if z < 0:
        z += 1 << 14
    x &= 0x3FFF
    z &= 0x3FFF
    dim &= 0xF
    return (dim << 28) | (z << 14) | x

class CDBPosition:
    def __init__(self, val: int):
        self.x = val & 0x3FFF
        self.z = (val >> 14) & 0x3FFF
        self.dimension = (val >> 28) & 0xF

    def to_signed(self):
        signed_size = 1 << 13
        unsigned_size = 1 << 14
        if self.x >= signed_size:
            self.x -= unsigned_size
        if self.z >= signed_size:
            self.z -= unsigned_size

    def parse_position(self):
        self.to_signed()
        return (self.x, self.z, self.dimension)

# --- File Handlers ---
def read_file_header(f):
    # <HHIIII> => 2 + 2 + 4*4 = 20 bytes
    hdr_fmt = "<HHIIII"
    size = struct.calcsize(hdr_fmt)
    data = f.read(size)
    if len(data) != size:
        raise ValueError("Bad FileHeader")
    return struct.unpack(hdr_fmt, data)

def read_first_chunk_position(cdb_path: str) -> tuple[int, int, int]:
    """read the first chunk position (x,z,dim) from first subfile's chunk header"""
    with open(cdb_path, "rb") as f:
        hdr_fmt = "<HHIIII"
        hdr_size = struct.calcsize(hdr_fmt)
        f.seek(0)
        hdr = f.read(hdr_size)
        if len(hdr) != hdr_size:
            raise ValueError("File too small to read header")
        # first subfile starts immediately after header; subfile header magic is 4 bytes
        first_chunk_header_offset = hdr_size + 4
        f.seek(first_chunk_header_offset)
        pos_bytes = f.read(4)
        if len(pos_bytes) != 4:
            raise ValueError("Cannot read first chunk position")
        position_val = struct.unpack("<I", pos_bytes)[0]
        pos = CDBPosition(position_val)
        return pos.parse_position()

# --- Chunk Writer ---
def write_chunk_to_subfile(f, offset, chunk_x, chunk_z, flat_map, size, dim=0):
    """
    Write a single chunk into the subfile at `offset`.
    flat_map: mapping (wx, y, wz) -> (rid, meta)
    size: [width, height, length] of structure
    offset: absolute position of the subfile start in the cdb file (subfile header magic goes here)
    """
    width, height, length = size

    # Build uncompressed BlockData
    raw = bytearray()

    # Number of 16-block-high subchunks (one byte)
    subchunk_count = (height + 15) // 16
    if subchunk_count > 255:
        raise ValueError("Too many subchunks")
    raw.extend(struct.pack("<B", subchunk_count))

    # For each subchunk, write Subchunk:
    # Subchunk struct: uint8 constant0 (0x0), blocks[16][16][16] (4096 bytes),
    # blockData (16*16*16/2 = 2048 bytes), unknownBlockData (4096 bytes)
    for subchunk in range(subchunk_count):
        # constant0
        raw.extend(b"\x00")

        # blocks: iterate x=0..15,z=0..15,sub_y=0..15 -> local order
        for x in range(16):
            for z in range(16):
                for sub_y in range(16):
                    actual_y = subchunk * 16 + sub_y
                    if actual_y >= height:
                        raw.append(0)
                    else:
                        wx = chunk_x * 16 + x
                        wz = chunk_z * 16 + z
                        val = flat_map.get((wx, actual_y, wz), (0, 0))
                        raw.append(int(val[0]) & 0xFF)

        # blockData (nibbles packed into 2048 bytes)
        nibbles = bytearray(0x800)  # 2048 bytes
        for sub_y in range(16):
            for z in range(16):
                for x in range(16):
                    actual_y = subchunk * 16 + sub_y
                    if actual_y >= height:
                        nibble_value = 0
                    else:
                        wx = chunk_x * 16 + x
                        wz = chunk_z * 16 + z
                        nibble_value = int(flat_map.get((wx, actual_y, wz), (0, 0))[1]) & 0x0F
                    local_index = sub_y * 256 + z * 16 + x
                    byte_i = local_index >> 1
                    if (local_index & 1) == 0:
                        # even -> lower nibble
                        nibbles[byte_i] = (nibbles[byte_i] & 0xF0) | nibble_value
                    else:
                        # odd -> upper nibble
                        nibbles[byte_i] = (nibbles[byte_i] & 0x0F) | (nibble_value << 4)
        raw.extend(nibbles)

        # unknownBlockData (4096 bytes, typically zero)
        raw.extend(b"\x00" * 0x1000)

    # unknown0 (uint16[16][16]) -> 512 bytes (16*16*2)
    raw.extend(b"\x00" * 0x200)

    # Biomes (16x16 = 256)
    biomes = bytearray(256)
    # pick biome 0 for all
    for x in range(16):
        for z in range(16):
            biomes[x * 16 + z] = 0
    raw.extend(biomes)

    # compress the blockdata
    comp = zlib.compress(bytes(raw))

    # Build chunk header (Position (4), parameters (2), three uint16s (6) ) then 6 ChunkSection (6*16 = 96)
    hdr = bytearray()
    hdr += struct.pack("<I", encode_position(chunk_x, chunk_z, dim))
    hdr += struct.pack("<bb", 1, 0)  # parameters: unknown0=1, unknown1=0
    hdr += struct.pack("<HHH", 0, 0, 0)  # three unknown uint16s

    # section_offset is the offset from the *chunk header* to the compressed data (do NOT add +4)
    # position_within_chunk = size of chunk header (so far) + size of 6 ChunkSection entries
    position_within_chunk = 4 + len(hdr) + (6 * 16)
    # Per the struct comment, the stored ChunkSection.position = position_within_chunk + 0xC
    stored_section_position = position_within_chunk + 0xC
    hdr += struct.pack("<iiii", 0, stored_section_position, len(comp), len(raw))
    # remaining 5 sections empty: index=-1, position=-1, compressedSize=0, decompressedSize=0
    for _ in range(5):
        hdr += struct.pack("<iiii", -1, -1, 0, 0)

    # Write into the subfile at the given offset (subfile start)
    f.seek(offset)
    # Write subfile magic
    f.write(struct.pack("<I", MAGIC_CDB))
    # write chunk header (chunk header begins at offset+4)
    f.write(hdr)
    # write compressed block data immediately after header and 6*ChunkSection entries
    f.write(comp)

# --- Index Update helpers ---
def append_index_entry(cdb_path, pos_u32, slot_number, subfile_idx):
    """
    Append a single index entry for a single subfile to index.cdb next to the cdb files.
    Fields follow the original code's structure:
    "<I H H H H b b H"
    """
    index_path = Path(cdb_path).with_name("index.cdb")
    # ensure directory exists
    index_path.parent.mkdir(parents=True, exist_ok=True)

    entry = struct.pack(
        "<I H H H H b b H",
        pos_u32,                 # u32 position
        slot_number & 0xFFFF,    # slot
        subfile_idx & 0xFFFF,    # subfile index within slot
        0x20FF,                  # constant0
        0x000A,                  # constant1
        0x01,                    # param0 (unknown)
        0x00,                    # param1 (unknown)
        0x8000,                  # constant2
    )

    # append to index
    with open(index_path, "ab") as f:
        f.write(entry)

def duplicate_slt_file(src_path: Path, dest_num: int):
    """Copy src_path to same dir as slt{dest_num}.cdb and return new Path."""
    dest = src_path.parent / f"slt{dest_num}.cdb"
    shutil.copy2(src_path, dest)
    logger.info("Duplicated %s -> %s", src_path, dest)
    return dest

# --- Insert Structure (modified to support splitting across multiple slt files, always duplicating original) ---
def insert_structure(cdb_path, flat_map, size, start_x, start_z, dim=0, slot_number=0, slot_var=None):
    """
    Insert the structure into potentially multiple sltN.cdb files.
    - cdb_path: initial/suggested cdb path (string) — this file will be the duplication SOURCE
    - slot_number: initial slot number (int)
    - slot_var: optional tkinter StringVar to update GUI slot when we duplicate
    """
    current_cdb_path = Path(cdb_path)
    parent_dir = current_cdb_path.parent

    # Parse original slot number from filename sltN.cdb if possible
    slt_re = re.compile(r"^slt(\d+)\.cdb$", re.IGNORECASE)
    m = slt_re.match(current_cdb_path.name)
    try:
        orig_slot_num = int(m.group(1)) if m else int(slot_number or 0)
    except Exception:
        orig_slot_num = int(slot_number or 0)

    # duplicate counter: 0 means using original file; when we need a duplicate, we set dup_count=1 -> slt(orig+1)
    dup_count = 0
    orig_path = Path(cdb_path)  # always duplicate this as source

    # open initial file
    f = open(current_cdb_path, "r+b")
    try:
        s0, s1, count, footer, subsize, unk0 = read_file_header(f)
    except Exception:
        f.close()
        raise

    chunks_x = (size[0] + 15) // 16
    chunks_z = (size[2] + 15) // 16

    # local index within the current slt file (resets to 0 when duplicating)
    local_idx = 0

    for cz in range(chunks_z):
        for cx in range(chunks_x):
            # check if the next subfile write in the current file would exceed max size
            offset = 20 + local_idx * subsize
            # if we would exceed the target file size, duplicate the original slt and switch
            if offset + subsize > MAX_CDB_SIZE:
                # close current file handle before duplicating
                try:
                    f.close()
                except Exception:
                    pass

                # increment duplicate count and create new filename slt(orig_slot_num + dup_count)
                dup_count += 1
                new_num = orig_slot_num + dup_count
                new_path = duplicate_slt_file(orig_path, new_num)

                # switch to new file and set slot_number to match the destination file number
                current_cdb_path = new_path
                slot_number = new_num

                # update GUI slot_var if provided (set to new slot_number)
                if slot_var is not None:
                    try:
                        slot_var.set(str(slot_number))
                    except Exception:
                        pass

                # open new file handle and re-read header (subsize likely same)
                f = open(current_cdb_path, "r+b")
                try:
                    s0, s1, count, footer, subsize, unk0 = read_file_header(f)
                except Exception:
                    f.close()
                    raise

                local_idx = 0
                offset = 20 + local_idx * subsize

            # write the chunk into the currently-open cdb file at local_idx
            write_chunk_to_subfile(
                f, offset, start_x + cx, start_z + cz, flat_map, size, dim
            )

            # append an index entry for this single chunk with the current slot and subfile index
            pos_u32 = encode_position(start_x + cx, start_z + cz, dim)
            append_index_entry(str(current_cdb_path), pos_u32, slot_number, local_idx)

            logger.debug("Wrote chunk (%d,%d) into %s at subfile idx %d (slot %d)",
                         start_x + cx, start_z + cz, current_cdb_path, local_idx, slot_number)

            # move to next local index within this cdb file
            local_idx += 1

    # close final file handle
    try:
        f.close()
    except Exception:
        pass

    logger.info("Insertion complete across slt files.")

# --- NBT parsing + palette validation ---
def parse_nbt_structure(nbt_path):
    try:
        nbt_file = nbtlib.load(nbt_path)
        blocks = nbt_file.get("blocks", [])
        size = [int(d) for d in nbt_file.get("size", [0, 0, 0])]
        palette = nbt_file.get("palette", [])
    except Exception as e:
        logger.error(f"Failed NBT parse: {e}\n{traceback.format_exc()}")
        raise ValueError("Invalid NBT")

    # Build state→name map (palmap: state index -> canonical name)
    palmap = {}
    for i, entry in enumerate(palette):
        name = str(entry.get("Name", "minecraft:air"))
        props = entry.get("Properties", {}) or {}
        palmap[i] = format_block_name(name, props)

    # Use a sparse mapping for blocks (non-flat)
    width, height, length = size
    non_flat = {}

    # Scatter each block
    for b in blocks:
        x, y, z = int(b["pos"][0]), int(b["pos"][1]), int(b["pos"][2])
        state = int(b.get("state", 0))
        name = palmap.get(state, "minecraft:air")
        rid, meta = get_block_id(name)
        non_flat[(x, y, z)] = (rid, meta)

    return non_flat, size, palmap

def parse_schematic(schematic_path):
    """
    Parse an old-style .schematic (numeric Blocks/Data). Returns (non_flat, size, palmap)
    non_flat is a mapping (x,y,z) -> (rid, meta) where rid and meta are numeric (ints).
    size is [Width,Height,Length]. palmap is empty dict (no namespace palette present).
    """
    try:
        nbt_file = nbtlib.load(schematic_path)
    except Exception as e:
        logger.error("Failed to load schematic: %s", e)
        raise ValueError("Invalid schematic")

    # canonical schematic tags
    try:
        width = int(nbt_file["Width"])
        height = int(nbt_file["Height"])
        length = int(nbt_file["Length"])
    except Exception as e:
        logger.error("Schematic missing size fields: %s", e)
        raise ValueError("Invalid schematic (missing size)")

    # Blocks: a byte array with one entry per block
    blocks_raw = bytes(nbt_file.get("Blocks", b""))
    if not blocks_raw or len(blocks_raw) != width * height * length:
        # allow Data length check later; if blocks length mismatched raise
        if len(blocks_raw) != width * height * length:
            logger.warning("Blocks length (%d) != expected (%d). Proceeding but results may be wrong.",
                           len(blocks_raw), width * height * length)

    # Data: typically one byte per block (0-15). If missing, default to zero.
    data_raw = bytes(nbt_file.get("Data", b""))
    if not data_raw or len(data_raw) != len(blocks_raw):
        # If Data missing or different length, fill with zeros (safe fallback)
        data_raw = bytes([0]) * len(blocks_raw)

    # AddBlocks: optional extension for IDs >255. Two common representations:
    #  - same-length byte array (one extra high byte per block)
    #  - packed nibbles (length ~= ceil(len(blocks)/2))
    add_raw = nbt_file.get("AddBlocks")
    add_bytes = bytes(add_raw) if add_raw is not None else None

    # Build full ids list
    full_ids = []
    blen = len(blocks_raw)
    if add_bytes:
        # if add_bytes same length, simple combine
        if len(add_bytes) == blen:
            for i in range(blen):
                full_ids.append((add_bytes[i] << 8) | blocks_raw[i])
        else:
            # try unpacking packed nibbles: each byte contains two 4-bit highs for two blocks
            # fallback safe handling: extract nibble per block if possible
            for i in range(blen):
                ai = i >> 1
                if ai < len(add_bytes):
                    b = add_bytes[ai]
                    if (i & 1) == 0:
                        high = b & 0x0F
                    else:
                        high = (b >> 4) & 0x0F
                    full_ids.append((high << 8) | blocks_raw[i])
                else:
                    # no add info -> just low byte
                    full_ids.append(blocks_raw[i])
    else:
        # no AddBlocks
        full_ids = list(blocks_raw)

    # Build mapping: schematic order is typically x + (z * Width) + (y * Width * Length)
    non_flat = {}
    idx = 0
    for y in range(height):
        for z in range(length):
            for x in range(width):
                if idx < len(full_ids):
                    rid = int(full_ids[idx])
                else:
                    rid = 0
                if idx < len(data_raw):
                    meta = int(data_raw[idx]) & 0x0F
                else:
                    meta = 0
                non_flat[(x, y, z)] = (rid, meta)
                idx += 1

    size = [width, height, length]
    palmap = {}  # schematics use numeric ids; no namespace palette
    return non_flat, size, palmap

def validate_palette_against_json(palmap):
    missing = []
    for i, key in palmap.items() if isinstance(palmap, dict) else enumerate(palmap):
        # If palmap is dict: i=state, key=canonical name; else key=name
        canonical = key if isinstance(key, str) else palmap[i]
        if canonical not in inverse_block_map:
            missing.append(canonical)
    logger.info("Palette validation: %d missing entries", len(missing))
    if missing:
        for k in missing[:50]:
            logger.warning("Missing palette mapping: %s", k)
    return missing

# --- GUI / JSON loader ---
def browse_json():
    """Load blocks.json and populate block_map & inverse_block_map robustly."""
    p = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
    if not p:
        return
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        block_map.clear()
        inverse_block_map.clear()

        raw_blocks = data.get("blocks", {})
        if not isinstance(raw_blocks, dict):
            messagebox.showerror("Error", "blocks entry missing or not an object")
            return

        for k, v in raw_blocks.items():
            # parse key like "1:0" -> (rid,meta)
            try:
                rid_meta = tuple(int(x) for x in k.split(":"))
                if len(rid_meta) != 2:
                    raise ValueError
            except Exception:
                logger.warning("Skipping invalid block key in JSON: %r", k)
                continue

            # normalize the value to canonical name + props
            name, props = normalize_block_value(v)

            # store original mapping (stringified)
            block_map[k] = v

            # canonical key used by palette formatting
            canonical = format_block_name(name, props)

            # populate inverse map so format_block_name(name, props) -> (rid, meta)
            inverse_block_map[canonical] = rid_meta

        messagebox.showinfo("OK", f"Loaded block map ({len(inverse_block_map)} entries)")
        logger.debug("Loaded blocks.json: %s entries", len(inverse_block_map))

    except Exception as e:
        logger.exception("Failed loading JSON")
        messagebox.showerror("Error", f"Failed loading JSON: {e}")

# --- GUI app ---
def run_gui():
    root = tk.Tk()
    root.title("NBT→CDB Inserter")

    cdb_var = tk.StringVar()
    nbt_var = tk.StringVar()
    json_var = tk.StringVar()
    x_var = tk.StringVar()
    z_var = tk.StringVar()
    dim_var = tk.StringVar()
    slot_var = tk.StringVar()

    tk.Label(root, text="CDB:").grid(row=0, column=0, sticky="w")
    tk.Entry(root, textvariable=cdb_var, width=50).grid(row=0, column=1)

    def pick_cdb():
        path = filedialog.askopenfilename(filetypes=[("CDB", "*.cdb")])
        if not path:
            return
        cdb_var.set(path)
        # reload block map if JSON already selected
        if json_var.get():
            browse_json()
        try:
            x, z, d = read_first_chunk_position(path)
            x_var.set(str(x))
            z_var.set(str(z))
            dim_var.set(str(d))
        except Exception:
            logger.debug("Couldn't read first chunk position from CDB")

    tk.Button(root, text="…", command=pick_cdb).grid(row=0, column=2)

    tk.Label(root, text="NBT / Schematic:").grid(row=1, column=0, sticky="w")
    tk.Entry(root, textvariable=nbt_var, width=50).grid(row=1, column=1)
    tk.Button(
        root,
        text="…",
        command=lambda: nbt_var.set(filedialog.askopenfilename(
            filetypes=[("NBT/Schematic", "*.nbt *.schematic"), ("NBT","*.nbt"), ("Schematic","*.schematic")]
        ))
    ).grid(row=1, column=2)

    tk.Label(root, text="blocks.json:").grid(row=2, column=0, sticky="w")
    tk.Entry(root, textvariable=json_var, width=50).grid(row=2, column=1)
    tk.Button(root, text="…", command=lambda: [
        json_var.set(filedialog.askopenfilename(filetypes=[("JSON", "*.json")])),
        browse_json()
    ]).grid(row=2, column=2)

    tk.Label(root, text="Start chunk X:").grid(row=3, column=0, sticky="w")
    tk.Entry(root, textvariable=x_var).grid(row=3, column=1)
    tk.Label(root, text="Start chunk Z:").grid(row=4, column=0, sticky="w")
    tk.Entry(root, textvariable=z_var).grid(row=4, column=1)
    tk.Label(root, text="Dimension:").grid(row=5, column=0, sticky="w")
    tk.Entry(root, textvariable=dim_var).grid(row=5, column=1)
    tk.Label(root, text="Slot Number:").grid(row=6, column=0, sticky="w")
    tk.Entry(root, textvariable=slot_var).grid(row=6, column=1)

    def on_insert():
        try:
            if not inverse_block_map:
                logger.debug("No blocks.json loaded; prompting user")
                messagebox.showwarning("Warning", "No blocks.json loaded — block ids will default to air (0:0).")

            input_path = nbt_var.get().strip()
            if not input_path:
                messagebox.showerror("Error", "No NBT / schematic file selected.")
                return

            # detect .schematic (old numeric) vs namespace nbt
            if input_path.lower().endswith(".schematic"):
                flat_map, size, palmap = parse_schematic(input_path)
                missing = []  # numeric schematic: skip palette validation
            else:
                flat_map, size, palmap = parse_nbt_structure(input_path)
                missing = validate_palette_against_json(palmap)

            if missing:
                # warn user but continue (air fallback for missing)
                messagebox.showwarning("Palette mismatch", f"{len(missing)} palette entries missing in blocks.json (see log). They will become air.")
            dim = int(dim_var.get()) if dim_var.get() else 0
            slot = int(slot_var.get()) if slot_var.get() else 0
            # pass slot_var so insert_structure can increment the GUI's slot when duplicating
            insert_structure(cdb_var.get(), flat_map, size, int(x_var.get()), int(z_var.get()), dim=dim, slot_number=slot, slot_var=slot_var)
            messagebox.showinfo("Done", "Structure inserted")
        except Exception as e:
            logger.error("Insert error:\n" + traceback.format_exc())
            messagebox.showerror("Error", str(e))

    tk.Button(root, text="Insert", command=on_insert).grid(row=7, column=1, pady=8)
    root.mainloop()

if __name__ == "__main__":
    run_gui()
