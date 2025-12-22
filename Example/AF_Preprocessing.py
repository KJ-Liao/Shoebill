#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
- Identify & remove terminal His-tags (HHHHH)
- Trim AlphaFold PDBs accordingly (0-based span), then renumber residues & atom serials from 1 when writing
- Parse predicted_aligned_error.json, build PAE arrays, save as .npy

Folder layout:
  TE_Sequence.fasta
  AF_Result/
    <targetFolder_containing_sequence_id>/
      unrelaxed_rank_001.pdb
      ...
      predicted_aligned_error.json

Outputs:
  Processed_AF_Result/<targetFolder>/
    processed unrelaxed_rank_00X.pdb
    PAE_Img.npy
"""

import os
import sys
import json
import glob
import shutil
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from Bio import SeqIO
from Bio.PDB import PDBParser, PDBIO, Atom
from Bio.PDB.StructureBuilder import StructureBuilder

# -----------------------------
# 1) FASTA parsing & His-tag identification
# -----------------------------

@dataclass
class SeqRecordInfo:
    Header:     str
    Sequence:   str
    HHH:        int     # 1 if trimmed due to His-tag, else 0
    NewSeq:     str
    Diff:       int
    PreCut:     int
    PostCut:    int

def _find_his_positions(seq: str, motif="HHHHH") -> List[int]:
    """Return indices of all 'motif' occurrences in seq."""
    positions = []
    start = 0
    while True:
        idx = seq.find(motif, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + 1
    return positions

def process_fasta(fasta_path: str) -> List[SeqRecordInfo]:
    out = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        header = rec.description
        seq = str(rec.seq).upper()
        L = len(seq)
        low_B = int(round(L * 0.10))
        top_B = int(round(L * 0.90))

        idxs = _find_his_positions(seq, "HHHHH") 
        HHH = 0
        NewSeq = seq
        start0 = 0
        end0_excl = L

        if idxs:
            n_terminal_hits = [i for i in idxs if i < low_B]
            c_terminal_hits = [i for i in idxs if i >= top_B]
            if n_terminal_hits:
                last_idx = max(n_terminal_hits)
                HHH = 1
                start0 = last_idx + 5
                end0_excl = L
                NewSeq = seq[start0:end0_excl]
            elif c_terminal_hits:
                first_idx = min(c_terminal_hits)
                HHH = 1
                start0 = 0
                end0_excl = first_idx
                NewSeq = seq[start0:end0_excl]
            else:
                # His in the middle — keep original
                HHH = 0
                start0, end0_excl = 0, L
                NewSeq = seq
        else:
            # No His-tag
            HHH = 0
            start0, end0_excl = 0, L
            NewSeq = seq

        Diff = len(NewSeq) - L
        out.append(SeqRecordInfo(header, seq, HHH, NewSeq, Diff, start0, end0_excl))
    return out

# -----------------------------
# 2) PDB trimming & renumbering
# -----------------------------

def trim_and_renumber_pdb_by_span(pdb_in: str, pdb_out: str, start0: int, end0_excl: int) -> None:
    """
    Output PDB is fully renumbered: residue IDs 1..N and atom serials 1..M.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("AF", pdb_in)

    sb = StructureBuilder()
    sb.init_structure("TRIM")
    sb.init_model(0)

    atom_serial = 1
    for model in structure:
        for chain in model:
            new_chain_id = chain.id
            sb.init_chain(new_chain_id)
            new_resseq = 0

            for residue in chain:
                hetflag, resseq, icode = residue.id  # (' ', 123, ' ')
                if hetflag != " ":
                    continue
                i0 = resseq - 1
                if not (start0 <= i0 < end0_excl):
                    continue

                new_resseq += 1
                sb.init_seg("    ")
                sb.init_residue(residue.resname, " ", new_resseq, " ")

                for atom in residue:
                    new_atom = Atom.Atom(
                        atom.name,
                        atom.coord.copy(),
                        atom.bfactor,
                        atom.occupancy,
                        atom.altloc,
                        atom.fullname,
                        atom_serial,
                        element=atom.element
                    )
                    sb.structure[0][new_chain_id][(" ", new_resseq, " ")].add(new_atom)
                    atom_serial += 1

    io = PDBIO()
    io.set_structure(sb.get_structure())
    os.makedirs(os.path.dirname(pdb_out), exist_ok=True)
    io.save(pdb_out)

# -----------------------------
# 3) PAE handling
# -----------------------------

def load_pae_json(json_path: str) -> np.ndarray:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pae = np.array(data["predicted_aligned_error"], dtype=float)
    # Keep parity with your MATLAB behavior where the final matrix was transposed:
    return pae.T

def trim_pae(pae: np.ndarray, start0: int, end0_excl: int) -> np.ndarray:
    start0 = max(0, start0)
    end0_excl = min(pae.shape[0], end0_excl)
    if start0 >= end0_excl:
        return pae[0:0, 0:0]
    return pae[start0:end0_excl, start0:end0_excl]

# -----------------------------
# 4) AF folder discovery & main loop
# -----------------------------

def sanitize_id_from_header(header: str) -> str:
    h = header.lstrip(">")
    return h.split()[0]

def find_target_folder(base: str, id_key: str) -> Optional[str]:
    if not os.path.isdir(base):
        return None
    candidates = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and id_key in d]
    return candidates[0] if candidates else None

def process_one_target(af_path: str, out_path: str, target_root: str,
                       start0: int, end0_excl: int, do_trim: bool) -> None:
    """
    Handle up to 5 PDBs: unrelaxed_rank_001..005, and the PAE JSON.
    """
    out_dir = os.path.join(out_path, target_root)
    os.makedirs(out_dir, exist_ok=True)

    # Collect PDBs from the provided af_path (not hard-coded)
    pdbs = sorted(glob.glob(os.path.join(af_path, target_root, "*unrelaxed_rank_*.pdb")))

    if do_trim:
        for pdb_in in pdbs:
            pdb_out = os.path.join(out_dir, os.path.basename(pdb_in))
            trim_and_renumber_pdb_by_span(pdb_in, pdb_out, start0, end0_excl)
    else:
        for pdb_in in pdbs:
            shutil.copy2(pdb_in, out_dir)

    # PAE (always save; trim if do_trim)
    pae_jsons = glob.glob(os.path.join(af_path, target_root, "*predicted_aligned_error*.json"))
    if pae_jsons:
        pae = load_pae_json(pae_jsons[0])
        if do_trim:
            pae = trim_pae(pae, start0, end0_excl)
        out_file = os.path.join(out_dir, "PAE_Img.csv")
        np.savetxt(out_file, pae, delimiter=",")
        
# -----------------------------
# main
# -----------------------------

def main():
    if len(sys.argv) != 4:
        print("Usage: python XXX.py <fasta_path> <af_path> <out_path>")
        sys.exit(1)

    fasta_path, af_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    seq_infos = process_fasta(fasta_path)
    os.makedirs(out_path, exist_ok=True)

    for info in seq_infos:
        id_key = sanitize_id_from_header(info.Header)
        target_folder = find_target_folder(af_path, id_key)
        if not target_folder:
            print(f"[WARN] No AF folder found for '{id_key}' in '{af_path}'. Skipping.")
            continue
        do_trim = (info.Diff != 0)
        process_one_target(af_path, out_path, target_folder, info.PreCut, info.PostCut, do_trim)

if __name__ == "__main__":
    main()
