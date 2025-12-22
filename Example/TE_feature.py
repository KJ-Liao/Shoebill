#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run:
  python TE_feature.py TE_Sequence.fasta Processed_AF_Result TE_feature.csv --bin ./bin

Requires:
  - Python: biopython, numpy, pandas, scipy
  - External tools in --bin:
      EDTSurf, mkdssp, MakeShape, Shape2Zernike, korpe, korp6Dv1.bin
"""

import os
import re
import math
import argparse
import subprocess
from collections import Counter
from typing import List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial.distance import cdist, pdist, squareform

from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.SeqUtils import molecular_weight

# -----------------------------
# Index libraries
# -----------------------------
RES_LIST = np.array(list("ARNDCQEGHILKMFPSTWYV"))

PKA_EMBOSS = {
    "N_term": 8.6,  "C_term": 3.6, "C": 8.5, "D": 3.9, "E": 4.1, "Y": 10.1, "H": 6.5, "K": 10.8, "R": 12.5
}

MASA = np.array([121, 265, 187, 187, 148, 214, 214,  97, 216, 195, 191, 230, 203, 228, 154, 143, 163, 264, 255, 165], dtype=float)

HP_IDX = np.array([ 1.8, -4.5, -3.5, -3.5,  2.5, -3.5, -3.5, -0.4, -3.2,  4.5,
                    3.8, -3.9,  1.9,  2.8, -1.6, -0.8, -0.7, -0.9, -1.3,  4.2], dtype=float)

CG_IDX = np.array([0.0,  1.0,  0.0, -1.0,  0.0,  0.0, -1.0,  0.0,  0.12, 0.0,
                   0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0], dtype=float)

MS_IDX = np.array([ 0.00, -1.88, -1.03, -0.78, -0.85, -1.73, -1.46,  0.00, -0.95, -0.76,
                   -0.71, -1.89, -1.46, -0.62, -0.06, -1.11, -1.08, -0.99, -1.13, -0.43], dtype=float)

FX_IDX = np.array([0.984, 1.008, 1.048, 1.068, 0.906, 1.037, 1.094, 1.031, 0.950, 0.927,
                   0.935, 1.102, 0.952, 0.915, 1.049, 1.046, 0.997, 0.904, 0.929, 0.931], dtype=float)

# -----------------------------
# Helpers
# -----------------------------

def run_cmd(cmd: List[str]) -> Tuple[int, str]:
    """Run external command; return (returncode, stdout+stderr)."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
        return p.returncode, out
    except FileNotFoundError:
        return f"[ERROR] Command not found: {' '.join(cmd)}"

def ensure_executable(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Executable not found: {path}")
    if os.name != "nt" and not os.access(path, os.X_OK):
        raise PermissionError(f"Executable is not +x: {path}")

def aacount_20(seq: str) -> np.ndarray:
    """Count AAs in RES_LIST order (20 standard)."""
    seq = ''.join([c for c in seq if c in RES_LIST])
    counts = np.zeros(20, dtype=float)
    if not seq:
        return counts
    # vectorized counting
    for i, aa in enumerate(RES_LIST):
        counts[i] = seq.count(aa)
    return counts

def read_pdb_ca_coords(pdb_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (coords_Nx3, aa3_list, b_factors_N)."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("M", pdb_path)
    coords = []
    aa3 = []
    bfac = []
    for model in structure:
        # use first model
        for chain in model:
            for res in chain:
                het, resseq, icode = res.id
                if het != " ":
                    continue
                for atom in res:
                    if atom.get_name() == "CA":
                        coords.append(atom.get_coord())
                        aa3.append(res.get_resname())
                        bfac.append(atom.get_bfactor())
        break
    if len(coords) == 0:
        raise ValueError(f"No CA atoms found in {pdb_path}")
    return np.array(coords, dtype=float), np.array(aa3), np.array(bfac, dtype=float)

def aa3_to_aa1(aa3: np.ndarray) -> str:
    three_to_one = {
        'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
        'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
        'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
        'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
    }
    aa1 = []
    for x in aa3:
        aa1.append(three_to_one.get(x.upper(), 'X'))
    return ''.join(aa1).replace('X','')  # drop unknowns for your metrics

def _net_charge(seq: str, pH: float, pka: dict) -> float:
    cnt = Counter(seq)
    
    f_N = 1 / (1 + 10**(pH - pka["N_term"]))
    f_K = 1 / (1 + 10**(pH - pka["K"]))
    f_R = 1 / (1 + 10**(pH - pka["R"]))
    f_H = 1 / (1 + 10**(pH - pka["H"]))
    pos = f_N + cnt["K"] * f_K + cnt["R"] * f_R + cnt["H"] * f_H
    
    f_C = 1 / (1 + 10**(pka["C_term"] - pH))
    f_D = 1 / (1 + 10**(pka["D"] - pH))
    f_E = 1 / (1 + 10**(pka["E"] - pH))
    f_Cys = 1 / (1 + 10**(pka["C"] - pH))
    f_Y = 1 / (1 + 10**(pka["Y"] - pH))
    neg = f_C + cnt["D"] * f_D + cnt["E"] * f_E + cnt["C"] * f_Cys + cnt["Y"] * f_Y
    
    return pos - neg

def isoelectric_point(seq: str, pka: dict = PKA_EMBOSS) -> float:
    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2
        q = _net_charge(seq, mid, pka)
        if abs(q) < 1e-6:
            return mid
        if q > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi)/2

def kabsch_align(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (R,T) to align P->Q (Nx3 each) with Kabsch."""
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    C = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1,1,d])
    R = V @ D @ Wt
    T = Q.mean(axis=0) - P.mean(axis=0) @ R
    return R, T

def compute_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """RMSD after optimal superposition of P (move) onto Q (fixed)."""
    R, T = kabsch_align(P, Q)
    P_aln = P @ R + T
    diff = Q - P_aln
    return float(np.sqrt((diff*diff).sum(axis=1).mean()))

def parse_edtsurf_stdout(stdout: str) -> Tuple[float, float]:
    """Find 'Total area ... and volume ...' line and parse numbers."""
    for line in stdout.splitlines():
        if "Total area" in line and "and volume" in line:
            # Extract floats in that line
            # Example: 'Total area  1234.56 and volume 789.0'
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            if len(nums) >= 2:
                return float(nums[-2]), float(nums[-1])
    raise RuntimeError("Failed to parse EDTSurf total area/volume from output")

def read_ply_vertices_faces(ply_path: str):
    """
    Robust ASCII PLY reader for EDTSurf outputs.
    """
    with open(ply_path, "r") as f:
        # --- read header
        header = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("Invalid PLY: no end_header")
            header.append(line.rstrip("\n"))
            if line.strip() == "end_header":
                break

        n_vert = n_face = None
        for h in header:
            if h.startswith("element vertex"):
                n_vert = int(h.split()[-1])
            elif h.startswith("element face"):
                n_face = int(h.split()[-1])

        if n_vert is None or n_face is None:
            raise RuntimeError("Invalid PLY: missing element counts")

        # --- read vertices (take first 3 floats per line)
        V = np.zeros((n_vert, 3), dtype=float)
        for i in range(n_vert):
            line = f.readline()
            if not line:
                raise RuntimeError("Unexpected EOF in vertex list")
            toks = line.strip().split()
            try:
                V[i, 0] = float(toks[0])
                V[i, 1] = float(toks[1])
                V[i, 2] = float(toks[2])
            except Exception as e:
                raise RuntimeError(f"Bad vertex line {i}: {line!r}") from e

        # --- read faces
        F = np.zeros((n_face, 4), dtype=int)
        for i in range(n_face):
            line = f.readline()
            if not line:
                raise RuntimeError("Unexpected EOF in face list")
            toks = line.strip().split()
            try:
                cnt = int(toks[0])
                i1 = int(toks[1]) if len(toks) > 1 else 0
                i2 = int(toks[2]) if len(toks) > 2 else 0
                i3 = int(toks[3]) if len(toks) > 3 else 0
            except Exception as e:
                raise RuntimeError(f"Bad face line {i}: {line!r}") from e
            F[i] = [cnt, i1, i2, i3]

    return V, F

def quantile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return 0.0
    return float(np.quantile(x, q, method='hazen'))

def write_jerhoud_zm(table: np.ndarray, order: int, out_path: str) -> None:
    """
    table : (K,5) array with columns [n, l, m, rea, ima]
            n,l,m should be integers; rea,ima are floats.
    order : maximum order (same 'Order' you used to generate moments)
    out_path : output .zm file path
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # sanitize types
    tbl = np.asarray(table, dtype=float)
    tbl = np.nan_to_num(tbl, nan=0.0)

    with open(out_path, "w", newline="\n") as f:
        f.write("ZM\n")
        f.write(f"ORTHO {int(order)} COMPLEX\n")
        for row in tbl:
            n, l, m, rea, ima = int(row[0]), int(row[1]), int(row[2]), float(row[3]), float(row[4])
            f.write(f"{n:d} {l:d} {m:d} {rea:.8e} {ima:.8e}\n")

# -----------------------------
# Core per-sequence processing
# -----------------------------

@dataclass
class SeqResult:
    id: str
    feat: np.ndarray

def process_one_sequence(seq_id: str, proc_root: str, bin_path: str) -> SeqResult:
    """
    seq_id: FASTA header (first token)
    proc_root: Processed_AF_Result
    bin_path: folder containing EDTSurf, mkdssp, MakeShape, Shape2Zernike, korpe, korp6Dv1.bin
    """
    target_dir = os.path.join(proc_root, seq_id)
    if not os.path.isdir(target_dir):
        raise FileNotFoundError(f"Target folder not found: {target_dir}")

    # --- Read main PDB (rank_001) & basics
    pdb001 = None
    files = os.listdir(target_dir)
    for name in files:
        if "unrelaxed_rank_001" in name and name.endswith(".pdb"):
            pdb001 = os.path.join(target_dir, name)
            break
    if pdb001 is None:
        raise FileNotFoundError(f"unrelaxed_rank_001*.pdb not found in {target_dir}")

    CA, aa3, bfac = read_pdb_ca_coords(pdb001)
    AF_seq = aa3_to_aa1(aa3)         # model sequence (1-letter)
    L = len(AF_seq)

    # --- Amino acid composition & global properties
    AAC = aacount_20(AF_seq)
    Count_Acd = float(AF_seq.count('D') + AF_seq.count('E'))
    Count_Bas = float(AF_seq.count('K') + AF_seq.count('R') + AF_seq.count('H'))
    Count_Aro = float(AF_seq.count('F') + AF_seq.count('Y') + AF_seq.count('W'))
    Count_Pol = float(AF_seq.count('S') + AF_seq.count('T') + AF_seq.count('N') + AF_seq.count('Q') + AF_seq.count('C') + AF_seq.count('Y'))
    Count_nPol = float(AF_seq.count('A') + AF_seq.count('V') + AF_seq.count('L') + AF_seq.count('I') + AF_seq.count('M') + AF_seq.count('P') + AF_seq.count('G'))

    pI = isoelectric_point(AF_seq)
    Count_AA = np.concatenate([AAC, [Count_Acd, Count_Bas, Count_Aro, Count_Pol, Count_nPol]])
    Ratio_AA = Count_AA / max(L, 1)

    MHCMF = np.array([
        MASA @ AAC, HP_IDX @ AAC, CG_IDX @ AAC, MS_IDX @ AAC, FX_IDX @ AAC
    ], dtype=float)
    AVG_MHCMF = MHCMF / max(L, 1)

    # --- Surface generation with EDTSurf
    edtsurf = os.path.join(bin_path, "EDTSurf")
    ensure_executable(edtsurf)

    # Produce surface (PLY)
    # Output prefix: <seq_id>_surf (.ply produced by EDTSurf)
    surface_ply = os.path.join(target_dir, f"{seq_id}_surf.ply")
    
    status, out = run_cmd([
        edtsurf, "-i", pdb001, "-c", "1", "-h", "2", "-o",
        os.path.join(target_dir, f"{seq_id}_surf")
    ])
    if not os.path.isfile(surface_ply):
        raise RuntimeError(f"Surface calculation (EDTSurf) failed for {seq_id}")

    # Parse area & volume from stdout
    try:
        Surf_area, Volume = parse_edtsurf_stdout(out)
    except Exception as e:
        raise RuntimeError(
            f"Surface calculation (EDTSurf) failed for {seq_id}"
        ) from e

    Vtx, Face = read_ply_vertices_faces(surface_ply)

    # Surface residues (threshold: 1.8 Å CA-vertex)
    D = cdist(CA, Vtx)   # len(CA) x nVert
    Surf_Idx = (D.min(axis=1) < 1.8)
    Surf_Res = ''.join(np.array(list(AF_seq))[Surf_Idx])
    Surf_Coord = CA[Surf_Idx, :]

    # Surface compositions
    Surf_AAC = aacount_20(Surf_Res)
    Surf_Count_Acd = float(Surf_Res.count('D') + Surf_Res.count('E'))
    Surf_Count_Bas = float(Surf_Res.count('K') + Surf_Res.count('R') + Surf_Res.count('H'))
    Surf_Count_Aro = float(Surf_Res.count('F') + Surf_Res.count('Y') + Surf_Res.count('W'))
    Surf_Count_Pol = float(Surf_Res.count('S') + Surf_Res.count('T') + Surf_Res.count('N') + Surf_Res.count('Q') + Surf_Res.count('C') + Surf_Res.count('Y'))
    Surf_Count_nPol = float(Surf_Res.count('A') + Surf_Res.count('V') + Surf_Res.count('L') + Surf_Res.count('I') + Surf_Res.count('M') + Surf_Res.count('P') + Surf_Res.count('G'))

    Surf_Count_AA = np.concatenate([Surf_AAC, [Surf_Count_Acd, Surf_Count_Bas, Surf_Count_Aro, Surf_Count_Pol, Surf_Count_nPol]])
    Surf_Ratio_AA_1 = Surf_Count_AA / max(len(Surf_Res), 1)
    Surf_Ratio_AA_2 = Surf_Count_AA / max(Surf_area, 1e-9)

    Surf_MHCMF = np.array([
        MASA @ Surf_AAC, HP_IDX @ Surf_AAC, CG_IDX @ Surf_AAC, MS_IDX @ Surf_AAC, FX_IDX @ Surf_AAC
    ], dtype=float)
    AVG_Surf_MHCMF_1 = Surf_MHCMF / max(len(Surf_Res), 1)
    AVG_Surf_MHCMF_2 = Surf_MHCMF / max(Surf_area, 1e-9)

    # Surface residues (threshold: 1.8 Å CA-vertex)
    D = cdist(CA, Vtx)                 # len(CA) x nVert
    Surf_Idx = (D.min(axis=1) < 1.8)
    if not Surf_Idx.any():
        raise RuntimeError(f"Surface calculation (EDTSurf) failed for {seq_id} (no surface residues found)")
        
    Surf_Res = ''.join(np.array(list(AF_seq))[Surf_Idx])
    Surf_Coord = CA[Surf_Idx, :]
    
    # Surface property distributions in patches (8 Å)
    Surf_DISM = pdist(Surf_Coord)
    Surf_Ind = squareform(Surf_DISM) < 8.0
        
    def map_prop(seq: str, idx_array: np.ndarray) -> np.ndarray:
        out = np.zeros(len(seq), dtype=float)
        for i, aa in enumerate(seq):
            pos = np.where(RES_LIST == aa)[0]
            out[i] = idx_array[pos[0]] if pos.size == 1 else 0.0
        return out

    Surf_MASA_Info = map_prop(Surf_Res, MASA)
    Surf_HP_Info   = map_prop(Surf_Res, HP_IDX)
    Surf_CG_Info   = map_prop(Surf_Res, CG_IDX)
    Surf_MS_Info   = map_prop(Surf_Res, MS_IDX)
    Surf_FX_Info   = map_prop(Surf_Res, FX_IDX)

    def patch_sum(info: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mat = np.zeros((len(info), len(info)), dtype=float)
        mat[Surf_Ind] = np.tile(info, (len(info), 1))[Surf_Ind]
        return mat, mat.sum(axis=1)

    _, MASA_Dist = patch_sum(Surf_MASA_Info)
    _, HP_Dist   = patch_sum(Surf_HP_Info)
    _, CG_Dist   = patch_sum(Surf_CG_Info)
    _, MS_Dist   = patch_sum(Surf_MS_Info)
    _, FX_Dist   = patch_sum(Surf_FX_Info)

    def dist_stats(vec: np.ndarray) -> np.ndarray:
        if vec.size == 0:
            # Defensive check
            raise RuntimeError("Surface calculation (EDTSurf) failed (empty distribution)")
    
        q75, q25 = np.percentile(vec, [75, 25], method="hazen")
        q10 = np.percentile(vec, 10, method="hazen")
        q90 = np.percentile(vec, 90, method="hazen")
    
        return np.array([
            float(q75 - q25),   # IQR
            float(q90 - q10),   # 90th - 10th
            float(q10),         # 10th percentile
            float(q90),         # 90th percentile
        ])

    Surf_Dist = np.concatenate([
        dist_stats(Surf_MASA_Info),
        dist_stats(Surf_HP_Info),
        dist_stats(Surf_CG_Info),
        dist_stats(Surf_MS_Info),
        dist_stats(Surf_FX_Info),
        dist_stats(MASA_Dist),
        dist_stats(HP_Dist),
        dist_stats(CG_Dist),
        dist_stats(MS_Dist),
        dist_stats(FX_Dist),
        ])

    # --- Structural level
    len_Model = float(L)
    len_Surf  = float(len(Surf_Res))
    MW        = float(molecular_weight(AF_seq, seq_type='protein'))
    log_MW    = math.log(MW) if MW > 0 else 0.0

    diffs = CA - CA.mean(axis=0)
    Rg = float(np.sqrt((diffs**2).sum(axis=1).mean()))

    RVS  = (Volume / Surf_area) if Surf_area > 0 else 0.0
    Dens = (MW / Volume) if Volume > 0 else 0.0

    if Surf_DISM.size == 0:
        raise RuntimeError(f"Surface calculation (EDTSurf) failed for {seq_id}")

    MX_len = float(Surf_DISM.max())
    MD_len = float(np.median(Surf_DISM))
    if MD_len <= 0:
        raise RuntimeError(f"Surface calculation (EDTSurf) failed for {seq_id} (median distance ≤ 0)")
    MX_MD_Ratio = MX_len / MD_len

    # --- DSSP (calls mkdssp)
    mkdssp = os.path.join(bin_path, "mkdssp")
    ensure_executable(mkdssp)
    status, dssp_out = run_cmd([mkdssp, pdb001, "--output-format=dssp"])
    if status != 0:
        raise RuntimeError(f"DSSP failed for {seq_id}")

    # Parse DSSP lines
    lines = dssp_out.splitlines()
    hdr_idx = next(i for i, s in enumerate(lines) if "#  RESIDUE AA STRUCTURE" in s)
    line = lines[hdr_idx]
    p_struct = line.find("STRUCTURE")
    p_acc = line.find("ACC")
    DSSP_seq = []
    DSSP_ACC = []
    for i in range(len(CA)):
        row = lines[hdr_idx + 1 + i]
        DSSP_seq.append(row[p_struct])
        DSSP_ACC.append(float(row[p_acc:p_acc+3]))
    DSSP_seq = ''.join(DSSP_seq)
    DSSP_ACC = np.array(DSSP_ACC, dtype=float)

    # DSSP global counts
    Count_Helix = float(sum(ch in "HGIP" for ch in DSSP_seq))
    Count_Strand = float(sum(ch in "BE" for ch in DSSP_seq))
    Count_Others = float(len(DSSP_seq) - Count_Helix - Count_Strand)
    Count_Loose  = float(sum(ch in "S " for ch in DSSP_seq))
    Count_AH = float(DSSP_seq.count('H'))
    Count_BS = float(DSSP_seq.count('E'))
    Count_OS = float(len(DSSP_seq) - Count_AH - Count_BS)
    Count_GGG = len(re.findall("(?=GGG)", DSSP_seq))
    Count_HE  = float(DSSP_seq.count("HE"))

    Count_2nd_Struct = np.array([Count_Helix, Count_Strand, Count_Others, Count_Loose, Count_AH, Count_BS, Count_OS, Count_GGG, Count_HE])
    Ratio_2nd_Struct = Count_2nd_Struct / max(len(DSSP_seq), 1)

    # Surface DSSP
    Surf_DSSP_Seq = ''.join(np.array(list(DSSP_seq))[Surf_Idx])
    Surf_Count_Helix = float(sum(ch in "HGIP" for ch in Surf_DSSP_Seq))
    Surf_Count_Strand = float(sum(ch in "BE" for ch in Surf_DSSP_Seq))
    Surf_Count_Others = float(len(Surf_DSSP_Seq) - Surf_Count_Helix - Surf_Count_Strand)
    Surf_Count_Loose  = float(sum(ch in "S " for ch in Surf_DSSP_Seq))
    Surf_Count_AH     = float(Surf_DSSP_Seq.count('H'))
    Surf_Count_BS     = float(Surf_DSSP_Seq.count('E'))
    Surf_Count_OS     = float(len(Surf_DSSP_Seq) - Surf_Count_AH - Surf_Count_BS)
    Surf_Count_2nd_Struct = np.array([Surf_Count_Helix, Surf_Count_Strand, Surf_Count_Others, Surf_Count_Loose, Surf_Count_AH, Surf_Count_BS, Surf_Count_OS])
    Surf_Ratio_2nd_Struct = Surf_Count_2nd_Struct / max(len(Surf_DSSP_Seq), 1)

    # Surface area by DSSP classes (use ACC as area proxy)
    Surf_DSSP_ACC = DSSP_ACC[Surf_Idx]
    Surf_Count_Helix_area = float(Surf_DSSP_ACC[[c in "HGIP" for c in Surf_DSSP_Seq]].sum())
    Surf_Count_Strand_area = float(Surf_DSSP_ACC[[c in "BE" for c in Surf_DSSP_Seq]].sum())
    Surf_Count_Others_area = float(Surf_DSSP_ACC.sum() - Surf_Count_Helix_area - Surf_Count_Strand_area)
    Surf_Count_Loose_area  = float(Surf_DSSP_ACC[[c in "S " for c in Surf_DSSP_Seq]].sum())
    Surf_Count_AH_area     = float(Surf_DSSP_ACC[[c == "H" for c in Surf_DSSP_Seq]].sum())
    Surf_Count_BS_area     = float(Surf_DSSP_ACC[[c == "E" for c in Surf_DSSP_Seq]].sum())
    Surf_Count_OS_area     = float(Surf_DSSP_ACC.sum() - Surf_Count_AH_area - Surf_Count_BS_area)
    Surf_Count_2nd_Struct_area = np.array([
        Surf_Count_Helix_area, Surf_Count_Strand_area, Surf_Count_Others_area,
        Surf_Count_Loose_area, Surf_Count_AH_area, Surf_Count_BS_area, Surf_Count_OS_area
    ])
    Surf_Ratio_2nd_Struct_area = Surf_Count_2nd_Struct_area / max(Surf_DSSP_ACC.sum(), 1e-9)

    # Runs stats (Loops / Loose / H / E)
    Con_Loops = DSSP_seq.replace(' ', 'L')
    Con_Loose = Con_Loops.replace('S', 'L')
    def runs_len(s: str, ch: str) -> List[int]:
        return [len(m.group(0)) for m in re.finditer(fr"{ch}+", s)]
    Loops_lengths = runs_len(Con_Loops, 'L')
    Loose_lengths = runs_len(Con_Loose, 'L')
    H_lengths     = runs_len(Con_Loose, 'H')
    E_lengths     = runs_len(Con_Loose, 'E')
    L_lengths     = runs_len(Con_Loose, 'L')
    # DSSP_Lengths_Vec (6x5 flattened) per your bins
    def lengths_vec(vals: List[int]) -> np.ndarray:
        v = np.zeros(6, dtype=float)
        arr = np.array(vals, dtype=float)
        v[0] = (arr < 5).sum()
        v[1] = (arr >= 5).sum()
        v[2] = ((arr >= 5) & (arr < 10)).sum()
        v[3] = (arr >= 10).sum()
        v[4] = ((arr >= 10) & (arr < 20)).sum()
        v[5] = (arr >= 20).sum()
        return v
    DSSP_Lengths_Table = np.vstack([
        lengths_vec(Loops_lengths),
        lengths_vec(Loose_lengths),
        lengths_vec(H_lengths),
        lengths_vec(E_lengths),
        lengths_vec(L_lengths),
    ]).T  # 6x5
    DSSP_Lengths_Vec = DSSP_Lengths_Table.reshape(-1, order='F')  # 30x1

    # Disorder region AA (AADR)
    AADR1 = ''.join(np.array(list(AF_seq))[[c not in ('H','E') for c in DSSP_seq]])
    AADR2 = ''.join(np.array(list(AF_seq))[[c not in ('H','G','I','P','B','E') for c in DSSP_seq]])
    SAADR1 = ''.join(np.array(list(Surf_Res))[[c not in ('H','E') for c in Surf_DSSP_Seq]])
    SAADR2 = ''.join(np.array(list(Surf_Res))[[c not in ('H','G','I','P','B','E') for c in Surf_DSSP_Seq]])
    AADRs = [AADR1, AADR2, SAADR1, SAADR2]
    AADR_Table = np.zeros((25, 4), dtype=float)
    for i, s in enumerate(AADRs):
        aac = aacount_20(s)
        AADR_Table[0:20, i] = aac
        AADR_Table[20, i] = s.count('D') + s.count('E')
        AADR_Table[21, i] = s.count('K') + s.count('R') + s.count('H')
        AADR_Table[22, i] = s.count('F') + s.count('Y') + s.count('W')
        AADR_Table[23, i] = s.count('S') + s.count('T') + s.count('N') + s.count('Q') + s.count('C') + s.count('Y')
        AADR_Table[24, i] = s.count('A') + s.count('V') + s.count('L') + s.count('I') + s.count('M') + s.count('P') + s.count('G')
    AADR_Vec = AADR_Table.reshape(-1, order='F')  # 100x1

    # RSA-like (normalized ACC by MASA per AA)
    RSA_Table = np.zeros(L, dtype=float)
    MASA_Table = np.zeros(L, dtype=float)
    HP_Table = np.zeros(L, dtype=float)
    CG_Table = np.zeros(L, dtype=float)
    MS_Table = np.zeros(L, dtype=float)
    FX_Table = np.zeros(L, dtype=float)
    for i, aa in enumerate(AF_seq):
        pos = np.where(RES_LIST == aa)[0]
        if pos.size == 1:
            p = pos[0]
            RSA_Table[i] = DSSP_ACC[i] / MASA[p]
            MASA_Table[i] = MASA[p]
            HP_Table[i] = HP_IDX[p]
            CG_Table[i] = CG_IDX[p]
            MS_Table[i] = MS_IDX[p]
            FX_Table[i] = FX_IDX[p]
    RSA_Results = np.zeros((6,3), dtype=float)
    for col, rsa in enumerate([0.60, 0.65, 0.70]):
        mask = (RSA_Table >= rsa)
        frac = mask.mean() if L > 0 else 0.0
        RSA_Results[0, col] = frac
        def safe_mean(arr): 
            return float(arr.mean()) if arr.size>0 else 0.0
        RSA_Results[1, col] = frac * safe_mean(MASA_Table[mask])
        RSA_Results[2, col] = frac * safe_mean(HP_Table[mask])
        RSA_Results[3, col] = frac * safe_mean(CG_Table[mask])
        RSA_Results[4, col] = frac * safe_mean(MS_Table[mask])
        RSA_Results[5, col] = frac * safe_mean(FX_Table[mask])
    RSA_Results_Vec = RSA_Results.reshape(-1, order='F')  # 18x1

    # --- pLDDT (from B-factors)
    pLDDT = bfac.copy()
    Surf_pLDDT = pLDDT[Surf_Idx]
    # loop-length bins for thresholds [40,50,60,70,80], merging gap<=5
    pLDDT_LL_Table = np.zeros((6,5), dtype=float)
    pLDDT_OT_Table = np.zeros((29,5), dtype=float)
    for col, thres in enumerate([40, 50, 60, 70, 80]):
        # === 1) 等同 bwconncomp：找連續區塊（pLDDT < thres）
        mask = (pLDDT < thres)
        labeled, num = ndimage.label(mask)
        slices = ndimage.find_objects(labeled)   # 每個 slice 對應一段連續 True

        # === 2) 等同 regionprops(...,'BoundingBox') → 取 length, start, end
        # MATLAB: start = ceil(BB(1)), end = start + BB(3) - 1
        # 這裡用 0-based：start = sl.start, end = sl.stop-1, length = sl.stop - sl.start
        loose_tab = []
        for sl in slices:
            s = sl[0].start
            e = sl[0].stop - 1
            length = e - s + 1
            loose_tab.append((length, s, e))

        # === 3) 依「下一段起點 − 目前終點 ≤ 5」合併，長度採“各段長度相加”
        merged = []
        i = 0
        while i < len(loose_tab):
            cur_len, cur_s, cur_e = loose_tab[i]
            j = i + 1
            while j < len(loose_tab) and (loose_tab[j][1] - cur_e) <= 5:
                # 延伸終點到較大的 end；長度為分段長度相加（與 MATLAB 相同）
                cur_e   = max(cur_e, loose_tab[j][2])
                cur_len = cur_len + loose_tab[j][0]
                j += 1
            merged.append((cur_len, cur_s, cur_e))
            i = j
    
        # === 4) 之後你需要的衍生量（例如各段長度 + 分箱）
        if merged:
            lengths = np.array([seg_len for (seg_len, _, _) in merged], dtype=float)
        else:
            merged = [(0, 0, 0)]
            lengths = np.array([0.0])

        # LL 分箱（保持你原本的 6 個箱）
        pLDDT_LL_Table[0, col] = (lengths < 5).sum()
        pLDDT_LL_Table[1, col] = (lengths >= 5).sum()
        pLDDT_LL_Table[2, col] = ((lengths >= 5) & (lengths < 10)).sum()
        pLDDT_LL_Table[3, col] = (lengths >= 10).sum()
        pLDDT_LL_Table[4, col] = ((lengths >= 10) & (lengths < 20)).sum()
        pLDDT_LL_Table[5, col] = (lengths >= 20).sum()

        # === 5) 產生 Domain_pLDDT（'L' / 'R'）
        domain = np.array(['R'] * L)
        for seg_len, s, e in merged:
            domain[s:e+1] = 'L'

        # 若還有後續要用 robust regions of 'R'
        robust_regions = re.findall(r"R+", ''.join(domain))

        # Domain mask 'L' vs 'R'
        L_mask = (domain == 'L')
        L_seq = ''.join(np.array(list(AF_seq))[L_mask])
        L_aac = aacount_20(L_seq)

        pLDDT_OT_Table[0, col] = len(robust_regions)
        pLDDT_OT_Table[1, col] = L_mask.mean() if L>0 else 0.0
        pLDDT_OT_Table[2, col] = MASA @ L_aac
        pLDDT_OT_Table[3, col] = HP_IDX @ L_aac
        pLDDT_OT_Table[4, col] = CG_IDX @ L_aac
        pLDDT_OT_Table[5, col] = MS_IDX @ L_aac
        pLDDT_OT_Table[6, col] = FX_IDX @ L_aac
        pLDDT_OT_Table[7:27, col] = L_aac  # 20 rows
        pLDDT_OT_Table[27, col] = (pLDDT < thres).mean() if L>0 else 0.0
        pLDDT_OT_Table[28, col] = (Surf_pLDDT < thres).mean() if Surf_pLDDT.size>0 else 0.0

    pLDDT_LL_Vec = pLDDT_LL_Table.reshape(-1, order='F')     # 30
    pLDDT_OT_Vec = pLDDT_OT_Table.reshape(-1, order='F')     # 145
    pLDDT_Stat = np.array([
        float(pLDDT.mean()), float(np.median(pLDDT)), quantile(pLDDT, 0.25),
        float(Surf_pLDDT.mean()) if Surf_pLDDT.size>0 else 0.0,
        float(np.median(Surf_pLDDT)) if Surf_pLDDT.size>0 else 0.0,
        quantile(Surf_pLDDT, 0.25) if Surf_pLDDT.size>0 else 0.0
    ])

    # --- PAE
    # read PAE_Img.csv (matrix) and compute spectra & stats like MATLAB
    pae_csv = os.path.join(target_dir, "PAE_Img.csv")
    if not os.path.isfile(pae_csv):
        raise FileNotFoundError(f"PAE CSV not found: {pae_csv}")
    PAE_Img = np.loadtxt(pae_csv, delimiter=",")
    PAE_Img = np.transpose(PAE_Img)

    PAE_LL_Table = np.zeros((6, 3), dtype=float)
    PAE_OT_Table = np.zeros((36, 3), dtype=float)

    for col, window_size in enumerate([5, 10, 15]):
        # === PAE_Spectrum：中心 i 的 +/- window_size 列平均（與 MATLAB 相同）
        PAE_Spectrum = np.zeros(L, dtype=float)
        for i in range(L):
            a = max(0, i - window_size)
            b = min(L - 1, i + window_size)
            PAE_Spectrum[i] = float(PAE_Img[a:b+1, i].mean())
        Surf_PAE_Spectrum = PAE_Spectrum[Surf_Idx] if Surf_Idx.any() else np.array([])

        # === 「鬆散區段」定義：PAE_Spectrum > window_size*0.5
        thr = window_size * 0.5
        mask = (PAE_Spectrum > thr)

        # 1) 等價 MATLAB 的 bwconncomp
        labeled, num = ndimage.label(mask)
        slices = ndimage.find_objects(labeled)   # 每個 slice 是一段連續 True

        # 2) 等價 regionprops(...,'BoundingBox') → 取 (length, start, end)
        # MATLAB: start = ceil(BB(1)); end = start + BB(3) - 1
        # 這裡 0-based：start = sl.start; end = sl.stop-1; length = stop - start
        loose_tab = []
        for sl in slices:
            s = sl[0].start
            e = sl[0].stop - 1
            length = e - s + 1
            loose_tab.append((length, s, e))
    
        # 3) 依「下一段起點 − 目前終點 ≤ 5」進行合併；
        #    合併後長度 = 各段 length 相加（與你 MATLAB while 迴圈一致）
        merged = []
        i = 0
        while i < len(loose_tab):
            cur_len, cur_s, cur_e = loose_tab[i]
            j = i + 1
            while j < len(loose_tab) and (loose_tab[j][1] - cur_e) <= 5:
                cur_e   = max(cur_e,  loose_tab[j][2])
                cur_len = cur_len +  loose_tab[j][0]
                j += 1
            merged.append((cur_len, cur_s, cur_e))
            i = j

        # 3') 無任何區段時的佔位（對應 MATLAB 的 [0,1,1]；這裡 0-based → (0,0,0)）
        if not merged:
            PAE_Merged = [(0, 0, 0)]
        else:
            PAE_Merged = merged

        # 4) Domain_PAE：標 'L'（只對有效長度 >0 的段落）
        domain = np.array(['R'] * L)
        for seg_len, s, e in PAE_Merged:
            domain[s:e+1] = 'L'

        # 5) PAE loop-length 分箱統計（與 MATLAB 相同六個箱）
        valid_lengths = [seg_len for seg_len, _, _ in PAE_Merged if seg_len > 0]
        lengths = np.array(valid_lengths, dtype=float) if valid_lengths else np.array([0.0])

        PAE_LL_Table[0, col] = (lengths < 5).sum()
        PAE_LL_Table[1, col] = (lengths >= 5).sum()
        PAE_LL_Table[2, col] = ((lengths >= 5) & (lengths < 10)).sum()
        PAE_LL_Table[3, col] = (lengths >= 10).sum()
        PAE_LL_Table[4, col] = ((lengths >= 10) & (lengths < 20)).sum()
        PAE_LL_Table[5, col] = (lengths >= 20).sum()

        # 6) 其餘 OT 指標（維持你原先 Python 版的寫法，對齊 MATLAB 含義）
        robust_regions = re.findall(r"R+", ''.join(domain))
        L_mask = (domain == 'L')
        if L_mask.any():
            L_seq = ''.join(np.array(list(AF_seq))[L_mask])
            L_aac = aacount_20(L_seq)   # 長度=20；順序需與 MASA/HP_IDX/... 一致
        else:
            L_aac = np.zeros(20, dtype=float)

        PAE_OT_Table[0,  col] = len(robust_regions)
        PAE_OT_Table[1,  col] = L_mask.mean() if L > 0 else 0.0
        PAE_OT_Table[2,  col] = float(MASA   @ L_aac)
        PAE_OT_Table[3,  col] = float(HP_IDX @ L_aac)
        PAE_OT_Table[4,  col] = float(CG_IDX @ L_aac)
        PAE_OT_Table[5,  col] = float(MS_IDX @ L_aac)
        PAE_OT_Table[6,  col] = float(FX_IDX @ L_aac)
        PAE_OT_Table[7:27, col] = L_aac

        # fraction of PAE_Img > window_size across the full matrix（與 MATLAB 相同含義）
        PAE_OT_Table[27, col] = float((PAE_Img > window_size).sum()) / (L * L if L > 0 else 1)

        # Loose percentage（注意 MATLAB 這裡用 < window_size*0.5）
        PAE_OT_Table[28, col] = (PAE_Spectrum < thr).mean() if L > 0 else 0.0
        PAE_OT_Table[29, col] = (Surf_PAE_Spectrum < thr).mean() if Surf_PAE_Spectrum.size > 0 else 0.0

        # Spectrum 統計（分面向與 MATLAB 一致；分位數建議 method='hazen'）
        PAE_OT_Table[30, col] = float(PAE_Spectrum.mean())
        PAE_OT_Table[31, col] = float(np.median(PAE_Spectrum))
        PAE_OT_Table[32, col] = float(np.quantile(PAE_Spectrum, 0.25, method="hazen")) if PAE_Spectrum.size>0 else 0.0
        PAE_OT_Table[33, col] = float(Surf_PAE_Spectrum.mean()) if Surf_PAE_Spectrum.size>0 else 0.0
        PAE_OT_Table[34, col] = float(np.median(Surf_PAE_Spectrum)) if Surf_PAE_Spectrum.size>0 else 0.0
        PAE_OT_Table[35, col] = float(np.quantile(Surf_PAE_Spectrum, 0.25, method="hazen")) if Surf_PAE_Spectrum.size>0 else 0.0

    PAE_LL_Vec = PAE_LL_Table.reshape(-1, order='F')  # 18
    PAE_OT_Vec = PAE_OT_Table.reshape(-1, order='F')  # 108

    # --- RMSD among rank_002..005 vs rank_001
    pdb_list = sorted([os.path.join(target_dir, f) for f in files if f.endswith(".pdb") and "unrelaxed_rank_" in f])
    RMSD_list = []
    for p in pdb_list[1:]:
        CA2, _, _ = read_pdb_ca_coords(p)
        n = min(CA.shape[0], CA2.shape[0])
        RMSD_list.append(compute_rmsd(CA2[:n], CA[:n]))
    if len(RMSD_list) == 0:
        AVG_RMSD = 0.0
        Max_RMSD = 0.0
    else:
        AVG_RMSD = float(np.mean(RMSD_list))
        Max_RMSD = float(np.max(RMSD_list))

    # --- 3D Zernike (Jerhoud)
    # 1) write OFF
    off_path = os.path.join(target_dir, f"{seq_id}.off")
    with open(off_path, "w") as f:
        f.write("OFF\n")
        f.write(f"{Vtx.shape[0]} {Face.shape[0]} 0\n")
        for v in Vtx:
            f.write(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for row in Face.astype(int):
            f.write(f"{row[0]} {row[1]} {row[2]} {row[3]}\n")

    make_shape = os.path.join(bin_path, "MakeShape")
    shape2zern = os.path.join(bin_path, "Shape2Zernike")
    ensure_executable(make_shape)
    ensure_executable(shape2zern)

    # 2) MakeShape
    out_off = os.path.join(target_dir, f"{seq_id}_0.8.off")
    status, _ = run_cmd([make_shape, "-l", off_path, "-cr", "0.8", "-o", out_off])
    if status != 0:
        raise RuntimeError(f"3DZD calculation (MakeShape) failed for {seq_id}")

    # 3) Shape2Zernike
    order = 20
    zm_path = os.path.join(target_dir, f"{seq_id}_0.8.zm")
    status, _ = run_cmd([shape2zern, "-o", zm_path, "-t", "8", "-a", "8", str(order), out_off, "-v"])
    if status != 0:
        raise RuntimeError(f"3DZD calculation (Shape2Zernike) failed for {seq_id}")

    # 4) Restore complex moments from .zm
    with open(zm_path, "r") as f:
        lines = f.readlines()

    # find the ORTHO line
    pos = next(i for i, s in enumerate(lines) if s.startswith("ORTHO"))

    rows = []
    for line in lines[pos+1:]:
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) == 4:
            n, l, m, rea = parts
            ima = "0.0"
        elif len(parts) == 5:
            n, l, m, rea, ima = parts[:5]
        else:
            continue
        rows.append([float(n), float(l), float(m), float(rea), float(ima)])

    arr = np.array(rows, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0)

    # Build Table
    rows = []
    for n in range(order+1):
        l_start = n % 2
        for l in range(l_start, n+1, 2):
            for m in range(-l, l+1):
                rows.append([n, l, m])
    Table = np.zeros((len(rows), 5), dtype=float)
    Table[:,0:3] = np.array(rows, dtype=float)
    for i, (n,l,m) in enumerate(rows):
        # exact
        idx = np.where((arr[:,0]==n) & (arr[:,1]==l) & (arr[:,2]==m))[0]
        if idx.size > 0:
            Table[i,3:5] = arr[idx[0], 3:5]
        else:
            idx2 = np.where((arr[:,0]==n) & (arr[:,1]==l) & (arr[:,2]==abs(m)))[0]
            if idx2.size > 0:
                re_, im_ = arr[idx2[0], 3:5]
                if int(abs(m)) % 2 == 1:
                    im_ = -im_
                if m < 0:
                    re_, im_ = re_, -im_
                Table[i,3:5] = [re_, im_]

    # 5) Moments2Descriptors
    Ridx = (Table[:,2] <= 0)
    Ordered_ZM = Table[Ridx].copy()
    Ordered_ZM[:,2] = Ordered_ZM[:,1] + Ordered_ZM[:,2]  # l+m

    Order = int(Table[:,0].max())
    nInvariants = int(math.floor(((Order+2)/2.0)**2))
    Input_Descriptor = np.zeros(nInvariants, dtype=float)

    count = 0
    for n in range(Order+1):
        for l in range(n % 2, n+1, 2):
            sum_mom = 0.0
            for m in range(-l, l+1):
                idx = np.where((Ordered_ZM[:,0]==n) & (Ordered_ZM[:,1]==l) & (Ordered_ZM[:,2]==abs(m)))[0]
                if idx.size == 0:
                    mom = 0+0j
                else:
                    re_, im_ = Ordered_ZM[idx[0], 3], Ordered_ZM[idx[0], 4]
                    mom = complex(re_, im_)
                    if m < 0:
                        mom = mom.conjugate()
                        if abs(m) % 2 == 1:
                            mom = -mom
                sum_mom += abs(mom)**2
            Input_Descriptor[count] = math.sqrt(sum_mom)
            count += 1

    out_zm = os.path.join(target_dir, f"{seq_id}_Jerhoud_All.zm")
    write_jerhoud_zm(Table, Order, out_zm)
    
    # --- KORPE energy
    korpe = os.path.join(bin_path, "korpe")
    korp_bin = os.path.join(bin_path, "korp6Dv1.bin")
    ensure_executable(korpe)
    # create 5 Sample_i.pdb and Script.txt from available rank files
    pdb_list_full = [os.path.join(target_dir, f) for f in files if ("unrelaxed_rank_" in f and f.endswith(".pdb"))]
    pdb_list_full = sorted(pdb_list_full)[:5]
    script_path = os.path.join(target_dir, "Script.txt")
    sample_names = []
    with open(script_path, "w") as sf:
        for i, src in enumerate(pdb_list_full, start=1):
            dst = os.path.join(target_dir, f"Sample_{i}.pdb")
            with open(src, "r") as fsrc, open(dst, "w") as fdst:
                fdst.write(fsrc.read())
            sf.write(dst[:-4] + "\n")
            sample_names.append(dst)

    status, _ = run_cmd([korpe, script_path, "--score_file", korp_bin, "-o", os.path.join(target_dir, "KORPE")])
    # clean temp files regardless of success
    for p in sample_names + [script_path]:
        try: os.remove(p)
        except: pass
    if status != 0:
        raise RuntimeError(f"KORPE energy calculation failed for {seq_id}")

    # parse KORPE_score.txt
    korpe_score = os.path.join(target_dir, "KORPE_score.txt")
    KORPE_ENG = []
    if os.path.isfile(korpe_score):
        with open(korpe_score, "r") as f:
            for line in f:
                if "AF_Result" in line or "Processed_AF_Result" in line:
                    parts = line.strip().split()
                    try:
                        KORPE_ENG.append(float(parts[-1]))
                    except:
                        pass
    AVG_KORPE_ENG = float(np.mean(KORPE_ENG)) if KORPE_ENG else 0.0
    Min_KORPE_ENG = float(np.min(KORPE_ENG)) if KORPE_ENG else 0.0
    Max_KORPE_ENG = float(np.max(KORPE_ENG)) if KORPE_ENG else 0.0

    # --- Pack features (order mirrors your MATLAB Feat vector)
    Feat = np.concatenate([
        [pI],
        Count_AA, Ratio_AA, MHCMF, AVG_MHCMF,
        Surf_Count_AA, Surf_Ratio_AA_1, Surf_Ratio_AA_2,
        Surf_MHCMF, AVG_Surf_MHCMF_1, AVG_Surf_MHCMF_2,
        Surf_Dist,
        [len_Model, len_Surf, MW, log_MW, Volume, Surf_area, Rg, RVS, Dens, MX_len, MD_len, MX_MD_Ratio],
        Count_2nd_Struct, Ratio_2nd_Struct,
        Surf_Count_2nd_Struct, Surf_Ratio_2nd_Struct,
        Surf_Count_2nd_Struct_area, Surf_Ratio_2nd_Struct_area,
        DSSP_Lengths_Vec, AADR_Vec, RSA_Results_Vec,
        pLDDT_LL_Vec, pLDDT_OT_Vec, pLDDT_Stat,
        PAE_LL_Vec, PAE_OT_Vec,
        [AVG_RMSD, Max_RMSD],
        Input_Descriptor,
        [AVG_KORPE_ENG, Min_KORPE_ENG, Max_KORPE_ENG]
    ]).astype(float)

    return SeqResult(id=seq_id, feat=Feat)

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Extract TE features from processed AlphaFold results.")
    ap.add_argument("fasta", help="FASTA file (headers must match subfolder names under Processed_AF_Result)")
    ap.add_argument("processed_root", help="Processed AF results root (e.g., Processed_AF_Result)")
    ap.add_argument("out_csv", help="Output CSV filename (e.g., TE_feature.csv)")
    ap.add_argument("--bin", default="./bin", help="Folder containing external tools (default: ./bin)")
    args = ap.parse_args()

    # sanity checks for tools
    for exe in ["EDTSurf", "mkdssp", "MakeShape", "Shape2Zernike", "korpe"]:
        path = os.path.join(args.bin, exe)
        if not os.path.isfile(path):
            print(f"[WARN] Missing executable: {path} (this seq will fail at that step)")

    rows = []
    max_len = 0
    seq_ids = []
    for rec in SeqIO.parse(args.fasta, "fasta"):
        seq_id = rec.description.split()[0].lstrip(">")
        seq_ids.append(seq_id)

    for seq_id in seq_ids:
        try:
            res = process_one_sequence(seq_id, args.processed_root, args.bin)
            rows.append(res)
            max_len = max(max_len, res.feat.size)
            print(f"[OK] {seq_id} ({res.feat.size} features)")
        except Exception as e:
            print(f"[FAIL] {seq_id}: {e}")
            # still write a row with NaNs so downstream shape is consistent
            rows.append(SeqResult(id=seq_id, feat=np.array([], dtype=float)))

    # Build DataFrame with consistent columns (pad with NaN to max_len)
    data = []
    for r in rows:
        v = np.full(max_len, np.nan, dtype=float)
        v[:min(max_len, r.feat.size)] = r.feat[:min(max_len, r.feat.size)]
        data.append(v)
    df = pd.DataFrame(data, index=[r.id for r in rows])
    df.insert(0, "ID", df.index)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(args.out_csv, index=False)
    print(f"[DONE] Saved features to {args.out_csv}")

if __name__ == "__main__":
    main()