from __future__ import print_function

import numpy as np
import torch
import torch.utils
import gemmi

restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK",
}
restype_str_to_int = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
    "X": 20,
}
restype_int_to_str = {
    0: "A",
    1: "C",
    2: "D",
    3: "E",
    4: "F",
    5: "G",
    6: "H",
    7: "I",
    8: "K",
    9: "L",
    10: "M",
    11: "N",
    12: "P",
    13: "Q",
    14: "R",
    15: "S",
    16: "T",
    17: "V",
    18: "W",
    19: "Y",
    20: "X",
}
alphabet = list(restype_str_to_int)

element_list = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mb",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Uut",
    "Fl",
    "Uup",
    "Lv",
    "Uus",
    "Uuo",
]
element_list = [item.upper() for item in element_list]
# element_dict = dict(zip(element_list, range(1,len(element_list))))
element_dict_rev = dict(zip(range(1, len(element_list)), element_list))


def get_seq_rec(S: torch.Tensor, S_pred: torch.Tensor, mask: torch.Tensor):
    """
    S : true sequence shape=[batch, length]
    S_pred : predicted sequence shape=[batch, length]
    mask : mask to compute average over the region shape=[batch, length]

    average : averaged sequence recovery shape=[batch]
    """
    match = S == S_pred
    average = torch.sum(match * mask, dim=-1) / torch.sum(mask, dim=-1)
    return average


def get_score(S: torch.Tensor, log_probs: torch.Tensor, mask: torch.Tensor):
    """
    S : true sequence shape=[batch, length]
    log_probs : predicted sequence shape=[batch, length]
    mask : mask to compute average over the region shape=[batch, length]

    average_loss : averaged categorical cross entropy (CCE) [batch]
    loss_per_resdue : per position CCE [batch, length]
    """
    S_one_hot = torch.nn.functional.one_hot(S, 21)
    loss_per_residue = -(S_one_hot * log_probs).sum(-1)  # [B, L]
    average_loss = torch.sum(loss_per_residue * mask, dim=-1) / (
        torch.sum(mask, dim=-1) + 1e-8
    )
    return average_loss, loss_per_residue


def write_full_PDB(
    save_path: str,
    X: np.ndarray,
    X_m: np.ndarray,
    b_factors: np.ndarray,
    R_idx: np.ndarray,
    chain_letters: np.ndarray,
    S: np.ndarray,
    other_atoms=None,
    icodes=None,
    force_hetatm=False,
):
    """
    save_path : path where the PDB will be written to
    X : protein atom xyz coordinates shape=[length, 14, 3]
    X_m : protein atom mask shape=[length, 14]
    b_factors: shape=[length, 14]
    R_idx: protein residue indices shape=[length]
    chain_letters: protein chain letters shape=[length]
    S : protein amino acid sequence shape=[length]
    other_atoms: other atoms parsed by prody
    icodes: a list of insertion codes for the PDB; e.g. antibody loops
    """

    restype_1to3 = {
        "A": "ALA",
        "R": "ARG",
        "N": "ASN",
        "D": "ASP",
        "C": "CYS",
        "Q": "GLN",
        "E": "GLU",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "L": "LEU",
        "K": "LYS",
        "M": "MET",
        "F": "PHE",
        "P": "PRO",
        "S": "SER",
        "T": "THR",
        "W": "TRP",
        "Y": "TYR",
        "V": "VAL",
        "X": "UNK",
    }
    restype_INTtoSTR = {
        0: "A",
        1: "C",
        2: "D",
        3: "E",
        4: "F",
        5: "G",
        6: "H",
        7: "I",
        8: "K",
        9: "L",
        10: "M",
        11: "N",
        12: "P",
        13: "Q",
        14: "R",
        15: "S",
        16: "T",
        17: "V",
        18: "W",
        19: "Y",
        20: "X",
    }
    restype_name_to_atom14_names = {
        "ALA": ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
        "ARG": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD",
            "NE",
            "CZ",
            "NH1",
            "NH2",
            "",
            "",
            "",
        ],
        "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", ""],
        "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", ""],
        "CYS": ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
        "GLN": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD",
            "OE1",
            "NE2",
            "",
            "",
            "",
            "",
            "",
        ],
        "GLU": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD",
            "OE1",
            "OE2",
            "",
            "",
            "",
            "",
            "",
        ],
        "GLY": ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
        "HIS": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "ND1",
            "CD2",
            "CE1",
            "NE2",
            "",
            "",
            "",
            "",
        ],
        "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", ""],
        "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", ""],
        "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", ""],
        "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", ""],
        "PHE": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD1",
            "CD2",
            "CE1",
            "CE2",
            "CZ",
            "",
            "",
            "",
        ],
        "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""],
        "SER": ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""],
        "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", ""],
        "TRP": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD1",
            "CD2",
            "CE2",
            "CE3",
            "NE1",
            "CZ2",
            "CZ3",
            "CH2",
        ],
        "TYR": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD1",
            "CD2",
            "CE1",
            "CE2",
            "CZ",
            "OH",
            "",
            "",
        ],
        "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", ""],
        "UNK": ["", "", "", "", "", "", "", "", "", "", "", "", "", ""],
    }

    S_str = [restype_1to3[AA] for AA in [restype_INTtoSTR[AA] for AA in S]]
    
    out_model = gemmi.Model("")
    residues_for_chain = {}  # Chain name -> [residues]
    for i, AA in enumerate(S_str):
        chain_name = chain_letters[i]
        if chain_name not in residues_for_chain:
            residues_for_chain[chain_name] = []

        residue = gemmi.Residue()
        residues_for_chain[chain_name].append(residue)
        residue.seqid = gemmi.SeqId(R_idx[i], icodes[i] if icodes[i] else " ")
        residue.name = AA
        
        for j, atom_name in enumerate(restype_name_to_atom14_names[AA]):
            if not atom_name:  # 14 long lists padded with empty strings.
                break
            
            atom = gemmi.Atom()
            atom.name = atom_name
            atom.element = gemmi.Element(atom_name[:1])
            atom.b_iso = b_factors[i,j]
            atom.pos = gemmi.Position(*X[i,j].tolist())
            residue.add_atom(atom)
    
    for cra in other_atoms:
        chain_name = cra.chain.name
        if chain_name not in residues_for_chain:
            residues_for_chain[chain_name] = []
        
        for residue in residues_for_chain[chain_name]:
            if residue.seqid == cra.residue.seqid:
                break
        else:
            residue = cra.residue.clone()
            del residue[:]
            residues_for_chain[chain_name].append(residue)
        
        residue.add_atom(cra.atom)
    
    for chain_name in residues_for_chain:
        chain = gemmi.Chain(chain_name)
        for residue in residues_for_chain[chain_name]:
            chain.add_residue(residue)
        out_model.add_chain(chain)

    out_struct = gemmi.Structure()
    out_struct.add_model(out_model)
    out_struct.write_pdb(save_path + ".pdb", gemmi.PdbWriteOptions(cryst1_record=False))
    cif = out_struct.make_mmcif_document()
    cif.write_file(save_path + ".cif")
    

def parse_PDB(
    input_path: str,
    device: str = "cpu",
    chains: list = [],
    parse_all_atoms: bool = False,
    parse_atoms_with_zero_occupancy: bool = False
):
    """
    input_path : path for the input PDB
    device: device for the torch.Tensor
    chains: a list specifying which chains need to be parsed; e.g. ["A", "B"]
    parse_all_atoms: if False parse only N,CA,C,O otherwise all 37 atoms
    parse_atoms_with_zero_occupancy: if True atoms with zero occupancy will be parsed
    """
    element_list = [
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mb",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I",
        "Xe",
        "Cs",
        "Ba",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
        "Rf",
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Ds",
        "Rg",
        "Cn",
        "Uut",
        "Fl",
        "Uup",
        "Lv",
        "Uus",
        "Uuo",
    ]
    element_list = [item.upper() for item in element_list]
    element_dict = dict(zip(element_list, range(1, len(element_list))))
    restype_3to1 = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }
    restype_STRtoINT = {
        "A": 0,
        "C": 1,
        "D": 2,
        "E": 3,
        "F": 4,
        "G": 5,
        "H": 6,
        "I": 7,
        "K": 8,
        "L": 9,
        "M": 10,
        "N": 11,
        "P": 12,
        "Q": 13,
        "R": 14,
        "S": 15,
        "T": 16,
        "V": 17,
        "W": 18,
        "Y": 19,
        "X": 20,
    }

    atom_order = {
        "N": 0,
        "CA": 1,
        "C": 2,
        "CB": 3,
        "O": 4,
        "CG": 5,
        "CG1": 6,
        "CG2": 7,
        "OG": 8,
        "OG1": 9,
        "SG": 10,
        "CD": 11,
        "CD1": 12,
        "CD2": 13,
        "ND1": 14,
        "ND2": 15,
        "OD1": 16,
        "OD2": 17,
        "SD": 18,
        "CE": 19,
        "CE1": 20,
        "CE2": 21,
        "CE3": 22,
        "NE": 23,
        "NE1": 24,
        "NE2": 25,
        "OE1": 26,
        "OE2": 27,
        "CH2": 28,
        "NH1": 29,
        "NH2": 30,
        "OH": 31,
        "CZ": 32,
        "CZ2": 33,
        "CZ3": 34,
        "NZ": 35,
        "OXT": 36,
    }
    
    restype_3to1 = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }


    if not parse_all_atoms:
        atom_types = ["N", "CA", "C", "O"]
    else:
        atom_types = [
            "N",
            "CA",
            "C",
            "CB",
            "O",
            "CG",
            "CG1",
            "CG2",
            "OG",
            "OG1",
            "SG",
            "CD",
            "CD1",
            "CD2",
            "ND1",
            "ND2",
            "OD1",
            "OD2",
            "SD",
            "CE",
            "CE1",
            "CE2",
            "CE3",
            "NE",
            "NE1",
            "NE2",
            "OE1",
            "OE2",
            "CH2",
            "NH1",
            "NH2",
            "OH",
            "CZ",
            "CZ2",
            "CZ3",
            "NZ",
        ]

    structure = gemmi.read_structure(input_path)  # Works with cif or pdb, detected by file extension.
    CRAs = []
    for model in structure:
        CRAs += [cra for cra in model.all() if
                (not chains or cra.chain in chains) and
                (parse_atoms_with_zero_occupancy or cra.atom.occ > 0) and
                cra.atom.altloc in ["\x00", "", "A"]]  # TODO how does original handle altlocs?
    
    
    protein_cras = [cra for cra in CRAs if cra.residue.name in restype_3to1]
    backbone = [cra for cra in protein_cras if (cra.atom.name in ["N", "CA", "C", "O"])]
    other_cras = [cra for cra in CRAs if cra.residue.name not in restype_3to1 and not cra.residue.is_water()]
    water_cras = [cra for cra in CRAs if cra.residue.is_water()]
    
    CA_cras = [cra for cra in protein_cras if cra.atom.name == "CA"]
    CA_residues = []
    chains = []
    for model in structure:
        for chain in model:
            chains.append(chain)
            for res in chain:
                if res.get_ca():
                    CA_residues.append(res)
    
    xyz_37 = np.zeros([len(CA_residues), 37, 3], np.float32)
    xyz_37_m = np.zeros([len(CA_residues), 37], np.int32)
    seen_residues = {}
    for atom_name in atom_types:
        matching_cras = [cra for cra in protein_cras if cra.atom.name == atom_name]
        for cra in matching_cras:
            if cra.residue.get_ca():
                key = cra.chain.name + str(cra.residue.seqid)
                if key not in seen_residues:
                    seen_residues[key] = len(seen_residues)
                index = seen_residues[key]
                xyz_37[index, atom_order[atom_name], :] = cra.atom.pos.tolist()
                xyz_37_m[index, atom_order[atom_name]] = 1
                index += 1

    N = xyz_37[:, atom_order["N"], :]
    CA = xyz_37[:, atom_order["CA"], :]
    C = xyz_37[:, atom_order["C"], :]
    O = xyz_37[:, atom_order["O"], :]

    N_m = xyz_37_m[:, atom_order["N"]]
    CA_m = xyz_37_m[:, atom_order["CA"]]
    C_m = xyz_37_m[:, atom_order["C"]]
    O_m = xyz_37_m[:, atom_order["O"]]

    mask = N_m * CA_m * C_m * O_m  # must all 4 atoms exist

    R_idx = np.array([res.seqid.num for res in CA_residues], dtype=np.int32)
    S = [cra.residue.name for cra in CA_cras]
    S = [restype_3to1[AA] if AA in restype_3to1 else "X" for AA in S]
    S = np.array([restype_STRtoINT[AA] for AA in list(S)], np.int32)
    X = np.concatenate([N[:, None], CA[:, None], C[:, None], O[:, None]], 1)

    if other_cras:
        Y = np.array([cra.atom.pos.tolist() for cra in other_cras], dtype=np.float32)
        Y_t = [cra.atom.element.name for cra in other_cras]
        Y_t = np.array(
            [
                element_dict[y_t.upper()] if y_t.upper() in element_list else 0
                for y_t in Y_t
            ],
            dtype=np.int32,
        )
        Y_m = (Y_t != 1) * (Y_t != 0)
        Y = Y[Y_m, :]
        Y_t = Y_t[Y_m]
        Y_m = Y_m[Y_m]
    else:
        Y = np.zeros([1, 3], np.float32)
        Y_t = np.zeros([1], np.int32)
        Y_m = np.zeros([1], np.int32)

    output_dict = {}
    output_dict["X"] = torch.tensor(X, device=device, dtype=torch.float32)
    output_dict["mask"] = torch.tensor(mask, device=device, dtype=torch.int32)
    output_dict["Y"] = torch.tensor(Y, device=device, dtype=torch.float32)
    output_dict["Y_t"] = torch.tensor(Y_t, device=device, dtype=torch.int32)
    output_dict["Y_m"] = torch.tensor(Y_m, device=device, dtype=torch.int32)

    output_dict["R_idx"] = torch.tensor(R_idx, device=device, dtype=torch.int32)
    output_dict["chain_labels"] = torch.tensor([chains.index(cra.chain) for cra in CA_cras], device=device, dtype=torch.int32)
    output_dict["chain_letters"] = np.array([cra.chain.name for cra in CA_cras])

    mask_c = []
    chain_list = list(set(output_dict["chain_letters"]))
    chain_list.sort()
    for chain in chain_list:
        mask_c.append(
            torch.tensor(
                [chain == item for item in output_dict["chain_letters"]],
                device=device,
                dtype=bool,
            )
        )

    output_dict["mask_c"] = mask_c
    output_dict["chain_list"] = chain_list

    output_dict["S"] = torch.tensor(S, device=device, dtype=torch.int32)

    output_dict["xyz_37"] = torch.tensor(xyz_37, device=device, dtype=torch.float32)
    output_dict["xyz_37_m"] = torch.tensor(xyz_37_m, device=device, dtype=torch.int32)

    CA_icodes = np.array([ca.residue.seqid.icode.strip() for ca in CA_cras])
    
    # We need to return the parent structure as well to avoid things being garbage collected.
    return output_dict, backbone, other_cras, CA_icodes, water_cras, structure


def get_nearest_neighbours(CB, mask, Y, Y_t, Y_m, number_of_ligand_atoms):
    device = CB.device
    mask_CBY = mask[:, None] * Y_m[None, :]  # [A,B]
    L2_AB = torch.sum((CB[:, None, :] - Y[None, :, :]) ** 2, -1)
    L2_AB = L2_AB * mask_CBY + (1 - mask_CBY) * 1000.0

    nn_idx = torch.argsort(L2_AB, -1)[:, :number_of_ligand_atoms]
    L2_AB_nn = torch.gather(L2_AB, 1, nn_idx)
    D_AB_closest = torch.sqrt(L2_AB_nn[:, 0])

    Y_r = Y[None, :, :].repeat(CB.shape[0], 1, 1)
    Y_t_r = Y_t[None, :].repeat(CB.shape[0], 1)
    Y_m_r = Y_m[None, :].repeat(CB.shape[0], 1)

    Y_tmp = torch.gather(Y_r, 1, nn_idx[:, :, None].repeat(1, 1, 3))
    Y_t_tmp = torch.gather(Y_t_r, 1, nn_idx)
    Y_m_tmp = torch.gather(Y_m_r, 1, nn_idx)

    Y = torch.zeros(
        [CB.shape[0], number_of_ligand_atoms, 3], dtype=torch.float32, device=device
    )
    Y_t = torch.zeros(
        [CB.shape[0], number_of_ligand_atoms], dtype=torch.int32, device=device
    )
    Y_m = torch.zeros(
        [CB.shape[0], number_of_ligand_atoms], dtype=torch.int32, device=device
    )

    num_nn_update = Y_tmp.shape[1]
    Y[:, :num_nn_update] = Y_tmp
    Y_t[:, :num_nn_update] = Y_t_tmp
    Y_m[:, :num_nn_update] = Y_m_tmp

    return Y, Y_t, Y_m, D_AB_closest


def featurize(
    input_dict,
    cutoff_for_score=8.0,
    use_atom_context=True,
    number_of_ligand_atoms=16,
    model_type="protein_mpnn",
):
    output_dict = {}
    if model_type == "ligand_mpnn":
        mask = input_dict["mask"]
        Y = input_dict["Y"]
        Y_t = input_dict["Y_t"]
        Y_m = input_dict["Y_m"]
        N = input_dict["X"][:, 0, :]
        CA = input_dict["X"][:, 1, :]
        C = input_dict["X"][:, 2, :]
        b = CA - N
        c = C - CA
        a = torch.cross(b, c, axis=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        Y, Y_t, Y_m, D_XY = get_nearest_neighbours(
            CB, mask, Y, Y_t, Y_m, number_of_ligand_atoms
        )
        mask_XY = (D_XY < cutoff_for_score) * mask * Y_m[:, 0]
        output_dict["mask_XY"] = mask_XY[None,]
        if "side_chain_mask" in list(input_dict):
            output_dict["side_chain_mask"] = input_dict["side_chain_mask"][None,]
        output_dict["Y"] = Y[None,]
        output_dict["Y_t"] = Y_t[None,]
        output_dict["Y_m"] = Y_m[None,]
        if not use_atom_context:
            output_dict["Y_m"] = 0.0 * output_dict["Y_m"]
    elif (
        model_type == "per_residue_label_membrane_mpnn"
        or model_type == "global_label_membrane_mpnn"
    ):
        output_dict["membrane_per_residue_labels"] = input_dict[
            "membrane_per_residue_labels"
        ][None,]

    R_idx_list = []
    count = 0
    R_idx_prev = -100000
    for R_idx in list(input_dict["R_idx"]):
        if R_idx_prev == R_idx:
            count += 1
        R_idx_list.append(R_idx + count)
        R_idx_prev = R_idx
    R_idx_renumbered = torch.tensor(R_idx_list, device=R_idx.device)
    output_dict["R_idx"] = R_idx_renumbered[None,]
    output_dict["R_idx_original"] = input_dict["R_idx"][None,]
    output_dict["chain_labels"] = input_dict["chain_labels"][None,]
    output_dict["S"] = input_dict["S"][None,]
    output_dict["chain_mask"] = input_dict["chain_mask"][None,]
    output_dict["mask"] = input_dict["mask"][None,]

    output_dict["X"] = input_dict["X"][None,]

    if "xyz_37" in list(input_dict):
        output_dict["xyz_37"] = input_dict["xyz_37"][None,]
        output_dict["xyz_37_m"] = input_dict["xyz_37_m"][None,]

    return output_dict
