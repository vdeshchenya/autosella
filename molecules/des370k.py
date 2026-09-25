"""Select DES370K train dimers and distinct valid dimers when available."""

import concurrent.futures as cf
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rdkit import Chem

from molecules import gfn_worker
from molecules import pipeline_helpers as helpers

# DES370K monomer classes from pages 5-9 of the Supplementary Information for
# Donchev et al.: https://doi.org/10.1038/s41597-021-00833-x
SMILES_TO_CLASS = {
    # acids (4)
    "CCC(=O)O": "acids",
    "CC(=O)O": "acids",
    "OC=O": "acids",
    "OC(=O)CC(=O)O": "acids",
    # alcohols (10)
    "CCCO": "alcohols",
    "CCC(O)C": "alcohols",
    "CCO": "alcohols",
    "CC(O)C": "alcohols",
    "CO": "alcohols",
    "OC1CCCC1": "alcohols",
    "OC1CCCCC1": "alcohols",
    "OCCCCO": "alcohols",
    "OCCCO": "alcohols",
    "OCCO": "alcohols",
    # alkanes (14)
    "C1CCCC1": "alkanes",
    "C1CCCCC1": "alkanes",
    "C": "alkanes",
    "CC1CCCC1": "alkanes",
    "CC1CCCCC1": "alkanes",
    "CC": "alkanes",
    "CCC": "alkanes",
    "CC(C)C": "alkanes",
    "CCCC": "alkanes",
    "CC(C)(C)C": "alkanes",
    "CCC(C)C": "alkanes",
    "CCCCC": "alkanes",
    "CCC(C)(C)C": "alkanes",
    "CCCCCC": "alkanes",
    # alkenes (12)
    "C=C": "alkenes",
    "CC=C": "alkenes",
    "CC=CC": "alkenes",
    "CC(=C)C": "alkenes",
    "CCC=C": "alkenes",
    "CC=C(C)C": "alkenes",
    "CCC=CC": "alkenes",
    "CCC(=C)C": "alkenes",
    "CC(=C(C)C)C": "alkenes",
    "CCC=C(C)C": "alkenes",
    "CCC(=CC)C": "alkenes",
    "CCC(=C(C)C)C": "alkenes",
    # amides (28)
    "CCCNC=O": "amides",
    "CCC(=O)N": "amides",
    "CCC(=O)NC": "amides",
    "CCC(=O)N(C)C": "amides",
    "CCC(=O)N(CC)C": "amides",
    "CCNC=O": "amides",
    "CCNC(=O)C": "amides",
    "CCN(C=O)CC": "amides",
    "CCN(C(=O)C)C": "amides",
    "CCNC(=O)CC": "amides",
    "CCN(C(=O)C)CC": "amides",
    "CC(=O)N": "amides",
    "CC(=O)N(C)C": "amides",
    "CNC=O": "amides",
    "CNC(=O)C": "amides",
    "CN(C=O)CC": "amides",
    "CNC(=O)CC(=O)N": "amides",
    "CNC(=O)CC(=O)NC": "amides",
    "CNC(=O)CNC=O": "amides",
    "CNC(=O)CNC(=O)C": "amides",
    "CNC(=O)C(NC(=O)C)C": "amides",
    "NC=O": "amides",
    "NC(=O)CC(=O)N": "amides",
    "O=CN(C)C": "amides",
    "O=CNCCC(=O)N": "amides",
    "O=CNCCC(=O)NC": "amides",
    "O=CNCCNC=O": "amides",
    "O=CNCC(=O)N": "amides",
    # amines (20)
    "C1CCCN1": "amines",
    "C1CCCNC1": "amines",
    "CCCN": "amines",
    "CCCNC": "amines",
    "CCCN(C)C": "amines",
    "CCN": "amines",
    "CCN(C)C": "amines",
    "CCNCC": "amines",
    "CCN(CC)C": "amines",
    "CN": "amines",
    "CNC": "amines",
    "CN(C)C": "amines",
    "CNCC": "amines",
    "CNCCCN": "amines",
    "CNCCCNC": "amines",
    "CNCCN": "amines",
    "CNCCNC": "amines",
    "N": "amines",
    "NCCCN": "amines",
    "NCCN": "amines",
    # ammoniums (6)
    "CC[NH3+]": "ammoniums",
    "C[N+](C)(C)C": "ammoniums",
    "C[NH2+]C": "ammoniums",
    "C[NH3+]": "ammoniums",
    "C[NH+](C)C": "ammoniums",
    "[NH4+]": "ammoniums",
    # benzene (3)
    "c1ccccc1": "benzene",
    "Cc1ccccc1": "benzene",
    "CCc1ccccc1": "benzene",
    # carboxylates (3)
    "[O-]C=O": "carboxylates",
    "[O-]C(=O)C": "carboxylates",
    "[O-]C(=O)CC": "carboxylates",
    # esters (10)
    "CCCOC=O": "esters",
    "CCC(=O)OC": "esters",
    "CCOC(=O)CC": "esters",
    "CCOC(=O)C": "esters",
    "CCOC=O": "esters",
    "COC(=O)C": "esters",
    "COC=O": "esters",
    "O=COCCCOC=O": "esters",
    "O=COCCOC=O": "esters",
    "O=COCOC=O": "esters",
    # ethers (15)
    "C1CCCO1": "ethers",
    "C1CCCOC1": "ethers",
    "C1CCOCO1": "ethers",
    "C1OCCO1": "ethers",
    "CCCOC": "ethers",
    "CCCOCOC": "ethers",
    "CCOCC": "ethers",
    "COCCCOC": "ethers",
    "COCC": "ethers",
    "COCCOC": "ethers",
    "COC": "ethers",
    "COCOCC": "ethers",
    "COCOC": "ethers",
    "O1CCOCC1": "ethers",
    "O1COCOC1": "ethers",
    # guanidiniums (3)
    "CCNC(=[NH2+])N": "guanidiniums",
    "CNC(=[NH2+])N": "guanidiniums",
    "NC(=[NH2+])N": "guanidiniums",
    # imidazolium (3)
    "c1[nH]cc[nH+]1": "imidazolium",
    "Cc1c[nH]c[nH+]1": "imidazolium",
    "CCc1c[nH]c[nH+]1": "imidazolium",
    # ketones (6)
    "CCC(=O)CC": "ketones",
    "CCC(=O)C": "ketones",
    "CCC=O": "ketones",
    "CC(=O)C": "ketones",
    "CC=O": "ketones",
    "C=O": "ketones",
    # monoatomics (14)
    "[Ar]": "monoatomics",
    "[Br-]": "monoatomics",
    "[Ca+2]": "monoatomics",
    "[Cl-]": "monoatomics",
    "[F-]": "monoatomics",
    "[He]": "monoatomics",
    "[I-]": "monoatomics",
    "[K+]": "monoatomics",
    "[Kr]": "monoatomics",
    "[Li+]": "monoatomics",
    "[Mg+2]": "monoatomics",
    "[Na+]": "monoatomics",
    "[Ne]": "monoatomics",
    "[Xe]": "monoatomics",
    # other (191)
    "Brc1ccc(cc1)Br": "other",
    "Brc1ccccc1": "other",
    "BrC(Br)Br": "other",
    "BrCBr": "other",
    "BrCCBr": "other",
    "CBr": "other",
    "CC(Br)Br": "other",
    "CCBr": "other",
    "CCCC#CC": "other",
    "CCCC(Cl)(Cl)Cl": "other",
    "CCCC(Cl)Cl": "other",
    "CCCCCl": "other",
    "CCC#CC": "other",
    "CCCC#C": "other",
    "CCCC(F)(F)F": "other",
    "CCCC(F)F": "other",
    "CCCCF": "other",
    "CCC(Cl)(Cl)Cl": "other",
    "CCC(Cl)Cl": "other",
    "CCCCl": "other",
    "CCCC#N": "other",
    "CC#CC": "other",
    "CCC#C": "other",
    "CCC(F)(F)F": "other",
    "CCC(F)F": "other",
    "CCCF": "other",
    "CC(Cl)(Cl)Cl": "other",
    "CC(Cl)Cl": "other",
    "CCCl": "other",
    "CCC#N": "other",
    "CC#C": "other",
    "CC(F)(F)F": "other",
    "CC(F)F": "other",
    "CCF": "other",
    "CC(I)I": "other",
    "CCI": "other",
    "CCl": "other",
    "CC#N": "other",
    "CCOP(=O)(OC)OC": "other",
    "CCOP(=O)(OC)[O-]": "other",
    "CCOP(=O)(OC)O": "other",
    "C#C": "other",
    "CF": "other",
    "CI": "other",
    "Clc1ccc(cc1)Cl": "other",
    "Clc1cccc(c1)Cl": "other",
    "Clc1ccccc1Cl": "other",
    "Clc1ccccc1": "other",
    "Clc1cc(Cl)c(c(c1Cl)Cl)Cl": "other",
    "Clc1cc(Cl)cc(c1)Cl": "other",
    "Clc1c(Cl)c(Cl)c(c(c1Cl)Cl)Cl": "other",
    "ClC(C(Cl)(Cl)Cl)(Cl)Cl": "other",
    "ClC(C(Cl)(Cl)Cl)Cl": "other",
    "ClCC(Cl)(Cl)Cl": "other",
    "ClCC(Cl)Cl": "other",
    "ClCCCl": "other",
    "ClC(Cl)Cl": "other",
    "ClCCl": "other",
    "CNCCCOC=O": "other",
    "CNCCCOC": "other",
    "CNCCC(=O)NC": "other",
    "CNCCC(=O)N": "other",
    "CNCCC(=O)O": "other",
    "CNCCCO": "other",
    "CNCCCSC": "other",
    "CNCCCS": "other",
    "CNCCNC=O": "other",
    "CNCCOC=O": "other",
    "CNCCOC": "other",
    "CNCC(=O)NC": "other",
    "CNCC(=O)N": "other",
    "CNCC(=O)O": "other",
    "CNCCO": "other",
    "CNCCSC": "other",
    "CNCCS": "other",
    "CNC(=O)CCN": "other",
    "CNC(=O)CC(=O)O": "other",
    "CNC(=O)CCO": "other",
    "CNC(=O)CCS": "other",
    "CNC(=O)CN": "other",
    "CNC(=O)COC=O": "other",
    "CNC(=O)CO": "other",
    "CNCOC=O": "other",
    "CNCOC": "other",
    "CNC(=O)CS": "other",
    "CNCSC": "other",
    "C#N": "other",
    "COCCCN": "other",
    "COCCCOC=O": "other",
    "COCCC(=O)NC": "other",
    "COCCC(=O)N": "other",
    "COCCC(=O)O": "other",
    "COCCCO": "other",
    "COCCCSC": "other",
    "COCCCS": "other",
    "COCCNC=O": "other",
    "COCCN": "other",
    "COCCOC=O": "other",
    "COCC(=O)NC": "other",
    "COCC(=O)N": "other",
    "COCC(=O)O": "other",
    "COCCO": "other",
    "COCCSC": "other",
    "COCCS": "other",
    "COCNC=O": "other",
    "COCN": "other",
    "COCOC=O": "other",
    "COCO": "other",
    "COCSC": "other",
    "COCS": "other",
    "COP(=O)(OC)OC": "other",
    "COP(=O)(OC)[O-]": "other",
    "COP(=O)(OC)O": "other",
    "COP(=O)(O)O": "other",
    "COP(=O)(OP(=O)(O)O)[O-]": "other",
    "CSCCCNC=O": "other",
    "CSCCCN": "other",
    "CSCCCOC=O": "other",
    "CSCCC(=O)N": "other",
    "CSCCC(=O)O": "other",
    "CSCCCO": "other",
    "CSCCN": "other",
    "CSCCOC=O": "other",
    "CSCC(=O)NC": "other",
    "CSCC(=O)N": "other",
    "CSCC(=O)O": "other",
    "CSCCO": "other",
    "CSCNC=O": "other",
    "CSCN": "other",
    "CSCOC=O": "other",
    "CSCO": "other",
    "Fc1ccc(cc1)F": "other",
    "Fc1cccc(c1)F": "other",
    "Fc1ccccc1F": "other",
    "Fc1ccccc1": "other",
    "Fc1cc(F)c(c(c1F)F)F": "other",
    "Fc1cc(F)cc(c1)F": "other",
    "Fc1c(F)c(F)c(c(c1F)F)F": "other",
    "FC(C(F)(F)F)(F)F": "other",
    "FC(C(F)(F)F)F": "other",
    "FCC(F)(F)F": "other",
    "FCC(F)F": "other",
    "FCCF": "other",
    "FC(F)F": "other",
    "FCF": "other",
    "ICCI": "other",
    "ICI": "other",
    "NCCCOC=O": "other",
    "NCCC(=O)N": "other",
    "NCCC(=O)O": "other",
    "NCCCO": "other",
    "NCCCS": "other",
    "NCCNC=O": "other",
    "NCCOC=O": "other",
    "NCC(=O)N": "other",
    "NCC(=O)O": "other",
    "NCCO": "other",
    "NCCS": "other",
    "NC(=O)CC(=O)O": "other",
    "NC(=O)CCO": "other",
    "NC(=O)CCS": "other",
    "NC(=O)CO": "other",
    "NCOC=O": "other",
    "NC(=O)CS": "other",
    "OCCCNC=O": "other",
    "OCCCOC=O": "other",
    "OCCC(=O)O": "other",
    "OCCCS": "other",
    "OCCNC=O": "other",
    "OCCOC=O": "other",
    "OCC(=O)O": "other",
    "OCCS": "other",
    "O=CNCCC(=O)O": "other",
    "O=CNCCOC=O": "other",
    "O=CNCC(=O)O": "other",
    "O=CNCOC=O": "other",
    "O=COCCC(=O)NC": "other",
    "O=COCCC(=O)N": "other",
    "O=COCCC(=O)O": "other",
    "O=COCC(=O)N": "other",
    "O=COCC(=O)O": "other",
    "OC(=O)CCS": "other",
    "OCOC=O": "other",
    "OC(=O)CS": "other",
    "OP(=O)(O)O": "other",
    "[O-]P(=O)(OP(=O)(OC)O)O": "other",
    "SCCCOC=O": "other",
    "SCCNC=O": "other",
    "SCCOC=O": "other",
    "SCOC=O": "other",
    "[H][H]": "other",
    # phenol (3)
    "Cc1ccc(cc1)O": "phenol",
    "CCc1ccc(cc1)O": "phenol",
    "Oc1ccccc1": "phenol",
    # pyridine (3)
    "c1cccnc1": "pyridine",
    "c1ccncn1": "pyridine",
    "n1ccncc1": "pyridine",
    # pyrrole (9)
    "c1ccc2c(c1)[nH]cc2": "pyrrole",
    "c1ccc[nH]1": "pyrrole",
    "c1ncc[nH]1": "pyrrole",
    "Cc1cnc[nH]1": "pyrrole",
    "Cc1c[nH]c2c1cccc2": "pyrrole",
    "Cc1c[nH]cn1": "pyrrole",
    "CCc1cnc[nH]1": "pyrrole",
    "CCc1c[nH]c2c1cccc2": "pyrrole",
    "CCc1c[nH]cn1": "pyrrole",
    # sulfides (21)
    "C1CCCS1": "sulfides",
    "C1CCCSC1": "sulfides",
    "C1CCSCS1": "sulfides",
    "C1CCSSC1": "sulfides",
    "C1CSSC1": "sulfides",
    "C1SCCS1": "sulfides",
    "CCCSCSC": "sulfides",
    "CCCSC": "sulfides",
    "CCCSSC": "sulfides",
    "CCSCC": "sulfides",
    "CCSSCC": "sulfides",
    "CCSSC": "sulfides",
    "CSCCCSC": "sulfides",
    "CSCCSC": "sulfides",
    "CSCC": "sulfides",
    "CSCSCC": "sulfides",
    "CSCSC": "sulfides",
    "CSC": "sulfides",
    "CSSC": "sulfides",
    "S1CCSCC1": "sulfides",
    "S1CSCSC1": "sulfides",
    # thiols (13)
    "CCCSS": "thiols",
    "CCCS": "thiols",
    "CCSS": "thiols",
    "CCS": "thiols",
    "CSCCCS": "thiols",
    "CSCCS": "thiols",
    "CSCS": "thiols",
    "CSS": "thiols",
    "CS": "thiols",
    "SCCCS": "thiols",
    "SCCS": "thiols",
    "SS": "thiols",
    "S": "thiols",
    # water (1)
    "O": "water",
}


def canonical_unmapped_smiles(molecule):
    molecule = Chem.Mol(molecule)
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    molecule = Chem.RemoveHs(molecule)
    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=False,
    )


def describe_des370k_dimer(mapped_smiles, lookup):
    molecule = Chem.MolFromSmiles(mapped_smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse SPICE SMILES {mapped_smiles!r}")

    fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
    if len(fragments) != 2:
        raise ValueError(
            f"Expected two monomers, found {len(fragments)} in {mapped_smiles!r}"
        )

    monomers = []
    for fragment in fragments:
        canonical = canonical_unmapped_smiles(fragment)
        try:
            source_smiles, class_name = lookup[canonical]
        except KeyError as exc:
            raise KeyError(
                f"No DES370K class for canonical monomer {canonical!r} "
                f"from {mapped_smiles!r}"
            ) from exc
        monomers.append(
            {
                "smiles": source_smiles,
                "mapped_smiles": Chem.MolToSmiles(
                    fragment,
                    canonical=True,
                    isomericSmiles=True,
                    allHsExplicit=True,
                ),
                "class": class_name,
            }
        )
    return monomers


def load_des370k_candidates(hdf5_path):
    """Read metadata once, retaining all dimers but none of their coordinates."""
    import h5py

    lookup = {}
    for source_smiles, class_name in SMILES_TO_CLASS.items():
        molecule = Chem.MolFromSmiles(source_smiles)
        if molecule is None:
            raise ValueError(f"RDKit could not parse class SMILES {source_smiles!r}")
        canonical = canonical_unmapped_smiles(molecule)
        if canonical in lookup:
            raise RuntimeError(
                f"Distinct DES370K SMILES {lookup[canonical][0]!r} and "
                f"{source_smiles!r} canonicalize to {canonical!r}"
            )
        lookup[canonical] = (source_smiles, class_name)

    by_pair = {}
    system_count = 0
    with h5py.File(hdf5_path, "r") as h5:
        for index, (name, group) in enumerate(h5.items(), start=1):
            subset = group["subset"].asstr()[0]
            if subset.startswith("SPICE DES370K Single Points Dataset"):
                system_count += 1
                smiles = group["smiles"].asstr()[0]
                monomers = describe_des370k_dimer(smiles, lookup)
                pair = tuple(sorted(item["class"] for item in monomers))
                if "other" not in pair:
                    by_pair.setdefault(pair, []).append(
                        {
                            "hdf5_group": name,
                            "dimer_mapped_smiles": smiles,
                            "monomers": monomers,
                            "n_conformers": int(group["conformations"].shape[0]),
                        }
                    )
            if index % 10000 == 0:
                print(f"Scanned {index:,}/{len(h5):,} HDF5 groups", flush=True)

    if not by_pair:
        raise RuntimeError(f"No classified DES370K dimers found in {hdf5_path}")
    for pair, dimers in by_pair.items():
        dimers.sort(
            key=lambda item: helpers.stable_rank(
                helpers.CONF_SELECTION_SEED, "dimer", *pair, item["hdf5_group"]
            )
        )
    return by_pair, system_count


def des370k_candidates(
    hdf5_path,
    pair,
    dimers,
    split,
    *,
    train_monomers=(),
):
    dimers = list(dimers)
    if train_monomers:
        dimers.sort(
            key=lambda dimer: sum(
                monomer["smiles"] in train_monomers for monomer in dimer["monomers"]
            )
        )

    for dimer in dimers:
        for index in helpers.ranked_conformers(
            dimer["n_conformers"], "conformer", dimer["hdf5_group"]
        ):
            numbers, positions, charge, smiles = helpers.read_spice_conformer(
                hdf5_path, dimer["hdf5_group"], index
            )
            yield {
                **dimer,
                "numbers": numbers,
                "positions": positions,
                "charge": charge,
                "smiles": smiles,
                "conformer_index": index,
                "xyz_file": f"des370k_{pair[0]}__{pair[1]}__{split}.xyz",
                "split": split,
            }


def build_des370k():
    inputs = helpers.ensure_spice_inputs()
    hdf5_path = inputs["spice_hdf5"]
    output_dir = helpers.MOLECULES_DIR
    by_pair, system_count = load_des370k_candidates(hdf5_path)
    workers = gfn_worker.default_worker_count()
    gfn_worker.check_runtime()
    with cf.ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        return _select_des370k(
            hdf5_path, output_dir, by_pair, system_count, pool=pool, workers=workers
        )


def _select_des370k(hdf5_path, output_dir, by_pair, system_count, *, pool, workers):
    """Finish training before ranking validation, reusing the same worker pool."""
    train_streams = (
        (
            " / ".join(pair) + " [train]",
            des370k_candidates(hdf5_path, pair, by_pair[pair], "train"),
        )
        for pair in sorted(by_pair)
    )
    train_results = _select_pairs(train_streams, output_dir, pool=pool, workers=workers)
    valid_results = {}
    if train_results:
        valid_streams = _validation_streams(hdf5_path, by_pair, train_results)
        valid_results = _select_pairs(
            valid_streams, output_dir, pool=pool, workers=workers
        )

    selected = _publish_selection(
        output_dir, system_count, train_results, valid_results
    )
    _report_coverage(by_pair, train_results, valid_results, len(selected))
    return selected


def _select_pairs(streams, output_dir, *, pool, workers):
    results = {}
    for candidate, result in helpers.select_candidates(
        streams, output_dir / "data" / "tmp", pool=pool, workers=workers
    ):
        pair = tuple(sorted(monomer["class"] for monomer in candidate["monomers"]))
        results[pair] = (candidate, result)
    return results


def _validation_streams(hdf5_path, by_pair, train_results):
    # Rank validation against monomers from the complete training pass.
    train_monomers = {
        monomer["smiles"]
        for candidate, _ in train_results.values()
        for monomer in candidate["monomers"]
    }
    valid_streams = []
    for pair, (train_candidate, _) in sorted(train_results.items()):
        train_index = next(
            index
            for index, dimer in enumerate(by_pair[pair])
            if dimer["hdf5_group"] == train_candidate["hdf5_group"]
        )
        # Earlier dimers already failed during training; never retry those
        # or the accepted training dimer for validation.
        valid_streams.append(
            (
                " / ".join(pair) + " [valid]",
                des370k_candidates(
                    hdf5_path,
                    pair,
                    by_pair[pair][train_index + 1 :],
                    "valid",
                    train_monomers=train_monomers,
                ),
            )
        )
    return valid_streams


def _publish_selection(output_dir, system_count, train_results, valid_results):
    selected = []
    baselines = {}
    for pair in sorted(train_results):
        pair_results = [train_results[pair]]
        if pair in valid_results:
            pair_results.append(valid_results[pair])
        for candidate, result in pair_results:
            key = helpers.publish_candidate(output_dir, candidate, result)
            baselines[key] = result["baseline"]
            selected.append(
                {
                    **{
                        key: value
                        for key, value in candidate.items()
                        if key not in {"numbers", "positions", "smiles"}
                    },
                    "charge": result["charge"],
                }
            )

    selected.sort(key=lambda item: item["xyz_file"])
    helpers.write_json(
        output_dir / "data" / "des370k_selected_dimers.json",
        {
            "des370k_system_count": system_count,
            "selected_dimer_count": len(selected),
            "selected_dimers": selected,
        },
    )
    helpers.merge_baselines(output_dir, baselines)
    return selected


def _report_coverage(by_pair, train_results, valid_results, selected_count):
    missing_train_pairs = sorted(by_pair.keys() - train_results.keys())
    missing_valid_pairs = sorted(train_results.keys() - valid_results.keys())
    print(
        f"DES370K: accepted train for {len(train_results)}/{len(by_pair)} class "
        f"pairs; valid for {len(valid_results)}/{len(train_results)} accepted "
        f"train pairs ({selected_count} dimers)",
        flush=True,
    )
    if missing_train_pairs:
        print(
            "Skipped exhausted DES370K train class pairs: "
            + "; ".join(" / ".join(pair) for pair in missing_train_pairs),
            flush=True,
        )
    if missing_valid_pairs:
        print(
            "DES370K train pairs without an accepted valid dimer: "
            + "; ".join(" / ".join(pair) for pair in missing_valid_pairs),
            flush=True,
        )


if __name__ == "__main__":
    build_des370k()
