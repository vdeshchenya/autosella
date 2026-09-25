"""Source downloads, raw SPICE reading, ranking, and accepted outputs."""

import hashlib
import json
import os
import tarfile
import tempfile
from io import BytesIO
import shutil
from collections import deque
from pathlib import Path

import requests

from molecules import gfn_worker

MOLECULES_DIR = Path(__file__).resolve().parent
CONF_SELECTION_SEED = 42
N_TRAIN_PUBCHEM = 225
N_VALID_PUBCHEM = 250
MAX_CONFORMERS = 10


SPICE_HDF5_URL = "https://zenodo.org/records/10975225/files/SPICE-2.0.1.hdf5"
SPICE_HDF5_CHECKSUM = "md5:bfba2224b6540e1390a579569b475510"

SPICE_DATA_DIR = MOLECULES_DIR / "data" / "spice"
SPICE_HDF5_PATH = SPICE_DATA_DIR / "SPICE-2.0.1.hdf5"

ROWANSCI_INPUT_DIR = MOLECULES_DIR / "data" / "xyz_rowansci_benchmark"
ROWANSCI_COMMIT = "83c604e3e626ff3df2094eff8a2340064e76601c"
ROWANSCI_ARCHIVE_URL = (
    "https://codeload.github.com/rowansci/benchmark-structures/"
    f"tar.gz/{ROWANSCI_COMMIT}"
)
ROWANSCI_ARCHIVE_SHA256 = (
    "06b0ea7900876278e4087f4d2ffee930cbaf8275b71255117fc29374e6d29412"
)
ROWANSCI_FILE_SHA256 = {
    "abemaciclib.xyz": "3aba8276526f857afc41efb8a4f9be790c189a300e6e79d299595feb7758ff1b",
    "acalabrutinib.xyz": "2529b156bde67afdcbfe3e43f19cf436ff11aabc9b63965a9229fe8510eabe05",
    "apalutamide.xyz": "78c4a4cd856c0606eb41a10a88af66135ac8b9ebc9b25b6295f443d76a142f35",
    "apixaban.xyz": "21015cfc6d2fb8cbb5d9408981097484245e06b09395461752ee2291820d1eef",
    "brexpiprazole.xyz": "793564a7b5a03a4484c27577c935c00b2e8e37355248f1ab14d4fd373514f6b5",
    "cariprazine.xyz": "02b4c1a8fc5860d2ef5d5ee1fff3f4035c86620b25244e3568b38db96e18ee87",
    "dapagliflozin.xyz": "a248cdbb145deb45e42ca6968f2c4f7e3fcf0fab1519bb6dd677c40730f42ecc",
    "empagliflozin.xyz": "5dced95c8d3747d8195670a834c7adb36f6fa82e60979fce60fc467a4f36e94b",
    "enzalutamide.xyz": "72181b15037b9b1cd1e7c2604df0d3fa9f2df59c317f843f9f1cf8f2abd97262",
    "ibrutinib.xyz": "dae1ca3aa261f3c723a28d219db54261c539f51238a23e3775051dff00cbacf6",
    "lenalidomide.xyz": "34dc6d9976def1900845498d451b7cd2d6f04e35e11f80dbe60c50b85226eae2",
    "lenvatinib.xyz": "473ce2e9ce4d3aa1c790235a732f591f6252af50c54e4282c6bd50efaea5e4f6",
    "lisdexamphetamine.xyz": "5a3cbeaaed6753d8c59d401818d6194efaaaa26c3c8b7fba182a357095220ad4",
    "nintedanib.xyz": "4123a491309a3f49f79f0984666f4fc34e6e9b8219f10095c88e45406c1b60e6",
    "olaparib.xyz": "8f483bb1ff578b31aefc6f30cabe493bdb374485a3a0f6dff6132560a1061700",
    "osimertinib.xyz": "3a369393f48c9e88efeefcb556caf44f47e5eedb49b1c7fbd5589c1bcd5d44a6",
    "palbociclib.xyz": "365db71b454927aca1bccb8c2547828a6ed2a7cfd3ac92aed07f312bd566a5b3",
    "paliperidone_palmitate.xyz": "dac71e8ff687780bab7f013faaeabbac5a7288df6133fc4ee6f2579b590b4e63",
    "pomalidomide.xyz": "f8eaa0a60c45139b062be66df63478924fe2ef46370e8f477a52d4204f784f75",
    "ribociclib.xyz": "174391b82dc697c03bef866e3729c61b35cd7562c532b56dee7705669e123424",
    "rivaroxaban.xyz": "fdb1a23c4bca573005bd8262241a68a5b01036ffbe0b635d9fabbbffdd2b0d91",
    "tafamidis.xyz": "190b2849fe2ac1a929a9b608e09c264a2eb7ff9bd057a0d528a5b8c3e8e5aa41",
    "upadacitinib.xyz": "aa4de87544d61dd1acfe4d89082781b8e9bb7091acd93e8a007c904af52c4879",
    "venetoclax.xyz": "f46aeefb15d42116a67e7c5c0834aaad7e6ea9398c5d64ed2ec8d01616368a60",
    "zanubrutinib.xyz": "8dce7cce6afed5436285d8ed77c79111b28f6b3398b80600d7c6e20c31604597",
}
ROWANSCI_SOURCE = {
    "repository": "https://github.com/rowansci/benchmark-structures",
    "commit": ROWANSCI_COMMIT,
    "archive_url": ROWANSCI_ARCHIVE_URL,
    "archive_sha256": ROWANSCI_ARCHIVE_SHA256,
    "files": ROWANSCI_FILE_SHA256,
}
EXPECTED_ROWANSCI_XYZ_FILES = 25

_DOWNLOAD_CHUNK_SIZE = 1024 * 1024


def _new_digest(algorithm: str):
    # Zenodo publishes an MD5 for this artifact; it is used only for integrity.
    return hashlib.new(algorithm, usedforsecurity=False)


def file_checksum(path: str | Path, algorithm: str) -> str:
    """Return a streaming checksum without loading a large file into memory."""
    digest = _new_digest(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(_DOWNLOAD_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_expected_checksum(expected_checksum: str) -> tuple[str, str]:
    try:
        algorithm, expected = expected_checksum.split(":", 1)
    except ValueError as exc:
        raise ValueError(
            "expected_checksum must have the form 'algorithm:hex-digest'"
        ) from exc
    _new_digest(algorithm)
    return algorithm, expected.lower()


def verify_checksum(path: str | Path, expected_checksum: str) -> None:
    """Raise when a file does not match an ``algorithm:hex-digest`` value."""
    algorithm, expected = _parse_expected_checksum(expected_checksum)
    actual = file_checksum(path, algorithm)
    if actual.lower() != expected:
        raise RuntimeError(
            f"Checksum mismatch for {Path(path)}: expected "
            f"{expected_checksum}, got {algorithm}:{actual}"
        )


def download_verified_file(
    url: str,
    output_path: str | Path,
    expected_checksum: str,
) -> Path:
    """Download to a temporary file, verify it, then publish it atomically."""
    output = Path(output_path)
    if output.is_file():
        verify_checksum(output, expected_checksum)
        print(f"Verified existing input: {output}")
        return output
    if output.exists():
        raise RuntimeError(f"Input path exists but is not a file: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".part")
    partial.unlink(missing_ok=True)
    print(f"Downloading {url} -> {output}")
    algorithm, expected = _parse_expected_checksum(expected_checksum)
    digest = _new_digest(algorithm)

    try:
        with requests.get(url, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with partial.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        stream.write(chunk)
                        digest.update(chunk)
        actual = digest.hexdigest()
        if actual.lower() != expected:
            raise RuntimeError(
                f"Checksum mismatch for downloaded {output}: expected "
                f"{expected_checksum}, got {algorithm}:{actual}"
            )
        os.replace(partial, output)
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    print(f"Downloaded and verified: {output}")
    return output


def ensure_spice_inputs() -> dict[str, Path]:
    """Ensure the pinned SPICE HDF5 file exists in the shared data directory."""
    return {
        "spice_hdf5": download_verified_file(
            SPICE_HDF5_URL,
            SPICE_HDF5_PATH,
            SPICE_HDF5_CHECKSUM,
        )
    }


def ensure_rowansci_inputs() -> list[Path]:
    """Verify pinned RowanSci contents, downloading again if the cache differs."""
    expected_files = ROWANSCI_SOURCE["files"]
    if len(expected_files) != EXPECTED_ROWANSCI_XYZ_FILES:
        raise RuntimeError("Pinned RowanSci manifest has an unexpected input count")
    existing = sorted(ROWANSCI_INPUT_DIR.glob("*.xyz"))
    if {path.name for path in existing} == set(expected_files) and all(
        file_checksum(path, "sha256") == expected_files[path.name] for path in existing
    ):
        return existing

    print("Downloading RowanSci optimization structures ...", flush=True)
    response = requests.get(ROWANSCI_ARCHIVE_URL, timeout=60)
    response.raise_for_status()
    if (
        hashlib.sha256(response.content).hexdigest()
        != ROWANSCI_SOURCE["archive_sha256"]
    ):
        raise RuntimeError("Checksum mismatch for pinned RowanSci archive")
    with tarfile.open(fileobj=BytesIO(response.content), mode="r:gz") as archive:
        members = sorted(
            (
                member
                for member in archive.getmembers()
                if member.isfile()
                and Path(member.name).parent.name == "optimization"
                and member.name.endswith(".xyz")
            ),
            key=lambda member: member.name,
        )
        if len(members) != EXPECTED_ROWANSCI_XYZ_FILES:
            raise RuntimeError(
                f"Expected {EXPECTED_ROWANSCI_XYZ_FILES} RowanSci XYZ files, "
                f"found {len(members)}"
            )
        if {Path(member.name).name for member in members} != set(expected_files):
            raise RuntimeError("Pinned RowanSci archive has unexpected XYZ filenames")
        contents = {}
        for member in members:
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Could not extract {member.name}")
            name = Path(member.name).name
            contents[name] = extracted.read()
            if hashlib.sha256(contents[name]).hexdigest() != expected_files[name]:
                raise RuntimeError(f"Checksum mismatch for RowanSci input {name}")

        # Validate every member before replacing any cached input.
        ROWANSCI_INPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="rowansci-", dir=ROWANSCI_INPUT_DIR.parent
        ) as staging:
            for name, content in contents.items():
                (Path(staging) / name).write_bytes(content)
            ROWANSCI_INPUT_DIR.mkdir(parents=True, exist_ok=True)
            for name in contents:
                os.replace(Path(staging) / name, ROWANSCI_INPUT_DIR / name)
        for old_input in ROWANSCI_INPUT_DIR.glob("*.xyz"):
            if old_input.name not in expected_files:
                old_input.unlink()
    return [ROWANSCI_INPUT_DIR / name for name in sorted(expected_files)]


def read_spice_conformer(hdf5_path, group_name, index):
    """Return atomic numbers, positions in nm, charge, and source SMILES."""
    import h5py
    from rdkit import Chem
    from molecules.utils import BOHR_PER_NM

    with h5py.File(hdf5_path, "r") as h5:
        group = h5[group_name]
        numbers = group["atomic_numbers"][:]
        positions = group["conformations"][index].astype(float) / BOHR_PER_NM
        smiles = group["smiles"].asstr()[0]
    charge = int(Chem.GetFormalCharge(Chem.MolFromSmiles(smiles, sanitize=False)))
    return numbers, positions, charge, smiles


def stable_rank(seed, *parts):
    """Return the same molecule-local SHA-256 rank as the original preparation."""
    payload = "\0".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode("utf-8")).digest()


def ranked_conformers(n_conformers, *parts):
    return sorted(
        range(n_conformers),
        key=lambda index: stable_rank(CONF_SELECTION_SEED, *parts, index),
    )[:MAX_CONFORMERS]


def spice_candidates(hdf5_path, group, n_conformers, *, xyz_file=None):
    for index in ranked_conformers(n_conformers, group):
        numbers, positions, charge, smiles = read_spice_conformer(
            hdf5_path, group, index
        )
        yield {
            "numbers": numbers,
            "positions": positions,
            "charge": charge,
            "smiles": smiles,
            "hdf5_group": group,
            "conformer_index": index,
            "xyz_file": xyz_file or f"{group}.xyz",
        }


def select_first_candidate(candidates, work_dir):
    """Try conformers sequentially and stop immediately at the first pass."""
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        result = gfn_worker.run_candidate(candidate, work_dir)
        if result["status"] == "fatal":
            raise RuntimeError(
                f"{candidate['xyz_file']}: infrastructure failure at "
                f"{result['stage']}: {result['error']}"
            )
        if result["status"] == "ok":
            return candidate, result
        reason = result["error"]
        if result["stage"] != "energy":
            reason = f"{result['stage']} {reason}"
        print(
            f"Rejected {candidate['xyz_file']} "
            f"(conformer {candidate['conformer_index']}): {reason}",
            flush=True,
        )
    return None


def canonical_unmapped_smiles(mapped_smiles):
    """Remove atom maps and explicit hydrogens, preserving stereochemistry."""
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(mapped_smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse mapped SMILES {mapped_smiles!r}")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    molecule = Chem.RemoveHs(molecule)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def check_connectivity(atomic_numbers, final_positions, reference_bonds):
    """Check final/reference mismatches using relative covalent-radius distances."""
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    from molecules.utils import ANGSTROM_TO_NM

    positions = np.asarray(final_positions) / ANGSTROM_TO_NM
    molecule = Chem.RWMol()
    conformer = Chem.Conformer(len(atomic_numbers))
    for index, (number, position) in enumerate(zip(atomic_numbers, positions)):
        molecule.AddAtom(Chem.Atom(int(number)))
        conformer.SetAtomPosition(index, position)
    molecule.AddConformer(conformer)
    rdDetermineBonds.DetermineConnectivity(molecule, useVdw=False)
    final = Chem.GetAdjacencyMatrix(molecule)
    periodic_table = Chem.GetPeriodicTable()
    formed, removed = [], []
    for i, j in np.argwhere(np.triu(reference_bonds != final, 1)):
        radius_sum = (
            periodic_table.GetRcovalent(int(atomic_numbers[i]))
            + periodic_table.GetRcovalent(int(atomic_numbers[j]))
        )
        g_ij = np.linalg.norm(positions[i] - positions[j]) / radius_sum
        pair = [int(i + 1), int(j + 1)]
        if reference_bonds[i, j] == 1:
            if g_ij >= 1:
                removed.append(pair)
        elif g_ij <= 1:
            formed.append(pair)
    if formed or removed:
        raise ValueError(f"issue: formed ({formed}), removed ({removed})")


def check_atom_separation(initial_positions, final_positions):
    """Reject a final system diameter over 2 times its initial value."""
    import numpy as np

    def diameter(positions):
        positions = np.asarray(positions)
        distances = positions[:, None, :] - positions[None, :, :]
        return np.linalg.norm(distances, axis=-1).max(initial=0.0)

    initial = diameter(initial_positions)
    final = diameter(final_positions)
    if final > 2 * initial:
        raise ValueError(
            f"Maximum atom separation increased from {initial:.6g} to "
            f"{final:.6g} nm (limit: 2 times initial)"
        )


def _run_batch(candidates, work_dir, pool):
    """Run one bounded batch; preserve input order and surface every fatal error."""
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    futures = []
    try:
        for candidate in candidates:
            futures.append(pool.submit(gfn_worker.run_candidate, candidate, work_dir))
        results = [future.result() for future in futures]
        for candidate, result in zip(candidates, results):
            if result["status"] == "fatal":
                raise RuntimeError(
                    f"{candidate['xyz_file']}: infrastructure failure at "
                    f"{result['stage']}: {result['error']}"
                )
            if result["status"] == "error":
                reason = result["error"]
                if result["stage"] != "energy":
                    reason = f"{result['stage']} {reason}"
                label = candidate["xyz_file"]
                details = f"conformer {candidate.get('conformer_index', '-')}"
                if "monomers" in candidate:
                    label = " / ".join(sorted(
                        monomer["class"] for monomer in candidate["monomers"]
                    ))
                    details = f"{candidate['hdf5_group']}, {details}"
                print(
                    f"Rejected {label} ({details}): {reason}",
                    flush=True,
                )
        return results
    except BaseException:
        # Drain submitted work and remove successes that cannot be published.
        for future in futures:
            if future.cancel():
                continue
            try:
                result = future.result()
            except BaseException:
                continue
            if result["status"] == "ok":
                shutil.rmtree(result["work_dir"])
        raise

def select_candidates(streams, work_dir, pool, workers):
    """Try one conformer per molecule/class pair in each parallel batch."""
    if workers < 1:
        raise ValueError("workers must be positive")
    pending = deque((label, iter(candidates)) for label, candidates in streams)
    selected = []
    try:
        while pending:
            batch = []
            retry = []
            while pending and len(batch) < workers:
                label, candidates = pending.popleft()
                candidate = next(candidates, None)
                if candidate is not None:
                    batch.append(candidate)
                    retry.append((label, candidates))
            results = _run_batch(batch, work_dir, pool)
            for candidate, result, stream in zip(batch, results, retry):
                if result["status"] == "ok":
                    selected.append((candidate, result))
                    message = f"Accepted {stream[0]}"
                    if "monomers" in candidate:
                        message += (
                            f": {candidate['hdf5_group']}, "
                            f"conformer {candidate['conformer_index']}"
                        )
                    print(message, flush=True)
                else:
                    pending.append(stream)
        return selected
    except BaseException:
        for _, result in selected:
            shutil.rmtree(result["work_dir"])
        raise


def write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def publish_candidate(output_dir, candidate, result, save_final=True):
    """Only successful, accepted candidates enter the active XYZ directories."""
    output_dir = Path(output_dir)
    name = candidate["xyz_file"]
    key = Path(name).stem
    for directory, source, filename in (
        ("xyz", result["input_path"], name),
        ("xyz_final", result.get("final_path") if save_final else None, f"{key}.xyz"),
    ):
        if source is None:
            continue
        destination = output_dir / directory / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    shutil.rmtree(result["work_dir"])
    return key


def merge_baselines(output_dir, baselines):
    path = Path(output_dir) / "data" / "baseline_XTB.json"
    combined = json.loads(path.read_text()) if path.exists() else {}
    combined.update(baselines)
    write_json(path, combined)
