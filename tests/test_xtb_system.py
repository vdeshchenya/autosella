"""Tests for xtb_molecular_system.XTBMolecularSystem.

Integration cases require the xtb binary on PATH. Failure-classification
cases inject subprocess responses and run without an installed xTB binary.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.requires_xtb
def test_construction_from_xyz(water_xyz_path):
    from xtb_molecular_system import XTBMolecularSystem

    sys_ = XTBMolecularSystem(str(water_xyz_path))
    assert list(sys_.atomic_numbers) == [8, 1, 1]
    assert sys_.initial_positions.shape == (3, 3)


@pytest.mark.requires_xtb
def test_compute_returns_finite_energy_and_forces(water_xyz_path):
    from xtb_molecular_system import XTBMolecularSystem

    sys_ = XTBMolecularSystem(str(water_xyz_path))
    E, F = sys_.compute(sys_.initial_positions)

    assert np.isfinite(E)
    assert F.shape == (3, 3)
    assert np.all(np.isfinite(F))


@pytest.mark.requires_xtb
def test_compute_gives_higher_energy_on_perturbed(water_xyz_path):
    """Relaxed water has lower energy than a stretched variant."""
    from xtb_molecular_system import XTBMolecularSystem

    sys_ = XTBMolecularSystem(str(water_xyz_path))
    E_relaxed, _ = sys_.compute(sys_.initial_positions)

    perturbed = sys_.initial_positions.copy()
    perturbed[1, 0] += 0.1  # stretch one O-H bond by ~1 Å (0.1 nm)
    E_stretched, _ = sys_.compute(perturbed)

    # The stretched geometry should have higher energy (within tolerance;
    # water equilibrium OH is ~0.96 Å and the fixture uses 0.95 Å)
    assert E_stretched > E_relaxed


@pytest.mark.requires_xtb
def test_compute_rejects_bad_input(water_xyz_path):
    """NaN input raises rather than returning a sentinel.

    The old contract returned `(inf, zeros)` on any failure, which let a broken
    geometry flow silently into the optimizer and be scored. compute() now
    fails loudly; the worker turns that into a per-task error and an invalid run.
    """
    from xtb_molecular_system import XTBMolecularSystem

    sys_ = XTBMolecularSystem(str(water_xyz_path))
    bad = np.full_like(sys_.initial_positions, np.nan)
    with pytest.raises(ValueError, match="NaN or Inf"):
        sys_.compute(bad)


@pytest.mark.requires_xtb
def test_compute_rejects_wrong_shape(water_xyz_path):
    from xtb_molecular_system import XTBMolecularSystem

    sys_ = XTBMolecularSystem(str(water_xyz_path))
    with pytest.raises(ValueError, match="must have shape"):
        sys_.compute(np.zeros((5, 3)))


@pytest.mark.requires_xtb
def test_charge_is_passed_through_to_xtb(tmp_path, water_xyz_path):
    """The XYZ `charge=` tag must actually change the xTB potential.

    This is the end-to-end check that the charge survives
    read_xyz_charge -> _write_xyz -> `xtb --chrg`. Same geometry, different
    declared charge => different energy.
    """
    from xtb_molecular_system import XTBMolecularSystem

    body = water_xyz_path.read_text().split("\n")
    energies = {}
    for charge in (0, 1):
        path = tmp_path / f"water_q{charge}.xyz"
        path.write_text("\n".join([body[0], f"charge={charge}", *body[2:]]))
        sys_ = XTBMolecularSystem(str(path))
        assert sys_.charge == charge
        energies[charge], _ = sys_.compute(sys_.initial_positions)

    assert np.isfinite(energies[0]) and np.isfinite(energies[1])
    assert abs(energies[1] - energies[0]) > 1.0  # kJ/mol; cation is far higher


# Exact diagnostics retained in worker_recovery_local_20260908.json. Keep the
# SCF marker explicit: generic "did not converge" may describe another failure.
SCF_FAILURE = "-1- scf: Self consistent charge iterator did not converge\n"
SCF_ITERATION_LIMIT = "   *** convergence criteria cannot be satisfied within 250 iterations ***\n"


@pytest.fixture
def mocked_xtb(monkeypatch, water_xyz_path):
    import xtb_molecular_system as xtb

    monkeypatch.setattr(xtb, "_resolve_xtb_binary", lambda: "/test/xtb")
    return xtb, str(water_xyz_path)


@pytest.mark.parametrize("method, gfn", [("GFN2-xTB", "2"), ("GFN1-xTB", "1")])
def test_xtb_command_preserves_method_units_charge_and_thread_caps(
    mocked_xtb, monkeypatch, tmp_path, method, gfn
):
    from pathlib import Path
    from subprocess import CompletedProcess

    xtb, _ = mocked_xtb
    xyz = tmp_path / "selenium.xyz"
    xyz.write_text("1\ncharge=-2\nSe 1.0 -2.0 3.0\n")

    def completed_xtb(command, *, cwd, env, **kwargs):
        assert command[command.index("--gfn") + 1] == gfn
        assert "--gfnff" not in command
        assert command[command.index("--chrg") + 1] == "-2"
        assert command[command.index("--acc") + 1] == "0.2"
        assert command[command.index("--parallel") + 1] == "2"
        for name in (
            "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            assert env[name] == "2"
        assert env["OMP_DYNAMIC"] == env["MKL_DYNAMIC"] == "FALSE"
        written_xyz = Path(command[1]).read_text().splitlines()
        assert written_xyz[:2] == ["1", "charge=-2"]
        assert written_xyz[2].split()[0] == "Se"
        np.testing.assert_allclose(
            np.array(written_xyz[2].split()[1:], dtype=float), [1.0, -2.0, 3.0]
        )
        Path(cwd, "gradient").write_text(
            "$grad\nSCF energy = -1.5D+00\n1.0D-03 -2.0D-03 3.0D-03\n$end\n"
        )
        return CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(xtb.subprocess, "run", completed_xtb)
    system = xtb.XTBMolecularSystem(str(xyz), method=method, num_threads=2)
    energy, forces = system.compute(system.initial_positions)
    assert energy == -1.5 * xtb.HARTREE_TO_KJ
    np.testing.assert_allclose(forces, [[-1e-3 * xtb.FORCE_CONV,
                                       2e-3 * xtb.FORCE_CONV,
                                       -3e-3 * xtb.FORCE_CONV]])


def test_evolution_calculator_rejects_preparation_potential(mocked_xtb):
    xtb, xyz = mocked_xtb
    with pytest.raises(ValueError, match="Unknown xTB method"):
        xtb.XTBMolecularSystem(xyz, method="GFN-FF")


@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize("scf_stream", ["stdout", "stderr"])
def test_scf_nonconvergence_is_numerical_in_both_output_modes(
    mocked_xtb, monkeypatch, capsys, verbose, scf_stream
):
    from subprocess import CompletedProcess, PIPE

    xtb, xyz = mocked_xtb
    stdout = SCF_ITERATION_LIMIT + (SCF_FAILURE if scf_stream == "stdout" else "")
    stderr = "abnormal termination of xtb\n" + (SCF_FAILURE if scf_stream == "stderr" else "")

    def failed_xtb(command, **kwargs):
        assert kwargs["stdout"] == kwargs["stderr"] == PIPE
        assert kwargs["text"] is True
        return CompletedProcess(command, 1, stdout, stderr)

    monkeypatch.setattr(xtb.subprocess, "run", failed_xtb)
    system = xtb.XTBMolecularSystem(xyz, verbose=verbose)
    with pytest.raises(xtb.NumericalCalculationError) as failure:
        system.compute(system.initial_positions)
    assert "status 1" in str(failure.value)
    assert SCF_FAILURE.strip() in str(failure.value)
    assert SCF_ITERATION_LIMIT.strip() in str(failure.value)
    captured = capsys.readouterr()
    assert captured.out == (stdout if verbose else "")
    assert captured.err == (stderr if verbose else "")


@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize("returncode, stdout", [
    (-9, SCF_ITERATION_LIMIT + SCF_FAILURE),
    (1, "geometry optimization did not converge\n"),
    (1, SCF_ITERATION_LIMIT),
    (2, "permission denied opening calculator data\n"),
])
def test_unrecognized_or_signal_exit_remains_runtime_error(
    mocked_xtb, monkeypatch, capsys, verbose, returncode, stdout
):
    from subprocess import CompletedProcess

    xtb, xyz = mocked_xtb
    stderr = "abnormal termination of xtb\n"
    monkeypatch.setattr(xtb.subprocess, "run", lambda command, **kwargs:
                        CompletedProcess(command, returncode, stdout, stderr))
    system = xtb.XTBMolecularSystem(xyz, verbose=verbose)
    with pytest.raises(RuntimeError, match=f"status {returncode}"):
        system.compute(system.initial_positions)
    captured = capsys.readouterr()
    assert captured.out == (stdout if verbose else "")
    assert captured.err == (stderr if verbose else "")


def test_scf_classification_uses_output_before_diagnostic_truncation(mocked_xtb, monkeypatch):
    from subprocess import CompletedProcess

    xtb, xyz = mocked_xtb
    stdout = SCF_ITERATION_LIMIT + SCF_FAILURE + "trailing diagnostic\n" * 500
    monkeypatch.setattr(xtb.subprocess, "run", lambda command, **kwargs:
                        CompletedProcess(command, 1, stdout, ""))
    system = xtb.XTBMolecularSystem(xyz)
    with pytest.raises(xtb.NumericalCalculationError):
        system.compute(system.initial_positions)


@pytest.mark.parametrize("exception", [FileNotFoundError, PermissionError, OSError])
def test_process_launch_io_errors_propagate_unchanged(mocked_xtb, monkeypatch, exception):
    xtb, xyz = mocked_xtb

    def launch_failure(*args, **kwargs):
        raise exception("calculator process could not be launched")

    monkeypatch.setattr(xtb.subprocess, "run", launch_failure)
    system = xtb.XTBMolecularSystem(xyz)
    with pytest.raises(exception, match="could not be launched"):
        system.compute(system.initial_positions)


def test_verbose_success_replays_both_streams_and_returns_calculation(
    mocked_xtb, monkeypatch, capsys
):
    from pathlib import Path
    from subprocess import CompletedProcess

    xtb, xyz = mocked_xtb
    stdout, stderr = "completed calculation\n", "calculator diagnostic\n"

    def completed_xtb(command, *, cwd, **kwargs):
        Path(cwd, "gradient").write_text("$grad\nSCF energy = -1.0\n" + "0 0 0\n" * 3 + "$end\n")
        return CompletedProcess(command, 0, stdout, stderr)

    monkeypatch.setattr(xtb.subprocess, "run", completed_xtb)
    system = xtb.XTBMolecularSystem(xyz, verbose=True)
    energy, forces = system.compute(system.initial_positions)
    assert energy == -xtb.HARTREE_TO_KJ
    np.testing.assert_array_equal(forces, np.zeros((3, 3)))
    captured = capsys.readouterr()
    assert captured.out == stdout
    assert captured.err == stderr
