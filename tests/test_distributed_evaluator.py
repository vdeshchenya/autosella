"""Evaluator integration with durable recovery, using a deterministic fake service."""
import pytest
import validate as vd

SOURCE = {"kind": "source", "source": "def minimize_func(*args): return None"}

class Service:
    def __init__(self, payloads):
        self.payloads = payloads
        self.submitted = {}
        self.results = {}
    def get_result(self, task_id):
        return self.results.get(task_id)
    def task_status(self, task_id):
        return {"state": "finished" if task_id in self.results else "missing"}
    def submit(self, task, task_id=None):
        if task_id not in self.submitted:
            self.submitted[task_id] = task
            self.results[task_id] = self.payloads[task["mol_name"]]
        return task_id

@pytest.fixture
def evaluator(monkeypatch, tmp_path):
    monkeypatch.setattr(vd, "discover_xtb_molecules", lambda *_: (["A", "B"], {
        k: {"n_steps": 10, "improvement": 20., "n_atoms": n,
            "xyz_path": f"/approved/{k}.xyz", "initial_energy": -100., "final_energy": -120.}
        for k, n in [("A", 3), ("B", 50)]}))
    return vd.Evaluator(molecules_dir=tmp_path)

def payload(name):
    return {"result": {"mol_name": name, "rel_steps": .5, "rel_energy": 1., "converged": False},
            "error": None, "error_kind": None}

def test_evaluator_dispatch_and_resume(evaluator, monkeypatch, tmp_path):
    service = Service({k: payload(k) for k in ["A", "B"]})
    monkeypatch.setattr(vd, "RemoteOptimizationClient", lambda **kw: service)
    state = tmp_path / "state.json"
    result = evaluator.evaluate(SOURCE, state_path=state)
    assert result["status"] == "complete"
    assert len(result["results"]) == 2
    assert len(service.submitted) == 2
    for task in service.submitted.values():
        assert task["max_steps"] == vd.MAX_FORCE_CALLS
        assert task["mode"] == "xtb"
        assert task["optimizer_spec"]["kind"] == "source"
    again = evaluator.evaluate(SOURCE, state_path=state)
    assert again["results"] == result["results"]
    assert len(service.submitted) == 2

def test_optimizer_error_is_terminal(evaluator, monkeypatch, tmp_path):
    service = Service({"A": payload("A"), "B": {"result": None, "error": "optimizer raised", "error_kind": "optimizer_error"}})
    monkeypatch.setattr(vd, "RemoteOptimizationClient", lambda **kw: service)
    out = evaluator.evaluate(SOURCE, state_path=tmp_path / "state.json")
    assert out["status"] == "complete"
    assert out["num_errors"] == 1
    assert len(service.submitted) == 2

def test_outage_is_pending_and_preserves_ids(evaluator, monkeypatch, tmp_path):
    def unavailable(**kw):
        raise ConnectionError("offline")
    monkeypatch.setattr(vd, "RemoteOptimizationClient", unavailable)
    monkeypatch.setattr("distributed_validate.recovery.time.sleep", lambda _: None)
    state = tmp_path / "state.json"
    out = evaluator.evaluate(SOURCE, state_path=state)
    assert out["status"] == "pending"
    assert out["num_errors"] == 0
    assert out["results"] == []
    assert state.exists()

def test_source_top_level_does_not_run_on_coordinator(evaluator, monkeypatch, tmp_path):
    marker = tmp_path / "executed"
    source = {"kind": "source", "source": f"open({str(marker)!r}, 'w').write('bad')\ndef minimize_func(*args): pass"}
    service = Service({k: payload(k) for k in ["A", "B"]})
    monkeypatch.setattr(vd, "RemoteOptimizationClient", lambda **kw: service)
    evaluator.evaluate(source, state_path=tmp_path / "state.json")
    assert not marker.exists()

def test_empty_dataset_does_not_connect(evaluator, monkeypatch):
    evaluator.molecules = []
    monkeypatch.setattr(vd, "RemoteOptimizationClient", lambda **kw: pytest.fail("unexpected Redis"))
    out = evaluator.evaluate(SOURCE)
    assert out == {"status": "complete", "results": [], "num_errors": 0, "errors": []}
