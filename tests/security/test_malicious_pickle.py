"""
Regression tests: verify RestrictedUnpickler blocks malicious payloads.
These tests MUST pass before any deployment.
"""

import io
import os
import pickle
import sys
import tempfile
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class MaliciousPayload:
    """A pickle payload that would execute os.system if unpickled unsafely."""
    def __reduce__(self):
        return (os.system, ("echo MALICIOUS_CODE_EXECUTED > /tmp/pwned",))


def make_malicious_pickle() -> bytes:
    return pickle.dumps(MaliciousPayload())


def test_restricted_unpickler_blocks_os_system():
    """RestrictedUnpickler must raise UnpicklingError for os.system payloads."""
    from celery_worker import RestrictedUnpickler
    malicious_bytes = make_malicious_pickle()
    with pytest.raises(pickle.UnpicklingError):
        RestrictedUnpickler(io.BytesIO(malicious_bytes)).load()
    # Verify file was NOT created (payload did not execute)
    assert not os.path.exists("/tmp/pwned"), "Malicious payload executed — CRITICAL SECURITY FAILURE"


def test_restricted_unpickler_blocks_subprocess():
    """RestrictedUnpickler must block subprocess module access."""
    import subprocess
    class SubprocessPayload:
        def __reduce__(self):
            return (subprocess.check_output, (["id"],))
    from celery_worker import RestrictedUnpickler
    with pytest.raises(pickle.UnpicklingError):
        RestrictedUnpickler(io.BytesIO(pickle.dumps(SubprocessPayload()))).load()


def test_safe_torch_load_rejects_malicious_file():
    """safe_torch_load must reject a file containing a malicious pickle."""
    from celery_worker import safe_torch_load
    malicious_bytes = make_malicious_pickle()
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        f.write(malicious_bytes)
        tmp_path = f.name
    try:
        with pytest.raises(Exception):  # Must raise — not silently load
            safe_torch_load(tmp_path)
    finally:
        os.unlink(tmp_path)
        assert not os.path.exists("/tmp/pwned"), "Malicious payload executed during safe_torch_load"


def test_audit_ledger_immutability():
    """audit_ledger.py must not expose DELETE or UPDATE operations."""
    import inspect
    import audit_ledger
    source = inspect.getsource(audit_ledger)
    assert "DELETE" not in source.upper() or "never DELETE" in source.lower(), \
        "audit_ledger.py contains a DELETE statement — immutability violated"
    assert "UPDATE" not in source.upper() or "never UPDATE" in source.lower(), \
        "audit_ledger.py contains an UPDATE statement — immutability violated"
