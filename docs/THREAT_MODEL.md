# Threat Model — Gemini Trojan Detection System

## System Overview

The system accepts uploaded neural network checkpoints (.pth, .onnx) from potentially untrusted clients, analyzes them for trojan backdoors using a suite of ML defenses, and returns a signed audit report.

## Trust Boundaries

| Boundary | From | To | Trust Level |
|----------|------|----|-------------|
| TB-1 | External client | FastAPI (api.py) | UNTRUSTED — validate all input |
| TB-2 | FastAPI | Celery worker | SEMI-TRUSTED — internal network only |
| TB-3 | Celery worker | Uploaded model file | UNTRUSTED — sandbox the load |
| TB-4 | Celery worker | meta_classifier.pkl | TRUSTED — hash-verified at startup |
| TB-5 | FastAPI | Redis | TRUSTED — internal network, password-protected |
| TB-6 | API report | External client | VERIFIED — Ed25519 signed |

## STRIDE Analysis

### Spoofing
- **S1**: Attacker impersonates a trusted client → **Mitigated** by API token auth (api.py)
- **S2**: Attacker submits a trojaned model as a clean model → **Partially mitigated** by detection pipeline; **Residual risk**: adaptive attackers with knowledge of the detector
- **S3**: Attacker replaces `meta_classifier.pkl` on disk → **Mitigated** by SHA256 hash verification on load

### Tampering
- **T1**: Malicious pickle in uploaded .pth executes arbitrary code → **Mitigated** by RestrictedUnpickler sandbox
- **T2**: Path traversal via ONNX filename to overwrite system files → **Mitigated** by filename sanitization (secrets.token_hex)
- **T3**: Tamper with audit report after generation → **Mitigated** by Ed25519 report signature
- **T4**: Redis task queue poisoning → **Mitigated** by Redis auth + internal network binding
- **T5**: Audit ledger modification → **Mitigated** by append-only SQLite WAL; no DELETE/UPDATE in code

### Repudiation
- **R1**: Attacker denies submitting a malicious model → **Mitigated** by audit_log(client_token_hash, model_sha256, timestamp)
- **R2**: Operator denies issuing a verdict → **Mitigated** by signed report with key fingerprint

### Information Disclosure
- **I1**: Stack traces leak internal paths → **Mitigated** by error scrubbing (trace_id only to client)
- **I2**: Redis data exposed to network → **Mitigated** by internal network binding + requirepass
- **I3**: Log files capture sensitive paths → **Mitigated** by structured logging (hashes only)
- **I4**: Container runs as root → **Mitigated** by non-root appuser (uid 10001) in Dockerfile

### Denial of Service
- **D1**: Attacker floods scan endpoint → **Mitigated** by rate limiting (slowapi, 10/min)
- **D2**: Attacker uploads 100GB model → **Mitigated** by 1GB upload cap
- **D3**: Malicious model causes OOM in worker → **Partially mitigated** by Docker memory limits (8GB); **Residual**: adversarial weight patterns may cause pathological defense runtime

### Elevation of Privilege
- **E1**: Pickle RCE escapes worker container → **Partially mitigated** by RestrictedUnpickler + cap_drop ALL + no-new-privileges; **Residual**: kernel exploits
- **E2**: SSRF via scan-local-path → **Mitigated** by realpath confinement to uploads/

## Residual Risks

| Risk | Likelihood | Impact | Accepted? |
|------|-----------|--------|-----------|
| Adaptive attacker evading meta-classifier | Medium | High | No — requires adversarial robustness testing (Phase 3) |
| Kernel exploit from inside container | Low | Critical | Yes — mitigated by host kernel patching SLA |
| Signing key compromise | Very Low | High | No — requires key rotation runbook (see RUNBOOK.md) |
| OOD model silently misclassified | High | Medium | Partial — OOD gate added, confidence thresholds TBD |

## Security Contacts

Report vulnerabilities to: [security contact TBD]
