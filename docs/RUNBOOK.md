# Operational Runbook — Gemini Trojan Detection

## Key Rotation Procedures

### Rotate API_TOKEN
1. Generate new token: `openssl rand -hex 32`
2. Update in production environment: `docker secret rm api_token && echo <new> | docker secret create api_token -`
3. Restart api service: `docker compose up -d --no-deps api`
4. Update all clients with new token
5. Old token is immediately invalid (no grace period)

### Rotate Redis Password
1. Generate: `openssl rand -hex 24`
2. Update `REDIS_PASSWORD` secret
3. Restart redis, worker, api in order: `docker compose restart redis worker api`

### Rotate Audit Signing Key (Ed25519)
1. Generate new keypair: `openssl genpkey -algorithm ed25519 -out new_audit_signing_key.pem`
2. Export public key: `openssl pkey -in new_audit_signing_key.pem -pubout -out new_audit_pub.pem`
3. Old reports remain verifiable with the old public key — publish both keys at `/api/v1/audit-keys` with `valid_from`/`valid_until` dates
4. Mount new key into api/worker containers; restart

## Incident Response

### Suspected Malicious Model Upload
1. Immediately quarantine: `mv uploads/<task_id>.pth quarantine/`
2. Check audit ledger: `sqlite3 audit_ledger.db "SELECT * FROM audit_log WHERE task_id='<id>'"`
3. Revoke client token if identified
4. Preserve the quarantined file for forensic analysis
5. Re-run scan in isolated environment with additional defenses

### Worker Crash / OOM
1. `docker compose ps` — check worker status
2. `docker compose logs --tail=100 worker`
3. If OOM: check `model_size_bytes` in audit_ledger — if > 4GB, adjust `--max-tasks-per-child` and memory limit
4. Restart: `docker compose restart worker`

### Audit Ledger Corruption
1. Stop all writes: `docker compose stop worker api`
2. Run SQLite integrity check: `sqlite3 audit_ledger.db "PRAGMA integrity_check"`
3. Restore from backup: `cp audit_ledger.db.backup audit_ledger.db`
4. Restart services

## Backup Procedures

### Daily Backup Checklist
- [ ] `audit_ledger.db` → off-site encrypted backup
- [ ] `meta_classifier.pkl` → version-controlled in git-LFS (immutable)
- [ ] `uploads/` → retain for 30 days, then purge

## SLA Targets

| Metric | Target |
|--------|--------|
| Scan completion (ResNet-18) | < 5 minutes P95 |
| API availability | 99.5% (excl. maintenance) |
| Incident response (critical) | < 2 hours |
| Key rotation (after breach) | < 1 hour |
