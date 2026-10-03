"""已登记消费者批次处理租约测试。

覆盖：从消费者连续水位的下一批领取未归档连续区间（至多 20 批、返回各批
观测明细）、未登记消费者 409、同消费者未终结租约下同键重试 200/其他键
409（lease_active / lease_expired）、幂等键全局唯一、无可领批次不建租约、
归档后登记的消费者从 archived_through 起步；完成租约逐项核对摘要（缺批次/
多批次/摘要变化/租约过期整体拒绝且不推进），通过则同事务写确认、推进
水位、释放租约，重复完成幂等；不同消费者可独立租同一区间，同一区间不
分给同一消费者两个租约；归档不得清除仍被（未过期或已过期未回收的）
租约占用的明细；过期租约显式回收后可重新领取；租约查询（单个/列表/
只看 active）；并发领取不产生重叠租约；verify-chain 覆盖租约占用。
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

THRESHOLDS = {"tolerance": 0.5, "max_drift": 5e-4, "max_missed": 3, "max_false": 3}
DEVICE = "dev-01"


@pytest.fixture()
def client(tmp_path):
    app = create_app(db_path=str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


def obs(seq, n=3, base=1_700_000_000.0, step=0.5):
    return [
        {"local_id": f"s{seq}-o{i:03d}", "captured_at": base + i * step}
        for i in range(n)
    ]


def upload_chain(client, device, n):
    """上传 n 个链式批次，返回各批摘要（下标 0 = seq1）。"""
    digests = []
    prev = ""
    for seq in range(1, n + 1):
        r = client.post(
            f"/field-devices/{device}/batches",
            json={
                "batch_id": f"b-{seq}",
                "sequence": seq,
                "previous_digest": prev,
                "observations": obs(seq),
                **THRESHOLDS,
            },
        )
        assert r.status_code == 201, r.text
        digests.append(r.json()["digest"])
        prev = digests[-1]
    return digests


def register(client, device, consumer_id):
    return client.post(
        f"/field-devices/{device}/consumers", json={"consumer_id": consumer_id}
    )


def claim(client, device, consumer_id, key, *, max_batches=20, lease_seconds=3600):
    return client.post(
        f"/field-devices/{device}/leases",
        json={
            "consumer_id": consumer_id,
            "max_batches": max_batches,
            "lease_seconds": lease_seconds,
            "idempotency_key": key,
        },
    )


def complete(client, device, lease_id, consumer_id, items):
    return client.post(
        f"/field-devices/{device}/leases/{lease_id}/complete",
        json={"consumer_id": consumer_id, "items": items},
    )


def recycle(client, device, lease_id, consumer_id):
    return client.post(
        f"/field-devices/{device}/leases/{lease_id}/recycle",
        json={"consumer_id": consumer_id},
    )


def lease_items(lease_body, digests):
    return [
        {"sequence": b["sequence"], "batch_digest": digests[b["sequence"] - 1]}
        for b in lease_body["batches"]
    ]


def st(client, device):
    return client.get(f"/field-devices/{device}/archive-status").json()


# ---------------- 领取 ----------------


def test_claim_requires_registered_consumer(client):
    upload_chain(client, DEVICE, 2)
    r = claim(client, DEVICE, "ghost", "k1")
    assert r.status_code == 409
    d = r.json()["detail"]
    assert d["reason"] == "consumer_not_registered"
    assert d["consumer_id"] == "ghost"
    # 未登记不产生租约
    assert client.get(f"/field-devices/{DEVICE}/leases").json()["count"] == 0


def test_claim_returns_continuous_unarchived_range_with_details(client):
    digests = upload_chain(client, DEVICE, 5)
    register(client, DEVICE, "w1")
    r = claim(client, DEVICE, "w1", "k1", max_batches=3)
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "leased"
    assert body["lease_status"] == "active"
    assert (body["from_sequence"], body["to_sequence"], body["batch_count"]) == (1, 3, 3)
    assert body["expired"] is False and body["expires_at"] is not None
    assert body["watermark"] == 0 and body["head_sequence"] == 5
    assert [b["sequence"] for b in body["batches"]] == [1, 2, 3]
    for i, b in enumerate(body["batches"], start=1):
        assert b["digest"] == digests[i - 1]
        assert b["observation_count"] == 3
        # active 租约返回观测明细与四项阈值，处理进程无需再读链
        assert len(b["observations"]) == 3
        assert b["observations"][0] == {"captured_at": 1_700_000_000.0,
                                        "local_id": f"s{i}-o000"}
        assert set(b["thresholds"]) == set(THRESHOLDS)


def test_claim_respects_max_batches_limit(client):
    upload_chain(client, DEVICE, 25)
    register(client, DEVICE, "w1")
    assert claim(client, DEVICE, "w1", "k", max_batches=20).json()["to_sequence"] == 20


def test_claim_validation_ranges(client):
    register(client, DEVICE, "w1")
    upload_chain(client, DEVICE, 1)
    for bad in (
        {"consumer_id": "w1", "max_batches": 0, "lease_seconds": 60, "idempotency_key": "k"},
        {"consumer_id": "w1", "max_batches": 21, "lease_seconds": 60, "idempotency_key": "k"},
        {"consumer_id": "w1", "max_batches": 5, "lease_seconds": 0, "idempotency_key": "k"},
        {"consumer_id": "w1", "max_batches": 5, "lease_seconds": 3601, "idempotency_key": "k"},
        {"consumer_id": "w1", "max_batches": 5, "lease_seconds": 60, "idempotency_key": "  "},
    ):
        assert client.post(f"/field-devices/{DEVICE}/leases", json=bad).status_code == 422


def test_claim_no_batches_creates_no_lease(client):
    register(client, DEVICE, "w1")
    # 设备还没有批次
    r = claim(client, DEVICE, "w1", "k1")
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "no_batches_available"
    assert body["lease_id"] is None and body["batches"] == []
    assert body["next_sequence"] == 1 and body["watermark"] == 0
    # 未建租约：消费者可立刻用别的键再领（不存在 active 冲突）
    digests = upload_chain(client, DEVICE, 2)
    r2 = claim(client, DEVICE, "w1", "k2")
    assert r2.status_code == 201 and r2.json()["to_sequence"] == 2
    # 空领取不占用幂等键
    r3 = claim(client, DEVICE, "w1", "k1", max_batches=1)
    assert r3.status_code == 409  # 此时是 active 租约冲突（lease_active），非键占用


def test_claim_starts_after_consumer_watermark(client):
    digests = upload_chain(client, DEVICE, 6)
    register(client, DEVICE, "w1")
    # 乱序确认 1..2，水位 2
    client.post(f"/field-devices/{DEVICE}/acknowledgments",
                json={"consumer_id": "w1", "sequence": 2, "batch_digest": digests[1]})
    client.post(f"/field-devices/{DEVICE}/acknowledgments",
                json={"consumer_id": "w1", "sequence": 1, "batch_digest": digests[0]})
    r = claim(client, DEVICE, "w1", "k1", max_batches=10)
    assert (r.json()["from_sequence"], r.json()["to_sequence"]) == (3, 6)


def test_idempotent_claim_retry_returns_same_lease(client):
    digests = upload_chain(client, DEVICE, 4)
    register(client, DEVICE, "w1")
    r1 = claim(client, DEVICE, "w1", "k1", max_batches=2, lease_seconds=3600)
    assert r1.status_code == 201
    first = r1.json()
    # 即使改了 max_batches/lease_seconds，同键重试也返回原租约（HTTP 200）
    r2 = claim(client, DEVICE, "w1", "k1", max_batches=4, lease_seconds=10)
    assert r2.status_code == 200
    again = r2.json()
    assert again["lease_id"] == first["lease_id"]
    assert (again["from_sequence"], again["to_sequence"]) == (1, 2)
    assert again["expires_at"] == first["expires_at"]
    assert [b["digest"] for b in again["batches"]] == digests[:2]
    # 只存在一个租约
    assert client.get(f"/field-devices/{DEVICE}/leases").json()["count"] == 1


def test_second_key_while_lease_active_conflicts(client):
    upload_chain(client, DEVICE, 4)
    register(client, DEVICE, "w1")
    claim(client, DEVICE, "w1", "k1", max_batches=2)
    r = claim(client, DEVICE, "w1", "k2", max_batches=2)
    assert r.status_code == 409
    d = r.json()["detail"]
    assert d["reason"] == "lease_active"
    assert d["active_lease_status"] == "active"
    assert d["active_lease_id"] == 1
    assert (d["active_from_sequence"], d["active_to_sequence"]) == (1, 2)
    # 冲突不创建第二个租约
    assert client.get(f"/field-devices/{DEVICE}/leases").json()["count"] == 1


def test_expired_lease_same_key_returns_it_other_key_conflicts(client):
    upload_chain(client, DEVICE, 3)
    register(client, DEVICE, "w1")
    r1 = claim(client, DEVICE, "w1", "k1", max_batches=2, lease_seconds=1)
    lease_id = r1.json()["lease_id"]
    # 租约批次不被清除：等待过期期间明细仍在
    import time
    time.sleep(1.1)
    # 同键重试仍返回原租约（标记 expired=True）
    r2 = claim(client, DEVICE, "w1", "k1", max_batches=2, lease_seconds=60)
    assert r2.status_code == 200
    assert r2.json()["lease_id"] == lease_id and r2.json()["expired"] is True
    # 其他键：409 lease_expired，提示显式回收
    r3 = claim(client, DEVICE, "w1", "k2", max_batches=2, lease_seconds=60)
    assert r3.status_code == 409
    d = r3.json()["detail"]
    assert d["reason"] == "lease_expired"
    assert d["active_lease_status"] == "expired"


def test_idempotency_key_global_unique(client):
    digests = upload_chain(client, "dev-a", 2)
    upload_chain(client, "dev-b", 2)
    register(client, "dev-a", "w1")
    register(client, "dev-b", "w1")
    assert claim(client, "dev-a", "w1", "shared-key").status_code == 201
    r = claim(client, "dev-b", "w1", "shared-key")
    assert r.status_code == 409
    d = r.json()["detail"]
    assert d["reason"] == "idempotency_key_in_use"
    assert d["occupied_device_id"] == "dev-a"
    assert d["occupied_by_lease_id"] == 1


def test_distinct_consumers_lease_same_range_independently(client):
    digests = upload_chain(client, DEVICE, 4)
    register(client, DEVICE, "w1")
    register(client, DEVICE, "w2")
    r1 = claim(client, DEVICE, "w1", "k1", max_batches=3)
    r2 = claim(client, DEVICE, "w2", "k2", max_batches=2)
    assert r1.status_code == 201 and r2.status_code == 201
    assert (r1.json()["from_sequence"], r1.json()["to_sequence"]) == (1, 3)
    assert (r2.json()["from_sequence"], r2.json()["to_sequence"]) == (1, 2)


def test_concurrent_claims_same_consumer_never_overlap(client):
    upload_chain(client, DEVICE, 40)
    register(client, DEVICE, "w1")
    n = 16
    barrier = threading.Barrier(n)
    results = []
    lock = threading.Lock()

    def worker(i):
        barrier.wait()
        r = claim(client, DEVICE, "w1", f"k{i}", max_batches=5, lease_seconds=3600)
        with lock:
            results.append(r)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    created = [r for r in results if r.status_code == 201]
    conflicts = [r for r in results if r.status_code == 409]
    # 恰好一个租约创建成功，其余全部 409（部分唯一索引 + 写锁）
    assert len(created) == 1
    assert len(conflicts) == n - 1
    assert all(r.json()["detail"]["reason"] == "lease_active" for r in conflicts)
    active = client.get(f"/field-devices/{DEVICE}/leases",
                        params={"active_only": True}).json()["leases"]
    assert len(active) == 1 and active[0]["to_sequence"] - active[0]["from_sequence"] == 4


# ---------------- 完成租约 ----------------


def test_complete_writes_acks_advances_watermark_and_releases(client):
    digests = upload_chain(client, DEVICE, 3)
    register(client, DEVICE, "w1")
    register(client, DEVICE, "w2")
    lid = claim(client, DEVICE, "w1", "k1", max_batches=3).json()["lease_id"]
    r = complete(client, DEVICE, lid, "w1", [
        {"sequence": s, "batch_digest": digests[s - 1]} for s in (1, 2, 3)
    ])
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "completed" and body["lease_status"] == "completed"
    assert body["completed_at"] is not None
    assert body["watermark"] == 3 and body["safe_sequence"] == 0  # w2 尚未确认
    assert body["first_gap"] == 1 and body["blocking_consumers"] == ["w2"]
    # 确认已落库
    assert st(client, DEVICE)["consumers"][0]["watermark"] == 3
    # 租约释放：该消费者可立即领取下一个（链头处无批次 → no_batches）
    nxt = claim(client, DEVICE, "w1", "k2")
    assert nxt.json()["status"] == "no_batches_available"


def test_complete_advances_only_over_first_gap(client):
    """租约区间从水位下一批开始；完成后即使有更靠后的乱序确认，水位停在缺口。"""
    digests = upload_chain(client, DEVICE, 5)
    register(client, DEVICE, "w1")
    # 乱序确认 seq5（水位仍 0）
    client.post(f"/field-devices/{DEVICE}/acknowledgments",
                json={"consumer_id": "w1", "sequence": 5, "batch_digest": digests[4]})
    lid = claim(client, DEVICE, "w1", "k1", max_batches=3).json()["lease_id"]
    r = complete(client, DEVICE, lid, "w1", [
        {"sequence": s, "batch_digest": digests[s - 1]} for s in (1, 2, 3)
    ])
    assert r.json()["watermark"] == 3  # seq4 是缺口，不越过
    assert r.json()["first_gap"] == 4


def test_complete_duplicate_is_idempotent(client):
    digests = upload_chain(client, DEVICE, 2)
    register(client, DEVICE, "w1")
    lid = claim(client, DEVICE, "w1", "k1").json()["lease_id"]
    items = [{"sequence": s, "batch_digest": digests[s - 1]} for s in (1, 2)]
    r1 = complete(client, DEVICE, lid, "w1", items)
    first_at = r1.json()["completed_at"]
    r2 = complete(client, DEVICE, lid, "w1", items)
    assert r2.status_code == 200
    assert r2.json()["status"] == "completed"
    assert r2.json()["completed_at"] == first_at
    assert r2.json()["watermark"] == 2


def test_complete_rejects_missing_and_extra_sequences(client):
    digests = upload_chain(client, DEVICE, 4)
    register(client, DEVICE, "w1")
    lid = claim(client, DEVICE, "w1", "k1", max_batches=3).json()["lease_id"]

    # 缺批次：只交 1,3
    r = complete(client, DEVICE, lid, "w1", [
        {"sequence": 1, "batch_digest": digests[0]},
        {"sequence": 3, "batch_digest": digests[2]},
    ])
    assert r.status_code == 409
    d = r.json()["detail"]
    assert d["reason"] == "items_do_not_match_lease"
    assert d["missing_sequences"] == [2] and d["unexpected_sequences"] == []

    # 多批次：塞入区间外的 4
    r = complete(client, DEVICE, lid, "w1", [
        {"sequence": s, "batch_digest": digests[s - 1]} for s in (1, 2, 3, 4)
    ])
    assert r.status_code == 409
    d = r.json()["detail"]
    assert d["missing_sequences"] == [] and d["unexpected_sequences"] == [4]

    # 整体拒绝：未写任何确认、水位不推进、租约仍 active
    body = client.get(f"/field-devices/{DEVICE}/leases/{lid}").json()
    assert body["lease_status"] == "active"
    assert st(client, DEVICE)["consumers"][0]["watermark"] == 0


def test_complete_rejects_digest_mismatch_without_progress(client):
    digests = upload_chain(client, DEVICE, 3)
    register(client, DEVICE, "w1")
    lid = claim(client, DEVICE, "w1", "k1", max_batches=3).json()["lease_id"]
    items = [{"sequence": s, "batch_digest": digests[s - 1]} for s in (1, 2, 3)]
    items[1]["batch_digest"] = "ab" * 32  # seq2 摘要错误
    r = complete(client, DEVICE, lid, "w1", items)
    assert r.status_code == 409
    d = r.json()["detail"]
    assert d["reason"] == "digest_mismatch"
    assert d["sequence"] == 2
    assert d["provided_digest"] == "ab" * 32
    assert d["expected_digest"] == digests[1]
    # 同事务整体拒绝：即使 seq1/seq3 摘要正确，也不写确认、不推进水位
    assert st(client, DEVICE)["consumers"][0]["watermark"] == 0
    body = client.get(f"/field-devices/{DEVICE}/leases/{lid}").json()
    assert body["lease_status"] == "active"


def test_complete_rejects_duplicate_sequence(client):
    register(client, DEVICE, "w1")
    upload_chain(client, DEVICE, 2)
    lid = claim(client, DEVICE, "w1", "k1").json()["lease_id"]
    r = complete(client, DEVICE, lid, "w1", [
        {"sequence": 1, "batch_digest": "ab" * 32},
        {"sequence": 1, "batch_digest": "cd" * 32},
    ])
    assert r.status_code == 422


def test_complete_expired_lease_rejected(client):
    digests = upload_chain(client, DEVICE, 2)
    register(client, DEVICE, "w1")
    lid = claim(client, DEVICE, "w1", "k1", lease_seconds=1).json()["lease_id"]
    import time
    time.sleep(1.1)
    r = complete(client, DEVICE, lid, "w1", [
        {"sequence": s, "batch_digest": digests[s - 1]} for s in (1, 2)
    ])
    assert r.status_code == 409
    assert r.json()["detail"]["reason"] == "lease_expired"
    # 不推进水位
    assert st(client, DEVICE)["consumers"][0]["watermark"] == 0


def test_complete_recycled_lease_rejected(client):
    digests = upload_chain(client, DEVICE, 2)
    register(client, DEVICE, "w1")
    lid = claim(client, DEVICE, "w1", "k1", lease_seconds=1).json()["lease_id"]
    import time
    time.sleep(1.1)
    assert recycle(client, DEVICE, lid, "w1").status_code == 200
    r = complete(client, DEVICE, lid, "w1", [
        {"sequence": s, "batch_digest": digests[s - 1]} for s in (1, 2)
    ])
    assert r.status_code == 409
    assert r.json()["detail"]["reason"] == "lease_recycled"
    assert st(client, DEVICE)["consumers"][0]["watermark"] == 0


def test_complete_consumer_and_device_must_match(client):
    digests = upload_chain(client, DEVICE, 2)
    register(client, DEVICE, "w1")
    register(client, DEVICE, "w2")
    lid = claim(client, DEVICE, "w1", "k1").json()["lease_id"]
    items = [{"sequence": s, "batch_digest": digests[s - 1]} for s in (1, 2)]
    r = complete(client, DEVICE, lid, "w2", items)
    assert r.status_code == 409
    assert r.json()["detail"]["reason"] == "lease_consumer_mismatch"
    # 其他设备查不到该租约 → 404
    r = complete(client, "dev-other", lid, "w1", items)
    assert r.status_code == 404


def test_complete_then_reclaim_starts_after_watermark(client):
    digests = upload_chain(client, DEVICE, 5)
    register(client, DEVICE, "w1")
    lid1 = claim(client, DEVICE, "w1", "k1", max_batches=2).json()["lease_id"]
    complete(client, DEVICE, lid1, "w1", [
        {"sequence": s, "batch_digest": digests[s - 1]} for s in (1, 2)
    ])
    lid2 = claim(client, DEVICE, "w1", "k2", max_batches=2).json()["lease_id"]
    body = client.get(f"/field-devices/{DEVICE}/leases/{lid2}").json()
    assert (body["from_sequence"], body["to_sequence"]) == (3, 4)
    complete(client, DEVICE, lid2, "w1", [
        {"sequence": s, "batch_digest": digests[s - 1]} for s in (3, 4)
    ])
    # 只剩 seq5
    lid3 = claim(client, DEVICE, "w1", "k3", max_batches=20).json()["lease_id"]
    body3 = client.get(f"/field-devices/{DEVICE}/leases/{lid3}").json()
    assert (body3["from_sequence"], body3["to_sequence"]) == (5, 5)


# ---------------- 回收过期租约 ----------------


def test_recycle_rules(client):
    upload_chain(client, DEVICE, 2)
    register(client, DEVICE, "w1")

    # 未过期不可回收
    lid = claim(client, DEVICE, "w1", "k1", lease_seconds=3600).json()["lease_id"]
    r = recycle(client, DEVICE, lid, "w1")
    assert r.status_code == 409 and r.json()["detail"]["reason"] == "lease_not_expired"

    # 消费者不匹配
    register(client, DEVICE, "w2")
    r = recycle(client, DEVICE, lid, "w2")
    assert r.status_code == 409 and r.json()["detail"]["reason"] == "lease_consumer_mismatch"

    # 不存在 / 跨设备
    assert recycle(client, DEVICE, 999, "w1").status_code == 404
    assert recycle(client, "dev-x", lid, "w1").status_code == 404


def test_recycle_expired_then_reclaim(client):
    digests = upload_chain(client, DEVICE, 3)
    register(client, DEVICE, "w1")
    lid = claim(client, DEVICE, "w1", "k1", lease_seconds=1, max_batches=2).json()["lease_id"]
    import time
    time.sleep(1.1)
    r = recycle(client, DEVICE, lid, "w1")
    assert r.status_code == 200 and r.json()["status"] == "recycled"
    assert r.json()["recycled_at"] is not None
    # 重复回收幂等
    assert recycle(client, DEVICE, lid, "w1").status_code == 200
    # 回收后可重新领取同一区间（水位未推进，仍从 1 开始）
    r2 = claim(client, DEVICE, "w1", "k2", max_batches=20, lease_seconds=3600)
    assert r2.status_code == 201
    assert (r2.json()["from_sequence"], r2.json()["to_sequence"]) == (1, 3)
    assert r2.json()["lease_id"] != lid


def test_cannot_recycle_completed_lease(client):
    digests = upload_chain(client, DEVICE, 2)
    register(client, DEVICE, "w1")
    lid = claim(client, DEVICE, "w1", "k1").json()["lease_id"]
    complete(client, DEVICE, lid, "w1", [
        {"sequence": s, "batch_digest": digests[s - 1]} for s in (1, 2)
    ])
    r = recycle(client, DEVICE, lid, "w1")
    assert r.status_code == 409 and r.json()["detail"]["reason"] == "lease_not_active"


# ---------------- 归档不得清除租约占用的明细 ----------------


def test_archive_blocked_while_lease_active(client):
    digests = upload_chain(client, DEVICE, 3)
    register(client, DEVICE, "w1")
    register(client, DEVICE, "w2")
    lid1 = claim(client, DEVICE, "w1", "k1", max_batches=3).json()["lease_id"]
    lid2 = claim(client, DEVICE, "w2", "k2", max_batches=2).json()["lease_id"]
    # 两个消费者都在带外确认全部批次（safe_sequence=3），但租约仍占用区间
    for cid in ("w1", "w2"):
        for s in (1, 2, 3):
            client.post(f"/field-devices/{DEVICE}/acknowledgments",
                        json={"consumer_id": cid, "sequence": s,
                              "batch_digest": digests[s - 1]})
    assert st(client, DEVICE)["safe_sequence"] == 3
    r = client.post(f"/field-devices/{DEVICE}/archive",
                    json={"through_sequence": 3})
    assert r.status_code == 409
    d = r.json()["detail"]
    assert d["reason"] == "leases_occupying_batches"
    occupied = {(o["lease_id"], o["consumer_id"]) for o in d["occupied_leases"]}
    assert occupied == {(lid1, "w1"), (lid2, "w2")}
    # 明细未被清除
    detail = client.get(f"/field-devices/{DEVICE}/batches/1").json()
    assert detail["archived"] is False and len(detail["observations"]) == 3
    assert client.get(f"/field-devices/{DEVICE}/archive-status").json()["archived_through"] == 0

    # 完成两个租约后归档放行
    complete(client, DEVICE, lid1, "w1", lease_items(
        client.get(f"/field-devices/{DEVICE}/leases/{lid1}").json(), digests))
    complete(client, DEVICE, lid2, "w2", lease_items(
        client.get(f"/field-devices/{DEVICE}/leases/{lid2}").json(), digests))
    r2 = client.post(f"/field-devices/{DEVICE}/archive",
                     json={"through_sequence": 3})
    assert r2.status_code == 201 and r2.json()["archived_through"] == 3


def test_expired_unrecycled_lease_still_blocks_archive(client):
    digests = upload_chain(client, DEVICE, 2)
    register(client, DEVICE, "w1")
    lid = claim(client, DEVICE, "w1", "k1", lease_seconds=1).json()["lease_id"]
    # 带外确认使 safe_sequence=2
    for s in (1, 2):
        client.post(f"/field-devices/{DEVICE}/acknowledgments",
                    json={"consumer_id": "w1", "sequence": s,
                          "batch_digest": digests[s - 1]})
    import time
    time.sleep(1.1)
    # 已过期但未回收：仍然占用，归档被挡
    r = client.post(f"/field-devices/{DEVICE}/archive",
                    json={"through_sequence": 2})
    assert r.status_code == 409
    assert r.json()["detail"]["reason"] == "leases_occupying_batches"
    occ = r.json()["detail"]["occupied_leases"][0]
    assert occ["lease_id"] == lid and occ["expired"] is True
    # 回收后归档放行
    recycle(client, DEVICE, lid, "w1")
    r2 = client.post(f"/field-devices/{DEVICE}/archive",
                     json={"through_sequence": 2})
    assert r2.status_code == 201


def test_archive_partial_range_not_touching_lease_allowed(client):
    """租约占用 seq3..4 时，归档此前已确认的 seq1..2 不受影响。"""
    digests = upload_chain(client, DEVICE, 4)
    register(client, DEVICE, "w1")
    # 先确认 1,2 并完成（水位 2），再租 3,4
    for s in (1, 2):
        client.post(f"/field-devices/{DEVICE}/acknowledgments",
                    json={"consumer_id": "w1", "sequence": s,
                          "batch_digest": digests[s - 1]})
    lid = claim(client, DEVICE, "w1", "k1", max_batches=2).json()["lease_id"]
    assert (client.get(f"/field-devices/{DEVICE}/leases/{lid}").json()["from_sequence"],
            client.get(f"/field-devices/{DEVICE}/leases/{lid}").json()["to_sequence"]) == (3, 4)
    r = client.post(f"/field-devices/{DEVICE}/archive",
                    json={"through_sequence": 2})
    assert r.status_code == 201 and r.json()["archived_through"] == 2
    # 扩大到租约区间被挡
    r2 = client.post(f"/field-devices/{DEVICE}/archive",
                     json={"through_sequence": 3})
    assert r2.status_code == 409
    assert r2.json()["detail"]["reason"] == "leases_occupying_batches"


def test_lease_after_archive_starts_past_tombstones(client):
    """归档后登记的消费者从 archived_through 起步；租约只选未归档连续区间。"""
    digests = upload_chain(client, DEVICE, 5)
    register(client, DEVICE, "old")
    for s in range(1, 6):
        client.post(f"/field-devices/{DEVICE}/acknowledgments",
                    json={"consumer_id": "old", "sequence": s,
                          "batch_digest": digests[s - 1]})
    client.post(f"/field-devices/{DEVICE}/archive", json={"through_sequence": 3})
    register(client, DEVICE, "late")
    r = claim(client, DEVICE, "late", "k1", max_batches=20)
    body = r.json()
    assert body["status"] == "leased"
    assert (body["from_sequence"], body["to_sequence"]) == (4, 5)
    assert body["watermark"] == 3  # 从 archived_through 起步
    lid = body["lease_id"]
    r2 = complete(client, DEVICE, lid, "late", [
        {"sequence": 4, "batch_digest": digests[3]},
        {"sequence": 5, "batch_digest": digests[4]},
    ])
    assert r2.status_code == 201 and r2.json()["watermark"] == 5


# ---------------- 查询 ----------------


def test_get_and_list_leases(client):
    upload_chain(client, DEVICE, 4)
    register(client, DEVICE, "w1")
    register(client, DEVICE, "w2")
    lid1 = claim(client, DEVICE, "w1", "k1", max_batches=2).json()["lease_id"]
    lid2 = claim(client, DEVICE, "w2", "k2", max_batches=3).json()["lease_id"]

    g = client.get(f"/field-devices/{DEVICE}/leases/{lid1}")
    assert g.status_code == 200
    assert g.json()["consumer_id"] == "w1"
    assert g.json()["batches"][0]["observations"] is not None  # active 附明细

    assert client.get(f"/field-devices/{DEVICE}/leases/999").status_code == 404

    lst = client.get(f"/field-devices/{DEVICE}/leases")
    assert lst.json()["count"] == 2
    assert [x["lease_id"] for x in lst.json()["leases"]] == [lid1, lid2]

    f1 = client.get(f"/field-devices/{DEVICE}/leases",
                    params={"consumer_id": "w1"})
    assert [x["lease_id"] for x in f1.json()["leases"]] == [lid1]

    # 完成 lid1 后 active_only 只返回 lid2；已完成租约不再返回观测明细
    digests = [
        client.get(f"/field-devices/{DEVICE}/batches/{s}").json()["digest"]
        for s in (1, 2)
    ]
    complete(client, DEVICE, lid1, "w1", [
        {"sequence": 1, "batch_digest": digests[0]},
        {"sequence": 2, "batch_digest": digests[1]},
    ])
    active = client.get(f"/field-devices/{DEVICE}/leases",
                        params={"active_only": True})
    assert [x["lease_id"] for x in active.json()["leases"]] == [lid2]
    done = client.get(f"/field-devices/{DEVICE}/leases/{lid1}").json()
    assert done["lease_status"] == "completed"
    assert all(b["observations"] is None for b in done["batches"])


# ---------------- verify-chain 覆盖租约 ----------------


def test_verify_chain_reports_leases(client):
    digests = upload_chain(client, DEVICE, 3)
    register(client, DEVICE, "w1")
    claim(client, DEVICE, "w1", "k1", max_batches=3)
    v = client.post(f"/field-devices/{DEVICE}/verify-chain",
                    json={"from_sequence": 1, "to_sequence": 3})
    body = v.json()
    assert body["ok"] is True and body["issue_count"] == 0
    assert body["checked_lease_batches"] == 3
    assert body["snapshot"]["active_lease_count"] == 1


def test_concurrent_claims_distinct_consumers_disjoint_progress(client):
    """多个消费者各自并发领取：每个消费者拿到一个租约，区间不跨消费者冲突。"""
    digests = upload_chain(client, DEVICE, 10)
    consumers = ["w1", "w2", "w3", "w4"]
    for cid in consumers:
        register(client, DEVICE, cid)
    barrier = threading.Barrier(len(consumers))
    out = {}
    lock = threading.Lock()

    def worker(cid):
        barrier.wait()
        r = claim(client, DEVICE, cid, f"key-{cid}", max_batches=5, lease_seconds=3600)
        with lock:
            out[cid] = r

    ts = [threading.Thread(target=worker, args=(cid,)) for cid in consumers]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    for cid, r in out.items():
        assert r.status_code == 201, (cid, r.text)
        assert (r.json()["from_sequence"], r.json()["to_sequence"]) == (1, 5)
    # 全部完成后 safe_sequence=5
    for cid in consumers:
        lid = out[cid].json()["lease_id"]
        rr = complete(client, DEVICE, lid, cid, [
            {"sequence": s, "batch_digest": digests[s - 1]} for s in range(1, 6)
        ])
        assert rr.status_code == 201
    assert st(client, DEVICE)["safe_sequence"] == 5
