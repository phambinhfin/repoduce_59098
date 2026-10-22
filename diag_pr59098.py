

# ---------------------------------------------------------------------------
# PR #59098 diagnostic -- appended unchanged to vllm/v1/attention/backends/utils.py
# in BOTH images. Not part of the fix: it only reads the batch and logs.
#
# It watches every batch that holds both a fresh prompt (no cached tokens) and a
# request that already has cached tokens -- the only batch the bug can hit -- and
# logs the order the batch was sorted into, how the splitter classified each row,
# and how many cached tokens a request lost by being run as a plain prefill.
import threading as _pr59098_threading
import time as _pr59098_time

_pr59098_orig_split = split_decodes_prefills_and_extends
_pr59098 = {"mixed": 0, "hit": 0, "lost_reqs": 0, "lost_tokens": 0,
            "printed": 0, "thread": None}
_PR59098_DETAILED = 20


def _pr59098_totals_loop():
    last = None
    while True:
        _pr59098_time.sleep(10)
        s = _pr59098
        now = (s["mixed"], s["hit"], s["lost_reqs"], s["lost_tokens"])
        if now != last:
            last = now
            logger.warning(
                "[PR59098] TOTAL: %d mixed batches, %d of them lost context, "
                "%d requests lost %d cached tokens in total", *now)


def split_decodes_prefills_and_extends(common_attn_metadata, decode_threshold=1):
    ret = _pr59098_orig_split(common_attn_metadata, decode_threshold)
    try:
        _pr59098_check(common_attn_metadata, decode_threshold, ret)
    except Exception as e:  # the diagnostic must never break a run
        logger.warning("[PR59098] diagnostic error: %r", e)
    return ret


def _pr59098_check(m, threshold, ret):
    n = m.num_reqs
    qsl = m.query_start_loc_cpu[: n + 1]
    new = (qsl[1:] - qsl[:-1]).tolist()
    cached = [s - q for s, q in zip(m.seq_lens_cpu_upper_bound[:n].tolist(), new)]
    rows = [i for i in range(n) if new[i] > threshold]
    if not any(cached[i] == 0 for i in rows) or not any(cached[i] > 0 for i in rows):
        return

    num_decodes, num_extends, num_prefills = ret[0], ret[1], ret[2]
    first_prefill = n - num_prefills
    lost = [i for i in range(first_prefill, n) if new[i] > 0 and cached[i] > 0]
    lost_tokens = sum(cached[i] for i in lost)

    s = _pr59098
    s["mixed"] += 1
    if lost:
        s["hit"] += 1
        s["lost_reqs"] += len(lost)
        s["lost_tokens"] += lost_tokens
    if s["thread"] is None:
        s["thread"] = _pr59098_threading.Thread(target=_pr59098_totals_loop,
                                                daemon=True)
        s["thread"].start()
    if s["printed"] >= _PR59098_DETAILED:
        return
    s["printed"] += 1

    def describe(i):
        kind = ("decode" if i < num_decodes
                else "extend" if i < first_prefill else "prefill")
        mark = f"   <-- LOST {cached[i]} cached tokens" if i in lost else ""
        return f"    #{i:<3} new={new[i]:<5} cached={cached[i]:<6} -> {kind}{mark}"

    shown = rows[:12]
    logger.warning(
        "[PR59098] mixed batch %d: %s\n"
        "    order as sorted (%d decode-length rows not shown):\n%s%s",
        s["mixed"],
        f"CONTEXT LOST on {len(lost)} request(s), {lost_tokens} cached tokens "
        f"ignored" if lost else "no context lost",
        n - len(rows),
        "\n".join(describe(i) for i in shown),
        "\n    ..." if len(rows) > len(shown) else "",
    )
