"""A windowed DFLASH draft pool indexed by the target's sliding-window slots:
when it applies, how long the target keeps those slots, and the id remap of the
draft's writes and its compact req->token rows."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.speculative.dflash_utils import (
    dflash_swa_draft_kv_blocker,
    dflash_swa_retention_window,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

_HAS_CUDA = torch.cuda.is_available()

_SUPPORTED = dict(
    draft_window_size=4096,
    target_has_swa_slots=True,
    enable_unified_memory=False,
    disaggregation_mode="null",
    enable_hierarchical_cache=False,
    post_capture_kv_sizing=False,
    attn_dp_size=1,
    attn_cp_size=1,
    dcp_size=1,
    enable_kv_cache_sharding=False,
    enable_hisparse=False,
)


class TestBlocker(CustomTestCase):
    def test_supported_config_has_no_blocker(self):
        self.assertIsNone(dflash_swa_draft_kv_blocker(**_SUPPORTED))

    def test_each_unsupported_setting_blocks(self):
        for field, value in [
            ("draft_window_size", None),
            ("target_has_swa_slots", False),
            ("enable_unified_memory", True),
            ("disaggregation_mode", "decode"),
            ("enable_hierarchical_cache", True),
            ("post_capture_kv_sizing", True),
            ("attn_dp_size", 2),
            ("attn_cp_size", 2),
            ("dcp_size", 2),
            ("enable_kv_cache_sharding", True),
            ("enable_hisparse", True),
        ]:
            with self.subTest(field=field):
                self.assertIsNotNone(
                    dflash_swa_draft_kv_blocker(**{**_SUPPORTED, field: value})
                )


class _ReqKv:
    """The request KV state `free_swa_out_of_window_slots` reads and writes."""

    holds_kv = True
    cache_protected_len = 0
    req_pool_idx = 0

    def __init__(self):
        self.evicted = 0

    def get_evicted_seqlen(self, component_type):
        assert component_type == ComponentType.SWA
        return self.evicted

    def set_evicted_seqlen(self, component_type, length):
        self.evicted = length

    def swa_dead_lo(self, page_size):
        return 0


def _first_live_position(pre_len, window, page_size):
    """Positions below this have had their sliding-window slots freed."""
    from sglang.srt.mem_cache.common import free_swa_out_of_window_slots

    req = SimpleNamespace(kv=_ReqKv())
    free_swa_out_of_window_slots(
        req,
        pre_len,
        sliding_window_size=window,
        page_size=page_size,
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros((1, pre_len + 1), dtype=torch.int32)
        ),
        token_to_kv_pool_allocator=SimpleNamespace(
            free_swa_segment=lambda slots, start_pos: None
        ),
    )
    return req.kv.evicted


def _compact_draft_start(prefix_len, window, page_size):
    """First position the compact draft reads for a `prefix_len` prefix."""
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    fake_self = SimpleNamespace(
        device="cpu", draft_window_size=window, page_size=page_size
    )
    seq_lens = torch.tensor([prefix_len])
    visible = DFlashWorkerV2._compute_compact_draft_seq_lens(fake_self, seq_lens)
    return prefix_len - int(visible[0])


class TestRetentionWindow(CustomTestCase):
    def test_widens_to_the_draft_window(self):
        self.assertEqual(
            dflash_swa_retention_window(
                512, draft_window_size=4096, block_size=8, page_size=64
            ),
            4096 + 8 + 64,
        )
        self.assertEqual(
            dflash_swa_retention_window(
                8192, draft_window_size=4096, block_size=8, page_size=64
            ),
            8192,
        )

    def test_the_cache_keeps_the_retention_window(self):
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        retention = dflash_swa_retention_window(
            512, draft_window_size=4096, block_size=8, page_size=64
        )
        worker = SimpleNamespace(
            draft_kv_in_swa_slots=True,
            model_runner=SimpleNamespace(swa_retention_window=retention),
        )
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=64,
            sliding_window_size=512,
            draft_swa_window=DFlashWorkerV2.draft_swa_window.fget(worker),
        )
        self.assertEqual(params.swa_retain_window, retention)

    def test_eviction_never_frees_a_slot_the_draft_reads(self):
        """The draft reads its window back from a page boundary; the target
        frees sliding-window slots past the retention window. A frontier up
        to one verify block ahead of the draft's prefix must free nothing the
        draft still reads."""
        draft_window, block, page = 4096, 8, 64
        retention = dflash_swa_retention_window(
            512, draft_window_size=draft_window, block_size=block, page_size=page
        )
        for prefix_len in range(draft_window, draft_window + 3 * page):
            draft_start = _compact_draft_start(prefix_len, draft_window, page)
            for lead in range(block + 1):
                with self.subTest(prefix_len=prefix_len, lead=lead):
                    self.assertLessEqual(
                        _first_live_position(prefix_len + lead, retention, page),
                        draft_start,
                    )

    def test_draft_window_alone_is_too_tight(self):
        """Retaining exactly the draft window frees the draft's first page as
        soon as the frontier leads by a token at a page edge."""
        draft_window, page = 4096, 64
        prefix_len = draft_window + page - 1
        self.assertGreater(
            _first_live_position(prefix_len + 1, draft_window, page),
            _compact_draft_start(prefix_len, draft_window, page),
        )


@unittest.skipUnless(_HAS_CUDA, "needs CUDA")
class TestCompactRebuildLocMap(CustomTestCase):
    def test_rows_hold_mapped_ids(self):
        from sglang.kernels.ops.speculative.cache_locs import (
            rebuild_compact_draft_req_to_token_func,
        )

        device = torch.device("cuda")
        g = torch.Generator(device=device).manual_seed(0)
        bs, block, width, num_slots = 6, 8, 512, 4096
        target = torch.randint(
            0, num_slots, (2 * bs, width), generator=g, device=device
        ).to(torch.int32)
        verify_2d = torch.randint(
            0, num_slots, (bs, block), generator=g, device=device
        ).to(torch.int64)
        loc_map = torch.randperm(num_slots, generator=g, device=device)
        req_idx = torch.randperm(2 * bs, generator=g, device=device)[:bs]
        lens = torch.randint(1, 200, (bs,), generator=g, device=device).to(torch.int32)
        start = torch.randint(0, 200, (bs,), generator=g, device=device)
        draft = torch.full((2 * bs, 256), -1, dtype=torch.int32, device=device)

        rebuild_compact_draft_req_to_token_func(
            draft_req_to_token=draft,
            target_req_to_token=target,
            req_pool_indices=req_idx,
            suffix_start=start,
            draft_prefix_lens=lens,
            verify_out_cache_loc_2d=verify_2d,
            batch_size=bs,
            block_size=block,
            loc_map=loc_map,
        )
        for i in range(bs):
            row, n, s = int(req_idx[i]), int(lens[i]), int(start[i])
            expected = torch.cat(
                [loc_map[target[row, s : s + n].long()], loc_map[verify_2d[i]]]
            ).to(torch.int32)
            torch.testing.assert_close(draft[row, : n + block], expected)
            self.assertTrue(bool((draft[row, n + block :] == -1).all()))


class TestDraftWritesInSwaSlots(CustomTestCase):
    def _translator(self, rows=4, width=16):
        from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator

        return KVIndexTranslator(
            req_to_token=torch.zeros((rows, width), dtype=torch.int32),
            token_to_kv_pool_allocator=SimpleNamespace(),
            token_to_kv_pool=SimpleNamespace(),
            page_size=1,
            device="cpu",
            is_draft_worker=True,
        )

    def test_writes_take_the_sliding_window_ids_and_reads_stay_compact(self):
        from sglang.srt.mem_cache.kv_loc_plan import IdSpace

        full_to_swa = torch.arange(100, dtype=torch.int64).flip(0)
        swa_space = IdSpace(key=("swa", 0), write=lambda ids: full_to_swa[ids])
        target, draft = self._translator(), self._translator()
        draft.index_writes_by(swa_space)

        window = torch.tensor([3, 4, 5, 9], dtype=torch.int64)
        plan = target.plan(
            req_pool_indices=torch.tensor([1, 2]),
            seq_lens=torch.tensor([5, 7]),
            seq_lens_cpu=torch.tensor([5, 7]),
            write_virtual=window,
            read_extent=2,
        )
        batch = SimpleNamespace()
        plan.bind(batch, draft)
        torch.testing.assert_close(batch.out_cache_loc, full_to_swa[window])
        self.assertIsNone(batch.out_cache_loc_virtual)
        # The target's own full-attention writes keep the virtual ids.
        torch.testing.assert_close(plan.write_ids(target), window)

        table = draft.read_table(plan)
        self.assertIs(table.ids, draft.req_to_token)
        self.assertFalse(table.is_translated)


if __name__ == "__main__":
    unittest.main()
