"""CPU checks for batch arithmetic and the review-disjoint preparation contract."""

import unittest
from collections import Counter

import pandas as pd

from common import plan_batch
from split_train_test import partition_reviews
from download_and_dedup_data import Corpus, deduplicate, remove_token_overlaps


class TokenOverlapTests(unittest.TestCase):
    @staticmethod
    def tokenizer(texts, **kwargs):
        # A small deterministic tokenizer exercises the removal policy without downloads.
        return {"input_ids": [[ord(c) for c in ' '.join(text.lower().split())][:kwargs['max_length']]
                              for text in texts]}

    def test_remove_both_sides_but_keep_different_author_or_rating(self):
        keys = [("Review", "Good  food", 5, "same-user"),
                ("review", "Good food", 5, "same-user"),
                ("AUTHOR", "text", 5, "author-a"),
                ("author", "text", 5, "author-b"),
                ("RATING", "text", 5, "rating-user"),
                ("rating", "text", 1, "rating-user"),
                ("unique", "A", 1, "u1"), ("unique", "B", 2, "u2"),
                ("unique", "C", 3, "u3"), ("unique", "D", 4, "u4")]
        metadata = ("name", 0, 0, 1)
        products = {"A": {r: metadata for r in [0, 2, 4, 6, 7, 8, 9]},
                    "B": {r: metadata for r in [1, 3, 5]}}
        corpus = Corpus(keys, products, Counter({p: len(r) for p, r in products.items()}))
        result = deduplicate(corpus)
        remove_token_overlaps(corpus, result, holdout_products=1, tokenizer=self.tokenizer)
        self.assertEqual(set(result.products['A']), {2, 4, 6, 7, 8, 9})
        self.assertEqual(set(result.products['B']), {3, 5})
        report = result.summary['token_overlap']
        self.assertEqual((report['eval_rows_removed'], report['train_rows_removed']), (1, 1))
        self.assertEqual(len(report['pairs_excluded_for_author_or_rating_difference']), 2)
        self.assertEqual(result.row_ids[('A', 2)], 2)
        self.assertEqual(result.row_ids[('B', 3)], 9)

    def test_reject_holdout_membership_change(self):
        keys = [("A", "text", 5, "u"), ("a", "text", 5, "u"),
                ("b", "text", 5, "v"), ("c", "text", 5, "w")]
        metadata = ("name", 0, 0, 1)
        products = {"B": {r: metadata for r in [0, 3]}, "A": {2: metadata}, "C": {1: metadata}}
        corpus = Corpus(keys, products, Counter({p: len(r) for p, r in products.items()}))
        result = deduplicate(corpus)
        with self.assertRaisesRegex(ValueError, 'holdout'):
            remove_token_overlaps(corpus, result, holdout_products=1, tokenizer=self.tokenizer)


class BatchPlanningTests(unittest.TestCase):
    def test_all_supported_gpu_counts(self):
        for world in range(1, 9):
            with self.subTest(gpus=world):
                plan = plan_batch(world)
                self.assertEqual(plan["global_batch"], 256 // world * world)
                self.assertEqual(plan["global_batch"], world * plan["per_device_batch"] * plan["gradient_accumulation_steps"])
                self.assertLessEqual(plan["per_device_batch"], 32)
                self.assertGreater(plan["gradient_accumulation_steps"], 0)

    def test_reference_and_smaller_memory_configs(self):
        for world in [1, 2, 4, 8]:
            with self.subTest(gpus=world):
                self.assertEqual(plan_batch(world)["gradient_accumulation_steps"], 8 // world)
                smaller = plan_batch(world, per_device_batch=16)
                self.assertEqual(smaller["global_batch"], 256)
                self.assertEqual(smaller["gradient_accumulation_steps"], 16 // world)

    def test_explicit_batch_is_not_rounded(self):
        self.assertEqual(plan_batch(3, global_batch=288, per_device_batch=32)["gradient_accumulation_steps"], 3)
        for kwargs in [dict(world_size=3, global_batch=256),
                       dict(world_size=8, per_device_batch=33),
                       dict(world_size=8, global_batch=0),
                       dict(world_size=8, per_device_batch=0),
                       dict(world_size=0), dict(world_size=9)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                plan_batch(**kwargs)


class ProductSplitTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "Id": [6, 2, 5, 1, 4, 3],
            "ProductId": ["C", "A", "C", "A", "B", "B"],
            "UserId": [f"user{i}" for i in range(6)],
            "Summary": ["", "NA", "title", "title", "title", "title"],
            "Text": [f"review{i}" for i in range(6)],
            "Score": [1, 2, 3, 4, 5, 1],
        })

    def test_ties_order_and_no_row_loss(self):
        parts, _, heldout = partition_reviews(self.frame, holdout_products=1)
        self.assertEqual(heldout, ["A"])
        self.assertEqual(parts["eval"].Id.tolist(), [1, 2])
        self.assertEqual(parts["train"].Id.tolist(), [3, 4, 5, 6])
        self.assertEqual(parts["train"].review_index.tolist(), [0, 1, 2, 3])
        self.assertEqual(len(parts["train"]) + len(parts["eval"]), len(self.frame))
        self.assertTrue(set(parts["train"].ProductId).isdisjoint(parts["eval"].ProductId))
        self.assertIn("", parts["train"].Summary.tolist())
        self.assertIn("NA", parts["eval"].Summary.tolist())

    def test_input_row_order_does_not_change_split(self):
        original, _, _ = partition_reviews(self.frame, 1)
        shuffled, _, _ = partition_reviews(self.frame.sample(frac=1, random_state=12), 1)
        for name in original:
            pd.testing.assert_frame_equal(original[name], shuffled[name])

    def test_reject_shared_review_and_empty_training_partition(self):
        duplicate = self.frame.iloc[[0]].assign(Id=7, ProductId="A")
        with self.assertRaisesRegex(ValueError, "dedup"):
            partition_reviews(pd.concat([self.frame, duplicate]), 1)
        with self.assertRaises(ValueError):
            partition_reviews(self.frame, 3)


if __name__ == "__main__":
    unittest.main()
