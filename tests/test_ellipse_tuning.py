import ast
import copy
import os
import unittest

import cv2
import numpy as np


SOURCE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ellipseDetectionV4.py")


def load_target_functions():
    """Load selected functions from ellipseDetectionV4.py without running UI loop."""
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        src = f.read()

    tree = ast.parse(src, filename=SOURCE_FILE)

    wanted = {
        "ellipse_points",
        "support_score",
        "ellipse_inlier_score",
        "partial_arc_score",
        "check_white_infill",
        "maybe_add_candidate",
        "compute_cup_confidence",
        "clamp_roi",
    }

    selected_nodes = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]

    module_ast = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(module_ast)

    ns = {"np": np, "cv2": cv2}
    exec(compile(module_ast, SOURCE_FILE, "exec"), ns)

    # Tunables used by maybe_add_candidate
    ns.update(
        {
            "MIN_MAJOR_AXIS": 40,
            "MIN_MINOR_AXIS": 12,
            "MAX_MINOR_AXIS": 200,
            "MAX_AXIS": 260,
            "MIN_ASPECT": 0.18,
            "MAX_ASPECT": 1.0,
            "SAMPLE_N": 240,
            "MIN_SUPPORT_FALLBACK": 0.28,
            "MIN_INLIER_SCORE": 0.52,
            "STRICT_INLIER_SCORE": 0.72,
            "MIN_INLIER_POINTS": 30,
            "INLIER_TOL": 0.18,
            "MIN_WHITE_SCORE": 0.35,
            "WHITE_SAT_MAX": 80,
            "WHITE_VAL_MIN": 145,
        }
    )

    return ns


class TestEllipseTuning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_target_functions()

    def test_clamp_roi_bounds_and_small_roi_fallback(self):
        clamp_roi = self.ns["clamp_roi"]

        self.assertEqual(clamp_roi(-50, -10, 2000, 1000, 640, 480), (0, 0, 640, 480))
        self.assertEqual(clamp_roi(10, 10, 13, 13, 640, 480), (0, 0, 640, 480))
        self.assertEqual(clamp_roi(20, 30, 120, 130, 640, 480), (20, 30, 120, 130))

    def test_ellipse_inlier_score_high_for_true_ellipse(self):
        ellipse_points = self.ns["ellipse_points"]
        ellipse_inlier_score = self.ns["ellipse_inlier_score"]

        pts = ellipse_points(120, 100, 40, 20, 25, n=300).astype(np.float32)
        score = ellipse_inlier_score(pts, 120, 100, 80, 40, 25, tol=0.12)

        self.assertGreater(score, 0.9)

    def test_partial_arc_score_detects_continuous_arc(self):
        partial_arc_score = self.ns["partial_arc_score"]

        edges = np.zeros((240, 320), dtype=np.uint8)
        cv2.ellipse(edges, ((160, 120), (120, 60), 90), 255, 2, lineType=cv2.LINE_AA)

        score = partial_arc_score(edges, 160, 120, 120, 60, 90)
        self.assertGreater(score, 0.55)

    def test_maybe_add_candidate_respects_max_minor_and_max_aspect(self):
        maybe_add_candidate = self.ns["maybe_add_candidate"]
        ellipse_points = self.ns["ellipse_points"]

        # Synthetic edge map with one clear ellipse
        edges = np.zeros((240, 320), dtype=np.uint8)
        cv2.ellipse(edges, ((160, 120), (100, 60), 90), 255, 2)
        points_xy = ellipse_points(160, 120, 50, 30, 90, n=260).astype(np.float32)

        baseline = dict(self.ns)

        # Baseline: should pass
        candidates = []
        maybe_add_candidate(candidates, points_xy, edges, 160, 120, 100, 60, 90, 3, 0.30, frame_for_infill=None)
        self.assertGreaterEqual(len(candidates), 1)

        # Tighten max minor axis below 60 -> should reject
        self.ns.update(baseline)
        self.ns["MAX_MINOR_AXIS"] = 50
        candidates = []
        maybe_add_candidate(candidates, points_xy, edges, 160, 120, 100, 60, 90, 3, 0.30, frame_for_infill=None)
        self.assertEqual(len(candidates), 0)

        # Tighten max aspect below actual aspect (0.60) -> should reject
        self.ns.update(baseline)
        self.ns["MAX_MINOR_AXIS"] = 200
        self.ns["MAX_ASPECT"] = 0.55
        candidates = []
        maybe_add_candidate(candidates, points_xy, edges, 160, 120, 100, 60, 90, 3, 0.30, frame_for_infill=None)
        self.assertEqual(len(candidates), 0)

    def test_compute_cup_confidence_prefers_clustered_ellipses(self):
        compute_cup_confidence = self.ns["compute_cup_confidence"]

        ellipses = [
            {"cx": 100, "cy": 100, "MA": 100, "ma": 60, "angle": 90, "rank": 0.70},
            {"cx": 106, "cy": 103, "MA": 95, "ma": 58, "angle": 92, "rank": 0.66},
            {"cx": 96, "cy": 98, "MA": 104, "ma": 62, "angle": 88, "rank": 0.68},
            {"cx": 280, "cy": 30, "MA": 110, "ma": 40, "angle": 15, "rank": 0.90},
        ]

        out = compute_cup_confidence(copy.deepcopy(ellipses))
        clustered_conf = np.mean([out[0]["cup_confidence"], out[1]["cup_confidence"], out[2]["cup_confidence"]])
        isolated_conf = out[3]["cup_confidence"]

        self.assertGreater(clustered_conf, isolated_conf)
        for e in out:
            self.assertGreaterEqual(e["cup_confidence"], 0.0)
            self.assertLessEqual(e["cup_confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
