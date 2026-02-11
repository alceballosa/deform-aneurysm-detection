import logging
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from sklearn.metrics._ranking import confusion_matrix_at_thresholds
from tabulate import tabulate
from torch.multiprocessing import Process
from tqdm import tqdm

from log_utils import setup_logger
from metrics import label_files

"""
FROC (Free-Response Receiver Operating Characteristic) Evaluation Module

This module implements FROC curve computation and evaluation for medical image 
analysis, specifically for aneurysm detection. It provides tools to:
- Compute FROC curves from predictions and ground truth labels
- Calculate sensitivity at various false positive per image (FPpI) thresholds
- Perform bootstrap analysis for confidence interval estimation
- Generate evaluation reports and visualizations

The evaluation supports CSV data, handles 3D bounding boxes,
and includes parallel processing for bootstrap confidence intervals.
"""

# Constants
DISEASE = "aneurysm"
np.set_printoptions(linewidth=310)


class FROCEvaluator:
    """
    Evaluator for computing FROC (Free-Response Receiver Operating Characteristic) curves.

    This class handles the full evaluation pipeline for object detection in medical images:
    - Parses ground truth labels and predictions
    - Matches predictions to ground truth using IoU thresholds
    - Computes FROC curves showing sensitivity vs false positives per image
    - Performs bootstrap analysis for confidence intervals
    - Generates evaluation reports and visualization figures

    The evaluator supports both overlap-based (IoU) and intersection-over-minimum (IoM)
    matching strategies depending on the evaluation mode.
    """

    def __init__(
        self,
        label_file,
        preds,
        *,
        logger=None,
        iou_thr=0.4,
        out_dir=None,
        max_fppi=None,
        n_workers=8,
        n_bootstraps=10000,
        ci=0.95,
        n_fppi=10000,
        fppi_thrs=[
            0.125,
            0.25,
            0.5,
            1.0,
        ],
        seed=0,
        out_bs=500,
        save_curves=False,
        min_fppi=1e-4,
        fp_scale="linear",
        exp_name=None,
        use_world_xyz=True,
        mode="val",
    ):
        """
        Initialize the FROC evaluator.

        Args:
            label_file: Path to CSV file containing ground truth labels
            preds: DataFrame or path containing model predictions
            logger: Logger instance for output messages (default: creates new logger)
            iou_thr: IoU threshold for matching predictions to ground truth (default: 0.4)
            out_dir: Output directory for results, figures, and cache files (default: None)
            max_fppi: Maximum false positives per image for plotting (default: None)
            n_workers: Number of parallel workers for bootstrap computation (default: 8)
            n_bootstraps: Number of bootstrap samples for confidence intervals (default: 10000)
            ci: Confidence interval level, e.g. 0.95 for 95% CI (default: 0.95)
            n_fppi: Number of points in the FROC curve (default: 10000)
            fppi_thrs: List of FPpI thresholds at which to report sensitivity (default: [0.125, 0.25, 0.5, 1.0])
            seed: Random seed for reproducible bootstrap sampling (default: 0)
            out_bs: Batch size for saving bootstrap results to disk (default: 500)
            save_curves: Whether to save raw curve data to disk (default: False)
            min_fppi: Minimum false positives per image for plotting (default: 1e-4)
            fp_scale: Scale for FPpI axis, either 'linear' or 'log' (default: 'linear')
            exp_name: Name of experiment for tracking (default: None)
            use_world_xyz: Whether to use world coordinates (default: True)
            mode: Evaluation mode affecting IoU/IoM calculation (default: 'val')
        """
        assert fp_scale in ["linear", "log"]
        self._iou_thr = iou_thr
        out_dir = Path(out_dir)
        self._out_dir = out_dir
        self._mode = mode
        if out_dir is not None:
            os.makedirs(out_dir / "figures", exist_ok=True)
            os.makedirs(out_dir / "bt_cache", exist_ok=True)
            os.makedirs(out_dir / "curves", exist_ok=True)
        self._max_fppi = max_fppi
        self._min_fppi = min_fppi
        self.use_world_xyz = use_world_xyz
        self._fp_scale = fp_scale
        self._n_fppi = n_fppi
        self._fppi_thrs = np.array(fppi_thrs)
        self._metric_names = [f"Se@FPpI={x:.1f}" for x in fppi_thrs]
        self._n_workers = n_workers
        self._n_bootstraps = n_bootstraps
        self._ci = ci
        self._seed = seed
        self._out_bs = out_bs
        self._save_curves = save_curves
        self._logger = logging.getLogger(__name__) if logger is None else logger
        self._exp_name = exp_name
        self._gts, self._categories, self._images = self.parse_gt_csv(label_file)
        self._dts = self.parse_dt_csv(preds, self._categories)
        self.preds = preds
        self.label_file = pd.read_csv(label_file)
        # compute totol pos per category
        n_pos_per_cat = {}
        for cat in self._categories.values():
            n_pos = sum([len(x["box"]) for x in self._gts[cat].values()])
            n_pos_per_cat[cat] = n_pos
        # print(n_pos_per_cat)
        self._n_pos_per_cat = n_pos_per_cat
        self._images = sorted(
            list(
                set(
                    self.preds["seriesuid"].unique().tolist()
                    + self.label_file["seriesuid"].unique().tolist()
                )
            )
        )

    def evaluate(self):
        """
        Perform the evaluation by matching predictions to ground truth.

        This method:
        1. Computes pairwise IoU between all predictions and ground truth boxes
        2. Matches predictions to ground truth based on IoU threshold
        3. Generates detection statistics (matched/unmatched predictions and GT)
        4. Saves detection results to CSV file

        Results are stored in self._match_results for later FROC computation.
        """
        # compute iou

        ious = {}  # category -> img_id -> iou matrix
        for category in self._categories.values():
            per_cat_ious = {}
            for img_id in self._images:

                pred = self._dts[category].get(img_id, {"box": []})["box"]
                gt = self._gts[category].get(img_id, {"box": []})["box"]

                iou = None
                if len(pred) and len(gt):
                    iou = self._pairwise_iou(gt, pred)
                per_cat_ious[img_id] = iou
            ious[category] = per_cat_ious

        # matching per category per image detection
        # image_id -> category -> {
        #    'scores': list of score,
        #    'gts': list of corresponding gt
        # }
        match_result = defaultdict(dict)

        col_detections = "model", "seriesuid", "detected"

        list_detections = []

        for img_id in self._images:
            for category in self._categories.values():
                p_boxes = self._dts[category].get(img_id, {"box": []})["box"]
                p_scores = self._dts[category].get(img_id, {"score": []})["score"]
                gts = self._gts[category].get(img_id, [])
                dict_match, un_matched_gt = self._match(
                    p_boxes, p_scores, gts, ious[category][img_id], self._iou_thr
                )
                dict_match["un_matched_gt"] = un_matched_gt

                match_result[img_id][category] = dict_match
                for i in range(len(un_matched_gt) - 1):
                    if un_matched_gt[i] == 1:
                        list_detections.append((self._exp_name, img_id, False))
                    else:
                        list_detections.append((self._exp_name, img_id, True))

        df_detections = pd.DataFrame(list_detections, columns=col_detections)
        df_detections.to_csv(
            os.path.join(self._out_dir, "model_detections.csv"), index=False
        )
        print(self._out_dir)
        self._match_results = match_result

        # self._compute_froc(save_fig=True)

    def get_bootstrap_data(self):
        """
        Prepare data for bootstrap confidence interval computation.

        Returns:
            tuple: (match_list, n_pos, categories) where:
                - match_list: List of match results for each image
                - n_pos: List of dicts mapping category to number of positive instances per image
                - categories: List of category names
        """
        match_results = self._match_results
        match_list, n_pos = [], []
        categories = list(self._categories.values())
        gts = self._gts
        for img_id in self._images:
            match_list.append(match_results[img_id])
            pos = {c: len(gts[c].get(img_id, {"box": []})["box"]) for c in categories}
            n_pos.append(pos)

        return match_list, n_pos, categories

    def run_compute_froc(self, save_fig=False):
        """
        Compute FROC curves and report sensitivity at specified FPpI thresholds.

        This method computes the FROC curve for each category by:
        1. Collecting all matched predictions and ground truth across images
        2. Computing the FROC curve (recall vs FPpI)
        3. Interpolating sensitivity at specified FPpI thresholds
        4. Generating result tables and optionally saving figures

        Args:
            save_fig: If True, save FROC curve plots to the output directory (default: False)
        """
        fppi_thrs = self._fppi_thrs
        cats = []
        results = []
        n_imgs = len(self._images)
        print("computing froc ...")
        for category in tqdm(self._categories.values()):
            cats.append(category)
            gts, preds = [], []
            for match in self._match_results.values():
                gts.append(match[category]["gts"])
                preds.append(match[category]["scores"])
            gts = np.concatenate(gts)
            preds = np.concatenate(preds)

            recalls, FPpI, _ = compute_froc(
                preds, gts, self._n_pos_per_cat[category], n_imgs
            )
            results.append(np.interp(fppi_thrs, FPpI, recalls))
            print(np.interp(fppi_thrs, FPpI, _))
            thres = np.interp(fppi_thrs, FPpI, _)
            np.save(os.path.join(self._out_dir, f"thres_{category}.npy"), thres)
            print(os.path.join(self._out_dir, f"thres_{category}.npy"))
            if save_fig:
                self._save_fig(recalls, FPpI, category)

            if self._save_curves:
                cat_file_name = category.replace("/", "_").replace(" ", "_").lower()
                torch.save(
                    {"recall": recalls, "fppi": FPpI},
                    os.path.join(self._out_dir, "curves", f"curve_{cat_file_name}.pth"),
                )
        self._derive_results(cats, results)

    def _derive_results(self, classes, results):
        def f2str(x):
            if x < 0.9995:
                return f"{x:.3f}"[1:]
            return "1.00"

        def str2f(x):
            if x[0] == "1":
                return 1.0
            return float(f"0{x}")

        metric_names = self._metric_names
        results_table = []
        for cat, k_results in zip(classes, results):
            row = [cat] + [f2str(float(x)) for x in k_results]
            results_table.append(row)

        headers = ["Finding"] + metric_names
        df = pd.DataFrame(results_table, columns=headers)
        means = {}
        for metric in metric_names:
            means[metric] = f2str(df[metric].apply(lambda x: str2f(x[:4])).mean())
        means[headers[0]] = "Mean"
        df.loc["mean"] = means
        df.to_csv(os.path.join(self._out_dir, "froc.csv"), index=False, columns=headers)

        table = tabulate(
            results_table + [[means[x] for x in headers]],
            tablefmt="pipe",
            floatfmt=".3f",
            headers=headers,
            numalign="left",
        )
        # self._logger.info(f"Per-finding bbox FROC at iou {self._iou_thr} \n" + table)
        print((f"Per-finding bbox FROC at iou {self._iou_thr} \n" + table))

    def _compute_bootstrap(self):
        start = time.perf_counter()
        n_workers = self._n_workers
        n_bootstraps = self._n_bootstraps
        global_count = mp.Value("i", 0)
        # results_queue = mp.Queue()
        barrier = mp.Barrier(n_workers + 1)
        match_list, n_pos, categories = self.get_bootstrap_data()

        workers = [
            BootstrapWorker(
                pid=i,
                global_count=global_count,
                barrier=barrier,
                # results_queue=results_queue,
                n_bootstraps=n_bootstraps,
                seed=self._seed + i,
                match_list=match_list,
                n_pos=n_pos,
                categories=categories,
                out_dir=self._out_dir,
                out_bs=self._out_bs,
                # max_fppi=self._max_fppi,
                # min_fppi=self._min_fppi,
                # n_fppi=self._n_fppi
            )
            for i in range(n_workers)
        ]
        for worker in workers:
            worker.start()
        print("init time ", time.perf_counter() - start)
        barrier.wait()

        num_bootstraps = global_count.value
        print(f"got total of {num_bootstraps} bootstraps")
        # results = []
        # for i in range(num_bootstraps):
        #     results.append(results_queue.get())

        for worker in workers:
            worker.join()
        print("main run time", time.perf_counter() - start)

    def _process_bootstrap(self):
        n_bootstraps = self._n_bootstraps
        fppi_thrs = self._fppi_thrs
        scale = self._fp_scale
        max_fppi = self._max_fppi
        assert max_fppi is not None

        m_res, lb_res, ub_res = [], [], []
        cats = []
        mean_frocs = np.zeros((len(self._categories), n_bootstraps, len(fppi_thrs)))
        print("processing bootstrap results ...")
        for cat_id, cat in enumerate(tqdm(self._categories.values())):
            # print("processing ", cat)

            cats.append(cat)
            max_fppi_c = (
                float(max_fppi) if isinstance(max_fppi, (int, float)) else max_fppi[cat]
            )
            if scale == "linear":
                fppi_grid = np.linspace(self._min_fppi, max_fppi_c, num=self._n_fppi)
            else:
                fppi_grid = np.exp(
                    np.linspace(
                        np.log(self._min_fppi),
                        np.log(max_fppi_c),
                        num=self._n_fppi,
                    )
                )
            cat_file_name = cat.replace("/", "_").replace(" ", "_").lower()
            files = glob(
                os.path.join(self._out_dir, "bt_cache", f"worker*{cat_file_name}.pth")
            )
            files = sorted(files)

            bt_id = 0
            interpolated_recall = np.zeros((n_bootstraps, len(fppi_grid)))
            specific_recall = np.zeros((n_bootstraps, len(fppi_thrs)))
            for _file in files:
                results = torch.load(_file)
                for recall, fppi in zip(results["recalls"], results["fppi"]):
                    if bt_id == n_bootstraps:
                        break
                    interpolated_recall[bt_id] = np.interp(fppi_grid, fppi, recall)
                    specific_recall[bt_id] = np.interp(fppi_thrs, fppi, recall)

                    bt_id += 1

            mean_frocs[cat_id] = specific_recall

            # get bootstrap curve
            recall_m, recall_lb, recall_ub = get_mean_ci(interpolated_recall, self._ci)
            self._save_fig(recall_m, fppi_grid, cat, rec_ub=recall_ub, rec_lb=recall_lb)
            if self._save_curves:
                torch.save(
                    {
                        "m_recall": recall_m,
                        "lb_recall": recall_lb,
                        "ub_recall": recall_ub,
                        "fppi": fppi_grid,
                    },
                    os.path.join(
                        self._out_dir, "curves", f"bt_curve_{cat_file_name}.pth"
                    ),
                )
            # recall at specific threshold
            rec_m, rec_lb, rec_ub = get_mean_ci(specific_recall, self._ci)
            m_res.append(rec_m)
            lb_res.append(rec_lb)
            ub_res.append(rec_ub)
            # print("Done bt froc", cat)

        mean_frocs = mean_frocs.mean(axis=0)
        rec_m, rec_lb, rec_ub = get_mean_ci(mean_frocs, self._ci)
        m_res.append(rec_m)
        lb_res.append(rec_lb)
        ub_res.append(rec_ub)
        cats.append("Mean")

        self._derive_bt_results(cats, m_res, ub_res, lb_res)

    def _derive_bt_results(self, classes, m_results, ub_results, lb_results):
        """
        results: shape (K,M) K: num classes, M: num metrics
        """

        def f2str(x):
            if x < 0.9995:
                return f"{x:.3f}"[1:]
            return "1.00"

        def str2f(x):
            if x[0] == "1":
                return 1.0
            return float(f"0{x}")

        metric_names = self._metric_names
        results_table = []
        for cat, k_means, k_lbs, k_ubs in zip(
            classes, m_results, lb_results, ub_results
        ):
            row = [cat]
            for mean, lb, ub in zip(k_means, k_lbs, k_ubs):
                # print(mean, lb, ub)
                row.append(
                    f"{f2str(float(mean))}({f2str(float(lb))}--{f2str(float(ub))})"
                )
            results_table.append(row)

        headers = ["Finding"] + metric_names
        df = pd.DataFrame(results_table, columns=headers)
        # means = {}
        # for metric in metric_names:
        #     means[metric] = f2str(df[metric].apply(lambda x: str2f(x[:4])).mean())
        # means[headers[0]] = "Mean"
        # df.loc['mean'] = means
        df.to_csv(
            os.path.join(self._out_dir, "froc_bt.csv"), index=False, columns=headers
        )

        table = tabulate(
            # results_table + [[means[x] for x in headers]],
            results_table,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=headers,
            numalign="left",
        )
        self._logger.info("Per-finding bbox FROC \n" + table)

    def _save_fig(self, recalls, FPpI, category, *, rec_ub=None, rec_lb=None):
        assert self._out_dir is not None
        plt.close()
        path = os.path.join(
            self._out_dir, "figures", f"{category.replace('/', '_')}_froc.png"
        )

        path_recalls = os.path.join(
            self._out_dir, "figures", f"{category.replace('/', '_')}_recalls.npy"
        )
        np.save(path_recalls, recalls)
        path_FPpI = os.path.join(
            self._out_dir, "figures", f"{category.replace('/', '_')}_FPpI.npy"
        )
        np.save(path_FPpI, FPpI)
        plt.figure(figsize=(5, 5))
        plt.title(f"{category} FROC")
        plt.plot(FPpI, recalls)
        plt.xlabel("Average number of false positive per scan")
        plt.ylabel("Recall")
        if self._fp_scale == "log":
            plt.xscale("log")
        if rec_lb is not None:
            path = path.replace("froc.", "froc_bootstrap.")
            plt.plot(
                FPpI,
                rec_ub,
                "r--",
                FPpI,
                rec_lb,
                "r--",
            )
        xmax = FPpI.max()
        max_fppi = self._max_fppi
        if max_fppi:
            if isinstance(max_fppi, (float, int)):
                xmax = float(max_fppi)
            else:
                xmax = float(max_fppi[category])
        if self._fp_scale == "log":
            xmin = self._min_fppi
        else:
            xmin = -0.1
        plt.xlim(xmin, xmax)
        plt.ylim(xmin, 1.01)
        plt.grid(linestyle="--", which="both")
        plt.savefig(path)

    def _match(self, p_boxes, p_scores, gts, ious, iou_thr):
        """
        assess whether each pred is pos or neg
        args
            threshold: iou threshold to match

        return
        """
        assert len(p_boxes) == len(p_scores)
        # no_pred_score = -0.01
        if not (len(gts) or len(p_scores)):
            return {
                "scores": np.array([]),
                "gts": np.array([]),
                "all_un_matched_gt": np.array([]),
            }, []

        if not len(gts):
            scores = p_scores.numpy()
            gts = np.zeros_like(scores)
            return {"scores": scores, "gts": gts, "all_un_matched_gt": np.array([])}, []

        if not len(p_scores):
            # gts = np.ones(size=(len(gts),))
            # scores = no_pred_score * gts
            # return {"scores": scores, "gts": gts}
            return {
                "scores": np.array([]),
                "gts": np.array([]),
                "all_un_matched_gt": np.array([]),
            }, []

        un_matched_gt = torch.ones(size=(ious.size(0) + 1,))

        _, sorted_ids = torch.sort(p_scores, descending=True)
        all_un_matched_gts = []
        r_scores, r_gts = [], []
        for i in sorted_ids:
            i_iou = ious[:, i]
            matched_gt_id = -1
            best_iou = -1
            for gt_id, iou in enumerate(i_iou):
                if iou >= iou_thr and un_matched_gt[gt_id] > 0 and iou > best_iou:
                    matched_gt_id = gt_id
                    best_iou = iou

            # update un_matched_gt
            un_matched_gt[matched_gt_id] = 0
            all_un_matched_gts.append(
                [float(p_scores[i]), un_matched_gt.numpy().copy()]
            )

            # update results
            gt = 1.0 if matched_gt_id > -1 else 0.0
            r_gts.append(gt)
            r_scores.append(float(p_scores[i]))

        # add false negative (if any)
        # for not_detected in un_matched_gt[:-1]:
        #     if not_detected:
        #         r_scores.append(no_pred_score)
        #         r_gts.append(1.)

        return (
            {
                "scores": np.array(r_scores),
                "gts": np.array(r_gts),
                "all_un_matched_gt": all_un_matched_gts,
            },
            un_matched_gt.numpy(),
        )

    def _pairwise_iou(self, box_list1, box_list2):
        """
        Compute pairwise 3D IoU or IoM between two sets of bounding boxes.

        Depending on the evaluation mode, this computes either:
        - IoU (Intersection over Union) for standard evaluation
        - IoM (Intersection over Minimum) for certain hospital datasets

        Args:
            box_list1: Tensor of shape (N, 6) with boxes in corner format (x1, y1, z1, x2, y2, z2)
            box_list2: Tensor of shape (M, 6) with boxes in corner format

        Returns:
            Tensor of shape (N, M) containing pairwise IoU or IoM values

        Note:
            Assumes all boxes are non-empty (have positive volume).
        """
        # compute intersection
        width_height = torch.min(
            box_list1[:, None, 3:], box_list2[None, :, 3:]
        ) - torch.max(box_list1[:, None, :3], box_list2[None, :, :3])
        width_height.clamp_(min=0.0)
        intersection = width_height.prod(dim=2)  # (N,M)

        # compute area
        area1 = (box_list1[:, 3:] - box_list1[:, :3]).prod(dim=1)
        area2 = (box_list2[:, 3:] - box_list2[:, :3]).prod(dim=1)
        if self._mode not in ["hospital", "hospital140", "cta_rsna_ane"]:
            return intersection / (area1[:, None] + area2[None, :] - intersection)
        else:
            return intersection / (np.minimum(area1[:, None], area2[None, :]))

    def parse_gt_csv(self, path):
        """
        Parse ground truth labels from CSV file.

        Args:
            path: Path to CSV file containing ground truth annotations with columns:
                  seriesuid, coordX, coordY, coordZ, w, h, d

        Returns:
            tuple: (gts_dict, categories_dict, image_list) where:
                - gts_dict: dict mapping disease -> seriesuid -> {"box": tensor}
                - categories_dict: dict mapping category id to category name
                - image_list: list of all image/series IDs
        """
        results = {}
        data = pd.read_csv(path)

        id2disease = {1: DISEASE}
        all_imgs = []

        # self._logger.info(f"got {len(all_imgs)} gt images")

        # print("parsing ground truth ...")
        for seriesuid, rows in data.groupby("seriesuid"):
            all_imgs.append(seriesuid)
            box = np.array(rows[["coordX", "coordY", "coordZ", "w", "h", "d"]])
            if self._mode in ["hospital", "hospital140"]:
                sides = box[:, 3:]

                minimum_side = np.argmin(sides, axis=1)
                # get the second largest side value  using argsort
                second_largest_side = np.argsort(box[:, 3:], axis=1)[:, 1]

                sides[:, minimum_side] = sides[:, second_largest_side]
                box[:, 3:] = sides
                # for i in range(len(box)):
                #    box[i,3:] = box[i,3:].mean()

            box = xyzwhd2xyzxyz(torch.tensor(box))

            results[seriesuid] = {"box": box}

        return {DISEASE: results}, id2disease, all_imgs

    def parse_dt_csv(self, preds, id2disease):
        """
        Parse predictions from CSV DataFrame.

        Args:
            preds: DataFrame containing predictions with columns:
                   seriesuid, coordX, coordY, coordZ, w, h, d, probability
            id2disease: Dictionary mapping category IDs to disease names (unused but kept for API consistency)

        Returns:
            dict: Mapping disease -> seriesuid -> {"box": tensor, "score": tensor}
                  where boxes are sorted by descending prediction score
        """
        results = {}

        # print("parsing predictions ...")
        for seriesuid, rows in preds.groupby("seriesuid"):
            box_data = np.array(
                rows[["coordX", "coordY", "coordZ", "w", "h", "d", "probability"]]
            )
            results[seriesuid] = {"box": box_data[:, :6], "score": box_data[:, -1]}

        # convert to tensor and sort box
        for image in results.values():
            score, sorted_id = torch.sort(torch.tensor(image["score"]), descending=True)
            image["score"] = score

            box = torch.tensor(image["box"])[sorted_id]
            image["box"] = xyzwhd2xyzxyz(box)

        return {DISEASE: results}


def xyzwhd2xyzxyz(boxes):
    """
    Convert bounding boxes from center format to corner format.

    Transforms boxes from (center_x, center_y, center_z, width, height, depth)
    to (x1, y1, z1, x2, y2, z2) where (x1, y1, z1) is the minimum corner
    and (x2, y2, z2) is the maximum corner.

    Args:
        boxes: Tensor of shape (N, 6) with boxes in center format

    Returns:
        Tensor of shape (N, 6) with boxes in corner format
    """
    res = torch.zeros_like(boxes)
    res[:, :3] = boxes[:, :3] - boxes[:, 3:] / 2
    res[:, 3:] = boxes[:, :3] + boxes[:, 3:] / 2
    return res


def compute_froc(preds, gts, n_pos, n_imgs, *, outputs=None):
    """
    Compute FROC curve from predictions and ground truth labels.

    The FROC (Free-Response Receiver Operating Characteristic) curve plots
    sensitivity (recall) against the average number of false positives per image.

    Args:
        preds: Array of prediction scores/confidences
        gts: Array of ground truth labels (0 for false positive, 1 for true positive)
        n_pos: Total number of positive instances across all images
        n_imgs: Total number of images
        outputs: Optional path to save intermediate results as .pth file (default: None)

    Returns:
        tuple: (recalls, FPpI, thresholds) where:
            - recalls: Array of recall values (sensitivity)
            - FPpI: Array of false positives per image
            - thresholds: Array of score thresholds corresponding to each point
    """
    #return tns, fps, fns, tps, y_score[threshold_idxs]

    _, fps, _, tps, thrs = confusion_matrix_at_thresholds(gts, preds)

    if outputs:
        assert outputs[-4:] == ".pth"
        torch.save(
            {"fps": fps, "tps": tps, "thrs": thrs, "n_pos": n_pos, "n_imgs": n_imgs},
            outputs,
        )

    recalls = tps / n_pos
    fppi = fps / n_imgs
    return recalls.astype(np.float32), fppi.astype(np.float32), thrs.astype(np.float32)


def get_mean_ci(values, ci=0.95):
    """
    Compute mean and confidence intervals from bootstrap samples.

    This function calculates the mean and confidence interval bounds from
    a set of bootstrap samples. The confidence interval is computed using
    the percentile method.

    Args:
        values: Array of shape (N, d1, d2, ...) where N is the number of
                bootstrap samples and remaining dimensions are the data shape
        ci: Confidence interval level between 0 and 1 (default: 0.95 for 95% CI)

    Returns:
        tuple: (mean, lower_bound, upper_bound) where each has shape (d1, d2, ...)

    Note:
        This function sorts the input array in-place along the first axis.
    """

    n = len(values)
    tail = (1.0 - ci) / 2.0
    bound_id = math.floor(tail * n)

    values.sort(axis=0)
    mean = values.mean(axis=0)
    lb = values[bound_id]
    ub = values[-bound_id]

    return mean, lb, ub


class BootstrapWorker(Process):
    """
    Parallel worker process for computing bootstrap confidence intervals.

    This worker generates bootstrap samples by resampling images with replacement
    and computing FROC curves for each sample. Multiple workers run in parallel
    to speed up the bootstrap computation. Results are saved to disk in batches
    to manage memory usage.

    Attributes:
        process_id: Unique identifier for this worker process
        global_count: Shared counter tracking total bootstraps completed
        barrier: Synchronization barrier for coordinating workers
        out_dir: Directory for saving bootstrap results
        n_bootstraps: Total number of bootstrap samples to generate
        match_list: List of prediction-ground truth matches per image
        n_pos: List of positive instance counts per image and category
        categories: List of category names
        random_state: NumPy random state for reproducible sampling
        out_bs: Batch size for saving results to disk
    """

    def __init__(
        self,
        pid,
        global_count,
        barrier,
        out_dir,
        n_bootstraps,
        seed,
        match_list,
        n_pos,
        categories,
        out_bs,
    ):
        """
        Initialize the bootstrap worker.

        Args:
            pid: Process ID for this worker
            global_count: Multiprocessing Value for tracking global progress
            barrier: Multiprocessing Barrier for synchronization
            out_dir: Output directory for caching bootstrap results
            n_bootstraps: Total number of bootstrap samples to compute
            seed: Random seed for reproducible bootstrap sampling
            match_list: List of match results for each image
            n_pos: List of positive counts per image and category
            categories: List of category names to evaluate
            out_bs: Batch size for writing results to disk
        """
        super(BootstrapWorker, self).__init__()
        self.process_id = pid
        self.global_count = global_count
        self.barrier = barrier
        # self.logging_queue = logging_queue
        # self.results_queue = results_queue
        self.n_bootstraps = n_bootstraps
        self.match_list = match_list
        self.n_pos = n_pos
        self.categories = categories
        self.random_state = np.random.RandomState(seed)
        self.out_dir = out_dir
        self.out_bs = out_bs
        print(f"{pid} initialized")

    def run(self):
        start = time.perf_counter()
        result_count = 0
        all_results = []
        batch_id = 0
        while self.global_count.value < self.n_bootstraps:
            results = self.compute_froc_bootstrap()
            all_results.append(results)

            with self.global_count.get_lock():
                self.global_count.value += 1
                value = self.global_count.value
                if value % 500 == 0:
                    print(f"{str(datetime.now())}: Done {value} bootstraps")

            # self.results_queue.put(result)
            result_count += 1
            if len(all_results) > self.out_bs:
                batch = all_results[: self.out_bs]
                all_results = all_results[self.out_bs :]
                self.save_results(batch, batch_id)
                batch_id += 1
                del batch
        if len(all_results):
            self.save_results(all_results, batch_id)
        print(
            f"worker {self.process_id} processed {result_count} bootstraps in",
            time.perf_counter() - start,
        )
        self.barrier.wait()

    def save_results(self, batch, batch_id):
        n = len(batch)
        # transpose results and save to disc
        for cat in self.categories:
            recalls, fppi, n_pos = zip(*[x[cat] for x in batch])
            results = {
                "recalls": list(recalls),
                "fppi": list(fppi),
                "n_pos": list(n_pos),
            }
            cat = cat.replace("/", "_").replace(" ", "_").lower()
            path = os.path.join(
                self.out_dir,
                "bt_cache",
                f"worker_{self.process_id:0>2}_batch_{batch_id:0>3}_size_{n:0>4}_{cat}.pth",
            )
            torch.save(results, path)
        print(f"worker {self.process_id} saved {n} bootstraps for batch {batch_id}")
        del batch

    def compute_froc_bootstrap(self):
        """
        n_pos: list of (category -> value)
        match_list: list of (category -> {"scores": scores, "gts": gts})

        return category -> (recals, fppi, n_pos)
        """
        match_list = self.match_list
        n_pos = self.n_pos
        categories = self.categories
        n = len(n_pos)
        rand_ids = self.random_state.randint(n, size=n)
        #############
        # TODO: resample while there is no positive case for a category
        # even though it is very unlikely

        # compute num pos
        results = {}
        for cat in categories:
            pos = sum([n_pos[i][cat] for i in rand_ids])

            gts = [match_list[i][cat]["gts"] for i in rand_ids]
            gts = np.concatenate(gts)

            preds = [match_list[i][cat]["scores"] for i in rand_ids]
            preds = np.concatenate(preds)

            recalls, fppi, _ = compute_froc(preds, gts, pos, n)
            results[cat] = (recalls, fppi, pos)

        return results


def main():
    """
    Main entry point for running FROC evaluation from command line.

    This function processes multiple experiments and computes FROC curves for
    aneurysm detection predictions. It supports different evaluation modes for
    filtering predictions based on overlap with anatomical structures.

    Command line usage:
        python froc_overlap_2.py <exp_base_path> [mode]

    Args:
        sys.argv[1]: Path to base experiment directory containing inference results
        sys.argv[2]: (Optional) Evaluation mode for filtering predictions:
            - None or "0": Filter by brain overlap > 0.5
            - "base": No filtering
            - "1": Filter by enhanced brain overlap > 0.5
            - "2": Filter out predictions overlapping with veins
            - "3": Keep predictions with more artery than vein overlap
            - "4": Filter out vein overlap AND require enhanced brain > 0.5
            - "5": More artery than vein AND enhanced brain > 0.5

    The function will:
    - Iterate through all experiments in the base directory
    - Process all inference directories
    - Compute FROC curves at multiple IoU thresholds (0.1, 0.2, 0.3)
    - Save results, figures, and CSV files to output directories
    """
    # Configuration parameters
    max_fppi = 8.0
    min_fppi = 0.0
    fp_scale = "linear"
    fppi_thrs = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    n_bootstraps = 10000
    iou_thrs = [0.1, 0.2, 0.3]
    n_workers = 8

    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python froc_overlap_2.py <exp_base_path> [mode]")
        sys.exit(1)

    exp_base = Path(sys.argv[1])
    mode = None if len(sys.argv) < 3 else sys.argv[2]
    dataset_name = exp_base.name
    exps = list(exp_base.glob("*"))
    label_file = label_files[dataset_name]
    # Process each experiment directory
    for exp_dir in exps:
        # Find all inference directories
        inf_appends = sorted(
            [x.name.replace("inference_", "") for x in exp_dir.glob("inference_*")]
        )

        for inf_append in inf_appends:
            for iou_thr in iou_thrs:
                print(f"Running iou_thr: {iou_thr} at {inf_append}")

                # Determine output directory and predictions path based on mode

                if mode in ["base", "1", "2", "3", "4", "5"]:
                    out_dir = (
                        exp_dir / f"mode_{mode}" / f"iou{iou_thr:.1f}_froc_{inf_append}"
                    )
                    path_preds = (
                        exp_dir / f"inference_{inf_append}" / "predict_roi_jisoo.csv"
                    )
                else:
                    raise ValueError(f"Invalid mode: {mode}")

                # Load and filter predictions
                try:
                    preds = pd.read_csv(path_preds)

                    if mode == "base":
                        preds = preds.copy()
                    elif mode == "1":
                        preds = preds[preds["overlap_enhanced_brain"] > 0.5]
                    elif mode == "2":
                        preds = preds[preds["overlap_vein"] == 0]
                    elif mode == "3":
                        preds = preds[preds["overlap_vein"] <= preds["overlap_artery"]]
                    elif mode == "4":
                        preds = preds[preds["overlap_vein"] == 0]
                        preds = preds[preds["overlap_enhanced_brain"] > 0.5]
                    elif mode == "5":
                        preds = preds[
                            (preds["overlap_vein"] <= preds["overlap_artery"])
                            & (preds["overlap_enhanced_brain"] > 0.5)
                        ]

                except FileNotFoundError:
                    continue

                # Setup logger
                logger = setup_logger(output=out_dir, name=__name__ + str(iou_thr))

                # Create evaluator and run evaluation
                evaluator = FROCEvaluator(
                    label_file=label_file,
                    preds=preds,
                    logger=logger,
                    iou_thr=iou_thr,
                    out_dir=out_dir,
                    max_fppi=max_fppi,
                    fppi_thrs=fppi_thrs,
                    min_fppi=min_fppi,
                    n_bootstraps=n_bootstraps,
                    n_workers=n_workers,
                    fp_scale=fp_scale,
                    use_world_xyz=False,
                    exp_name=exp_dir.name + "_" + inf_append,
                    mode=dataset_name,
                )
                evaluator.evaluate()
                evaluator.run_compute_froc(save_fig=True)
                print()


if __name__ == "__main__":
    main()
