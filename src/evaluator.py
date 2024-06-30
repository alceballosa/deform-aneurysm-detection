import itertools
import logging
import os

import detectron2.utils.comm as comm
import pandas as pd
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.file_io import PathManager


class CTAEvaluator(DatasetEvaluator):
    def __init__(self, cfg, dataset_name, output_dir, distributed=True):
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.distributed = distributed
        self._logger = logging.getLogger(__name__)
        self.reset()

    def reset(self):
        self._preds = []
        self.scan_ids = []

    def process(self, inputs, outputs):
        """
        cache inputs and outputs

        args:
            inputs(inputs of model): list of input dict
            outputs(outputs of model): tensor of shape (N,M)
        """
        self.scan_ids.extend([x["scan_id"] for x in inputs])
        self._preds.append(outputs.detach().cpu())

    def evaluate(self):
        """
        run evaluation
        """
        self._logger.info("Evaluating using CTAEvaluator")
        # cummulate prediction

        if self.distributed:
            comm.synchronize()
            preds = comm.gather(self._preds, dst=0)
            preds = list(itertools.chain(*preds))

            scan_ids = comm.gather(self.scan_ids, dst=0)
            scan_ids = list(itertools.chain(*scan_ids))

            if not comm.is_main_process():
                return {}
            else:
                # remove duplicate data
                all_data = {}
                for pred, _id in zip(preds, scan_ids):
                    all_data[_id] = pred
                preds, scan_ids = [], []
                for k, v in all_data.items():
                    preds.append(v)
                    scan_ids.append(k)
        else:
            preds = self._preds
            scan_ids = self.scan_ids

        if self.output_dir:
            PathManager.mkdirs(self.output_dir)
        self.save_outputs(preds, scan_ids)
        # TODO implement FROC
        return {"FROC": 0.0}

    def save_outputs(self, preds, scan_ids):
        """
        ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability', 'w', 'h', 'd']
        """
        df = []
        for pred, scan_id in zip(preds, scan_ids):
            # N,8 (class_id, prob, z,y,x,d,h,w)
            pred = pred.numpy()
            for x in pred:
                df.append([scan_id] + list(x[1:]))
        columns = [
            "seriesuid",
            "probability",
            "coordZ",
            "coordY",
            "coordX",
            "d",
            "h",
            "w",
        ]
        df = pd.DataFrame(df, columns=columns)
        results_filename = os.path.join(self.output_dir, "predict.csv")
        df.to_csv(results_filename, index=False)
