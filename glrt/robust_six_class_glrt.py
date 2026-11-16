#!/usr/bin/env python3
"""Run the CV-selected robust six-class likelihood classifier."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Hashable, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glrt.robust_likelihoods import fit_laplace_models, fit_student_models, predict_robust  # noqa: E402
from glrt.six_class_glrt import (
    PROJECT_ROOT,
    UCI_ACTIVITIES,
    WISDM_ACTIVITIES,
    WISDM_TEST_USERS,
    classification_metrics,
    extract_features,
    read_uci_partition,
    read_wisdm_windows,
)


SELECTED_SHRINKAGE = {"UCI HAR": 0.03, "WISDM": 0.50}


def evaluate(name: str, train_windows: Sequence[Sequence[tuple[float,float,float]]],
             train_labels: Sequence[Hashable], test_windows: Sequence[Sequence[tuple[float,float,float]]],
             test_labels: Sequence[Hashable], class_order: Sequence[Hashable], sample_rate: float,
             distribution: str, degrees_freedom: float, shrinkage: float,
             output_dir: Path) -> dict[str, object]:
    train=[extract_features(window,sample_rate) for window in train_windows]
    test=[extract_features(window,sample_rate) for window in test_windows]
    if distribution == "student_t":
        standardizer,models=fit_student_models(train,train_labels,class_order,degrees_freedom,
                                               shrinkage,1e-6,False,30)
    else:
        standardizer,models=fit_laplace_models(train,train_labels,class_order,shrinkage,1e-6)
    results=[predict_robust(vector,standardizer,models) for vector in test]
    predicted=[result.label for result in results]
    metrics=classification_metrics(test_labels,predicted,class_order)
    metrics.update({"dataset":name,"distribution":distribution,"degrees_freedom":degrees_freedom if distribution=="student_t" else None,
                    "shrinkage":shrinkage,"feature_set":"full","train_windows":len(train_windows)})
    output_dir.mkdir(parents=True,exist_ok=True); stem="uci_har" if name=="UCI HAR" else "wisdm"
    (output_dir/f"{stem}_robust_glrt_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    with (output_dir/f"{stem}_robust_glrt_predictions.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.writer(handle); writer.writerow(("window","actual","predicted","correct","twice_log_glr_margin"))
        for index,(actual,result) in enumerate(zip(test_labels,results)):
            writer.writerow((index,actual,result.label,actual==result.label,result.glr_margin))
    print(f"{name}: accuracy={metrics['accuracy']:.2%}; balanced={metrics['balanced_accuracy']:.2%}; macro F1={metrics['macro_f1']:.2%}")
    return metrics


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",choices=("uci","wisdm","all"),default="all")
    parser.add_argument("--distribution",choices=("student_t","laplace"),default="student_t")
    parser.add_argument("--degrees-freedom",type=float,default=30.0)
    parser.add_argument("--shrinkage",type=float,default=None,
                        help="Override CV-selected dataset-specific shrinkage")
    parser.add_argument("--uci-root",type=Path,default=PROJECT_ROOT/"Datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset")
    parser.add_argument("--wisdm-file",type=Path,default=PROJECT_ROOT/"Datasets/WISDM_ar_latest/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt")
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parent/"results/robust")
    args=parser.parse_args()
    if args.degrees_freedom<=0 or (args.shrinkage is not None and not 0<=args.shrinkage<=1):
        parser.error("degrees of freedom must be positive and shrinkage must lie in [0,1]")
    if args.dataset in ("uci","all"):
        train_windows,train_labels=read_uci_partition(args.uci_root,"train")
        test_windows,test_labels=read_uci_partition(args.uci_root,"test")
        evaluate("UCI HAR",train_windows,train_labels,test_windows,test_labels,UCI_ACTIVITIES,50.0,
                 args.distribution,args.degrees_freedom,args.shrinkage if args.shrinkage is not None else SELECTED_SHRINKAGE["UCI HAR"],args.output_dir)
    if args.dataset in ("wisdm","all"):
        windows,_=read_wisdm_windows(args.wisdm_file)
        train=[row for row in windows if row[0] not in WISDM_TEST_USERS]; test=[row for row in windows if row[0] in WISDM_TEST_USERS]
        evaluate("WISDM",[row[2] for row in train],[row[1] for row in train],[row[2] for row in test],[row[1] for row in test],WISDM_ACTIVITIES,20.0,
                 args.distribution,args.degrees_freedom,args.shrinkage if args.shrinkage is not None else SELECTED_SHRINKAGE["WISDM"],args.output_dir)


if __name__=="__main__": main()
