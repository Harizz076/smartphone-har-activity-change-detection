#!/usr/bin/env python3
"""Compare Gaussian, Student-t, and Laplace six-class likelihood models.

This is an exploratory supplementary-classification experiment. Configuration
selection uses balanced, subject-disjoint training folds; the held-out test
partition is evaluated only after a likelihood configuration is selected.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.optimize_six_class_glrt import Dataset, group_folds, read_uci, read_wisdm  # noqa: E402
from glrt.robust_likelihoods import (  # noqa: E402
    fit_laplace_models,
    fit_student_models,
    predict_robust,
    student_model_from_estimate,
    student_t_em,
)
from glrt.six_class_glrt import (  # noqa: E402
    FEATURE_NAMES,
    classification_metrics,
    fit_models,
    fit_standardizer,
    predict,
)


OUTPUT = ROOT / "analysis/robust_likelihoods"
SHRINKAGES = (0.03, 0.10, 0.25, 0.50, 0.75)
DEGREES_FREEDOM = (3.0, 5.0, 10.0, 30.0)
RIDGE = 1e-6
EM_ITERATIONS = 30


@dataclass(frozen=True)
class Configuration:
    distribution: str
    covariance: str
    shrinkage: float
    degrees_freedom: float | None = None

    @property
    def key(self) -> str:
        tail = f"|nu={self.degrees_freedom:g}" if self.degrees_freedom is not None else ""
        return f"{self.distribution}|{self.covariance}|s={self.shrinkage:g}{tail}"


def configurations() -> tuple[Configuration, ...]:
    result = []
    for covariance in ("class_full", "class_diagonal"):
        for shrinkage in SHRINKAGES:
            result.append(Configuration("gaussian", covariance, shrinkage))
    for covariance in ("class_full", "class_diagonal"):
        for degrees in DEGREES_FREEDOM:
            for shrinkage in SHRINKAGES:
                result.append(Configuration("student_t", covariance, shrinkage, degrees))
    for shrinkage in SHRINKAGES:
        result.append(Configuration("laplace", "independent", shrinkage))
    return tuple(result)


def fit_configuration(train_vectors: Sequence[Sequence[float]], train_labels: Sequence[str],
                      class_order: Sequence[str], config: Configuration):
    if config.distribution == "gaussian":
        return fit_models(train_vectors, train_labels, class_order, config.shrinkage,
                          RIDGE, config.covariance)
    if config.distribution == "student_t":
        return fit_student_models(
            train_vectors, train_labels, class_order, config.degrees_freedom,
            config.shrinkage, RIDGE, config.covariance.endswith("diagonal"), EM_ITERATIONS
        )
    if config.distribution == "laplace":
        return fit_laplace_models(train_vectors, train_labels, class_order,
                                  config.shrinkage, RIDGE)
    raise ValueError(f"Unknown distribution {config.distribution!r}")


def predict_configuration(vector: Sequence[float], fitted, distribution: str):
    standardizer, models = fitted
    return predict(vector, standardizer, models) if distribution == "gaussian" else predict_robust(vector, standardizer, models)


def cross_validate(dataset: Dataset, config: Configuration,
                   assignment: dict[int, int], student_cache: dict[tuple[object, ...], object]) -> dict[str, object]:
    predictions: list[str | None] = [None] * len(dataset.train_features)
    margins = [0.0] * len(dataset.train_features)
    convergence = []
    for fold in range(3):
        train_indices = [index for index, group in enumerate(dataset.train_groups) if assignment[group] != fold]
        valid_indices = [index for index, group in enumerate(dataset.train_groups) if assignment[group] == fold]
        train_vectors=[dataset.train_features[index] for index in train_indices]
        train_labels=[dataset.train_labels[index] for index in train_indices]
        if config.distribution == "student_t":
            diagonal=config.covariance.endswith("diagonal")
            cache_key=(dataset.name,fold,diagonal,config.degrees_freedom)
            cached=student_cache.get(cache_key)
            if cached is None:
                standardizer=fit_standardizer(train_vectors)
                standardized=[standardizer.transform(vector) for vector in train_vectors]
                estimates=[]
                for label in dataset.class_order:
                    group=[vector for vector,actual in zip(standardized,train_labels) if actual==label]
                    location,scatter,iterations,converged=student_t_em(
                        group,config.degrees_freedom,diagonal,EM_ITERATIONS
                    )
                    estimates.append((label,len(group),location,scatter,iterations,converged))
                cached=(standardizer,estimates); student_cache[cache_key]=cached
            standardizer,estimates=cached
            models=tuple(student_model_from_estimate(
                label,count,location,scatter,config.degrees_freedom,config.shrinkage,
                RIDGE,diagonal,iterations,converged
            ) for label,count,location,scatter,iterations,converged in estimates)
            fitted=(standardizer,models)
        else:
            fitted = fit_configuration(train_vectors,train_labels,dataset.class_order,config)
        if config.distribution == "student_t":
            convergence.extend((model.em_converged, model.em_iterations) for model in fitted[1])
        for index in valid_indices:
            result = predict_configuration(dataset.train_features[index], fitted, config.distribution)
            predictions[index] = str(result.label); margins[index] = result.glr_margin
    if any(value is None for value in predictions):
        raise AssertionError("Cross-validation did not predict every training window")
    return {
        "config": config,
        "metrics": classification_metrics(dataset.train_labels, predictions, dataset.class_order),
        "predictions": predictions,
        "margins": margins,
        "em_convergence": convergence,
    }


def ranking_key(result: dict[str, object]) -> tuple[float, float, float]:
    metrics = result["metrics"]
    return metrics["balanced_accuracy"], metrics["macro_f1"], metrics["accuracy"]


def fit_final(dataset: Dataset, config: Configuration) -> dict[str, object]:
    fitted = fit_configuration(dataset.train_features, dataset.train_labels,
                               dataset.class_order, config)
    results = [predict_configuration(vector, fitted, config.distribution)
               for vector in dataset.test_features]
    predicted = [str(result.label) for result in results]
    metrics = classification_metrics(dataset.test_labels, predicted, dataset.class_order)
    convergence = []
    if config.distribution == "student_t":
        convergence = [{"label": str(model.label), "converged": model.em_converged,
                        "iterations": model.em_iterations} for model in fitted[1]]
    return {"metrics": metrics, "predictions": predicted,
            "margins": [result.glr_margin for result in results],
            "em_convergence": convergence}


def tail_diagnostics(dataset: Dataset, directory: Path) -> dict[str, float]:
    rows = []
    for activity in dataset.class_order:
        group = [vector for vector,label in zip(dataset.train_features,dataset.train_labels) if label==activity]
        for index,name in enumerate(FEATURE_NAMES):
            values=[vector[index] for vector in group]; center=fmean(values)
            variance=fmean((value-center)**2 for value in values); spread=math.sqrt(variance)
            if spread <= 1e-12:
                skewness=excess=tail=0.0
            else:
                standardized=[(value-center)/spread for value in values]
                skewness=fmean(value**3 for value in standardized)
                excess=fmean(value**4 for value in standardized)-3
                tail=sum(abs(value)>3 for value in standardized)/len(standardized)
            rows.append({"activity":activity,"feature":name,"skewness":skewness,
                         "excess_kurtosis":excess,"fraction_beyond_3sd":tail})
    with (directory/"training_tail_diagnostics.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=tuple(rows[0])); writer.writeheader(); writer.writerows(rows)
    absolute_skewness=sorted(abs(row["skewness"]) for row in rows)
    excess_kurtosis=sorted(row["excess_kurtosis"] for row in rows)
    tail_fractions=sorted(row["fraction_beyond_3sd"] for row in rows)
    p90=min(len(rows)-1,math.ceil(.9*len(rows))-1)
    return {
        "median_absolute_skewness": absolute_skewness[len(rows)//2],
        "p90_absolute_skewness": absolute_skewness[p90],
        "median_excess_kurtosis": excess_kurtosis[len(rows)//2],
        "p90_excess_kurtosis": excess_kurtosis[p90],
        "median_fraction_beyond_3sd": tail_fractions[len(rows)//2],
        "p90_fraction_beyond_3sd": tail_fractions[p90],
        "class_feature_cells": len(rows),
    }


def svg_start(width: int, height: int, title: str, description: str) -> list[str]:
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
            f'<title>{html.escape(title)}</title><desc>{html.escape(description)}</desc>',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:18px;font-weight:bold}.label{font-size:12px}.small{font-size:10px}</style>',
            f'<text class="title" x="{width/2}" y="27" text-anchor="middle">{html.escape(title)}</text>']


def save_svg(lines: list[str], path: Path) -> None:
    path.write_text("\n".join(lines+["</svg>"]),encoding="utf-8")


def plot_ranking(ranked: Sequence[dict[str, object]], path: Path) -> None:
    top=ranked[:25]; width,height,x0,y0,chart_w,row_h=1200,760,470,55,650,27
    colors={"gaussian":"#0072B2","student_t":"#D55E00","laplace":"#009E73"}
    low=min(result["metrics"]["balanced_accuracy"] for result in top)-.01
    lines=svg_start(width,height,"Robust likelihood validation ranking","Top configurations by subject-disjoint balanced accuracy; colors identify likelihood families.")
    for row,result in enumerate(top):
        config=result["config"]; score=result["metrics"]["balanced_accuracy"]; y=y0+row*row_h
        bar=chart_w*(score-low)/(1-low)
        lines += [f'<text class="small" x="{x0-8}" y="{y+16}" text-anchor="end">{html.escape(config.key)}</text>',
                  f'<rect x="{x0}" y="{y+3}" width="{bar:.1f}" height="18" fill="{colors[config.distribution]}"/>',
                  f'<text class="small" x="{x0+bar+5:.1f}" y="{y+16}">{score:.3f}</text>']
    save_svg(lines,path)


def plot_family_best(ranked: Sequence[dict[str, object]], path: Path) -> None:
    families=("gaussian","student_t","laplace"); colors=("#0072B2","#D55E00","#009E73")
    best=[max(result["metrics"]["balanced_accuracy"] for result in ranked if result["config"].distribution==family) for family in families]
    width,height,x0,y0,chart_w,chart_h=760,480,100,65,580,320; low=min(best)-.02
    lines=svg_start(width,height,"Best validation score by likelihood","Best subject-disjoint balanced accuracy within each likelihood family.")
    bar_w=110
    for index,(family,score,color) in enumerate(zip(families,best,colors)):
        x=x0+90+index*180; h=chart_h*(score-low)/(1-low); y=y0+chart_h-h
        lines += [f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" fill="{color}"/>',
                  f'<text class="label" x="{x+bar_w/2}" y="{y-8:.1f}" text-anchor="middle">{score:.3f}</text>',
                  f'<text class="label" x="{x+bar_w/2}" y="{y0+chart_h+22}" text-anchor="middle">{family}</text>']
    lines.append(f'<text class="small" x="{x0}" y="{height-18}">Axis begins at {low:.2f}; family tuning spaces differ.</text>')
    save_svg(lines,path)


def plot_subject_deltas(dataset: Dataset, selected_predictions: Sequence[str],
                        gaussian_predictions: Sequence[str], path: Path) -> None:
    values=[]
    for subject in sorted(set(dataset.test_groups)):
        indices=[index for index,group in enumerate(dataset.test_groups) if group==subject]
        robust=sum(dataset.test_labels[index]==selected_predictions[index] for index in indices)/len(indices)
        gaussian=sum(dataset.test_labels[index]==gaussian_predictions[index] for index in indices)/len(indices)
        values.append((subject,robust-gaussian))
    width,height,x0,y0,chart_w,chart_h=1000,500,80,65,850,330; maximum=max(.01,max(abs(value) for _,value in values))
    zero=y0+chart_h/2; bar_w=chart_w/len(values)*.7
    lines=svg_start(width,height,"Per-subject accuracy change: selected minus Gaussian","Positive bars favor the selected robust likelihood; each held-out subject has equal visual weight.")
    for index,(subject,value) in enumerate(values):
        x=x0+chart_w*(index+.5)/len(values)-bar_w/2; height_value=chart_h/2*abs(value)/maximum
        y=zero-height_value if value>=0 else zero
        color="#009E73" if value>=0 else "#D55E00"
        lines += [f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{height_value:.1f}" fill="{color}"/>',
                  f'<text class="small" x="{x+bar_w/2:.1f}" y="{(y-5 if value>=0 else y+height_value+13):.1f}" text-anchor="middle">{value:+.1%}</text>',
                  f'<text class="small" x="{x+bar_w/2:.1f}" y="{y0+chart_h+18}" text-anchor="middle">{subject}</text>']
    lines.append(f'<line x1="{x0}" y1="{zero}" x2="{x0+chart_w}" y2="{zero}" stroke="#444"/>')
    save_svg(lines,path)


def write_outputs(dataset: Dataset, ranked: Sequence[dict[str, object]], selected: dict[str, object],
                  final: dict[str, object], gaussian_result: dict[str, object], gaussian_final: dict[str, object],
                  assignment: dict[int,int], fold_sizes: Sequence[int],
                  directory: Path) -> None:
    with (directory/"configuration_search.csv").open("w",newline="",encoding="utf-8") as handle:
        fields=("rank","distribution","covariance","degrees_freedom","shrinkage","validation_accuracy","validation_balanced_accuracy","validation_macro_f1","em_converged_models","em_total_models")
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for rank,result in enumerate(ranked,1):
            config=result["config"]; metrics=result["metrics"]; convergence=result["em_convergence"]
            writer.writerow({"rank":rank,"distribution":config.distribution,"covariance":config.covariance,
                             "degrees_freedom":config.degrees_freedom,"shrinkage":config.shrinkage,
                             "validation_accuracy":metrics["accuracy"],"validation_balanced_accuracy":metrics["balanced_accuracy"],
                             "validation_macro_f1":metrics["macro_f1"],
                             "em_converged_models":sum(value[0] for value in convergence),"em_total_models":len(convergence)})
    config=selected["config"]
    diagnostics=tail_diagnostics(dataset,directory)
    summary={"dataset":dataset.name,"exploratory":True,"feature_set":"full",
             "selection_objective":["balanced_accuracy","macro_f1","accuracy"],
             "fold_window_counts":list(fold_sizes),"subject_folds":{str(group):fold for group,fold in sorted(assignment.items())},
             "candidate_count":len(ranked),"training_tail_diagnostics":diagnostics,
             "selected":{"distribution":config.distribution,"covariance":config.covariance,
                         "degrees_freedom":config.degrees_freedom,"shrinkage":config.shrinkage,
                         "validation_metrics":selected["metrics"],"em_convergence":selected["em_convergence"]},
             "test_metrics":final["metrics"],"final_em_convergence":final["em_convergence"],
             "gaussian_reference":{"configuration":gaussian_result["config"].key,
                                   "validation_metrics":gaussian_result["metrics"],
                                   "test_metrics":gaussian_final["metrics"]}}
    (directory/"selection_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    with (directory/"selected_test_predictions.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.writer(handle); writer.writerow(("window","subject","actual","selected_prediction","gaussian_prediction","selected_correct","gaussian_correct","selected_glr_margin"))
        for index,(subject,actual,chosen,gaussian,margin) in enumerate(zip(dataset.test_groups,dataset.test_labels,final["predictions"],gaussian_final["predictions"],final["margins"])):
            writer.writerow((index,subject,actual,chosen,gaussian,actual==chosen,actual==gaussian,margin))
    with (directory/"subject_accuracy_comparison.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.writer(handle); writer.writerow(("subject","windows","selected_accuracy","gaussian_accuracy","difference"))
        for subject in sorted(set(dataset.test_groups)):
            indices=[index for index,group in enumerate(dataset.test_groups) if group==subject]
            selected_accuracy=sum(dataset.test_labels[index]==final["predictions"][index] for index in indices)/len(indices)
            gaussian_accuracy=sum(dataset.test_labels[index]==gaussian_final["predictions"][index] for index in indices)/len(indices)
            writer.writerow((subject,len(indices),selected_accuracy,gaussian_accuracy,selected_accuracy-gaussian_accuracy))
    plot_ranking(ranked,directory/"01_likelihood_ranking.svg")
    plot_family_best(ranked,directory/"02_best_by_likelihood.svg")
    plot_subject_deltas(dataset,final["predictions"],gaussian_final["predictions"],directory/"03_subject_accuracy_delta.svg")


def run(dataset: Dataset) -> None:
    directory=OUTPUT/("uci_har" if dataset.name=="UCI HAR" else "wisdm")
    directory.mkdir(parents=True,exist_ok=True)
    assignment,fold_sizes=group_folds(dataset.train_groups,dataset.train_labels,dataset.class_order)
    configs=configurations(); results=[]; student_cache={}
    print(f"{dataset.name}: evaluating {len(configs)} full-feature likelihood configurations")
    for index,config in enumerate(configs,1):
        results.append(cross_validate(dataset,config,assignment,student_cache))
        if index % 10 == 0: print(f"  completed {index}/{len(configs)}")
    ranked=sorted(results,key=ranking_key,reverse=True); selected=ranked[0]
    final=fit_final(dataset,selected["config"])
    gaussian_result=next(result for result in ranked if result["config"].distribution=="gaussian")
    gaussian_final=fit_final(dataset,gaussian_result["config"])
    write_outputs(dataset,ranked,selected,final,gaussian_result,gaussian_final,assignment,fold_sizes,directory)
    config=selected["config"]
    print(f"{dataset.name}: selected {config.key}")
    print(f"  CV balanced={selected['metrics']['balanced_accuracy']:.2%}; test balanced={final['metrics']['balanced_accuracy']:.2%}; test accuracy={final['metrics']['accuracy']:.2%}")


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",choices=("uci","wisdm","all"),default="all")
    args=parser.parse_args(); OUTPUT.mkdir(parents=True,exist_ok=True)
    if args.dataset in ("uci","all"): run(read_uci())
    if args.dataset in ("wisdm","all"): run(read_wisdm())


if __name__ == "__main__": main()
