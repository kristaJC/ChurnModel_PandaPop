from pyspark.sql import functions as F, types as T, DataFrame
from pyspark.ml import Pipeline, PipelineModel

from pyspark.ml.tuning import CrossValidatorModel, TrainValidationSplitModel, ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.linalg import DenseVector, SparseVector, VectorUDT
from pyspark.ml.feature import VectorAssembler, StandardScaler, Imputer
from pyspark.ml.classification import LogisticRegression, GBTClassifier
from pyspark.storagelevel import StorageLevel
from datetime import datetime
import matplotlib.pyplot as plt
import mlflow
import pandas as pd


# =========================================================
# Helpers
# =========================================================

#TODO:  Probably add some error checking for the binary classification evaluator

# New - setting a default evaluator for binary classification CV
def _set_evaluator(estimator, 
                   evaluator, 
                   label_col, 
                   prediction_col, 
                   probability_col, 
                   collect_submodels=False, 
                   metric="areaUnderPR"):
    if isinstance(estimator, (CrossValidator, TrainValidationSplit)):

        evaluator.setLabelCol(label_col)
        evaluator.setRawPredictionCol(probability_col)
        evaluator.setMetricName(metric)

        # set Evaluator and collectSubModels flag
        estimator.setEvaluator(evaluator).setCollectSubModels(collect_submodels)
        print("setting estimator and evaluator")

        return estimator, evaluator
    else:
        #print("setting evaluator")
        #estimator.setEvaluator(evaluator)
        return estimator, evaluator



def _best_model(estimator):
    if isinstance(estimator, (CrossValidatorModel, TrainValidationSplitModel)):
        return estimator.bestModel
    return estimator


def _flatten_params(model_or_stage, prefix=""):
    params = {}
    if isinstance(model_or_stage, PipelineModel):
        for i, st in enumerate(model_or_stage.stages):
            params.update(_flatten_params(st, f"{prefix}{i}_{st.__class__.__name__}__"))
        return params
    try:
        for p, v in model_or_stage.extractParamMap().items():
            params[f"{prefix}{p.name}"] = str(v)
    except Exception:
        pass
    return params


def _positive_prob_col(df, col, pos_idx=1):
    dt = df.schema[col].dataType
    if isinstance(dt, (T.DoubleType, T.FloatType)):
        return F.col(col).cast("double")
    if isinstance(dt, VectorUDT):
        if hasattr(F, "vector_to_array"):
            return F.vector_to_array(F.col(col))[pos_idx]
        udf_extract = F.udf(lambda v: float(v[pos_idx]) if v else None, T.DoubleType())
        return udf_extract(F.col(col))
    return F.col(col).cast("double")


def _tp_fp_fn_tn(df, label, prob_expr, thr):
    pred = (prob_expr >= F.lit(thr)).cast("int")
    agg = df.agg(
        F.sum((F.col(label) == 1).cast("int") * pred).alias("tp"),
        F.sum((F.col(label) == 0).cast("int") * pred).alias("fp"),
        F.sum((F.col(label) == 1).cast("int") * (1 - pred)).alias("fn"),
        F.sum((F.col(label) == 0).cast("int") * (1 - pred)).alias("tn"),
    ).first()
    return agg.tp, agg.fp, agg.fn, agg.tn


def _metrics(tp, fp, fn, tn, beta=1.0):
    total = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec = tp / (tp + fn) if tp + fn > 0 else 0
    acc = (tp + tn) / total if total > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
    fb = (1 + beta**2) * prec * rec / (beta**2 * prec + rec) if prec + rec > 0 else 0
    return dict(precision=prec, recall=rec, accuracy=acc, f1=f1, fbeta=fb)


def _metrics_at_threshold(df, label, prob_expr, thr):
    tp, fp, fn, tn = _tp_fp_fn_tn(df, label, prob_expr, thr)
    m1 = _metrics(tp, fp, fn, tn, beta=1.0)
    m2 = _metrics(tp, fp, fn, tn, beta=2.0)
    return dict(threshold=thr, tp=tp, fp=fp, fn=fn, tn=tn,
                precision=m1["precision"], recall=m1["recall"],
                accuracy=m1["accuracy"], f1=m1["f1"], f2=m2["fbeta"])


def _confusion_plot(tp, fp, fn, tn, title):
    from sklearn.metrics import ConfusionMatrixDisplay
    import numpy as np
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(np.array([[tn, fp], [fn, tp]]), display_labels=[0, 1]) \
        .plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    return fig


# =========================================================
# Manual ROC (Spark ≥3.5 fallback)
# =========================================================
def _manual_roc_curve(df, label_col, score_col, n_points=100):
    thresholds = [i / n_points for i in range(n_points + 1)]
    total_pos = df.filter(F.col(label_col) == 1).count()
    total_neg = df.filter(F.col(label_col) == 0).count()
    points = []
    for thr in thresholds:
        pred = (F.col(score_col) >= thr).cast("int")
        agg = df.agg(
            F.sum((F.col(label_col) == 1).cast("int") * pred).alias("tp"),
            F.sum((F.col(label_col) == 0).cast("int") * pred).alias("fp")
        ).first()
        tpr = agg.tp / total_pos if total_pos > 0 else 0
        fpr = agg.fp / total_neg if total_neg > 0 else 0
        points.append((fpr, tpr))
    return points


# =========================================================
# Curves (Spark compute → matplotlib)
# =========================================================
def _binary_curves(df, label, prob):
    from pyspark.mllib.evaluation import BinaryClassificationMetrics
    rdd = df.select(F.col(prob).alias("score"), F.col(label).cast("float").alias("label")) \
             .rdd.map(lambda r: (float(r.score), float(r.label)))
    bcm = BinaryClassificationMetrics(rdd)

    roc_attr = getattr(bcm, "roc", None)
    pr_attr = getattr(bcm, "pr", None)

    if callable(roc_attr):  # Spark ≤3.4
        roc_pts = [(float(x), float(y)) for x, y in bcm.roc().collect()]
        pr_pts  = [(float(x), float(y)) for x, y in bcm.pr().collect()]
    else:  # Spark ≥3.5
        roc_pts = _manual_roc_curve(df, label, prob)
        pr_pts  = []  # PR opcional

    return dict(roc=roc_pts, pr=pr_pts,
                auc_roc=float(bcm.areaUnderROC),
                auc_pr=float(bcm.areaUnderPR))


def _plot_and_log_curves(df, label, prob, title_suffix):
    curves = _binary_curves(df, label, prob)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for name, (pts, auc, xlab, ylab) in {
        "roc": (curves["roc"], curves["auc_roc"], "FPR", "TPR"),
        "pr": (curves["pr"], curves["auc_pr"], "Recall", "Precision"),
    }.items():
        fig, ax = plt.subplots()
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, label=f"{name.upper()} (AUC={auc:.3f})")
        if name == "roc":
            ax.plot([0, 1], [0, 1], "--", label="Random")
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        ax.set_title(f"{name.upper()} Curve {title_suffix}"); ax.legend()
        path = f"{name}_{ts}.png"
        fig.savefig(path, bbox_inches="tight"); mlflow.log_artifact(path); plt.close(fig)
    mlflow.log_metrics({"auc_roc": curves["auc_roc"], "auc_pr": curves["auc_pr"]})


# =========================================================
# Main runner
# =========================================================
def run_spark_cv_with_logging_spark_only(
        estimator,
        train_df: DataFrame,
        test_df: DataFrame,
        val_df: DataFrame = None,
        *,
        label_col="label",
        features_col="features",
        probability_col="probability",
        prediction_col="prediction",
        positive_index=1,
        thresholds=None,
        run_name="spark-ml-experiment",
        extra_tags=None,
        additional_metrics=None,
        persist_eval=True,
        log_confusion=True,
        log_curves=True,
    ):
    thresholds = thresholds or [i / 100 for i in range(1, 100)]
    eval_df = val_df if val_df is not None else test_df
    eval_name = "validation" if val_df is not None else "test"
    

    try:
        with mlflow.start_run(run_name=run_name) as run:

            fitted = estimator.fit(train_df)
            model = _best_model(fitted)
            mlflow.spark.log_model(model, artifact_path="spark_model")

            scored = model.transform(eval_df).select(
                F.col(label_col).cast("int").alias(label_col),
                F.col(probability_col).alias(probability_col)
            )
            if persist_eval:
                scored.persist(StorageLevel.MEMORY_AND_DISK)

            p_pos = _positive_prob_col(scored, probability_col, positive_index)

            # --- Evaluadores Spark ---
            evaluator = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol=probability_col)
            roc_auc = evaluator.evaluate(scored, {evaluator.metricName: "areaUnderROC"})
            pr_auc = evaluator.evaluate(scored, {evaluator.metricName: "areaUnderPR"})

            # --- Sweep thresholds ---
            best_f1, best_t_f1 = -1, 0.5
            best_f2, best_t_f2 = -1, 0.5
            for t in thresholds:
                m = _metrics_at_threshold(scored, label_col, p_pos, t)
                mlflow.log_metrics({f"f1@{t:.2f}": m["f1"], f"f2@{t:.2f}": m["f2"]})
                if m["f1"] > best_f1: best_f1, best_t_f1 = m["f1"], t
                if m["f2"] > best_f2: best_f2, best_t_f2 = m["f2"], t

            at_f1 = _metrics_at_threshold(scored, label_col, p_pos, best_t_f1)
            at_f2 = _metrics_at_threshold(scored, label_col, p_pos, best_t_f2)
            metrics = dict(
                area_under_roc=roc_auc, area_under_pr=pr_auc,
                optimal_threshold_f1=best_t_f1, optimal_threshold_f2=best_t_f2,
                precision_at_f1=at_f1["precision"], recall_at_f1=at_f1["recall"],
                f1_at_f1=at_f1["f1"], accuracy_at_f1=at_f1["accuracy"],
                precision_at_f2=at_f2["precision"], recall_at_f2=at_f2["recall"],
                f1_at_f2=at_f2["f1"], accuracy_at_f2=at_f2["accuracy"],
            )
            if additional_metrics:
                for k, v in additional_metrics.items():
                    if isinstance(v, (int, float)): metrics[k] = float(v)
            mlflow.log_metrics(metrics)

            mlflow.log_params(_flatten_params(model))
            tags = {
                "eval_set": eval_name,
                "tuner": type(fitted).__name__,
                "positive_index": str(positive_index),
                "pipeline_type": model.__class__.__name__,
            }
            if extra_tags: tags.update(extra_tags)
            mlflow.set_tags(tags)

            # --- Charts ---
            scored_plot = scored.select(F.col(label_col), p_pos.alias("p_pos"))
            if log_confusion:
                for name, thr in [("f1", best_t_f1), ("f2", best_t_f2)]:
                    tp, fp, fn, tn = _tp_fp_fn_tn(scored_plot, label_col, F.col("p_pos"), thr)
                    fig = _confusion_plot(tp, fp, fn, tn, f"Confusion Matrix @ {name.upper()} ({eval_name})")
                    path = f"cm_{name}_{eval_name}.png"
                    fig.savefig(path, bbox_inches="tight"); mlflow.log_artifact(path); plt.close(fig)
            if log_curves:
                _plot_and_log_curves(scored_plot, label_col, "p_pos", f"({eval_name})")

            # --- Heldout test ---
            if val_df is not None:
                scored_test = model.transform(test_df).select(
                    F.col(label_col).cast("int").alias(label_col),
                    F.col(probability_col).alias(probability_col)
                )
                p_pos_test = _positive_prob_col(scored_test, probability_col, positive_index)
                test_plot = scored_test.select(F.col(label_col), p_pos_test.alias("p_pos"))
                for name, thr in [("f1", best_t_f1), ("f2", best_t_f2)]:
                    t = _metrics_at_threshold(test_plot, label_col, F.col("p_pos"), thr)
                    mlflow.log_metrics({f"test_{name}_precision": t["precision"],
                                        f"test_{name}_recall": t["recall"],
                                        f"test_{name}_f1": t["f1"],
                                        f"test_{name}_accuracy": t["accuracy"]})
                    if log_confusion:
                        tp, fp, fn, tn = _tp_fp_fn_tn(test_plot, label_col, F.col("p_pos"), thr)
                        fig = _confusion_plot(tp, fp, fn, tn, f"Confusion Matrix @ {name.upper()} (test)")
                        path = f"cm_test_{name}.png"
                        fig.savefig(path, bbox_inches="tight"); mlflow.log_artifact(path); plt.close(fig)
                if log_curves:
                    _plot_and_log_curves(test_plot, label_col, "p_pos", "(test)")

            print(f"Run complete: {run.info.run_id}")
            return {"run_id": run.info.run_id, "metrics": metrics, "tags": tags}
    finally:
        if persist_eval:
            scored.unpersist()


def _log_single_fitted_estimator(eval_df, 
                         estimator, # fitted estimator model
                         evaluator, 
                         probability_col, 
                         positive_index,
                         sub_model=False, # Flag to indicate if it is a submodel or not
                        ):
    if sub_model:
        model_base_path = "./submodels/"
        plot_base_path = "./submodels_plots/"
        metric_base_path = "./submodels_metrics/"
        params_base_path = "./submodels_params/"
    
    else:
        model_base_path = ""
        plot_base_path = "./plots/"
        metric_base_path = "./metrics/"
        params_base_path = "./params/"


    mlflow.log_model(estimator, artifact_path= f"{model_base_path}spark_model")

    # Get Metrics and log them
    scored = _get_scored(estimator, eval_df)
    metrics = _calculate_metrics(scored, evaluator, probability_col, positive_index)
    p_pos = _positive_prob_col(scored, probability_col, positive_index)

    mlflow.log_metrics(metrics, artifact_path= f"{metrics_base_path}metrics_{estimator.name}")
    mlflow.log_params({
            "estimator": estimator.__class__.__name__,
            "estimator_params": estimator.getEstimatorParamMaps(),
            "evaluator": estimator.__class__.__name__,
            "evaluator_params": estimator.getParams(),
        }, artifact_path=f"{params_base_path}params_{estimator.name}")
    
    ##TODO: Set this up to logo appropriately.
    _plot_all_charts(estimator, 
        scored, 
        evaluator, 
        probability_col, 
        positive_index,
        model_base_path,
        plot_base_path,
        metric_base_path,
        params_base_path
        )



def _log_model_submodels(eval_df, 
                         estimator, # fitted estimator model
                         evaluator, 
                         probability_col, 
                         positive_index,
                         collect_submodels=False):

    """
        Helper function to get and log all of the submodels trained for a CV pipeline; Best model is the parent model.
        If not cv, just logs the estimator model and returns the estimator
    """

    sub_models = []

    if isinstance(estimator, (CrossValidatorModel, TrainValidationSplitModel)):
        if collect_submodels:
            sub_models = estimator.subModels
        estimator = _best_model(estimator)

    #### Log the only or best model as the parent
    mlflow.log_model(estimator, artifact_path="spark_model")

    # Get Metrics and log them
    scored = _get_scored(estimator, eval_df)
    metrics = _calculate_metrics(scored, evaluator, probability_col, positive_index)

    mlflow.log_metrics(metrics, artifact_path=f"metrics_{estimator.name}")
    mlflow.log_params({
            "estimator": estimator.__class__.__name__,
            "estimator_params": estimator.getEstimatorParamMaps(),
            "evaluator": estimator.__class__.__name__,
            "evaluator_params": estimator.getParams(),
        }, artifact_path=f"params_{estimator.name}")
    _plot_all_charts(estimator, scored, evaluator, probability_col, positive_index)

    # For each submodel - log them separetly in the submodels directory
    if collect_submodels:
        for sub_model in sub_models:

            mlflow.log_model(sub_model, artifact_path = f"./submodels/{sub_model.name}")
            
            scored = _get_scored(sub_model, eval_df)

            metrics = _calculate_metrics(scored, evaluator, probability_col, positive_index)
            
            mlflow.log_metrics(metrics, artifact_path=f"./submodel_metrics/{sub_model.name}")
            mlflow.log_params(
                {
                    "estimator": sub_model.__class__.__name__,
                    "estimator_params": sub_model.getEstimatorParamMaps(),
                    "evaluator": sub_model.__class__.__name__,
                    "evaluator_params": sub_model.getParams(),
                }, 
                artifact_path = f"./submodel_params/{sub_model.name}"
            )

            ### TODO: LOG THE IMAGES/CHARTS
            _plot_all_charts(sub_model, scored, evaluator, probability_col, positive_index)
        
    return estimator


def _calculate_metrics_old(scored, evaluator, probability_col, positive_index):

    p_pos = _positive_prob_col(scored, probability_col, positive_index)

    # Get 
    roc_auc = evaluator.evaluate(scored, {evaluator.metricName: "areaUnderROC"})
    pr_auc = evaluator.evaluate(scored, {evaluator.metricName: "areaUnderPR"})

    # --- Sweep thresholds ---
    best_f1, best_t_f1 = -1, 0.5
    best_f2, best_t_f2 = -1, 0.5
    for t in thresholds:
        m = _metrics_at_threshold(scored, label_col, p_pos, t)
        mlflow.log_metrics({f"f1@{t:.2f}": m["f1"], f"f2@{t:.2f}": m["f2"]})
        if m["f1"] > best_f1: best_f1, best_t_f1 = m["f1"], t
        if m["f2"] > best_f2: best_f2, best_t_f2 = m["f2"], t

    at_f1 = _metrics_at_threshold(scored, label_col, p_pos, best_t_f1)
    at_f2 = _metrics_at_threshold(scored, label_col, p_pos, best_t_f2)
    metrics = dict(
        area_under_roc=roc_auc, 
        area_under_pr=pr_auc,
        optimal_threshold_f1=best_t_f1, 
        optimal_threshold_f2=best_t_f2,
        precision_at_f1=at_f1["precision"], 
        recall_at_f1=at_f1["recall"],
        f1_at_f1=at_f1["f1"], 
        accuracy_at_f1=at_f1["accuracy"],
        precision_at_f2=at_f2["precision"], 
        recall_at_f2=at_f2["recall"],
        f1_at_f2=at_f2["f1"], 
        accuracy_at_f2=at_f2["accuracy"],
        )
    
    return metrics


def _calculate_metrics_old(scored, evaluator, probability_col, positive_index):

    p_pos = _positive_prob_col(scored, probability_col, positive_index)

    # Get 
    roc_auc = evaluator.evaluate(scored, {evaluator.metricName: "areaUnderROC"})
    pr_auc = evaluator.evaluate(scored, {evaluator.metricName: "areaUnderPR"})

    # --- Sweep thresholds ---
    best_f1, best_t_f1 = -1, 0.5
    best_f2, best_t_f2 = -1, 0.5
    for t in thresholds:
        m = _metrics_at_threshold(scored, label_col, p_pos, t)
        mlflow.log_metrics({f"f1@{t:.2f}": m["f1"], f"f2@{t:.2f}": m["f2"]})
        if m["f1"] > best_f1: best_f1, best_t_f1 = m["f1"], t
        if m["f2"] > best_f2: best_f2, best_t_f2 = m["f2"], t

    at_f1 = _metrics_at_threshold(scored, label_col, p_pos, best_t_f1)
    at_f2 = _metrics_at_threshold(scored, label_col, p_pos, best_t_f2)
    metrics = dict(
        area_under_roc=roc_auc, 
        area_under_pr=pr_auc,
        optimal_threshold_f1=best_t_f1, 
        optimal_threshold_f2=best_t_f2,
        precision_at_f1=at_f1["precision"], 
        recall_at_f1=at_f1["recall"],
        f1_at_f1=at_f1["f1"], 
        accuracy_at_f1=at_f1["accuracy"],
        precision_at_f2=at_f2["precision"], 
        recall_at_f2=at_f2["recall"],
        f1_at_f2=at_f2["f1"], 
        accuracy_at_f2=at_f2["accuracy"],
        )
    
    return metrics


def _get_scored(model, eval_df):
    try:
            scored = model.transform(eval_df).select(
                F.col(label_col).cast("int").alias(label_col),
                F.col(probability_col).alias(probability_col)
            )
            if persist_eval:
                scored.persist(StorageLevel.MEMORY_AND_DISK)
            #DB
            print("Transforming validation set")
    except:
        print(f"ERROR:{probability_col} not found in eval_df")
        raise Exception(f"{probability_col} not found in eval_df")
    
    return scored

def _plot_all_charts_old(scored, metrics, label_col, sub_model=False, log_confusion=True):

    #if sub_model:
    #    eval_name = sub_model.name
    #    base_path = "./submodel_plots/"
    #else:
    #    eval_name = model.name
    #    base_path = ""

    # --- Charts ---
    scored_plot = scored.select(F.col(label_col), p_pos.alias("p_pos"))
    if log_confusion:
        for name, thr in [("f1", metrics['best_t_f1']), ("f2", metrics['best_t_f2'])]:
        #for name, thr in [("f1", best_t_f1), ("f2", best_t_f2)]:
            tp, fp, fn, tn = _tp_fp_fn_tn(scored_plot, label_col, F.col("p_pos"), thr)
            fig = _confusion_plot(tp, fp, fn, tn, f"Confusion Matrix @ {name.upper()} ({eval_name})")
            path = f"cm_{name}_{eval_name}.png"
            #fig.savefig(path, bbox_inches="tight"); 
            mlflow.log_artifact(base_path+path); 
            plt.close(fig)
    if log_curves:
        _plot_and_log_curves(scored_plot, label_col, "p_pos", f"({eval_name})")

    # --- Heldout test ---
    if val_df is not None:
        scored_test = model.transform(test_df).select(
            F.col(label_col).cast("int").alias(label_col),
            F.col(probability_col).alias(probability_col)
        )
        p_pos_test = _positive_prob_col(scored_test, probability_col, positive_index)
        test_plot = scored_test.select(F.col(label_col), p_pos_test.alias("p_pos"))
        for name, thr in [("f1", best_t_f1), ("f2", best_t_f2)]:
            t = _metrics_at_threshold(test_plot, label_col, F.col("p_pos"), thr)
            mlflow.log_metrics({f"test_{name}_precision": t["precision"],
                                f"test_{name}_recall": t["recall"],
                                f"test_{name}_f1": t["f1"],
                                f"test_{name}_accuracy": t["accuracy"]},
                               artifact_path = base_path)
            if log_confusion:
                tp, fp, fn, tn = _tp_fp_fn_tn(test_plot, label_col, F.col("p_pos"), thr)
                fig = _confusion_plot(tp, fp, fn, tn, f"Confusion Matrix @ {name.upper()} (test)")
                path = f"cm_test_{name}.png"
                #fig.savefig(path, bbox_inches="tight"); 
                mlflow.log_artifact(base_path + path); 
                plt.close(fig)
        if log_curves:
            _plot_and_log_curves(test_plot, label_col, "p_pos", "(test)")




##TODO: This is weird. Fix this 
def _plot_all_charts(scored, metrics, label_col, p_pos, sub_model=False, log_confusion=True,log_curves=True):

    #if sub_model:
    #    eval_name = sub_model.name
    #    base_path = "./submodel_plots/"
    #else:
    #    eval_name = model.name
    #    base_path = ""

    # --- Charts ---
    #scored_plot = scored.select(F.col(label_col), p_pos.alias("p_pos"))
    #if log_confusion:
        #for name, thr in [("f1", metrics['best_t_f1']), ("f2", metrics['best_t_f2'])]:
            #tp, fp, fn, tn = _tp_fp_fn_tn(scored_plot, label_col, F.col("p_pos"), thr)
            #fig = _confusion_plot(tp, fp, fn, tn, f"Confusion Matrix @ {name.upper()} ({eval_name})")
            #path = f"cm_{name}_{eval_name}.png"
            #fig.savefig(path, bbox_inches="tight"); 
            #mlflow.log_artifact(base_path+path); 
            #plt.close(fig)
    
    if log_curves:
        _plot_and_log_curves(scored_plot, label_col, "p_pos", f"({eval_name})")

    # --- Heldout test ---
    if val_df is not None:
        scored_test = model.transform(test_df).select(
            F.col(label_col).cast("int").alias(label_col),
            F.col(probability_col).alias(probability_col)
        )
        p_pos_test = _positive_prob_col(scored_test, probability_col, positive_index)
        test_plot = scored_test.select(F.col(label_col), p_pos_test.alias("p_pos"))
        for name, thr in [("f1", best_t_f1), ("f2", best_t_f2)]:
            t = _metrics_at_threshold(test_plot, label_col, F.col("p_pos"), thr)
            mlflow.log_metrics({f"test_{name}_precision": t["precision"],
                                f"test_{name}_recall": t["recall"],
                                f"test_{name}_f1": t["f1"],
                                f"test_{name}_accuracy": t["accuracy"]},
                               artifact_path = base_path)
            if log_confusion:
                tp, fp, fn, tn = _tp_fp_fn_tn(test_plot, label_col, F.col("p_pos"), thr)
                fig = _confusion_plot(tp, fp, fn, tn, f"Confusion Matrix @ {name.upper()} (test)")
                path = f"cm_test_{name}.png"
                #fig.savefig(path, bbox_inches="tight"); 
                mlflow.log_artifact(base_path + path); 
                plt.close(fig)
        if log_curves:
            _plot_and_log_curves(test_plot, label_col, "p_pos", "(test)")


# =========================================================
# Main runner - v2
# =========================================================
def run_spark_ml_training(
        estimator,
        train_df: DataFrame,
        test_df: DataFrame,
        val_df: DataFrame = None,
        *,
        label_col="label",
        evaluator = BinaryClassificationEvaluator(),
        features_col="features",
        probability_col="probability",
        prediction_col="prediction",
        positive_index=1,
        thresholds=None,
        run_name="spark-ml-experiment",
        extra_tags=None,
        additional_metrics=None,
        persist_eval=True,
        log_confusion=True,
        log_curves=True,
        collect_submodels=True,
        eval_metric = "areaUnderPR"
    ):
    

    thresholds = thresholds or [i / 100 for i in range(1, 100)]
    eval_df = val_df if val_df is not None else test_df
    eval_name = "validation" if val_df is not None else "test"

    metrics = {}
    tags = {
            "eval_set": eval_name,
            "positive_index": str(positive_index),
        }

    with mlflow.start_run(run_name=run_name) as run:
        try:
            # Before fitting CV, we must set the evaluator 
            estimator, evaluator = _set_evaluator(estimator, 
                                                evaluator, 
                                                label_col,
                                                prediction_col,
                                                probability_col,
                                                collect_submodels, 
                                                eval_metric)

            print("Fitting model...")
            fitted = estimator.fit(train_df)
            print("Done Fitting Model...")
        except:
            tags.update(
                    {
                        "run_id": run.info.run_id, 
                        "error":f"Error in fitting",
                        "status":"error"
                    }
                )
            mlflow.set_tags(tags)
            print("Error in fitting")

            return tags
        
        try:
            print("Logging Model")
            model = _log_model_submodels(eval_df,
                                         fitted, 
                                         evaluator, 
                                         probability_col, 
                                         positive_index,
                                         collect_submodels)

            #### add if CV - artifact path as parent model. 
            tags.update(
                {
                    "run_id": run.info.run_id, 
                    "pipeline_type": model.__class__.__name__,
                    "tuner": type(fitted).__name__,
                    'params': mlflow.log_params(_flatten_params(model))
                }
            )
            mlflow.set_tags(tags)
        except:
            tags.update(
                    {
                        "run_id": run.info.run_id, 
                        "error":f"Error in training",
                        "status":"error"
                    }
                )
            mlflow.set_tags(tags)

            return tags
    
        #### Transform evaluation set and get prediction probability and labels
        """try:
            scored = model.transform(eval_df).select(
                F.col(label_col).cast("int").alias(label_col),
                F.col(probability_col).alias(probability_col)
            )
            if persist_eval:
                scored.persist(StorageLevel.MEMORY_AND_DISK)
            #DB
            print("Transforming validation set")
        except:
            print(f"ERROR:{probability_col} not found in eval_df")
            return {**tags,
                    'error':f"{probability_col} not found in eval_df",
                    "status":"error",
                    }


        try:
            p_pos = _positive_prob_col(scored, probability_col, positive_index)

            # --- Evaluadores Spark ---
            #evaluator = BinaryClassificationEvaluator() # already set above
            roc_auc = evaluator.evaluate(scored, {evaluator.metricName: "areaUnderROC"})
            pr_auc = evaluator.evaluate(scored, {evaluator.metricName: "areaUnderPR"})

            # --- Sweep thresholds ---
            best_f1, best_t_f1 = -1, 0.5
            best_f2, best_t_f2 = -1, 0.5
            for t in thresholds:
                m = _metrics_at_threshold(scored, label_col, p_pos, t)
                mlflow.log_metrics({f"f1@{t:.2f}": m["f1"], f"f2@{t:.2f}": m["f2"]})
                if m["f1"] > best_f1: best_f1, best_t_f1 = m["f1"], t
                if m["f2"] > best_f2: best_f2, best_t_f2 = m["f2"], t

            at_f1 = _metrics_at_threshold(scored, label_col, p_pos, best_t_f1)
            at_f2 = _metrics_at_threshold(scored, label_col, p_pos, best_t_f2)
            metrics = dict(
                area_under_roc=roc_auc, 
                area_under_pr=pr_auc,
                optimal_threshold_f1=best_t_f1, 
                optimal_threshold_f2=best_t_f2,
                precision_at_f1=at_f1["precision"], 
                recall_at_f1=at_f1["recall"],
                f1_at_f1=at_f1["f1"], 
                accuracy_at_f1=at_f1["accuracy"],
                precision_at_f2=at_f2["precision"], 
                recall_at_f2=at_f2["recall"],
                f1_at_f2=at_f2["f1"], 
                accuracy_at_f2=at_f2["accuracy"],
            )
            if additional_metrics:
                for k, v in additional_metrics.items():
                    if isinstance(v, (int, float)): metrics[k] = float(v)
            mlflow.log_metrics(metrics)

            #mlflow.log_params(_flatten_params(model))
            """

            #model.get_metrics()

        if extra_tags: 
            tags.update(extra_tags)
        mlflow.set_tags(tags)

        # --- Charts ---
        scored_plot = scored.select(F.col(label_col), p_pos.alias("p_pos"))
        if log_confusion:
            for name, thr in [("f1", metrics['best_t_f1']), ("f2", metrics['best_t_f2'])]:
            #for name, thr in [("f1", best_t_f1), ("f2", best_t_f2)]:
                tp, fp, fn, tn = _tp_fp_fn_tn(scored_plot, label_col, F.col("p_pos"), thr)
                fig = _confusion_plot(tp, fp, fn, tn, f"Confusion Matrix @ {name.upper()} ({eval_name})")
                path = f"cm_{name}_{eval_name}.png"
                #fig.savefig(path, bbox_inches="tight"); 
                mlflow.log_artifact(path); 
                plt.close(fig)
        if log_curves:
            _plot_and_log_curves(scored_plot, label_col, "p_pos", f"({eval_name})")

        # --- Heldout test ---
        if val_df is not None:
            scored_test = model.transform(test_df).select(
                F.col(label_col).cast("int").alias(label_col),
                F.col(probability_col).alias(probability_col)
            )
            p_pos_test = _positive_prob_col(scored_test, probability_col, positive_index)
            test_plot = scored_test.select(F.col(label_col), p_pos_test.alias("p_pos"))
            for name, thr in [("f1", best_t_f1), ("f2", best_t_f2)]:
                t = _metrics_at_threshold(test_plot, label_col, F.col("p_pos"), thr)
                mlflow.log_metrics({f"test_{name}_precision": t["precision"],
                                    f"test_{name}_recall": t["recall"],
                                    f"test_{name}_f1": t["f1"],
                                    f"test_{name}_accuracy": t["accuracy"]})
                if log_confusion:
                    tp, fp, fn, tn = _tp_fp_fn_tn(test_plot, label_col, F.col("p_pos"), thr)
                    fig = _confusion_plot(tp, fp, fn, tn, f"Confusion Matrix @ {name.upper()} (test)")
                    path = f"cm_test_{name}.png"
                    #fig.savefig(path, bbox_inches="tight"); 
                    mlflow.log_artifact(path); 
                    plt.close(fig)
            if log_curves:
                _plot_and_log_curves(test_plot, label_col, "p_pos", "(test)")

        print(f"Run complete: {run.info.run_id}")

        if persist_eval:
            scored.unpersist()
        return {**tags, "status":"success"}
    

    if persist_eval:
        scored.unpersist()
    return {**tags,
            "error":"Failed at metric calculation and plotting phase",
            "status":"failed"}










def run_spark_ml_training_v2(
        estimator,
        train_df: DataFrame,
        test_df: DataFrame,
        val_df: DataFrame = None,
        *,
        label_col="label",
        evaluator = BinaryClassificationEvaluator(),
        features_col="features",
        probability_col="probability",
        prediction_col="prediction",
        positive_index=1,
        thresholds=None,
        run_name="spark-ml-experiment",
        extra_tags=None,
        additional_metrics=None,
        persist_eval=True,
        log_confusion=True,
        log_curves=True,
        collect_submodels=True,
        eval_metric = "areaUnderPR"
    ):
    

    thresholds = thresholds or [i / 100 for i in range(1, 100)]
    eval_df = val_df if val_df is not None else test_df
    eval_name = "validation" if val_df is not None else "test"

    metrics = {}
    tags = {
            "eval_set": eval_name,
            "positive_index": str(positive_index),
        }

    with mlflow.start_run(run_name=run_name) as run:
        try:
            # Before fitting CV, we must set the evaluator 
            estimator, evaluator = _set_evaluator(estimator, 
                                                evaluator, 
                                                label_col,
                                                prediction_col,
                                                probability_col,
                                                collect_submodels, 
                                                eval_metric)

            print("Fitting model...")
            fitted = estimator.fit(train_df)
            print("Done Fitting Model...")
        except:
            tags.update(
                    {
                        "run_id": run.info.run_id, 
                        "error":f"Error in fitting",
                        "status":"error"
                    }
                )
            mlflow.set_tags(tags)
            print("Error in fitting")

            return tags
        
        try:
            print("Logging Model")
            model = _log_model_submodels(eval_df,
                                         fitted, 
                                         evaluator, 
                                         probability_col, 
                                         positive_index,
                                         collect_submodels)

            #### add if CV - artifact path as parent model. 
            #mlflow.spark.log_model(model, artifact_path="spark_model")
            tags.update(
                {
                    "run_id": run.info.run_id, 
                    "pipeline_type": model.__class__.__name__,
                    "tuner": type(fitted).__name__,
                    'params': mlflow.log_params(_flatten_params(model))
                }
            )
            mlflow.set_tags(tags)
        except:
            tags.update(
                    {
                        "run_id": run.info.run_id, 
                        "error":f"Error in training",
                        "status":"error"
                    }
                )
            mlflow.set_tags(tags)

            return tags
    
        #### Transform evaluation set and get prediction probability and labels
        """try:
            scored = model.transform(eval_df).select(
                F.col(label_col).cast("int").alias(label_col),
                F.col(probability_col).alias(probability_col)
            )
            if persist_eval:
                scored.persist(StorageLevel.MEMORY_AND_DISK)
            #DB
            print("Transforming validation set")
        except:
            print(f"ERROR:{probability_col} not found in eval_df")
            return {**tags,
                    'error':f"{probability_col} not found in eval_df",
                    "status":"error",
                    }


        try:
            p_pos = _positive_prob_col(scored, probability_col, positive_index)

            # --- Evaluadores Spark ---
            #evaluator = BinaryClassificationEvaluator() # already set above
            roc_auc = evaluator.evaluate(scored, {evaluator.metricName: "areaUnderROC"})
            pr_auc = evaluator.evaluate(scored, {evaluator.metricName: "areaUnderPR"})

            # --- Sweep thresholds ---
            best_f1, best_t_f1 = -1, 0.5
            best_f2, best_t_f2 = -1, 0.5
            for t in thresholds:
                m = _metrics_at_threshold(scored, label_col, p_pos, t)
                mlflow.log_metrics({f"f1@{t:.2f}": m["f1"], f"f2@{t:.2f}": m["f2"]})
                if m["f1"] > best_f1: best_f1, best_t_f1 = m["f1"], t
                if m["f2"] > best_f2: best_f2, best_t_f2 = m["f2"], t

            at_f1 = _metrics_at_threshold(scored, label_col, p_pos, best_t_f1)
            at_f2 = _metrics_at_threshold(scored, label_col, p_pos, best_t_f2)
            metrics = dict(
                area_under_roc=roc_auc, 
                area_under_pr=pr_auc,
                optimal_threshold_f1=best_t_f1, 
                optimal_threshold_f2=best_t_f2,
                precision_at_f1=at_f1["precision"], 
                recall_at_f1=at_f1["recall"],
                f1_at_f1=at_f1["f1"], 
                accuracy_at_f1=at_f1["accuracy"],
                precision_at_f2=at_f2["precision"], 
                recall_at_f2=at_f2["recall"],
                f1_at_f2=at_f2["f1"], 
                accuracy_at_f2=at_f2["accuracy"],
            )
            if additional_metrics:
                for k, v in additional_metrics.items():
                    if isinstance(v, (int, float)): metrics[k] = float(v)
            mlflow.log_metrics(metrics)

            #mlflow.log_params(_flatten_params(model))
            """

            #model.get_metrics()

        if extra_tags: 
            tags.update(extra_tags)
        mlflow.set_tags(tags)

        # --- Charts ---
        scored_plot = scored.select(F.col(label_col), p_pos.alias("p_pos"))
        if log_confusion:
            for name, thr in [("f1", metrics['best_t_f1']), ("f2", metrics['best_t_f2'])]:
            #for name, thr in [("f1", best_t_f1), ("f2", best_t_f2)]:
                tp, fp, fn, tn = _tp_fp_fn_tn(scored_plot, label_col, F.col("p_pos"), thr)
                fig = _confusion_plot(tp, fp, fn, tn, f"Confusion Matrix @ {name.upper()} ({eval_name})")
                path = f"cm_{name}_{eval_name}.png"
                #fig.savefig(path, bbox_inches="tight"); 
                mlflow.log_artifact(path); 
                plt.close(fig)
        if log_curves:
            _plot_and_log_curves(scored_plot, label_col, "p_pos", f"({eval_name})")

        # --- Heldout test ---
        if val_df is not None:
            scored_test = model.transform(test_df).select(
                F.col(label_col).cast("int").alias(label_col),
                F.col(probability_col).alias(probability_col)
            )
            p_pos_test = _positive_prob_col(scored_test, probability_col, positive_index)
            test_plot = scored_test.select(F.col(label_col), p_pos_test.alias("p_pos"))
            for name, thr in [("f1", best_t_f1), ("f2", best_t_f2)]:
                t = _metrics_at_threshold(test_plot, label_col, F.col("p_pos"), thr)
                mlflow.log_metrics({f"test_{name}_precision": t["precision"],
                                    f"test_{name}_recall": t["recall"],
                                    f"test_{name}_f1": t["f1"],
                                    f"test_{name}_accuracy": t["accuracy"]})
                if log_confusion:
                    tp, fp, fn, tn = _tp_fp_fn_tn(test_plot, label_col, F.col("p_pos"), thr)
                    fig = _confusion_plot(tp, fp, fn, tn, f"Confusion Matrix @ {name.upper()} (test)")
                    path = f"cm_test_{name}.png"
                    #fig.savefig(path, bbox_inches="tight"); 
                    mlflow.log_artifact(path); 
                    plt.close(fig)
            if log_curves:
                _plot_and_log_curves(test_plot, label_col, "p_pos", "(test)")

        print(f"Run complete: {run.info.run_id}")

        if persist_eval:
            scored.unpersist()
        return {**tags, "status":"success"}
    

    if persist_eval:
        scored.unpersist()
    return {**tags,
            "error":"Failed at metric calculation and plotting phase",
            "status":"failed"}
    #finally:
    