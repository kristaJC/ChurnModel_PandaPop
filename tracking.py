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

import mlflow.spark


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
        #evaluator.setRawPredictionCol(probability_col)
        evaluator.setMetricName(metric)

        # set Evaluator and collectSubModels flag
        estimator.setEvaluator(evaluator).setCollectSubModels(collect_submodels)
        print("Setting estimator and evaluator for CV")

    print("Setting estimator and evaluator for pipeline/estimator")
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


def _plt_binary(pts, auc, plot_type, title_suffix=""):
    xlab, ylab = ("FPR", "TPR") if plot_type == "roc" else ("Recall", "Precision")

    fig, ax = plt.subplots()
    if pts:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, label=f"{plot_type.upper()} (AUC={auc:.3f})")
    if plot_type == "roc":
        ax.plot([0, 1], [0, 1], "--", label="Random")
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.set_title(f"{plot_type.upper()} Curve {title_suffix}")
    ax.legend()
    return fig


def _plt_binary_old(pts, auc, plot_type, title_suffix = ""):

    if plot_type == "roc":
        xlab, ylab = "FPR", "TPR"
    if plot_type == "pr":
        xlab,ylab= "Recall", "Precision"

    fig, ax = plt.subplots()
    if pts:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, label=f"{plot_type.upper()} (AUC={auc:.3f})")
    if type == "roc":
        ax.plot([0, 1], [0, 1], "--", label="Random")
    ax.set_xlabel(xlab) 
    ax.set_ylabel(ylab)
    ax.set_title(f"{plot_type.upper()} Curve {title_suffix}"); 
    ax.legend()
    
    return fig


def _get_curves(scored, label, prob, title_suffix=""):
    curves = _binary_curves(scored, label, prob)
    pr_pts  = curves.get('pr', [])
    auc_pr  = curves.get('auc_pr')
    pr_fig  = _plt_binary(pr_pts, auc_pr, 'pr', title_suffix)

    roc_pts = curves.get('roc', [])
    auc_roc = curves.get('auc_roc')
    roc_fig = _plt_binary(roc_pts, auc_roc, 'roc', title_suffix)
    return pr_fig, roc_fig

def _get_curves_old(scored,label, prob, title_suffix=""):

    curves_dict = _binary_curves(scored, label, prob,title_suffix)

    pr_pts = curves_dict.get('pr_pts'),
    auc_pr = curves_dict.get('auc_pr')
    pr_curve =  _plt_binary(pr_pts, auc_pr,'pr')


    roc_pts = curves_dict.get('roc_pts'), 
    auc_roc = curves_dict.get('auc_roc'),
    roc_curve = _plt_binary(roc_pts, auc_roc,'roc')
    
    return pr_curve, roc_curve


def _plt_confusion_matrix(df, 
                          label, 
                          prob_expr, 
                          thr=0.5, # defaults to the default threshold
                          title = "Confusion Matrix"):
    tp, fp, fn, tn = _tp_fp_fn_tn(df, label, prob_expr, thr)

    return _confusion_plot(tp, fp, fn, tn, title)

def _log_single_fitted_estimator(eval_df, 
                         estimator, # fitted estimator model
                         evaluator, 
                         probability_col, 
                         label_col,
                         positive_index,
                         sub_model=False, 
                         validation_flag=False,# Flag to indicate if it is a submodel or not
                        ):
    
    if validation_flag:
        model_base_path = "./validation/"
        plot_base_path = "./validation_plots/"
        metric_base_path = "./validation_metrics/"
        params_base_path = "./validation_params/"
    
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


    mlflow.spark.log_model(estimator, artifact_path= f"{model_base_path}spark_model")

    # Get Metrics and log them
    scored = _get_scored(estimator, eval_df,label_col, probability_col,persist_eval = True)
    p_pos = _positive_prob_col(scored, probability_col, positive_index)
    
    """ 
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
    """
    
    metrics = _calculate_metrics(scored, label_col, probability_col, evaluator, thresholds, positive_index)
    
    params = {
            "estimator": estimator.__class__.__name__,
            "estimator_params": estimator.getEstimatorParamMaps(),
            "evaluator": estimator.__class__.__name__,
            "evaluator_params": estimator.getParams(),
        }

    #Charts and curves: 
    confusion_matrix = _plt_confusion_matrix(scored, 
                                             label_col, 
                                             p_pos,
                                             thr=0.5,
                                             title=f'Confusion Matrix at default threshold=0.5')
    confusion_matrix_f1 = _plt_confusion_matrix(scored,
                                                label_col, 
                                                p_pos, 
                                                thr = metrics['optimal_threshold_f1'],
                                                title=f"Confusion Matrix at f1 threshold={metrics['optimal_threshold_f1']}")
    confusion_matrix_f2 = _plt_confusion_matrix(scored,
                                                label_col, 
                                                p_pos, 
                                                thr = metrics['optimal_threshold_f2'],
                                                title=f"Confusion Matrix at f2 threshold={metrics['optimal_threshold_f2']}")
    
    ## PR and ROC curves
    pr_curve, roc_curve = _get_curves(scored,label_col, probability_col, title_suffix="")


    ###### MLFlow logging 


    ## log metrics and parameters
    mlflow.log_metrics(metrics)
    #mlflow.log_params(params, artifact_path=f"{params_base_path}params_{estimator.name}")
    mlflow.log_dict(params, f"{params_base_path}params_{getattr(estimator,'uid','model')}.json")

    if sub_model:
        estimator_name = estimator.name + '_'
    else:
        estimator_name = ""
    # figures
    mlflow.log_figure(confusion_matrix,   f"{plot_base_path}{estimator_name}confusion_default.png")
    mlflow.log_figure(confusion_matrix_f1, f"{plot_base_path}{estimator_name}confusion_at_f1.png")
    mlflow.log_figure(confusion_matrix_f2, f"{plot_base_path}{estimator_name}confusion_at_f2.png")
    mlflow.log_figure(pr_curve,           f"{plot_base_path}{estimator_name}pr_curve.png")
    mlflow.log_figure(roc_curve,          f"{plot_base_path}{estimator_name}roc_curve.png")
        

    return estimator

def _log_model_submodels(eval_df, 
                         estimator, # fitted estimator model
                         evaluator, 
                         probability_col, 
                         label_col,
                         positive_index,
                         collect_submodels=False,
                         validation_flag = False):

    """
        Helper function to get and log all of the submodels trained for a CV pipeline; 
        Best model is the parent model.
        If not cv, just logs the estimator model and returns the estimator
    """


    sub_models = []

    if isinstance(estimator, (CrossValidatorModel, TrainValidationSplitModel)):
        if collect_submodels:
            sub_models = estimator.subModels
        estimator = _best_model(estimator)

    mlflow.spark.log_model(estimator,artifact_path = "estimator_spark_model")

    estimator = _log_single_fitted_estimator(eval_df, 
                            estimator, # fitted estimator model
                            evaluator, 
                            probability_col, 
                            label_col,
                            positive_index,
                            sub_model,
                            validation_flag, # Flag to indicate if it is a submodel or not
                        )


    # For each submodel - log them separetly in the submodels directory

    if collect_submodels and getattr(estimator, "subModels", None) is not None:
        for gi, fold_models in enumerate(estimator.subModels):           # grid index
            for fi, sm in enumerate(fold_models):                         # fold index
                estimator = _log_single_fitted_estimator(eval_df, 
                            estimator, # fitted estimator model
                            evaluator, 
                            probability_col, 
                            label_col,
                            positive_index,
                            sub_model=True, # Flag to indicate if it is a submodel or not
                        )
    
    """if collect_submodels:
        for sub_model in sub_models:

            estimator = _log_single_fitted_estimator(eval_df, 
                            estimator, # fitted estimator model
                            evaluator, 
                            probability_col, 
                            label_col,
                            positive_index,
                            sub_model=True, # Flag to indicate if it is a submodel or not
                        )"""
        
    return estimator



def _calculate_metrics(scored, label_col, probability_col, evaluator, thresholds, positive_index):

    p_pos = _positive_prob_col(scored, probability_col, positive_index)


    # Get 
    roc_auc = evaluator.evaluate(scored, {evaluator.metricName: "areaUnderROC"})
    pr_auc = evaluator.evaluate(scored, {evaluator.metricName: "areaUnderPR"})

    best_f1, best_t_f1 = -1, 0.5
    best_f2, best_t_f2 = -1, 0.5
    for t in thresholds:
        m = _metrics_at_threshold(scored, label_col, p_pos, t)

        metrics_at_thresh = {f"f1@{t:.2f}": m["f1"], f"f2@{t:.2f}": m["f2"]}
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


def _get_scored(model, eval_df, label_col, probability_col, persist_eval=True):
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
            print("Logging Models")
            model = _log_model_submodels(eval_df,
                                         fitted, 
                                         evaluator, 
                                         probability_col,
                                         label_col, 
                                         positive_index,
                                         collect_submodels)

            #### add if CV - artifact path as parent model. 
            mlflow.log_params(_flatten_params(model))
            tags.update({
                    "run_id": run.info.run_id,
                    "pipeline_type": model.__class__.__name__,
                    "tuner": type(fitted).__name__,
                })
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
    

        if extra_tags: 
            tags.update(extra_tags)
        mlflow.set_tags(tags)

        # --- Heldout test ---
        if val_df is not None:
            
            val_estimator = _log_model_submodels(val_df, 
                         estimator, # fitted estimator model
                         evaluator, 
                         probability_col, 
                         label_col,
                         positive_index,
                         collect_submodels=False, 
                         validation_flag = True)
    
        print(f"Run complete: {run.info.run_id}")

        if persist_eval:
            scored.unpersist()
        return {**tags, "status":"success"}
    

    #if persist_eval:
    #    scored.unpersist()
    return {**tags,
            "error":"Failed at metric calculation and plotting phase",
            "status":"failed"}
    