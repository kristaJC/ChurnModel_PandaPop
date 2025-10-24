# To Do: adapting to pyspark ml pipelines 
from pyspark.sql.functions import *
import pandas as pd
import numpy as np


import mlflow
from datetime import datetime
from typing import Optional, Dict, Union, List,Tuple

from pyspark.sql import DataFrame, functions as F, types as T
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.tuning import CrossValidatorModel, TrainValidationSplitModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.storagelevel import StorageLevel

from pyspark.ml.linalg import DenseVector, SparseVector, VectorUDT

import matplotlib.pyplot as plt

#TODO: Feature importances function is a little weird
#TODO: erroring out on F.vector_to_array

def _positive_prob_col(scored_df, probability_col: str, positive_index: int = 1):
    """
    Returns a Column with P(positive) as a scalar double, regardless of whether
    `probability_col` is a vector (size 2) or already a scalar.
    Works on Spark 2.4+ (no vector_to_array dependency).
    """
    dt = scored_df.schema[probability_col].dataType

    # If it's already numeric, just cast to double and return
    if isinstance(dt, (T.DoubleType, T.FloatType)):
        return F.col(probability_col).cast("double")

    # If it's a vector, try vector_to_array; else use a tiny udf fallback
    if isinstance(dt, VectorUDT):
        # try vector_to_array if available
        if hasattr(F, "vector_to_array"):
            return F.vector_to_array(F.col(probability_col))[positive_index]
        else:
            # Spark < 3.x fallback
            extract_udf = F.udf(lambda v: float(v[positive_index]) if v is not None else None, T.DoubleType())
            return extract_udf(F.col(probability_col))

    # Last resort: cast to double (may error if unsupported)
    return F.col(probability_col).cast("double")


"""
def _positive_prob_expr(probability_col: str, positive_index: int = 1):
    # scalar = vector_to_array(probability)[positive_index]
    return F.vector_to_array(F.col(probability_col))[positive_index]
"""

def _tp_fp_fn_tn(df: DataFrame, *, label_col: str, prob_col_expr, threshold: float) -> Tuple[int, int, int, int]:
    pred = (prob_col_expr >= F.lit(threshold)).cast("int")
    tp = F.sum((F.col(label_col) == 1).cast("int") * pred)
    fp = F.sum((F.col(label_col) == 0).cast("int") * pred)
    fn = F.sum((F.col(label_col) == 1).cast("int") * (1 - pred))
    tn = F.sum((F.col(label_col) == 0).cast("int") * (1 - pred))
    r = df.agg(tp.alias("tp"), fp.alias("fp"), fn.alias("fn"), tn.alias("tn")).first()
    return int(r.tp), int(r.fp), int(r.fn), int(r.tn)

def _metrics_from_counts(tp: int, fp: int, fn: int, tn: int, beta: float = 1.0) -> Dict[str, float]:
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy  = (tp + tn) / total if total > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fbeta = ((1 + beta**2) * precision * recall / (beta**2 * precision + recall)) if (precision + recall) > 0 else 0.0
    return dict(precision=precision, recall=recall, accuracy=accuracy, f1=f1, fbeta=fbeta)

def _metrics_at_threshold(df: DataFrame, *, label_col: str, prob_col_expr, threshold: float) -> Dict[str, float]:
    tp, fp, fn, tn = _tp_fp_fn_tn(df, label_col=label_col, prob_col_expr=prob_col_expr, threshold=threshold)
    m1 = _metrics_from_counts(tp, fp, fn, tn, beta=1.0)
    m2 = _metrics_from_counts(tp, fp, fn, tn, beta=2.0)
    return dict(threshold=threshold, tp=tp, fp=fp, fn=fn, tn=tn,
                precision=m1["precision"], recall=m1["recall"],
                accuracy=m1["accuracy"], f1=m1["f1"], f2=m2["fbeta"])

# --- Curves (Spark compute → matplotlib plots) ---

def _confusion_fig_from_counts(tp: int, fp: int, fn: int, tn: int, title: str):
    from sklearn.metrics import ConfusionMatrixDisplay
    import numpy as np
    cm = np.array([[tn, fp], [fn, tp]], dtype="int64")
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1]).plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    return fig

def _binary_curves(scored_df: DataFrame, *, label_col: str, prob_pos_col: str):
    # Distributed computation via mllib BinaryClassificationMetrics
    from pyspark.mllib.evaluation import BinaryClassificationMetrics
    rdd = scored_df.select(F.col(prob_pos_col).alias("score"),
                           F.col(label_col).cast("float").alias("label")).rdd.map(lambda r: (float(r.score), float(r.label)))
    bcm = BinaryClassificationMetrics(rdd)
    # PA 2025-10-24: changed method by property. -->
    # roc_pts = [(float(x), float(y)) for x, y in bcm.roc().collect()]   # (FPR, TPR)
    # pr_pts  = [(float(x), float(y)) for x, y in bcm.pr().collect()]    # (Recall, Precision)
    roc_pts = [(float(x), float(y)) for x, y in bcm.roc.collect()]  # (FPR, TPR)
    pr_pts  = [(float(x), float(y)) for x, y in bcm.pr.collect()]  # (Recall, Precision)
    # PA 2025-10-24: changed method by property. <--
    return dict(roc=roc_pts, pr=pr_pts, auc_roc=float(bcm.areaUnderROC), auc_pr=float(bcm.areaUnderPR))

def _plot_and_log_curves(scored_df: DataFrame, 
                         *, 
                         label_col: str, 
                         prob_pos_col: str, 
                         title_suffix: str = "", 
                         prefix: str = ""):
    curves = _binary_curves(scored_df, label_col=label_col, prob_pos_col=prob_pos_col)
    roc_pts, pr_pts = curves["roc"], curves["pr"]
    auc_roc, auc_pr = curves["auc_roc"], curves["auc_pr"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pf = f"{prefix}_" if prefix else ""

    # ROC
    fig, ax = plt.subplots()
    if roc_pts:
        xs, ys = zip(*roc_pts)
        ax.plot(xs, ys, label=f"ROC (AUC={auc_roc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve {title_suffix}"); ax.legend()
    roc_path = f"roc_{pf}{ts}.png"; fig.savefig(roc_path, bbox_inches="tight"); mlflow.log_artifact(f"artifacts/{roc_path}"); plt.close(fig)

    # PR
    fig, ax = plt.subplots()
    if pr_pts:
        xs, ys = zip(*pr_pts)  # recall, precision
        ax.plot(xs, ys, label=f"PR (AP={auc_pr:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve {title_suffix}"); ax.legend()
    pr_path = f"pr_{pf}{ts}.png"; fig.savefig(pr_path, bbox_inches="tight"); mlflow.log_artifact(f"artifacts/{pr_path}"); plt.close(fig)

    # Also log AUCs as metrics (you may already log via evaluator)
    #mlflow.log_metrics({"roc_auc_curve_mllib": auc_roc, "pr_auc_curve_mllib": auc_pr})



#TODO: This seems wonky, lets rewrite this function 
'''
def log_feature_importances_pd(
    best_model: PipelineModel,
    train_df: DataFrame,
    *,
    features_col: str = "features",
    top_n: int = 25
):
    """
    Extracts feature importances from the classifier stage, maps to feature names
    (using metadata from VectorAssembler), builds a pandas DataFrame of top-N, and logs to MLflow.
    """
    # --- 1️⃣ Extract feature names ---
    meta = train_df.select(features_col).schema[features_col].metadata
    names = []
    try:
        attrs = meta["ml_attr"]["attrs"]
        for group in attrs.values():
            for item in group:
                names.append(item["name"])
    except Exception:
        # fallback generic names
        first_row = train_df.select(features_col).limit(1).collect()[0][0]
        names = [f"f{i}" for i in range(first_row.size)]

    # --- 2️⃣ Find classifier stage ---
    clf_stage = None
    for st in reversed(best_model.stages):
        if hasattr(st, "featureImportances") or hasattr(st, "_java_obj"):
            clf_stage = st
            break

    if clf_stage is None:
        print("⚠️ No classifier stage found for feature importances.")
        return

    # --- 3️⃣ Extract importances ---
    importances = None
    if hasattr(clf_stage, "featureImportances"):
        v = clf_stage.featureImportances
        if isinstance(v, DenseVector):
            importances = v.toArray()
        elif isinstance(v, SparseVector):
            importances = v.toArray()
        else:
            importances = list(v)
    else:
        # XGBoost or other
        try:
            imp_map = clf_stage.getFeatureImportances()
            importances = [imp_map.get(f"f{i}", 0.0) for i in range(len(names))]
        except Exception:
            print("⚠️ Could not extract feature importances.")
            return

    # --- 4️⃣ Build pandas DataFrame ---
    df_imp = pd.DataFrame({"feature": names, "importance": importances})
    df_imp = df_imp.sort_values("importance", ascending=False)
    top_df = df_imp.head(top_n)

    # --- 5️⃣ Log to MLflow ---
    mlflow.log_params({
        **{f"top_{i+1}_feature": row.feature for i, row in top_df.iterrows()},
        **{f"top_{i+1}_importance": round(row.importance, 6) for i, row in top_df.iterrows()},
    })

    csv_path = "feature_importances_topN.csv"
    top_df.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path)

    print(f"✅ Logged top-{top_n} feature importances to MLflow.")
    return top_df
'''

def _best_from_tuner(fitted):
    if isinstance(fitted, (CrossValidatorModel, TrainValidationSplitModel)):
        return fitted.bestModel
    return fitted  # already a PipelineModel


def _flatten_params(model_or_stage, prefix: str = "") -> Dict[str, str]:
    out = {}
    if isinstance(model_or_stage, PipelineModel):
        for i, st in enumerate(model_or_stage.stages):
            out.update(_flatten_params(st, prefix=f"{prefix}{i}_{st.__class__.__name__}__"))
        return out
    try:
        pmap = model_or_stage.extractParamMap()
        for p, v in pmap.items():
            out[f"{prefix}{p.name}"] = str(v)
    except Exception:
        pass
    return out

"""
def _positive_prob_col(probability_col: str, positive_index: int = 1):
    # vector_to_array(probability)[pos]
    return vector_to_array(probability_col)[positive_index]
    #return F.vector_to_array(F.col(probability_col))[positive_index]
"""

def _metrics_at_threshold(
    df: DataFrame, label_col: str, prob_col_expr, threshold: float
) -> Dict[str, float]:
    """
    Compute precision/recall/accuracy/F1/F2 on Spark at a given threshold.
    """
    pred = (prob_col_expr >= F.lit(threshold)).cast("int")

    # TP, FP, FN, TN
    tp = F.sum((F.col(label_col) == 1).cast("int") * pred)
    fp = F.sum((F.col(label_col) == 0).cast("int") * pred)
    fn = F.sum((F.col(label_col) == 1).cast("int") * (1 - pred))
    tn = F.sum((F.col(label_col) == 0).cast("int") * (1 - pred))

    agg = df.agg(tp.alias("tp"), fp.alias("fp"), fn.alias("fn"), tn.alias("tn")).first()
    tp, fp, fn, tn = [int(agg[x]) for x in ("tp", "fp", "fn", "tn")]
    total = tp + fp + fn + tn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy  = (tp + tn) / total if total > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    beta = 2.0
    f2 = ((1 + beta**2) * precision * recall / (beta**2 * precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "f2": float(f2),
        "tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn),
    }


# --- Imports ---
import mlflow
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, Dict, Union, List, Tuple

from pyspark.sql import DataFrame, functions as F
from pyspark.storagelevel import StorageLevel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.tuning import CrossValidatorModel, TrainValidationSplitModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import round  

# (pandas allowed ONLY for feature importances)
import pandas as pd
from pyspark.ml.linalg import DenseVector, SparseVector

# --- Core helpers ---

def _best_from_tuner(fitted):
    if isinstance(fitted, (CrossValidatorModel, TrainValidationSplitModel)):
        return fitted.bestModel
    return fitted

def _flatten_params(model_or_stage, prefix: str = "") -> Dict[str, str]:
    from pyspark.ml import PipelineModel
    out = {}
    if isinstance(model_or_stage, PipelineModel):
        for i, st in enumerate(model_or_stage.stages):
            out.update(_flatten_params(st, prefix=f"{prefix}{i}_{st.__class__.__name__}__"))
        return out
    try:
        pmap = model_or_stage.extractParamMap()
        for p, v in pmap.items():
            out[f"{prefix}{p.name}"] = str(v)
    except Exception:
        pass
    return out

def _positive_prob_expr(probability_col: str, positive_index: int = 1):
    scalar = vector_to_array(probability_col)[positive_index]
    return scalar #F.vector_to_array(F.col(probability_col))[positive_index]

def _tp_fp_fn_tn(df: DataFrame, *, label_col: str, prob_col_expr, threshold: float) -> Tuple[int, int, int, int]:
    pred = (prob_col_expr >= F.lit(threshold)).cast("int")
    tp = F.sum((F.col(label_col) == 1).cast("int") * pred)
    fp = F.sum((F.col(label_col) == 0).cast("int") * pred)
    fn = F.sum((F.col(label_col) == 1).cast("int") * (1 - pred))
    tn = F.sum((F.col(label_col) == 0).cast("int") * (1 - pred))
    r = df.agg(tp.alias("tp"), fp.alias("fp"), fn.alias("fn"), tn.alias("tn")).first()
    return int(r.tp), int(r.fp), int(r.fn), int(r.tn)

def _metrics_from_counts(tp: int, fp: int, fn: int, tn: int, beta: float = 1.0) -> Dict[str, float]:
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy  = (tp + tn) / total if total > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fbeta = ((1 + beta**2) * precision * recall / (beta**2 * precision + recall)) if (precision + recall) > 0 else 0.0
    return dict(precision=precision, recall=recall, accuracy=accuracy, f1=f1, fbeta=fbeta)

def _metrics_at_threshold(df: DataFrame, *, label_col: str, prob_col_expr, threshold: float) -> Dict[str, float]:
    tp, fp, fn, tn = _tp_fp_fn_tn(df, label_col=label_col, prob_col_expr=prob_col_expr, threshold=threshold)
    m1 = _metrics_from_counts(tp, fp, fn, tn, beta=1.0)
    m2 = _metrics_from_counts(tp, fp, fn, tn, beta=2.0)
    return dict(threshold=threshold, tp=tp, fp=fp, fn=fn, tn=tn,
                precision=m1["precision"], recall=m1["recall"],
                accuracy=m1["accuracy"], f1=m1["f1"], f2=m2["fbeta"])

# --- Curves (Spark compute → matplotlib plots) ---

def _confusion_fig_from_counts(tp: int, fp: int, fn: int, tn: int, title: str):
    from sklearn.metrics import ConfusionMatrixDisplay
    import numpy as np
    cm = np.array([[tn, fp], [fn, tp]], dtype="int64")
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1]).plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    return fig

# PA 2025-10-24: changed. -->
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.ml.linalg import VectorUDT

def _binary_curves(scored_df: DataFrame, *, label_col: str, prob_pos_col: str):
    # --- Normalize: get scalar positive-class score as double ---
    field = next(f for f in scored_df.schema.fields if f.name == prob_pos_col)
    if isinstance(field.dataType, VectorUDT):
        score_expr = F.col(prob_pos_col)[1]  # assumes index 1 = positive class
    else:
        score_expr = F.col(prob_pos_col)
    score_col = score_expr.cast("double")

    df = (
        scored_df
        .select(
            score_col.alias("score"),
            F.col(label_col).cast("double").alias("label")
        )
        .where(F.col("score").isNotNull() & F.col("label").isNotNull())
    )

    # If nothing left after cleaning, return NaNs/empties
    if df.rdd.isEmpty():
        return {"roc": [], "pr": [], "auc_roc": float("nan"), "auc_pr": float("nan")}

    # --- Need both classes present ---
    pn_row = df.agg(
        F.sum(F.when(F.col("label") == 1, 1).otherwise(0)).alias("P"),
        F.sum(F.when(F.col("label") == 0, 1).otherwise(0)).alias("N"),
    ).first()
    P, N = int(pn_row.P or 0), int(pn_row.N or 0)

    if P == 0 or N == 0:
        return {"roc": [], "pr": [], "auc_roc": float("nan"), "auc_pr": float("nan")}

    # --- Sort by score desc, build cumulative TP/FP ---
    w = Window.orderBy(F.col("score").desc())
    ranked = (
        df
        .withColumn("is_pos", (F.col("label") == 1).cast("int"))
        .withColumn("is_neg", (F.col("label") == 0).cast("int"))
        .withColumn("cum_tp", F.sum(F.col("is_pos")).over(w))
        .withColumn("cum_fp", F.sum(F.col("is_neg")).over(w))
    )

    # --- One row per distinct score → max cumulative at that score ---
    by_t = (
        ranked.groupBy("score")
        .agg(
            F.max(F.col("cum_tp")).alias("TP"),
            F.max(F.col("cum_fp")).alias("FP"),
        )
        .orderBy(F.col("score").desc())
        .withColumn("TPR", F.col("TP") / F.lit(P))   # recall
        .withColumn("FPR", F.col("FP") / F.lit(N))
        .withColumn(
            "PREC",
            F.when((F.col("TP") + F.col("FP")) > 0,
                   F.col("TP") / (F.col("TP") + F.col("FP")))
             .otherwise(F.lit(1.0))
        )
        .select("FPR", "TPR", "PREC")
    )

    # --- Collect curves (ROC: x=FPR, y=TPR; PR: x=Recall, y=Precision) ---
    by_t_rows = by_t.collect()  # single collect for both curves

    roc_pts = [(0.0, 0.0)] + [(float(r.FPR), float(r.TPR)) for r in by_t_rows] + [(1.0, 1.0)]
    pr_pts  = [(float(r.TPR), float(r.PREC)) for r in by_t_rows]
    if not pr_pts or pr_pts[0][0] > 0.0:
        pr_pts = [(0.0, 1.0)] + pr_pts
    if pr_pts[-1][0] < 1.0:
        pr_pts.append((1.0, pr_pts[-1][1]))

    # --- Trapezoidal AUCs (pure Python; safe to use generator here) ---
    def area(points):
        if len(points) < 2:
            return float("nan")
        pts = sorted(points, key=lambda t: t[0])
        return float(sum((x2 - x1) * (y1 + y2) / 2.0
                         for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:])))

    return {"roc": roc_pts, "pr": pr_pts, "auc_roc": area(roc_pts), "auc_pr": area(pr_pts)}

# def _binary_curves(scored_df: DataFrame, *, label_col: str, prob_pos_col: str):
#     # Distributed computation via mllib BinaryClassificationMetrics
#     from pyspark.mllib.evaluation import BinaryClassificationMetrics
#     rdd = scored_df.select(F.col(prob_pos_col).alias("score"),
#                            F.col(label_col).cast("float").alias("label")).rdd.map(lambda r: (float(r.score), float(r.label)))
#     bcm = BinaryClassificationMetrics(rdd)
#     roc_pts = [(float(x), float(y)) for x, y in bcm.roc().collect()]   # (FPR, TPR)
#     pr_pts  = [(float(x), float(y)) for x, y in bcm.pr().collect()]    # (Recall, Precision)
    # return dict(roc=roc_pts, pr=pr_pts, auc_roc=float(bcm.areaUnderROC), auc_pr=float(bcm.areaUnderPR))
# PA 2025-10-24: changed. <--

def _plot_and_log_curves(scored_df: DataFrame, *, label_col: str, prob_pos_col: str, title_suffix: str = "", prefix: str = ""):
    curves = _binary_curves(scored_df, label_col=label_col, prob_pos_col=prob_pos_col)
    roc_pts, pr_pts = curves["roc"], curves["pr"]
    auc_roc, auc_pr = curves["auc_roc"], curves["auc_pr"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pf = f"{prefix}_" if prefix else ""

    # ROC
    fig, ax = plt.subplots()
    if roc_pts:
        xs, ys = zip(*roc_pts)
        ax.plot(xs, ys, label=f"ROC (AUC={auc_roc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve {title_suffix}"); ax.legend()
    roc_path = f"roc_{pf}{ts}.png"; fig.savefig(roc_path, bbox_inches="tight"); mlflow.log_artifact(f"artifacts/{roc_path}"); plt.close(fig)

    # PR
    fig, ax = plt.subplots()
    if pr_pts:
        xs, ys = zip(*pr_pts)  # recall, precision
        ax.plot(xs, ys, label=f"PR (AP={auc_pr:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve {title_suffix}"); ax.legend()
    pr_path = f"pr_{pf}{ts}.png"; fig.savefig(pr_path, bbox_inches="tight"); mlflow.log_artifactf(f"artifacts/{pr_path}"); plt.close(fig)

    # Also log AUCs as metrics (you may already log via evaluator)
    mlflow.log_metrics({"roc_auc_curve_mllib": auc_roc, "pr_auc_curve_mllib": auc_pr})

# --- Feature Importances (pandas allowed here only) ---

def _extract_feature_names_from_metadata(df_with_features: DataFrame, features_col: str) -> List[str]:
    names: List[str] = []
    try:
        meta = df_with_features.schema[features_col].metadata
        attrs = meta["ml_attr"]["attrs"]
        idx_to_name = {}
        for group in attrs.values():  # numeric/binary/nominal
            for item in group:        # {"idx": int, "name": str}
                idx_to_name[int(item["idx"])] = item.get("name", f"f{int(item['idx'])}")
        # Pack into list by index
        max_idx = max(idx_to_name) if idx_to_name else -1
        names = [idx_to_name.get(i, f"f{i}") for i in range(max_idx + 1)]
    except Exception:
        # fallback generic names by probing size
        row = df_with_features.select(features_col).limit(1).collect()
        if row:
            size = row[0][0].size
            names = [f"f{i}" for i in range(size)]
    return names

def _find_classifier_stage(model: PipelineModel):
    for st in reversed(model.stages):
        if hasattr(st, "featureImportances") or hasattr(st, "_java_obj"):
            return st
    return model.stages[-1]

def _extract_importances_from_stage(stage, size_hint: int) -> Optional[List[float]]:
    # 1) Spark tree models
    if hasattr(stage, "featureImportances"):
        v = stage.featureImportances
        return v.toArray().tolist() if isinstance(v, (DenseVector, SparseVector)) else list(v)
    # 2) XGBoost for Spark – try a few APIs
    for attr in ("getFeatureImportances", "featureImportances", "get_feature_importances"):
        if hasattr(stage, attr):
            try:
                val = getattr(stage, attr)()
                if isinstance(val, dict):  # {"f0":score, ...}
                    out = [0.0] * size_hint
                    for k, s in val.items():
                        if isinstance(k, str) and k.startswith("f") and k[1:].isdigit():
                            i = int(k[1:])
                            if 0 <= i < size_hint:
                                out[i] = float(s)
                    return out
                if isinstance(val, (list, tuple)):
                    return [float(x) for x in val]
            except Exception:
                pass
    # 3) Java fallback
    try:
        j = getattr(stage, "_java_obj", None)
        if j and hasattr(j, "getFeatureScore"):
            m = dict(j.getFeatureScore())  # {"f0":score,...}
            out = [0.0] * size_hint
            for k, s in m.items():
                if isinstance(k, str) and k.startswith("f") and k[1:].isdigit():
                    i = int(k[1:])
                    if 0 <= i < size_hint:
                        out[i] = float(s)
            return out
    except Exception:
        pass
    return None

def log_feature_importances_pd(
    best_model: PipelineModel,
    features_meta_df: DataFrame,   # a tiny DF (e.g., best_model.transform(train_df.limit(1))).select(features)
    *, features_col: str = "features", top_n: int = 25
):
    names = _extract_feature_names_from_metadata(features_meta_df, features_col)
    clf = _find_classifier_stage(best_model)
    importances = _extract_importances_from_stage(clf, size_hint=len(names))
    if not importances:
        mlflow.set_tag("feature_importances_available", "false")
        return None

    df_imp = pd.DataFrame({"feature": names, "importance": importances}).sort_values("importance", ascending=False)
    top_df = df_imp.head(top_n)

    # params and artifact
    mlflow.log_params({f"top_feature_{i+1}_name": r.feature for i, r in top_df.iterrows()})
    mlflow.log_params({f"top_feature_{i+1}_importance": float(r.importance) for i, r in top_df.iterrows()})
    path = "feature_importances_topN.csv"; top_df.to_csv(path, index=False); mlflow.log_artifact(f"artifacts/{path}")

    mlflow.set_tag("feature_importances_available", "true")
    mlflow.set_tag("feature_importances_count", str(len(importances)))
    return top_df

# --- Main runner ---

def run_spark_cv_with_logging_spark_only(
    estimator: Union[Pipeline, PipelineModel, CrossValidatorModel, TrainValidationSplitModel],
    train_df: DataFrame,
    test_df: DataFrame,
    val_df: Optional[DataFrame] = None,     # prefer tuning on validation
    *,
    label_col: str = "label",
    features_col: str = "features",
    probability_col: str = "probability",
    prediction_col: str = "prediction",
    positive_index: int = 1,
    thresholds: Optional[List[float]] = None,    # thresholds to sweep
    run_name: str = "spark-ml-search",
    extra_tags: Optional[Dict[str, str]] = None,
    additional_metrics: Optional[Dict[str, float]] = None,
    persist_eval: bool = True,
    # NEW: chart & importances controls
    log_confusion: bool = True,
    log_curves: bool = True,
    log_feature_importances: bool = True,
    top_n_features: int = 25,
):
    """
    Spark-only evaluation & threshold sweep; plots via matplotlib/sklearn; logs to MLflow.
    Uses pandas ONLY for feature importances (small).
    """
    # avoid sklearn autolog collisions if present
    try:
        import mlflow.sklearn
        mlflow.sklearn.autolog(disable=True)
    except Exception:
        pass

    tags = {}
    if thresholds is None:
        thresholds = [i / 100 for i in range(1, 100)]  # 0.01..0.99

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # ---- fit ----
        fitted = estimator.fit(train_df)
        best_model = _best_from_tuner(fitted)

        # ---- log model ----
        mlflow.spark.log_model(spark_model=best_model, artifact_path="spark_model")

        # ---- feature importances (optional; tiny pandas allowed) ----
        '''
        if log_feature_importances:
            meta_df = best_model.transform(train_df.limit(1)).select(features_col)
            log_feature_importances_pd(best_model, meta_df, features_col=features_col, top_n=top_n_features)
        '''

        # ---- choose eval frame ----
        eval_df = val_df if val_df is not None else test_df
        eval_name = "validation" if val_df is not None else "test"

        # ---- score & cache ----
        scored = best_model.transform(eval_df).select(
            F.col(label_col).cast("int").alias(label_col),
            F.col(prediction_col).alias(prediction_col),
            F.col(probability_col).alias(probability_col),
        )
        if persist_eval:
            scored = scored.persist(StorageLevel.MEMORY_AND_DISK)

        p_pos = _positive_prob_col(scored, probability_col=probability_col, positive_index=positive_index)

        # ---- AUCs from Spark evaluators ----
        evaluator_roc = BinaryClassificationEvaluator(
            labelCol=label_col, rawPredictionCol=probability_col, metricName="areaUnderROC"
        )
        evaluator_pr  = BinaryClassificationEvaluator(
            labelCol=label_col, rawPredictionCol=probability_col, metricName="areaUnderPR"
        )
        roc_auc = evaluator_roc.evaluate(scored)
        pr_auc  = evaluator_pr.evaluate(scored)

        # ---- threshold sweep (Spark counts) ----
        best_f1, best_t_f1 = -1.0, 0.5
        best_f2, best_t_f2 = -1.0, 0.5
        
        for t in thresholds:
            m = _metrics_at_threshold(scored, label_col=label_col, prob_col_expr=p_pos, threshold=t)
            # per-threshold metrics (optional granularity)
            mlflow.log_metrics({
                f"precision@{t:.2f}": m["precision"],
                f"recall@{t:.2f}": m["recall"],
                f"f1@{t:.2f}": m["f1"],
                f"f2@{t:.2f}": m["f2"],
                f"accuracy@{t:.2f}": m["accuracy"],
            })
            if m["f1"] > best_f1:
                best_f1, best_t_f1 = m["f1"], t
            if m["f2"] > best_f2:
                best_f2, best_t_f2 = m["f2"], t

        # ---- aggregate metrics at tuned thresholds ----
        at_f1 = _metrics_at_threshold(scored, label_col=label_col, prob_col_expr=p_pos, threshold=best_t_f1)
        at_f2 = _metrics_at_threshold(scored, label_col=label_col, prob_col_expr=p_pos, threshold=best_t_f2)

        metrics = {
            "area_under_roc": float(roc_auc),
            "area_under_pr": float(pr_auc),
            "optimal_threshold_f1": float(best_t_f1),
            "optimal_threshold_f2": float(best_t_f2),
            "precision_at_f1": float(at_f1["precision"]),
            "recall_at_f1": float(at_f1["recall"]),
            "f1_at_f1": float(at_f1["f1"]),
            "accuracy_at_f1": float(at_f1["accuracy"]),
            "precision_at_f2": float(at_f2["precision"]),
            "recall_at_f2": float(at_f2["recall"]),
            "f1_at_f2": float(at_f2["f1"]),
            "accuracy_at_f2": float(at_f2["accuracy"]),
        }

        # If you pass additional_metrics, only keep numerics
        if additional_metrics:
            for k, v in additional_metrics.items():
                if isinstance(v, (int, float)):
                    metrics[k] = float(v)

        mlflow.log_metrics(metrics)

        # Put eval_set in tags (not metrics)
        
        # ---- Params / Tags ----
        flat_params = _flatten_params(best_model)
        if flat_params:
            mlflow.log_params(flat_params)
        tags = {
            "pipeline_type": best_model.__class__.__name__,
            "tuner": (
                "CrossValidator" if isinstance(fitted, CrossValidatorModel)
                else "TrainValidationSplit" if isinstance(fitted, TrainValidationSplitModel)
                else "none"
            ),
            "threshold_source": eval_name,
            "positive_index": str(positive_index),
        }
        if extra_tags:
            tags.update(extra_tags)
        eval_name = "validation" if val_df is not None else "test"
        tags.update({"eval_set": eval_name})
        
        mlflow.set_tags(tags)

        # ---- Charts (confusion matrices + ROC/PR) ----
        # Build a scored DF with scalar prob for plotting helpers
        scored_for_plots = scored.select(
            F.col(label_col).alias(label_col),
            p_pos.alias("p_pos")
        )

        if log_confusion:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            # CM @ best F1
            tp, fp, fn, tn = _tp_fp_fn_tn(scored_for_plots, label_col=label_col, prob_col_expr=F.col("p_pos"), threshold=best_t_f1)
            fig = _confusion_fig_from_counts(tp, fp, fn, tn, f"Confusion Matrix @ F1={best_t_f1:.3f} ({eval_name})")
            cm_path = f"cm_{eval_name}_f1_{ts}.png"; fig.savefig(cm_path, bbox_inches="tight"); mlflow.log_artifact(f"artifacts/{cm_path}"); plt.close(fig)
            # CM @ best F2
            tp, fp, fn, tn = _tp_fp_fn_tn(scored_for_plots, label_col=label_col, prob_col_expr=F.col("p_pos"), threshold=best_t_f2)
            fig = _confusion_fig_from_counts(tp, fp, fn, tn, f"Confusion Matrix @ F2={best_t_f2:.3f} ({eval_name})")
            cm_path = f"cm_{eval_name}_f2_{ts}.png"; fig.savefig(cm_path, bbox_inches="tight"); mlflow.log_artifact(f"artifacts/{cm_path}"); plt.close(fig)

        if log_curves:
            _plot_and_log_curves(
                scored_for_plots, label_col=label_col, prob_pos_col="p_pos",
                title_suffix=f"({eval_name})", prefix=eval_name
            )

        # ---- Optional: evaluate held-out test with tuned thresholds when using val_df ----
        if val_df is not None:
            scored_test = best_model.transform(test_df).select(
                F.col(label_col).cast("int").alias(label_col),
                F.col(probability_col).alias(probability_col)
            )
            # Guido : here i modify the next line because "_positive_prob_col" was using scored as a parameter
            # but i understand that it should use "scored_test"
            p_pos_test = _positive_prob_col(scored_test,
                                            probability_col=probability_col,   positive_index=positive_index)
            scored_test_plots = scored_test.select(F.col(label_col).alias(label_col), p_pos_test.alias("p_pos"))

            t1 = _metrics_at_threshold(scored_test_plots, label_col=label_col, prob_col_expr=F.col("p_pos"), threshold=best_t_f1)
            t2 = _metrics_at_threshold(scored_test_plots, label_col=label_col, prob_col_expr=F.col("p_pos"), threshold=best_t_f2)
            mlflow.log_metrics({
                "test_precision_at_f1": t1["precision"], "test_recall_at_f1": t1["recall"],
                "test_f1_at_f1": t1["f1"], "test_accuracy_at_f1": t1["accuracy"],
                "test_precision_at_f2": t2["precision"], "test_recall_at_f2": t2["recall"],
                "test_f1_at_f2": t2["f1"], "test_accuracy_at_f2": t2["accuracy"],
            })

            if log_confusion:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                tp, fp, fn, tn = _tp_fp_fn_tn(scored_test_plots, label_col=label_col, prob_col_expr=F.col("p_pos"), threshold=best_t_f1)
                fig = _confusion_fig_from_counts(tp, fp, fn, tn, f"Confusion Matrix @ F1={best_t_f1:.3f} (heldout test)")
                cm_path = f"cm_test_f1_{ts}.png"; fig.savefig(cm_path, bbox_inches="tight"); mlflow.log_artifact(f"artifacts/{cm_path}"); plt.close(fig)

                tp, fp, fn, tn = _tp_fp_fn_tn(scored_test_plots, label_col=label_col, prob_col_expr=F.col("p_pos"), threshold=best_t_f2)
                fig = _confusion_fig_from_counts(tp, fp, fn, tn, f"Confusion Matrix @ F2={best_t_f2:.3f} (heldout test)")
                cm_path = f"cm_test_f2_{ts}.png"; fig.savefig(cm_path, bbox_inches="tight"); mlflow.log_artifact(f"artifacts/{cm_path}"); plt.close(fig)

            if log_curves:
                _plot_and_log_curves(scored_test_plots, label_col=label_col, prob_pos_col="p_pos",
                                     title_suffix="(heldout test)", prefix="heldout_test")

        if persist_eval:
            scored.unpersist()

        print(f"✔️ Run complete: {run_id}")
        return {
            "run_id": run_id,
            "eval_set": eval_name,
            "optimal_threshold_f1": float(best_t_f1),
            "optimal_threshold_f2": float(best_t_f2),
            "metrics": metrics,
            "params": flat_params if flat_params else {},
            "tags": tags,
        }


'''
def run_spark_ml_search(
    estimator: Union[Pipeline, PipelineModel, CrossValidatorModel, TrainValidationSplitModel],
    train_df: DataFrame,
    test_df: DataFrame,
    val_df: Optional[DataFrame] = None,     # prefer tuning on validation
    *,
    label_col: str = "label",
    features_col: str = "features",
    probability_col: str = "probability",
    prediction_col: str = "prediction",
    positive_index: int = 1,
    thresholds: Optional[List[float]] = None,    # thresholds to sweep
    run_name: str = "spark-ml-search",
    extra_tags: Optional[Dict[str, str]] = None,
    additional_metrics: Optional[Dict[str, float]] = None,
    persist_eval: bool = True,
    # NEW: chart & importances controls
    log_confusion: bool = True,
    log_curves: bool = True,
    log_feature_importances: bool = True,
    top_n_features: int = 25,
):
    """
    Spark-only evaluation & threshold sweep; plots via matplotlib/sklearn; logs to MLflow.
    Uses pandas ONLY for feature importances (small).
    """
    # avoid sklearn autolog collisions if present
    try:
        import mlflow.sklearn
        mlflow.sklearn.autolog(disable=True)
    except Exception:
        pass

    if thresholds is None:
        thresholds = [i / 100 for i in range(1, 100)]  # 0.01..0.99

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # ---- fit ----
        fitted = estimator.fit(train_df)
        best_model = _best_from_tuner(fitted)

        # ---- log model ----
        mlflow.spark.log_model(spark_model=best_model, artifact_path="spark_model")

        # ---- feature importances (optional; tiny pandas allowed) ----
        #TODO: 
        
        #if log_feature_importances:
        #    meta_df = best_model.transform(train_df.limit(1)).select(features_col)
        #    log_feature_importances_pd(best_model, meta_df, features_col=features_col, top_n=top_n_features)
        

        # ---- choose eval frame ----
        eval_df = val_df if val_df is not None else test_df
        eval_name = "validation" if val_df is not None else "test"

        # ---- score & cache ----
        scored = best_model.transform(eval_df).select(
            F.col(label_col).cast("int").alias(label_col),
            F.col(prediction_col).alias(prediction_col),
            F.col(probability_col).alias(probability_col),
        )
        if persist_eval:
            scored = scored.persist(StorageLevel.MEMORY_AND_DISK)

        p_pos = _positive_prob_col(scored, probability_col=probability_col, positive_index=positive_index)

        # ---- AUCs from Spark evaluators ----
        evaluator_roc = BinaryClassificationEvaluator(
            labelCol=label_col, rawPredictionCol=probability_col, metricName="areaUnderROC"
        )
        evaluator_pr  = BinaryClassificationEvaluator(
            labelCol=label_col, rawPredictionCol=probability_col, metricName="areaUnderPR"
        )
        roc_auc = evaluator_roc.evaluate(scored)
        pr_auc  = evaluator_pr.evaluate(scored)

        # ---- threshold sweep (Spark counts) ----
        best_f1, best_t_f1 = -1.0, 0.5
        best_f2, best_t_f2 = -1.0, 0.5
        
        for t in thresholds:
            m = _metrics_at_threshold(scored, label_col=label_col, prob_col_expr=p_pos, threshold=t)
            # per-threshold metrics (optional granularity)
            mlflow.log_metrics({
                f"precision@{t:.2f}": m["precision"],
                f"recall@{t:.2f}": m["recall"],
                f"f1@{t:.2f}": m["f1"],
                f"f2@{t:.2f}": m["f2"],
                f"accuracy@{t:.2f}": m["accuracy"],
            })
            if m["f1"] > best_f1:
                best_f1, best_t_f1 = m["f1"], t
            if m["f2"] > best_f2:
                best_f2, best_t_f2 = m["f2"], t

        # ---- aggregate metrics at tuned thresholds ----
        at_f1 = _metrics_at_threshold(scored, label_col=label_col, prob_col_expr=p_pos, threshold=best_t_f1)
        at_f2 = _metrics_at_threshold(scored, label_col=label_col, prob_col_expr=p_pos, threshold=best_t_f2)

        metrics = {
            "area_under_roc": float(roc_auc),
            "area_under_pr": float(pr_auc),
            "optimal_threshold_f1": float(best_t_f1),
            "optimal_threshold_f2": float(best_t_f2),
            "precision_at_f1": float(at_f1["precision"]),
            "recall_at_f1": float(at_f1["recall"]),
            "f1_at_f1": float(at_f1["f1"]),
            "accuracy_at_f1": float(at_f1["accuracy"]),
            "precision_at_f2": float(at_f2["precision"]),
            "recall_at_f2": float(at_f2["recall"]),
            "f1_at_f2": float(at_f2["f1"]),
            "accuracy_at_f2": float(at_f2["accuracy"]),
        }

        # If you pass additional_metrics, only keep numerics
        if additional_metrics:
            for k, v in additional_metrics.items():
                if isinstance(v, (int, float)):
                    metrics[k] = float(v)

        mlflow.log_metrics(metrics)

        # Put eval_set in tags (not metrics)
        
        mlflow.set_tags(tags)
        
        # ---- Params / Tags ----
        flat_params = _flatten_params(best_model)
        if flat_params:
            mlflow.log_params(flat_params)
        tags = {
            "pipeline_type": best_model.__class__.__name__,
            "tuner": (
                "CrossValidator" if isinstance(fitted, CrossValidatorModel)
                else "TrainValidationSplit" if isinstance(fitted, TrainValidationSplitModel)
                else "none"
            ),
            "threshold_source": eval_name,
            "positive_index": str(positive_index),
        }
        if extra_tags:
            tags.update(extra_tags)
        eval_name = "validation" if val_df is not None else "test"
        tags.update({"eval_set": eval_name})
        
        mlflow.set_tags(tags)

        # ---- Charts (confusion matrices + ROC/PR) ----
        # Build a scored DF with scalar prob for plotting helpers
        scored_for_plots = scored.select(
            F.col(label_col).alias(label_col),
            p_pos.alias("p_pos")
        )

        if log_confusion:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            # CM @ best F1
            tp, fp, fn, tn = _tp_fp_fn_tn(scored_for_plots, label_col=label_col, prob_col_expr=F.col("p_pos"), threshold=best_t_f1)
            fig = _confusion_fig_from_counts(tp, fp, fn, tn, f"Confusion Matrix @ F1={best_t_f1:.3f} ({eval_name})")
            cm_path = f"cm_{eval_name}_f1_{ts}.png"; fig.savefig(cm_path, bbox_inches="tight"); mlflow.log_artifact(cm_path); plt.close(fig)
            # CM @ best F2
            tp, fp, fn, tn = _tp_fp_fn_tn(scored_for_plots, label_col=label_col, prob_col_expr=F.col("p_pos"), threshold=best_t_f2)
            fig = _confusion_fig_from_counts(tp, fp, fn, tn, f"Confusion Matrix @ F2={best_t_f2:.3f} ({eval_name})")
            cm_path = f"cm_{eval_name}_f2_{ts}.png"; fig.savefig(cm_path, bbox_inches="tight"); mlflow.log_artifact(cm_path); plt.close(fig)

        if log_curves:
            _plot_and_log_curves(
                scored_for_plots, label_col=label_col, prob_pos_col="p_pos",
                title_suffix=f"({eval_name})", prefix=eval_name
            )

        # ---- Optional: evaluate held-out test with tuned thresholds when using val_df ----
        if val_df is not None:
            scored_test = best_model.transform(test_df).select(
                F.col(label_col).cast("int").alias(label_col),
                F.col(probability_col).alias(probability_col)
            )
            p_pos_test = _positive_prob_col(scored,
                                            probability_col=probability_col,   positive_index=positive_index)
            scored_test_plots = scored_test.select(F.col(label_col).alias(label_col), p_pos_test.alias("p_pos"))

            t1 = _metrics_at_threshold(scored_test_plots, label_col=label_col, prob_col_expr=F.col("p_pos"), threshold=best_t_f1)
            t2 = _metrics_at_threshold(scored_test_plots, label_col=label_col, prob_col_expr=F.col("p_pos"), threshold=best_t_f2)
            mlflow.log_metrics({
                "test_precision_at_f1": t1["precision"], "test_recall_at_f1": t1["recall"],
                "test_f1_at_f1": t1["f1"], "test_accuracy_at_f1": t1["accuracy"],
                "test_precision_at_f2": t2["precision"], "test_recall_at_f2": t2["recall"],
                "test_f1_at_f2": t2["f1"], "test_accuracy_at_f2": t2["accuracy"],
            })

            if log_confusion:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                tp, fp, fn, tn = _tp_fp_fn_tn(scored_test_plots, label_col=label_col, prob_col_expr=F.col("p_pos"), threshold=best_t_f1)
                fig = _confusion_fig_from_counts(tp, fp, fn, tn, f"Confusion Matrix @ F1={best_t_f1:.3f} (heldout test)")
                cm_path = f"cm_test_f1_{ts}.png"; fig.savefig(cm_path, bbox_inches="tight"); mlflow.log_artifact(cm_path); plt.close(fig)

                tp, fp, fn, tn = _tp_fp_fn_tn(scored_test_plots, label_col=label_col, prob_col_expr=F.col("p_pos"), threshold=best_t_f2)
                fig = _confusion_fig_from_counts(tp, fp, fn, tn, f"Confusion Matrix @ F2={best_t_f2:.3f} (heldout test)")
                cm_path = f"cm_test_f2_{ts}.png"; fig.savefig(cm_path, bbox_inches="tight"); mlflow.log_artifact(cm_path); plt.close(fig)

            if log_curves:
                _plot_and_log_curves(scored_test_plots, label_col=label_col, prob_pos_col="p_pos",
                                     title_suffix="(heldout test)", prefix="heldout_test")

        if persist_eval:
            scored.unpersist()

        print(f"✔️ Run complete: {run_id}")
        return {
            "run_id": run_id,
            "eval_set": eval_name,
            "optimal_threshold_f1": float(best_t_f1),
            "optimal_threshold_f2": float(best_t_f2),
            "metrics": metrics,
            "params": flat_params if flat_params else {},
            "tags": tags,
        }

'''