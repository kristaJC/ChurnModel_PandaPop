from pyspark.sql.functions import *
#from typing import Optional, Tuple
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window

import builtins
from typing import Optional, Dict, Union, List, Tuple, Any

from pyspark.ml.linalg import Vectors, VectorUDT, DenseVector
from pyspark.ml.feature import BucketedRandomProjectionLSH
import math
import random

#TODO: Test
def get_stratified_sets(df, 
                        upsample=True,
                        undersample=True, 
                        split=None,
                        P_TEST = 0.2, 
                        P_VAL=0.2,):

    # Start with simple stratified sampling... 
    strat_train, strat_val, strat_test = stratified_sampling(df, P_TEST, P_VAL)


    """     
    train_info_base = {
                            'sampling':'stratified', 
                            'split':split,
                            'P_TEST':P_TEST,
                            'P_VAL':P_VAL,
                            'P_TRAIN':1.0 - P_TEST-P_VAL,
                            'strategy':None}
    """     
                        
    all_sets = []
    all_sets.append({'dataset': strat_train,
                    'dataset_info':{
                            'sampling':'stratified', 
                            'split':split,
                            'P_TEST':P_TEST,
                            'P_VAL':P_VAL,
                            'P_TRAIN':1.0 - P_TEST-P_VAL,
                            'strategy':None,
                            'type':'training'
                        },
                'relevant_test_set': strat_test,
                'relevant_val_set': strat_val})


    if upsample:
        strat_train_up, train_info_up = upsample_minority(strat_train, split= split)
        #return_dict['dataset'] = strat_train
        return_dict = {**train_info_up, 'sampling':'upsample', 'split':split,'type':'training'}
        all_sets.append({'dataset': strat_train_up,
                'dataset_info': return_dict,
                'relevant_test_set': strat_test,
                'relevant_val_set': strat_val
        })
    if undersample:
        strat_train_under, train_info_under = undersample_majority(strat_train, split= split)
        return_dict = {**train_info_under, 'sampling':'undersample', 'split':split,'type':'training'}
        all_sets.append({'dataset': strat_train_under,
                    'dataset_info': return_dict,
                    'relevant_test_set': strat_test,
                    'relevant_val_set': strat_val
        })
    

    ## Returns a list of dictionaries with [{for stratified set, split dependent},{opt. upsampled},{opt.downsampled}]
    return all_sets






def stratified_sampling(df, 
                        P_TEST=0.2, 
                        P_VAL=0.2):


    # 1) Work at the ID level to avoid row overlaps/leakage
    ids = df.select('judi', 'label').distinct()

    # 2) Assign a stratified rank within each label using a stable random
    ids = ids.withColumn('u', F.rand(42))
    w = Window.partitionBy('label').orderBy(F.col('u'))
    ids = ids.withColumn('pr', F.percent_rank().over(w))

    # 3) Map percent-rank to splits (per class)
    ids = ids.withColumn(
        'split',
        F.when(F.col('pr') < P_TEST, F.lit('test'))
        .when(F.col('pr') < P_TEST + P_VAL, F.lit('val'))
        .otherwise(F.lit('train'))
    ).select('judi', 'split')

    # 4) Materialize the three sets (still imbalanced, as in reality)
    strat_train = df.join(ids.filter("split = 'train'"), on='judi', how='inner')
    strat_val  = df.join(ids.filter("split = 'val'"),   on='judi', how='inner')
    strat_test  = df.join(ids.filter("split = 'test'"),  on='judi', how='inner')

    return strat_train, strat_val, strat_test



def undersample_majority(
    df: DataFrame,
    label_col: str = "label",
    majority_label: Optional[Any] = None,
    ratio: float = 1.0,
    strategy: str = "sample",      # "sample" (fast, approx) or "limit" (exact)
    seed: int = 42,
    split = None,
    sampling = 'undersample',
) -> Tuple[DataFrame, Dict[str, Any]]:
    """
    Undersample the majority class in a binary-labeled Spark DataFrame.
    """
    if ratio <= 0:
        raise ValueError("ratio must be > 0 (majority:minority).")

    # Get class counts into a clean Python dict
    rows = df.groupBy(label_col).count().collect()
    counts: Dict[Any, int] = {r[label_col]: int(r["count"]) for r in rows}

    if len(counts) < 2:
        raise ValueError("Expected at least two classes in label_col.")

    labels = list(counts.keys())

    # Infer majority if not provided (use builtins.max to avoid shadowing)
    if majority_label is None:
        majority_label = builtins.max(labels, key=lambda k: counts[k])

    # In binary case, the other label is the minority
    if len(labels) == 2:
        minority_label = labels[0] if labels[1] == majority_label else labels[1]
    else:
        # If >2 classes, pick the smallest as minority
        minority_label = builtins.min((k for k in labels if k != majority_label), key=lambda k: counts[k])

    maj_n = counts[majority_label]
    min_n = counts[minority_label]

    if maj_n == 0 or min_n == 0:
        raise ValueError("Both classes must have non-zero counts.")

    # Target majority count after undersampling
    target_maj = int(builtins.min(min_n * ratio, maj_n))

    # Fast exit if already within target ratio
    if maj_n <= target_maj:
        return df, {
            "note": "No undersampling needed; majority already within target ratio.",
            "majority_label": majority_label,
            "minority_label": minority_label,
            "counts_before": counts,
            "counts_after": counts,
            "target_majority": target_maj,
            "strategy":None,
            "sampling":None,
            "split":None
        }

    # Split
    df_min = df.filter(F.col(label_col) == F.lit(minority_label))
    df_maj = df.filter(F.col(label_col) == F.lit(majority_label))

    if strategy == "sample":
        frac = target_maj / float(maj_n)
        # Guard: fraction must be in [0,1]
        frac = builtins.max(0.0, builtins.min(1.0, frac))
        df_maj_us = df_maj.sample(withReplacement=False, fraction=frac, seed=seed)
    elif strategy == "limit":
        df_maj_us = df_maj.orderBy(F.rand(seed)).limit(target_maj)
    else:
        raise ValueError("strategy must be 'sample' or 'limit'.")

    balanced = df_min.unionByName(df_maj_us)

    # Optional: shuffle
    balanced = balanced.orderBy(F.rand(seed))

    # Report counts after
    rows_after = balanced.groupBy(label_col).count().collect()
    after_counts: Dict[Any, int] = {r[label_col]: int(r["count"]) for r in rows_after}

    info = {
        "majority_label": majority_label,
        "minority_label": minority_label,
        "counts_before": counts,
        "counts_after": after_counts,
        "target_majority": target_maj,
        "achieved_ratio": after_counts.get(majority_label, 0) / float(after_counts.get(minority_label, 1)),
        "sampling":sampling,
        "strategy":strategy,
        "split":split
    }
    return balanced, info



def upsample_minority(
    df: DataFrame,
    label_col: str = "label",
    majority_label: Optional[Any] = None,
    ratio: float = 1.0,         # desired majority:minority AFTER oversampling (e.g., 1.0 => 1:1)
    strategy: str = "sample",   # "sample" (fast, approx) or "limit" (exact)
    seed: int = 42,
    split: int = None,
    sampling: str = "upsample",
) -> Tuple[DataFrame, Dict[str, Any]]:
    """
    Randomly oversample the minority class in a binary-labeled Spark DataFrame until
    the post-oversampling majority:minority ratio ~= `ratio`.

    NOTE: This is RANDOM OVER-SAMPLING (duplication), not SMOTE (synthetic examples).
    If you truly need SMOTE in Spark, you'd use a third-party transformer and call it
    inside your Pipeline.

    Returns
    -------
    (balanced_df, info)
    """
    if ratio <= 0:
        raise ValueError("ratio must be > 0 (majority:minority).")

    # --- Class counts -> plain dict (robust across Spark versions)
    rows = df.groupBy(label_col).count().collect()
    counts: Dict[Any, int] = {r[label_col]: int(r["count"]) for r in rows}
    labels = list(counts.keys())
    if len(labels) < 2:
        raise ValueError("Expected at least two classes in label_col.")

    # --- Majority / minority WITHOUT min/max(key=...) to avoid shadowing issues
    if majority_label is None:
        maj_lbl, maj_n = None, -1
        for k, v in counts.items():
            if v > maj_n:
                maj_lbl, maj_n = k, v
        majority_label = maj_lbl
    else:
        maj_n = counts[majority_label]

    min_lbl, min_n = None, None
    for k, v in counts.items():
        if k == majority_label:
            continue
        if (min_n is None) or (v < min_n):
            min_lbl, min_n = k, v
    minority_label = min_lbl

    if maj_n == 0 or min_n == 0:
        raise ValueError("Both classes must have non-zero counts.")

    # --- Target minority count to achieve maj:min == ratio
    #     maj / target_min = ratio  =>  target_min = ceil(maj / ratio)
    target_min = int(math.ceil(maj_n / float(ratio)))
    if target_min <= min_n:
        # Already at or better than the requested ratio
        return df, {
            "note": "No oversampling needed; classes already meet the requested ratio.",
            "majority_label": majority_label,
            "minority_label": minority_label,
            "counts_before": counts,
            "counts_after": counts,
            "target_minority": min_n,
            "sampling":None,
            "strategy": None,
            "split":split
        }

    # --- Split
    minority_df = df.filter(F.col(label_col) == F.lit(minority_label))
    majority_df = df.filter(F.col(label_col) == F.lit(majority_label))

    # --- Build oversampled minority
    need_extra = target_min - min_n

    if strategy == "sample":
        # Approximate: expected extra = fraction * min_n
        frac_extra = need_extra / float(min_n)
        if frac_extra < 0.0:
            frac_extra = 0.0
        # sample withReplacement=True to allow duplicates
        extra_df = minority_df.sample(withReplacement=True, fraction=frac_extra, seed=seed)
        up_minority = minority_df.unionByName(extra_df)

    elif strategy == "limit":
        # Exact: replicate rows K times then limit to target_min
        # K = ceil(target_min / min_n)
        K = int(math.ceil(target_min / float(min_n)))
        # explode an array of size K to replicate each minority row K times
        reps = F.array(*[F.lit(i) for i in range(K)])
        replicated = minority_df.withColumn("_rep", F.explode(reps)).drop("_rep")
        # randomize and take exact target_min rows
        up_minority = replicated.orderBy(F.rand(seed)).limit(target_min)

    else:
        raise ValueError("strategy must be 'sample' or 'limit'.")

    balanced = majority_df.unionByName(up_minority).orderBy(F.rand(seed))

    # --- Report counts after
    rows_after = balanced.groupBy(label_col).count().collect()
    after_counts: Dict[Any, int] = {r[label_col]: int(r["count"]) for r in rows_after}

    info = {
        "majority_label": majority_label,
        "minority_label": minority_label,
        "counts_before": counts,
        "counts_after": after_counts,
        "target_minority": target_min,
        "achieved_ratio": after_counts.get(majority_label, 0) / float(after_counts.get(minority_label, 1)),
        "sampling":sampling,
        "strategy": strategy,
        "split":split,
    }
    return balanced, info

"""

def _vec_add(a: DenseVector, b: DenseVector):
    return DenseVector([x + y for x, y in zip(a, b)])

def _vec_sub(a: DenseVector, b: DenseVector):
    return DenseVector([x - y for x, y in zip(a, b)])

def _vec_scale(a: DenseVector, s: float):
    return DenseVector([x * s for x in a])

@F.udf(VectorUDT())
def _interpolate(v1: DenseVector, v2: DenseVector, u: float):
    # returns v1 + u*(v2 - v1)
    return _vec_add(v1, _vec_scale(_vec_sub(v2, v1), float(u)))

def smote_oversample_spark(
    df: DataFrame,
    *,
    label_col: str = "label",
    features_col: str = "features",
    majority_label: Optional[Any] = None,
    ratio: float = 1.0,               # desired majority:minority AFTER SMOTE
    k_neighbors: int = 5,
    lsh_bucket_len: float = 2.0,      # tune to your feature scale
    lsh_tables: int = 2,
    max_pairs_dist: float = 10.0,     # candidate neighbor distance cap (tune)
    seed: int = 42
) -> Tuple[DataFrame, Dict[str, Any]]:
    # class counts
    rows = df.groupBy(label_col).count().collect()
    counts: Dict[Any, int] = {r[label_col]: int(r["count"]) for r in rows}
    if len(counts) < 2:
        raise ValueError("Need at least two classes.")
    # infer majority/minority (no key=)
    if majority_label is None:
        maj_lbl, maj_n = None, -1
        for k,v in counts.items():
            if v > maj_n:
                maj_lbl, maj_n = k, v
        majority_label = maj_lbl
    else:
        maj_n = counts[majority_label]

    min_lbl, min_n = None, None
    for k,v in counts.items():
        if k == majority_label: continue
        if (min_n is None) or (v < min_n):
            min_lbl, min_n = k, v
    minority_label = min_lbl

    if maj_n == 0 or min_n == 0:
        raise ValueError("Both classes must be non-zero.")

    target_min = int(math.ceil(maj_n / float(ratio)))
    need = target_min - min_n
    if need <= 0:
        return df, {"note": "No SMOTE needed", "counts_before": counts, "counts_after": counts}

    # isolate minority
    minority = df.filter(F.col(label_col) == F.lit(minority_label)) \
                 .select(F.monotonically_increasing_id().alias("_id"), features_col, label_col)

    # LSH for approx neighbor pairs (self-join)
    lsh = (BucketedRandomProjectionLSH(inputCol=features_col, outputCol="hashes",
                                       bucketLength=lsh_bucket_len, numHashTables=lsh_tables))
    lsh_model = lsh.fit(minority)

    # similarity join within distance; returns rows: (datasetA.*, datasetB.*, distCol)
    pairs = lsh_model.approxSimilarityJoin(
        minority, minority, max_pairs_dist, distCol="dist"
    ).filter(F.col("datasetA._id") != F.col("datasetB._id"))

    # keep top-k neighbors per anchor (smallest dist)
    from pyspark.sql.window import Window
    w = Window.partitionBy("datasetA._id").orderBy(F.col("dist").asc())
    knn = (pairs
           .withColumn("rn", F.row_number().over(w))
           .filter(F.col("rn") <= F.lit(k_neighbors))
           .select(
               F.col("datasetA._id").alias("_id"),
               F.col("datasetA."+features_col).alias("f1"),
               F.col("datasetB."+features_col).alias("f2"),
           ))

    # how many synthetics per anchor on average
    synthetics_per_anchor = float(need) / float(min_n)
    # sample a random u for interpolation
    randu = (F.rand(seed).alias("u"))

    # replicate anchors enough times, then trim to exact 'need'
    reps = int(math.ceil(synthetics_per_anchor))
    rep_arr = F.array(*[F.lit(i) for i in range(reps)])
    expanded = (knn
                .withColumn("_rep", F.explode(rep_arr))
                .withColumn("u", randu)
                .withColumn(features_col, _interpolate(F.col("f1").cast("vector"),
                                                       F.col("f2").cast("vector"),
                                                       F.col("u")))
                .select(F.col(features_col))
                .withColumn(label_col, F.lit(minority_label)))

    synthetic = expanded.orderBy(F.rand(seed)).limit(need)

    balanced = df.unionByName(synthetic)

    # counts after
    rows_after = balanced.groupBy(label_col).count().collect()
    after_counts = {r[label_col]: int(r["count"]) for r in rows_after}
    return balanced, {
        "majority_label": majority_label,
        "minority_label": minority_label,
        "counts_before": counts,
        "counts_after": after_counts,
        "target_minority": target_min,
        "synthetic_generated": need,
        "k_neighbors": k_neighbors,
        "lsh_bucket_len": lsh_bucket_len,
        "lsh_tables": lsh_tables,
        "max_pairs_dist": max_pairs_dist,
        "method": "LSH-SMOTE (approx)",
    }
"""




'''
def sample(
    df: DataFrame,
    undersample = True,
    oversample = True,
    P_TEST=0.20,
    P_VAL=0.20,      
    label_col: str = "label",
    majority_label: Optional[str] = None,
    ratio: float = 1.0,
    strategy: str = "sample",      # "sample" (fast, approx) or "limit" (exact)
    seed: int = 42):


    """
    Samples a Spark DataFrame into train, val, and test sets
    Then Oversamples and Undersamples as indicated
    """

    # Stratified Sampling
    strat_train, strat_val, strat_test = stratified_sampling(df, P_TEST, P_VAL)

    if undersample:
        strat_train_under, train_info_under = undersample_majority(strat_train, label_col, majority_label, ratio, strategy, seed)
    else:
        strat_train_under = None
        train_info_under = None

    if oversample:
        strat_train_over, train_info_over = upsample_minority(strat_train, label_col, majority_label, ratio, strategy, seed)
    else:
        strat_train_over = None
        train_info_over = None

    return strat_train, strat_val, strat_test, strat_train_under, strat_train_over, train_info_under, train_info_over
'''
    
