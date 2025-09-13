#!/usr/bin/env python3
"""
Bangla Text Analytics - One-File Streamlit App
Built according to specifications for near-duplicate detection, clustering, and search
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import hashlib
import os
import sys
import platform
import shutil
import requests
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
import warnings
from pyspark.sql import functions as F
from pyspark.sql.functions import desc
import os, urllib.request
from pathlib import Path
from matplotlib import font_manager
import streamlit as st
warnings.filterwarnings('ignore')
from pyspark.sql import functions as F
from pyspark.ml.linalg import Vectors
# Core ML/Spark imports
# Improved import handling with specific installation guidance
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, when, count, sum as spark_sum
    from pyspark.sql.types import *
    from pyspark.ml.feature import *
    from pyspark.ml.clustering import KMeans
    from pyspark.ml.feature import PCA
    from pyspark.ml import Pipeline
    from pyspark.ml.linalg import Vectors, VectorUDT
    SPARK_AVAILABLE = True
except ImportError as e:
    SPARK_AVAILABLE = False
    SPARK_ERROR = str(e)

# Plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from wordcloud import WordCloud
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    st.warning("Some plotting libraries missing. Install: pip install matplotlib seaborn wordcloud plotly")

# =====================================
# CONSTANTS & CONFIGURATION
# =====================================

APP_ROOT = Path(__file__).parent.resolve()
WORK_DIR = APP_ROOT / "work"

# Default configuration
DEFAULT_CONFIG = {
    'dataset_fraction': 1.0,
    'eda_fraction': 0.1,
    'random_seed': 42,
    'hash_dim_exp': 18,  # 2^18
    'idf_min_doc_freq': 3,
    'lsh_num_hash_tables': 8,
    'kmeans_k': 20,
    'top_k_search': 20,
    'min_jaccard': 0.2,
    'min_cosine': 0.2,
    'final_insights_jaccard': 0.8,
    'max_pairs_save': 100000,
    'ngram_n': 2,
    'top_n_terms': 50,
}

# Directory structure
DIRS_TO_CREATE = [
    "work/data_clean/base_prothomalo",
    "work/data_clean/preprocess_prothomalo/compact_nostop_udffree", 
    "work/eda_figs",
    "work/models",
    "work/data_results/ngram_analysis",
    "work/data_results/similarity_search_cache",
    "work/quality_monitoring/integrity_reports",
    "work/quality_monitoring/coverage_analysis", 
    "work/quality_monitoring/performance_benchmarks",
    "work/spark_local",
    "work/checkpoints",
    "work/warehouse",
    "work/parquet_sanity",
    "work/hadoop/bin"
]

# =====================================
# UTILITY FUNCTIONS
# =====================================

def create_directory_structure():
    """Create all required directories"""
    for dir_path in DIRS_TO_CREATE:
        full_path = APP_ROOT / dir_path
        full_path.mkdir(parents=True, exist_ok=True)

def get_system_info() -> Dict[str, Any]:
    """Get system information for preflight checks"""
    import psutil
    
    return {
        'os': platform.system(),
        'os_version': platform.version(),
        'python_version': sys.version,
        'cpu_count': os.cpu_count(),
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'disk_free_gb': round(psutil.disk_usage(str(APP_ROOT)).free / (1024**3), 2),
        'java_version': get_java_version(),
    }

def get_java_version() -> Optional[str]:
    """Check Java availability"""
    try:
        import subprocess
        result = subprocess.run(['java', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stderr.split('\n')[0]
        return None
    except:
        return None

def download_hadoop_utils():
    """Download winutils.exe and hadoop.dll for Windows"""
    if platform.system() != "Windows":
        return True, "Not needed on non-Windows systems"
    
    hadoop_dir = WORK_DIR / "hadoop" / "bin"
    hadoop_dir.mkdir(parents=True, exist_ok=True)
    
    winutils_path = hadoop_dir / "winutils.exe"
    hadoop_dll_path = hadoop_dir / "hadoop.dll"
    
    if winutils_path.exists() and hadoop_dll_path.exists():
        return True, "Already exists"
    
    try:
        # Download winutils.exe
        winutils_url = "https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.5/bin/winutils.exe"
        hadoop_dll_url = "https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.5/bin/hadoop.dll"
        
        with st.status("Downloading Hadoop utilities...") as status:
            # Download winutils.exe
            status.update(label="Downloading winutils.exe...")
            response = requests.get(winutils_url, stream=True)
            response.raise_for_status()
            with open(winutils_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Download hadoop.dll
            status.update(label="Downloading hadoop.dll...")
            response = requests.get(hadoop_dll_url, stream=True)
            response.raise_for_status()
            with open(hadoop_dll_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Set environment variables
            os.environ['HADOOP_HOME'] = str(hadoop_dir.parent)
            os.environ['PATH'] = f"{hadoop_dir};{os.environ.get('PATH', '')}"
            
            status.update(label="Setting permissions...", state="complete")
            
        return True, "Downloaded successfully"
    except Exception as e:
        return False, f"Download failed: {str(e)}"
import os, sys
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
from pyspark.sql import SparkSession
import streamlit as st
import os
import sys
import platform

@st.cache_resource
def get_spark_session():
    """Get or create Spark session with ML libraries enabled"""
    if not SPARK_AVAILABLE:
        return None
    
    try:
        # Set Hadoop home if on Windows
        if platform.system() == "Windows":
            hadoop_home = WORK_DIR / "hadoop"
            if hadoop_home.exists():
                os.environ['HADOOP_HOME'] = str(hadoop_home)
        
        # Calculate optimal settings
        cpu_count = os.cpu_count() or 4
        memory_gb = 4
        try:
            import psutil
            memory_gb = max(2, int(psutil.virtual_memory().total / (1024**3) * 0.6))
        except:
            pass
        
        # Create Spark session with ML support
        spark = (SparkSession.builder
            .appName("DocumentSearch")
            .master(f"local[{cpu_count}]")
            # Python executables
            .config("spark.pyspark.python", sys.executable)
            .config("spark.pyspark.driver.python", sys.executable)
            # Memory settings
            .config("spark.driver.memory", f"{memory_gb}g")
            .config("spark.driver.maxResultSize", f"{max(1, memory_gb//2)}g")
            # Performance settings
            .config("spark.sql.execution.arrow.pyspark.enabled", "false")
            .config("spark.sql.shuffle.partitions", "64")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            # Serialization
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            # Enable ML libraries
            .config("spark.jars.packages", "org.apache.spark:spark-mllib_2.12:3.5.0")
            .getOrCreate())

        # Better tracebacks
        try:
            spark.conf.set("spark.python.worker.faulthandler.enabled", "true")
            spark.conf.set("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
        except Exception:
            pass
        
        spark.sparkContext.setLogLevel("WARN")
        
        # Test ML libraries availability
        ml_available = test_ml_libraries(spark)
        # if ml_available:
        #     st.success("✅ PySpark ML libraries loaded successfully!")
        # else:
        #     st.warning("⚠️ ML libraries test failed, using fallback mode")
        
        return spark
        
    except Exception as e:
        st.error(f"Failed to create Spark session: {e}")
        return None

def test_ml_libraries(spark):
    """Test if ML libraries are working properly"""
    try:
        from pyspark.ml.feature import HashingTF, IDF, Normalizer
        from pyspark.ml.feature import MinHashLSH, BucketedRandomProjectionLSH
        from pyspark.ml import Pipeline
        
        # Create a simple test
        test_df = spark.createDataFrame([
            (["hello", "world"],),
            (["test", "tokens"],)
        ], ["tokens"])
        
        # Test HashingTF
        tf = HashingTF(inputCol="tokens", outputCol="features", numFeatures=1000)
        tf_result = tf.transform(test_df)
        
        # Test IDF
        idf = IDF(inputCol="features", outputCol="idf_features")
        idf_model = idf.fit(tf_result)
        idf_result = idf_model.transform(tf_result)
        
        # Test Normalizer
        normalizer = Normalizer(inputCol="idf_features", outputCol="norm_features", p=2.0)
        norm_result = normalizer.transform(idf_result)
        
        # Count results to ensure everything works
        count = norm_result.count()
        
        print(f"✅ ML libraries test passed ({count} rows processed)")
        return True
        
    except Exception as e:
        print(f"❌ ML libraries test failed: {e}")
        return False
    """Get or create Spark session with ML libraries and optimized settings"""
    if not SPARK_AVAILABLE:
        return None
    
    try:
        # Set Hadoop home if on Windows
        if platform.system() == "Windows":
            hadoop_home = WORK_DIR / "hadoop"
            if hadoop_home.exists():
                os.environ['HADOOP_HOME'] = str(hadoop_home)
        
        # Calculate optimal settings
        cpu_count = os.cpu_count() or 4
        memory_gb = 4  # Conservative default
        try:
            import psutil
            memory_gb = max(2, int(psutil.virtual_memory().total / (1024**3) * 0.6))
        except:
            pass
        
        # Create Spark session with ML libraries
        spark = (SparkSession.builder
            .appName("DocumentSearch")
            .master(f"local[{cpu_count}]")
            # Python executables
            .config("spark.pyspark.python", sys.executable)
            .config("spark.pyspark.driver.python", sys.executable)
            # Memory settings
            .config("spark.driver.memory", f"{memory_gb}g")
            .config("spark.driver.maxResultSize", f"{max(1, memory_gb//2)}g")
            .config("spark.executor.memory", f"{max(1, memory_gb//2)}g")
            # Performance settings
            .config("spark.sql.execution.arrow.pyspark.enabled", "false")
            .config("spark.sql.shuffle.partitions", "64")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            # Serialization
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.kryo.registrationRequired", "false")
            # ML Library packages - THIS IS THE KEY ADDITION
            .config("spark.jars.packages", "org.apache.spark:spark-mllib_2.12:3.5.0")
            .getOrCreate())

        # Better tracebacks if a worker dies
        try:
            spark.conf.set("spark.python.worker.faulthandler.enabled", "true")
            spark.conf.set("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
        except Exception:
            pass
        
        spark.sparkContext.setLogLevel("WARN")
        
        # Test if ML libraries are actually available
        try:
            from pyspark.ml.feature import HashingTF
            test_tf = HashingTF()
            print("✅ ML libraries successfully loaded")
        except Exception as e:
            print(f"⚠️  ML libraries test failed: {e}")
            # Don't return None here, as basic functionality might still work
        
        return spark
        
    except Exception as e:
        st.error(f"Failed to create Spark session: {e}")
        return None

# Alternative function if the package approach doesn't work
@st.cache_resource
def get_spark_session_local():
    """Fallback: Create Spark session assuming local PySpark installation with ML"""
    if not SPARK_AVAILABLE:
        return None
    
    try:
        # First, verify that PySpark ML is installed
        try:
            from pyspark.ml.feature import HashingTF, IDF, Normalizer
            from pyspark.ml.feature import MinHashLSH, BucketedRandomProjectionLSH
            from pyspark.ml import Pipeline
            print("✅ PySpark ML modules are importable")
        except ImportError as e:
            st.error(f"❌ PySpark ML modules not available: {e}")
            st.info("Try: pip uninstall pyspark && pip install pyspark[sql,ml]")
            return None
        
        # Set Hadoop home if on Windows
        if platform.system() == "Windows":
            hadoop_home = WORK_DIR / "hadoop"
            if hadoop_home.exists():
                os.environ['HADOOP_HOME'] = str(hadoop_home)
        
        # Calculate optimal settings
        cpu_count = os.cpu_count() or 4
        memory_gb = 4
        try:
            import psutil
            memory_gb = max(2, int(psutil.virtual_memory().total / (1024**3) * 0.6))
        except:
            pass
        
        spark = (SparkSession.builder
            .appName("DocumentSearch")
            .master(f"local[{cpu_count}]")
            .config("spark.pyspark.python", sys.executable)
            .config("spark.pyspark.driver.python", sys.executable)
            .config("spark.driver.memory", f"{memory_gb}g")
            .config("spark.driver.maxResultSize", f"{max(1, memory_gb//2)}g")
            .config("spark.sql.execution.arrow.pyspark.enabled", "false")
            .config("spark.sql.shuffle.partitions", "64")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .getOrCreate())

        # Test ML functionality
        try:
            test_df = spark.createDataFrame([
                (["hello", "world"],),
                (["test", "tokens"],)
            ], ["tokens"])
            
            tf = HashingTF(inputCol="tokens", outputCol="features")
            result = tf.transform(test_df)
            count = result.count()
            print(f"✅ ML libraries working (test count: {count})")
            
        except Exception as e:
            st.error(f"❌ ML libraries test failed: {e}")
            return None

        # Better tracebacks
        try:
            spark.conf.set("spark.python.worker.faulthandler.enabled", "true")
            spark.conf.set("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
        except Exception:
            pass
        
        spark.sparkContext.setLogLevel("WARN")
        return spark
        
    except Exception as e:
        st.error(f"Failed to create Spark session: {e}")
        return None

# Diagnostic function to help troubleshoot
def diagnose_ml_availability():
    """Check what ML functionality is available"""
    print("🔍 Diagnosing ML Library Availability")
    print("=" * 40)
    
    # Check PySpark ML imports
    ml_imports = [
        ("HashingTF", "pyspark.ml.feature"),
        ("IDF", "pyspark.ml.feature"), 
        ("Normalizer", "pyspark.ml.feature"),
        ("MinHashLSH", "pyspark.ml.feature"),
        ("BucketedRandomProjectionLSH", "pyspark.ml.feature"),
        ("Pipeline", "pyspark.ml")
    ]
    
    available_count = 0
    for cls_name, module in ml_imports:
        try:
            exec(f"from {module} import {cls_name}")
            print(f"✅ {cls_name}")
            available_count += 1
        except Exception as e:
            print(f"❌ {cls_name}: {e}")
    
    print(f"\nResult: {available_count}/{len(ml_imports)} ML classes available")
    
    if available_count == 0:
        print("\n💡 Recommendations:")
        print("   pip uninstall pyspark")
        print("   pip install pyspark[sql,ml]==3.5.0")
    elif available_count < len(ml_imports):
        print("\n💡 Try upgrading:")
        print("   pip install --upgrade pyspark")
    else:
        print("\n✅ All ML libraries should work!")
    
    return available_count == len(ml_imports)
def run_spark_sanity_tests(spark) -> Tuple[bool, str]:
    """Run basic Spark functionality tests"""
    if not spark:
        return False, "No Spark session available"
    
    try:
        with st.status("Running Spark sanity tests...") as status:
            # Test 1: Basic computation
            status.update(label="Testing basic computation...")
            df = spark.range(1000).select(col("id").alias("value"))
            result = df.agg(sum("value")).collect()[0][0]
            expected = sum(range(1000))
            if result != expected:
                return False, f"Computation test failed: {result} != {expected}"
            
            # Test 2: Parquet I/O
            status.update(label="Testing Parquet I/O...")
            test_dir = WORK_DIR / "parquet_sanity"
            test_path = test_dir / "test.parquet"
            if test_path.exists():
                shutil.rmtree(test_path)
            
            test_df = spark.range(100).withColumn("text", lit("test"))
            test_df.write.mode("overwrite").parquet(str(test_path))
            
            read_df = spark.read.parquet(str(test_path))
            if read_df.count() != 100:
                return False, "Parquet I/O test failed"
            
            # Test 3: Simple join
            status.update(label="Testing join operations...")
            df1 = spark.range(50).withColumn("key", col("id"))
            df2 = spark.range(50).withColumn("key", col("id"))
            joined = df1.join(df2, "key")
            if joined.count() != 50:
                return False, "Join test failed"
            
            status.update(label="All tests passed!", state="complete")
            
        return True, "All sanity tests passed"
    except Exception as e:
        return False, f"Sanity test failed: {str(e)}"

def get_artifact_status() -> Dict[str, Dict[str, Any]]:
    """Check status of all artifacts"""
    artifacts = {}
    
    # Dataset files
    dataset_file = APP_ROOT / "prothomalo_articles.jsonl"
    artifacts['dataset'] = {
        'exists': dataset_file.exists(),
        'size': dataset_file.stat().st_size if dataset_file.exists() else 0,
        'path': str(dataset_file)
    }
    
    # Compact parquet
    compact_dir = WORK_DIR / "data_clean/preprocess_prothomalo/compact_nostop_udffree"
    artifacts['compact'] = {
        'exists': compact_dir.exists() and any(compact_dir.iterdir()),
        'path': str(compact_dir)
    }
    
    # Models
    models_dir = WORK_DIR / "models"
    artifacts['idf_models'] = {
        'exists': models_dir.exists() and any(models_dir.glob("idf_hash*")),
        'path': str(models_dir)
    }
    
    artifacts['lsh_models'] = {
        'exists': models_dir.exists() and any(models_dir.glob("*lsh*")),
        'path': str(models_dir)
    }
    
    # Clustering results
    results_dir = WORK_DIR / "data_results"
    artifacts['kmeans_preds'] = {
        'exists': results_dir.exists() and any(results_dir.glob("kmeans_pred_k*")),
        'path': str(results_dir)
    }
    
    cfg = st.session_state.config
    wanted = Path(WORK_DIR) / "models" / f"minhash_ng{cfg.get('ngram_for_minhash',2)}_bin{cfg['hash_dim']}_h{cfg['lsh_num_hash_tables']}"
    fallback = Path(WORK_DIR) / "models" / "minhash_ng2_bin262144_h8"   # notebook path

    mh_path = next((p for p in [wanted, fallback] if _has_mlh_model(p)), None)
    artifacts['lsh_models'] = {
        'exists': mh_path is not None,
        'minhash_exists': mh_path is not None,
        'minhash_path': str(mh_path) if mh_path else None,
        'brp_exists': artifacts.get('lsh_models', {}).get('brp_exists', False)
    }
    return artifacts

# =====================================
# DATA LOADING & PROCESSING
# =====================================



import json
import gzip
import pandas as pd
import streamlit as st
from pathlib import Path

@st.cache_data
def load_dataset(fraction: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """Load dataset — handles mixed-format files (JSONL + GZIP garbage)"""
    dataset_file = APP_ROOT / "prothomalo_articles.jsonl"

    if not dataset_file.exists():
        st.error(f"Dataset file not found: {dataset_file}")
        return pd.DataFrame()

    st.subheader("🔍 Debug: Dataset Loading Strategy")
    st.write(f"**File:** `{dataset_file}`")
    st.write(f"**Size:** {dataset_file.stat().st_size:,} bytes")

    data = []
    line_num = 0

    # Open as binary to avoid premature decoding errors
    with open(dataset_file, 'rb') as f:
        buffer = b""
        while True:
            chunk = f.read(8192)  # Read in 8KB chunks
            if not chunk:
                break

            buffer += chunk

            # Split on newlines, keeping the last partial line
            lines = buffer.split(b'\n')
            buffer = lines[-1]  # Keep last incomplete line
            lines = lines[:-1]  # Process all complete lines

            for raw_line in lines:
                line_num += 1

                # Try to decode as UTF-8 — if fails, skip
                try:
                    line_str = raw_line.decode('utf-8').strip()
                except UnicodeDecodeError:
                    st.warning(f"⚠️ Skipped corrupted line {line_num}: contains non-UTF-8 bytes")
                    continue

                if not line_str:
                    continue

                try:
                    obj = json.loads(line_str)
                    data.append(obj)
                except json.JSONDecodeError:
                    st.warning(f"⚠️ Skipped malformed JSON at line {line_num}: {line_str[:60]}...")
                    continue

                # Progress indicator every 10k lines
                if line_num > 0 and line_num % 10000 == 0:
                    st.info(f"✅ Loaded {line_num:,} lines...")

    st.success(f"✅ Successfully loaded {len(data):,} articles after skipping corrupt lines.")

    df = pd.DataFrame(data)

    if df.empty:
        st.error("Dataset is empty after loading")
        return df

    if fraction < 1.0:
        df = df.sample(frac=fraction, random_state=seed).reset_index(drop=True)

    if 'doc_id' not in df.columns:
        df['doc_id'] = df.index.astype(str)

    st.success(f"🎉 Final: Loaded {len(df):,} articles for analysis!")
    return df

def build_compact_dataset(spark, df: pd.DataFrame, remove_stopwords: bool = True) -> bool:
    """Build compact dataset with preprocessing"""
    if df.empty:
        st.error("No data to process")
        return False

    try:
        with st.status("Building compact dataset...") as status:
            # --------------------- Find/validate text column ---------------------
            status.update(label="Converting to Spark DataFrame...")

            candidate_names = {'text', 'content', 'body', 'article'}
            text_col = next((c for c in df.columns if c.lower() in candidate_names), None)

            if not text_col:
                # fall back: first string-like column
                for c in df.columns:
                    dt = df[c].dtype
                    if getattr(pd.api.types, "is_string_dtype", None) and pd.api.types.is_string_dtype(dt):
                        text_col = c
                        break
                    if str(dt) == "object":
                        text_col = c
                        break

            if not text_col:
                st.error("No text column found in dataset")
                return False

            has_category = 'category' in df.columns
            has_doc_id   = 'doc_id' in df.columns

            # --------------------- Build rows & schema safely ---------------------
            schema = StructType([
                StructField("doc_id", StringType(), True),
                StructField("text",   StringType(), True),
            ])
            if has_category:
                schema.add(StructField("category", StringType(), True))

            spark_data = []
            for idx, row in df.iterrows():
                # safe doc_id (fallback to index if missing/blank)
                if has_doc_id and pd.notna(row.get('doc_id')) and str(row.get('doc_id')).strip():
                    doc_id_val = str(row['doc_id'])
                else:
                    doc_id_val = f"doc_{idx}"

                text_val = row.get(text_col)
                text_val = str(text_val) if pd.notna(text_val) else ""

                row_out = [doc_id_val, text_val]
                if has_category:
                    cat_val = row.get('category')
                    row_out.append(str(cat_val) if pd.notna(cat_val) else "unknown")

                spark_data.append(row_out)

            sdf = spark.createDataFrame(spark_data, schema)

            # --------------------- Preprocess & tokenize ---------------------
            status.update(label="Building preprocessing pipeline...")

            # Normalize text (basic cleaning)
            sdf = sdf.withColumn(
                "text_norm",
                regexp_replace(
                    regexp_replace(col("text"), r"[^\u0980-\u09FF\w\s]", " "),
                    r"\s+", " "
                )
            )

            tokenizer = RegexTokenizer(
                inputCol="text_norm",
                outputCol="tokens_raw",
                pattern=r"\s+",
                gaps=True
            )

            stages = [tokenizer]

            if remove_stopwords:
                bangla_stopwords = ["এই","এবং","বা","তার","তাই","যে","যা","কি","কে","কোন","থেকে","দিয়ে","সে","তিনি","আমি","আমার","তুমি","তোমার"]
                stopwords_remover = StopWordsRemover(
                    inputCol="tokens_raw",
                    outputCol="tokens_clean",
                    stopWords=bangla_stopwords
                )
                stages.append(stopwords_remover)
                base_token_col = "tokens_clean"
            else:
                base_token_col = "tokens_raw"

            status.update(label="Fitting preprocessing pipeline...")
            pipeline = Pipeline(stages=stages)
            model = pipeline.fit(sdf)

            status.update(label="Transforming data...")
            processed = model.transform(sdf)

            # Remove empty/whitespace tokens first
            processed = processed.withColumn(
                "tokens_filtered",
                expr(f"filter({base_token_col}, x -> length(trim(x)) > 0)")
            )

            # Fallback: if everything got stripped by stopwords/cleaning, use raw tokens
            processed = processed.withColumn(
                "tokens_final",
                expr("CASE WHEN size(tokens_filtered) > 0 THEN tokens_filtered ELSE tokens_raw END")
            ).drop("tokens_filtered")

            # Final projection
            final_cols = ["doc_id", "text_norm", "tokens_final"]
            if has_category:
                final_cols.append("category")
            final_sdf = processed.select(*final_cols)

            # --------------------- Guard against empty outputs ---------------------
            status.update(label="Validating output rows...")
            non_empty_cnt = final_sdf.where(expr("size(tokens_final) > 0")).limit(1).count()
            if non_empty_cnt == 0:
                st.error("All rows became empty after cleaning/tokenization. Check your text column selection and preprocessing.")
                return False

            # --------------------- Save (only if non-empty) ---------------------
            status.update(label="Saving compact dataset...")
            output_path = WORK_DIR / "data_prep/prothomalo_prep/compact_nostop_udffree"
            if output_path.exists():
                shutil.rmtree(output_path)

            # Write some partitions to ensure visible part files
            (final_sdf
             .repartition(8)
             .write
             .mode("overwrite")
             .parquet(str(output_path)))

            status.update(label="Compact dataset built successfully!", state="complete")
            return True

    except Exception as e:
        st.error(f"Error building compact dataset: {e}")
        return False
from pathlib import Path

def _first_parquet_dir_with_parts(paths):
    for p in paths:
        if p.exists() and any(p.glob("*.parquet")):
            return p
    return None
import streamlit as st
from pathlib import Path

@st.cache_resource(show_spinner=False)
def load_compact_dataset(_spark, dataset_fraction: float, random_seed: int):
    """Load compact dataset as a Spark DataFrame (cache as a resource, not data)."""
    def _first_parquet_dir_with_parts(paths):
        for p in paths:
            if p.exists() and any(p.glob("*.parquet")):
                return p
        return None

    candidates = [
        WORK_DIR / "data_prep/prothomalo_prep/compact_nostop_udffree",          # notebook path
        WORK_DIR / "data_clean/preprocess_prothomalo/compact_nostop_udffree",   # legacy app path
        Path.cwd() / "work" / "data_prep" / "prothomalo_prep" / "compact_nostop_udffree",
        Path.cwd() / "work" / "data_clean" / "preprocess_prothomalo" / "compact_nostop_udffree",
    ]
    p = _first_parquet_dir_with_parts(candidates)
    if not p:
        raise FileNotFoundError("No compact dataset with Parquet part files found. Build it first or update paths.")

    sdf = _spark.read.parquet(str(p))
    if 0.0 < float(dataset_fraction) < 1.0:
        sdf = sdf.sample(withReplacement=False, fraction=float(dataset_fraction), seed=int(random_seed))
    return sdf

# =====================================
# MODEL BUILDING & LOADING
# =====================================

@st.cache_resource
def build_or_load_idf_model(spark, hash_dim: int, min_doc_freq: int):
    """Build or load IDF model"""
    model_path = WORK_DIR / f"models/idf_hash{hash_dim}"
    
    if model_path.exists():
        try:
            from pyspark.ml.feature import IDFModel
            return IDFModel.load(str(model_path))
        except:
            pass
    
    # Need to build model
    sdf = load_compact_dataset(spark)
    if sdf is None:
        st.error("No compact dataset available. Please build it first.")
        return None
    
    try:
        with st.status("Building IDF model...") as status:
            status.update(label="Creating TF-IDF pipeline...")
            
            # HashingTF
            hashing_tf = HashingTF(
                inputCol="tokens_final",
                outputCol="tf_features",
                numFeatures=hash_dim,
                binary=False
            )
            
            # IDF
            idf = IDF(
                inputCol="tf_features",
                outputCol="idf_features",
                minDocFreq=min_doc_freq
            )
            
            # Transform with HashingTF
            status.update(label="Computing term frequencies...")
            tf_sdf = hashing_tf.transform(sdf)
            
            # Fit IDF
            status.update(label="Fitting IDF model...")
            idf_model = idf.fit(tf_sdf)
            
            # Save model
            status.update(label="Saving IDF model...")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            idf_model.write().overwrite().save(str(model_path))
            
            status.update(label="IDF model built successfully!", state="complete")
            
        return idf_model
        
    except Exception as e:
        st.error(f"Error building IDF model: {e}")
        return None

from pathlib import Path
import shutil
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import NGram, HashingTF, MinHashLSH, MinHashLSHModel
@st.cache_resource  
def build_or_load_minhash_lsh(
    spark, sdf, n: int, num_features: int, num_hash: int, prefer_path: str | None = None
) -> PipelineModel | None:
    """
    Returns a PipelineModel: [NGram -> HashingTF(binary) -> MinHashLSHModel].
    If a saved LSH model exists (from notebook), it is loaded and wrapped
    into a pipeline so .transform(sdf) and .stages[-1].approxNearestNeighbors work.
    Otherwise, fits a fresh LSH on the compact dataset.
    """
    # 1) Try to LOAD an existing LSH model (model-only) and wrap it
    candidates = []
    if prefer_path:
        candidates.append(Path(prefer_path))
    candidates += [
        Path(WORK_DIR) / "models" / f"minhash_ng{n}_bin{num_features}_h{num_hash}",
        Path(WORK_DIR) / "models" / "minhash_ng2_bin262144_h8",  # notebook default
    ]
    for p in candidates:
        try:
            if (p / "metadata").exists():
                lsh_model = MinHashLSHModel.load(str(p))
                # Build transformers to produce the expected input col
                ng = NGram(n=int(n), inputCol="tokens_final", outputCol=f"ng{n}")
                ht = HashingTF(inputCol=f"ng{n}", outputCol="features_bin",
                               numFeatures=int(num_features), binary=True)
                # Ensure IO cols on the loaded model
                lsh_model.setInputCol("features_bin")
                lsh_model.setOutputCol("minhash_features")
                return PipelineModel(stages=[ng, ht, lsh_model])
        except Exception:
            pass

    # 2) FIT a new pipeline if nothing to load
    ng = NGram(n=int(n), inputCol="tokens_final", outputCol=f"ng{n}")
    ht = HashingTF(inputCol=f"ng{n}", outputCol="features_bin",
                   numFeatures=int(num_features), binary=True)
    lsh = MinHashLSH(inputCol="features_bin", outputCol="minhash_features",
                     numHashTables=int(num_hash))
    pipe = Pipeline(stages=[ng, ht, lsh])

    # Keep it reasonable
    fit_df = sdf.select("doc_id", "tokens_final")
    model: PipelineModel = pipe.fit(fit_df)

    # Persist only the LSH stage like your notebook does
    save_dir = Path(WORK_DIR) / "models" / f"minhash_ng{n}_bin{num_features}_h{num_hash}"
    try:
        shutil.rmtree(save_dir, ignore_errors=True)
        model.stages[-1].write().overwrite().save(str(save_dir))
    except Exception:
        pass

    return model

@st.cache_resource
def build_or_load_brp_lsh(spark, hash_dim: int, min_doc_freq: int, num_hash_tables: int):
    """Build or load BucketedRandomProjectionLSH model"""
    model_path = WORK_DIR / f"models/brp_tfidf_hash{hash_dim}_l2"
    
    if model_path.exists():
        try:
            from pyspark.ml import PipelineModel
            return PipelineModel.load(str(model_path))
        except:
            pass
    
    # Need to build model
    sdf = load_compact_dataset(spark)
    if sdf is None:
        st.error("No compact dataset available. Please build it first.")
        return None
    
    try:
        with st.status("Building BRP-LSH model...") as status:
            status.update(label="Creating BRP-LSH pipeline...")
            
            # HashingTF
            hashing_tf = HashingTF(
                inputCol="tokens_final",
                outputCol="tf_features", 
                numFeatures=hash_dim,
                binary=False
            )
            
            # IDF
            idf = IDF(
                inputCol="tf_features",
                outputCol="idf_features",
                minDocFreq=min_doc_freq
            )
            
            # L2 Normalizer
            normalizer = Normalizer(
                inputCol="idf_features",
                outputCol="normalized_features",
                p=2.0
            )
            
            # BucketedRandomProjectionLSH
            brp_lsh = BucketedRandomProjectionLSH(
                inputCol="normalized_features",
                outputCol="brp_features",
                numHashTables=num_hash_tables,
                bucketLength=2.0
            )
            
            # Build pipeline
            pipeline = Pipeline(stages=[hashing_tf, idf, normalizer, brp_lsh])
            
            status.update(label="Fitting BRP-LSH pipeline...")
            pipeline_model = pipeline.fit(sdf)
            
            # Save model
            status.update(label="Saving BRP-LSH model...")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            pipeline_model.write().overwrite().save(str(model_path))
            
            status.update(label="BRP-LSH model built successfully!", state="complete")
            
        return pipeline_model
        
    except Exception as e:
        st.error(f"Error building BRP-LSH model: {e}")
        return None
def search_with_minhash(spark, sdf, query_text, minhash_model, top_k, min_jaccard):
    """Improved MinHashLSH search with proper query processing"""
    try:
        # Preprocess query text using the same pipeline as training data
        query_tokens = query_text.strip().split()
        
        # Filter empty tokens
        query_tokens = [token for token in query_tokens if token.strip()]
        
        if not query_tokens:
            return pd.DataFrame()
        
        # Create query DataFrame with the same schema as training data
        query_sdf = spark.createDataFrame(
            [("QUERY", query_text, query_tokens)],
            ["doc_id", "text_norm", "tokens_final"]
        )
        
        # Transform the dataset and query using the full pipeline
        transformed_sdf = minhash_model.transform(sdf)
        query_transformed = minhash_model.transform(query_sdf)
        
        # Get the query vector
        query_row = query_transformed.select("minhash_features").collect()
        if not query_row:
            return pd.DataFrame()
        
        query_vector = query_row[0][0]
        
        # Find approximate nearest neighbors
        candidates = minhash_model.stages[-1].approxNearestNeighbors(
            transformed_sdf, query_vector, top_k * 3, "distance"
        )
        
        # Calculate Jaccard similarity and filter
        results = candidates.withColumn(
            "jaccard_sim", 1 - col("distance")
        ).filter(
            col("jaccard_sim") >= min_jaccard
        ).orderBy(
            desc("jaccard_sim")
        ).limit(top_k)
        
        # Select final columns and convert to pandas
        final_results = results.select(
            "doc_id", "text_norm", "jaccard_sim"
        ).toPandas()
        
        # Add preview column
        if not final_results.empty:
            final_results['preview'] = final_results['text_norm'].str[:200] + "..."
            final_results['engine'] = 'MinHashLSH'
            final_results['rank'] = range(1, len(final_results) + 1)
        
        return final_results
        
    except Exception as e:
        st.warning(f"MinHashLSH search failed: {e}")
        return pd.DataFrame()

def search_with_brp(spark, sdf, query_text, brp_model, top_k, min_cosine):
    """Improved BRP-LSH search with proper query processing"""
    try:
        query_tokens = query_text.strip().split()
        query_tokens = [token for token in query_tokens if token.strip()]
        
        if not query_tokens:
            return pd.DataFrame()
        
        query_sdf = spark.createDataFrame(
            [("QUERY", query_text, query_tokens)],
            ["doc_id", "text_norm", "tokens_final"]
        )
        
        # Transform both dataset and query
        transformed_sdf = brp_model.transform(sdf)
        query_transformed = brp_model.transform(query_sdf)
        
        # Get query vector
        query_row = query_transformed.select("brp_features").collect()
        if not query_row:
            return pd.DataFrame()
        
        query_vector = query_row[0][0]
        
        # Find approximate nearest neighbors
        candidates = brp_model.stages[-1].approxNearestNeighbors(
            transformed_sdf, query_vector, top_k * 3, "distance"
        )
        
        # Calculate cosine similarity and filter
        results = candidates.withColumn(
            "cosine_sim", 1 - col("distance")  # Distance is 1 - cosine_sim
        ).filter(
            col("cosine_sim") >= min_cosine
        ).orderBy(
            desc("cosine_sim")
        ).limit(top_k)
        
        final_results = results.select(
            "doc_id", "text_norm", "cosine_sim"
        ).toPandas()
        
        if not final_results.empty:
            final_results['preview'] = final_results['text_norm'].str[:200] + "..."
            final_results['engine'] = 'BRP-LSH'
            final_results['rank'] = range(1, len(final_results) + 1)
        
        return final_results
        
    except Exception as e:
        st.warning(f"BRP-LSH search failed: {e}")
        return pd.DataFrame()
class ConfigManager:
    """Centralized configuration management"""
    
    # Default values
    DEFAULTS = {
        'dataset_fraction': 1.0,
        'eda_fraction': 0.1,
        'random_seed': 42,
        'hash_dim_exp': 18,
        'idf_min_doc_freq': 3,
        'lsh_num_hash_tables': 8,
        'kmeans_k': 10,
        'top_k_search': 20,
        'min_jaccard': 0.2,
        'min_cosine': 0.2,
        'final_insights_jaccard': 0.8,
        'max_pairs_save': 100000,
        'ngram_n': 2,
        'top_n_terms': 50,
    }
    
    # Valid ranges for validation
    RANGES = {
        'dataset_fraction': (0.01, 1.0),
        'eda_fraction': (0.01, 0.5),
        'hash_dim_exp': (11, 18),
        'idf_min_doc_freq': (1, 20),
        'lsh_num_hash_tables': (4, 32),
        'kmeans_k': (5, 50),
        'top_k_search': (5, 100),
        'min_jaccard': (0.0, 1.0),
        'min_cosine': (0.0, 1.0),
        'final_insights_jaccard': (0.6, 0.95),
        'max_pairs_save': (10000, 2000000),
        'ngram_n': (1, 3),
        'top_n_terms': (10, 200),
    }
    
    def __init__(self):
        self.config = self.DEFAULTS.copy()
    
    def update(self, key: str, value: Any) -> bool:
        """Update config value with validation"""
        if key not in self.DEFAULTS:
            st.warning(f"Unknown config key: {key}")
            return False
        
        if key in self.RANGES:
            min_val, max_val = self.RANGES[key]
            if not min_val <= value <= max_val:
                st.warning(f"{key} must be between {min_val} and {max_val}")
                return False
        
        self.config[key] = value
        return True
    
    def get(self, key: str) -> Any:
        """Get config value"""
        return self.config.get(key, self.DEFAULTS.get(key))
    
    def get_derived(self) -> Dict[str, Any]:
        """Get derived configuration values"""
        return {
            'hash_dim': 2 ** self.get('hash_dim_exp'),
            'distance_threshold': 1 - self.get('final_insights_jaccard'),
        }
    
    def save_to_session_state(self):
        """Save to Streamlit session state"""
        st.session_state.config = self.config.copy()
    
    def load_from_session_state(self):
        """Load from Streamlit session state"""
        if 'config' in st.session_state:
            self.config.update(st.session_state.config)

# Usage in sidebar
def create_sidebar_controls(config_manager: ConfigManager):
    """Create centralized sidebar controls"""
    with st.sidebar:
        st.header("Configuration")
        
        # Dataset controls
        st.subheader("Dataset")
        config_manager.update('dataset_fraction', 
            st.slider("Dataset Fraction", 0.01, 1.0, config_manager.get('dataset_fraction'))
        )
        config_manager.update('eda_fraction',
            st.slider("EDA Fraction", 0.01, 0.5, config_manager.get('eda_fraction'))
        )
        config_manager.update('random_seed',
            st.number_input("Random Seed", value=config_manager.get('random_seed'))
        )
        
        # Model controls
        st.subheader("Models")
        hash_dim_exp = st.selectbox(
            "Hash Dimension (2^n)",
            options=list(range(11, 19)),
            index=config_manager.get('hash_dim_exp') - 11
        )
        config_manager.update('hash_dim_exp', hash_dim_exp)
        
        config_manager.update('idf_min_doc_freq',
            st.slider("IDF Min Doc Freq", 1, 20, config_manager.get('idf_min_doc_freq'))
        )
        config_manager.update('lsh_num_hash_tables',
            st.slider("LSH Hash Tables", 4, 32, config_manager.get('lsh_num_hash_tables'))
        )
        
        # Save to session state
        config_manager.save_to_session_state()
        
        return config_manager
def load_dataset_streaming(spark, fraction: float = 1.0, seed: int = 42) -> 'pyspark.sql.DataFrame':
    """Load dataset directly to Spark DataFrame for memory efficiency"""
    dataset_file = APP_ROOT / "prothomalo_articles.jsonl"
    
    if not dataset_file.exists():
        st.error(f"Dataset file not found: {dataset_file}")
        return None
    
    try:
        # For large files, load directly with Spark
        if dataset_file.stat().st_size > 500 * 1024 * 1024:  # > 500MB
            st.info("Large dataset detected. Using Spark streaming loader...")
            
            # Use Spark's JSON reader with multiline support
            sdf = spark.read.option("multiline", "false").json(str(dataset_file))
            
            # Add doc_id if not present
            if 'doc_id' not in sdf.columns:
                from pyspark.sql.functions import monotonically_increasing_id
                sdf = sdf.withColumn('doc_id', monotonically_increasing_id().cast('string'))
            
            # Sample if needed
            if fraction < 1.0:
                sdf = sdf.sample(fraction=fraction, seed=seed)
            
            count = sdf.count()
            st.success(f"Loaded {count:,} articles directly to Spark")
            return sdf
        else:
            # Use existing pandas approach for smaller files
            df = load_dataset(fraction, seed)
            if df.empty:
                return None
            
            # Convert to Spark DataFrame
            return spark.createDataFrame(df)
            
    except Exception as e:
        st.error(f"Streaming dataset loading failed: {e}")
        # Fallback to pandas approach
        df = load_dataset(fraction, seed)
        if not df.empty:
            return spark.createDataFrame(df)
        return None
# =====================================
# STREAMLIT APP
# =====================================

def main():
    st.set_page_config(
        page_title="Bangla Text Analytics",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📚 Bangla Text Analytics")
    st.markdown("_Near-duplicate detection, clustering, and semantic search for Bangla text._")
    st.caption("Made by Saif, Rafid, Atkiya, and Mohua")

    # Create directory structure
    create_directory_structure()
    
    # Initialize session state
    if 'config' not in st.session_state:
        st.session_state.config = DEFAULT_CONFIG.copy()
    
    # Check Spark availability first
    if not SPARK_AVAILABLE:
        st.error("❌ PySpark is not available")
        
        with st.expander("Installation Instructions", expanded=True):
            st.markdown("""
            **To install PySpark:**
            
            ```bash
            pip install pyspark
            ```
            
            **If you encounter Java issues:**
            - Install Java 8 or higher
            - Set JAVA_HOME environment variable
            - On Windows, ensure Java is in your PATH
            
            **For Windows users:**
            - The app will automatically download required Hadoop utilities
            - Ensure you have proper permissions in your user directory
            """)
            
            if st.button("Check Java Installation"):
                java_version = get_java_version()
                if java_version:
                    st.success(f"Java found: {java_version}")
                else:
                    st.error("Java not found. Please install Java JDK 8+")
        
        st.info("After installing PySpark and Java, please restart the Streamlit app.")
        return
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Global Controls")
        
        # Dataset controls
        st.subheader("Dataset Configuration")
        st.session_state.config['dataset_fraction'] = st.slider(
            "Dataset Fraction", 0.01, 1.0, st.session_state.config['dataset_fraction']
        )
        st.session_state.config['eda_fraction'] = st.slider(
            "EDA Fraction", 0.01, 0.5, st.session_state.config['eda_fraction']
        )
        st.session_state.config['random_seed'] = st.number_input(
            "Random Seed", value=st.session_state.config['random_seed']
        )
        
        # Model controls
        st.subheader("Model Configuration")
        hash_dim_exp = st.selectbox(
            "Hash Dimension (2^n)", 
            options=[11, 12, 13, 14, 15, 16, 17, 18],
            index=[11, 12, 13, 14, 15, 16, 17, 18].index(st.session_state.config['hash_dim_exp'])
        )
        st.session_state.config['hash_dim_exp'] = hash_dim_exp
        st.session_state.config['hash_dim'] = 2 ** hash_dim_exp
        
        st.session_state.config['idf_min_doc_freq'] = st.slider(
            "IDF Min Doc Freq", 1, 20, st.session_state.config['idf_min_doc_freq']
        )
        st.session_state.config['lsh_num_hash_tables'] = st.slider(
            "LSH Hash Tables", 4, 32, st.session_state.config['lsh_num_hash_tables']
        )
        
        # Run Order status
        st.subheader("🔄 Run Order")
        artifacts = get_artifact_status()
        
        steps = [
            ("Build Compact Dataset", "compact", "dataset_overview"),
            ("Fit/Load Models", "idf_models", "models"),
            ("Search & Explore", "lsh_models", "search"),
            ("Parameter Tuning", "kmeans_preds", "tuning"),
            ("Final Insights", "kmeans_preds", "insights")
        ]
        
        for step_name, artifact_key, tab_key in steps:
            status = "✅" if artifacts.get(artifact_key, {}).get('exists', False) else "❌"
            if st.button(f"{status} {step_name}", key=f"jump_{tab_key}"):
                st.session_state.active_tab = tab_key

    # Main content area
    tab_names = [
        "🚀 Preflight", "📊 Dataset Overview", "📈 Text Analytics", 
        "🔍 Document Search", "🎯 Clustering Analysis", "⚙️ Parameter Tuning", 
        "💎 Final Insights", "📋 Quality & Monitoring"
    ]
    
    tabs = st.tabs(tab_names)
    
    # Get Spark session for use across tabs
    try:
        spark = get_spark_session()
    except Exception as e:
        st.error(f"❌ Failed to create Spark session: {e}")
        spark = None
    
    # Preflight Tab
    with tabs[0]:
        show_preflight_tab(spark)
    
    # Check if preflight is OK before enabling other tabs
    preflight_ok = spark is not None
    
    if not preflight_ok:
        for i in range(1, len(tabs)):
            with tabs[i]:
                st.error("⚠️ Please complete preflight checks first!")
        return
    
    # Dataset Overview Tab  
    with tabs[1]:
        try:
            show_dataset_overview_tab(spark)
        except Exception as e:
            st.error(f"❌ Error in Dataset Overview: {e}")
            with st.expander("Error Details"):
                import traceback
                st.code(traceback.format_exc())
    
    # Text Analytics Tab
    with tabs[2]:
        try:
            show_text_analytics_tab(spark)
        except Exception as e:
            st.error(f"❌ Error in Text Analytics: {e}")
            with st.expander("Error Details"):
                import traceback
                st.code(traceback.format_exc())
    
    # Document Search Tab
    with tabs[3]:
        try:
            show_document_search_tab(spark)
        except Exception as e:
            st.error(f"❌ Error in Document Search: {e}")
            with st.expander("Error Details"):
                import traceback
                st.code(traceback.format_exc())
    
    # Clustering Analysis Tab
    with tabs[4]:
        try:
            show_clustering_tab(spark)
        except Exception as e:
            st.error(f"❌ Error in Clustering Analysis: {e}")
            with st.expander("Error Details"):
                import traceback
                st.code(traceback.format_exc())
    
    # Parameter Tuning Tab
    with tabs[5]:
        try:
            show_parameter_tuning_tab(spark)
        except Exception as e:
            st.error(f"❌ Error in Parameter Tuning: {e}")
            with st.expander("Error Details"):
                import traceback
                st.code(traceback.format_exc())
    
    # Final Insights Tab
    with tabs[6]:
        try:
            show_final_insights_tab(spark)
        except Exception as e:
            st.error(f"❌ Error in Final Insights: {e}")
            with st.expander("Error Details"):
                import traceback
                st.code(traceback.format_exc())
    
    # Quality & Monitoring Tab
    with tabs[7]:
        try:
            show_quality_monitoring_tab(spark)
        except Exception as e:
            st.error(f"❌ Error in Quality & Monitoring: {e}")
            with st.expander("Error Details"):
                import traceback
                st.code(traceback.format_exc())

def show_preflight_tab(spark):
    """Show preflight checks and system status (no nested expanders)."""
    import platform
    from pathlib import Path
    import pandas as pd
    import streamlit as st

    # Resolve APP_ROOT if not global
    try:
        _ = APP_ROOT
    except NameError:
        try:
            APP_ROOT = Path(__file__).resolve().parent
        except Exception:
            APP_ROOT = Path.cwd()

    st.header("🚀 Preflight Checks")

    # ========== System information ==========
    with st.expander("💻 System Information", expanded=True):
        sys_info = get_system_info()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Operating System", f"{sys_info.get('os','?')} {sys_info.get('os_version','')}")
            st.metric("CPU Cores", sys_info.get('cpu_count', '?'))
            st.metric("Memory (GB)", sys_info.get('memory_gb', '?'))
        with col2:
            pyver = str(sys_info.get('python_version','')).split()[0] if sys_info.get('python_version') else "?"
            st.metric("Python Version", pyver)
            st.metric("Disk Free (GB)", sys_info.get('disk_free_gb', '?'))
            java_ok = bool(sys_info.get('java_version'))
            st.metric("Java JDK", "✅ Available" if java_ok else "❌ Missing")

        if sys_info.get('java_version'):
            st.success(f"Java: {sys_info['java_version']}")
        else:
            st.error("Java JDK not found. Please install Java 8 or higher.")

    # ========== Windows-specific checks ==========
    if platform.system() == "Windows":
        with st.expander("🪟 Windows Configuration", expanded=True):
            hadoop_dir = APP_ROOT / "hadoop" / "bin"  # same level as app.py
            winutils_exists = (hadoop_dir / "winutils.exe").exists()
            hadoop_dll_exists = (hadoop_dir / "hadoop.dll").exists()

            c1, c2 = st.columns(2)
            with c1:
                st.metric("winutils.exe", "✅" if winutils_exists else "❌")
            with c2:
                st.metric("hadoop.dll", "✅" if hadoop_dll_exists else "❌")

            if not (winutils_exists and hadoop_dll_exists):
                st.warning("Hadoop utilities missing. Download required for Spark functionality.")
                if st.button("📥 Download Hadoop Utils", key="btn_dl_hadoop"):
                    success, message = download_hadoop_utils()
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)

    # ========== Spark session (details only) ==========
    with st.expander("⚡ Spark Session", expanded=True):
        if spark:
            st.success("✅ Spark session created successfully")

            # Show selected Spark configuration
            try:
                spark_conf = dict(spark.sparkContext.getConf().getAll())
            except Exception:
                spark_conf = {}
            important = [
                'spark.app.name', 'spark.master', 'spark.driver.memory',
                'spark.sql.shuffle.partitions', 'spark.sql.adaptive.enabled'
            ]
            rows = [{'Parameter': k, 'Value': spark_conf.get(k, '')} for k in important]
            config_df = pd.DataFrame(rows)
            if not config_df.empty:
                st.dataframe(config_df, hide_index=True, use_container_width=True)
        else:
            st.error("❌ Spark session failed to initialize")
            st.info("Please check Java installation and try restarting the app.")

    # ========== Spark sanity tests (top-level; NOT inside an expander) ==========
    st.subheader("🧪 Spark Sanity Tests")
    run_test = st.button("Run Spark Sanity Tests", key="btn_sanity")
    result_box = st.container()
    if run_test:
        try:
            success, message = run_spark_sanity_tests(spark)
            if success:
                result_box.success(message)
            else:
                result_box.error(message)
        except Exception as e:
            result_box.error(f"Sanity test failed: {e}")
            import traceback
            result_box.code(traceback.format_exc())

    # ========== Dataset check ==========
    with st.expander("📁 Dataset Status", expanded=True):
        dataset_file = APP_ROOT / "prothomalo_articles.jsonl"
        if dataset_file.exists():
            size_mb = dataset_file.stat().st_size / (1024 * 1024)
            st.success(f"✅ Dataset found: {size_mb:.1f} MB")
        else:
            st.error(f"❌ Dataset not found: {dataset_file}")
            st.info("Please ensure 'prothomalo_articles.jsonl' is in the same directory as app.py")
def show_dataset_overview_tab(spark):
    """Show a clean, fast overview of the raw dataset with pretty visuals, handy actions,
       and an optimized Spark-based EDA section (Bangla-friendly tokenization)."""
    import re
    from io import StringIO
    from contextlib import redirect_stdout, redirect_stderr
    from pathlib import Path

    import streamlit as st
    import pandas as pd
    import numpy as np

    # Plotly (optional for charts)
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        PLOTTING_AVAILABLE = True
    except Exception:
        PLOTTING_AVAILABLE = False
        px, go = None, None  # type: ignore

    st.header("📊 Dataset Overview")

    # -----------------------------
    # Load dataset (capture verbose logs into a dropdown)
    # -----------------------------
    config = st.session_state.get("config", {})
    dataset_fraction = float(config.get("dataset_fraction", 1.0))
    random_seed = int(config.get("random_seed", 42))
    eda_fraction = float(config.get("eda_fraction", 0.20))  # for EDA sampling

    buf_out, buf_err = StringIO(), StringIO()
    try:
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            df = load_dataset(dataset_fraction, random_seed)
    except Exception as e:
        st.error(f"❌ Could not load dataset: {e}")
        return

    logs = (buf_out.getvalue() + "\n" + buf_err.getvalue()).strip()

    if df is None or (hasattr(df, "empty") and df.empty):
        st.error("❌ No dataset available. Please check the dataset source.")
        if logs:
            with st.expander("📜 Detailed Load Log (click to expand)", expanded=False):
                st.code(logs)
        return

    # Tiny summary from logs (best-effort)
    loaded_match = re.search(r"Successfully loaded\s+([\d,]+)\s+articles", logs, flags=re.IGNORECASE)
    loaded_total = loaded_match.group(1) if loaded_match else f"{len(df):,}"
    skipped_count = sum(1 for ln in logs.splitlines() if "Skipped" in ln)

    c1, c2, c3 = st.columns(3)
    c1.metric("Loaded Articles", loaded_total)
    c2.metric("Skipped Lines", f"{skipped_count:,}")
    c3.metric("Data Fraction", f"{dataset_fraction:.0%}")

    if logs:
        with st.expander("📜 Detailed Load Log (click to expand)", expanded=False):
            st.code(logs)

    # -----------------------------
    # High-level statistics
    # -----------------------------
    with st.expander("📈 Dataset Statistics", expanded=True):
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Total Documents", f"{len(df):,}")

        with c2:
            if "text" in df.columns:
                non_null_text = df["text"].dropna()
                if not non_null_text.empty:
                    token_lengths = non_null_text.astype(str).str.split().str.len()
                    st.metric("Avg Tokens / Doc", f"{float(token_lengths.mean()):.0f}")
                else:
                    st.metric("Avg Tokens / Doc", "—")
            else:
                st.metric("Avg Tokens / Doc", "—")

        with c3:
            if "category" in df.columns:
                st.metric("Categories", f"{df['category'].nunique():,}")
            else:
                st.metric("Categories", "—")

        with c4:
            memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            st.metric("Memory Usage", f"{memory_mb:.1f} MB")

        # Quick percentiles (if text exists)
        if "text" in df.columns and df["text"].notna().any():
            lengths = df["text"].dropna().astype(str).str.split().str.len()
            p50, p90, p95 = lengths.quantile([0.5, 0.9, 0.95]).astype(int).tolist()
            st.caption(f"**Token length percentiles** — P50: **{p50}** • P90: **{p90}** • P95: **{p95}**")

    # -----------------------------
    # Category distribution (Top 20)
    # -----------------------------
    if "category" in df.columns and PLOTTING_AVAILABLE:
        with st.expander("📊 Category Distribution (Top 20)", expanded=True):
            category_counts = df["category"].value_counts().head(20)
            if not category_counts.empty:
                fig = px.bar(
                    x=category_counts.values,
                    y=category_counts.index,
                    orientation="h",
                    title="Top 20 Categories by Document Count",
                )
                fig.update_layout(
                    height=500,
                    xaxis_title="Documents",
                    yaxis_title="Category",
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No category values to display.")

    # -----------------------------
    # Text length distribution
    # -----------------------------
    if PLOTTING_AVAILABLE and "text" in df.columns:
        with st.expander("📏 Text Length Distribution", expanded=True):
            # Sample for faster EDA
            try:
                frac = float(np.clip(eda_fraction, 0.01, 1.0))
                eda_df = df.sample(frac=frac, random_state=random_seed) if len(df) > 1_000 else df.copy()
            except Exception:
                eda_df = df.copy()

            non_null_text = eda_df["text"].dropna() if "text" in eda_df.columns else pd.Series(dtype=str)
            if not non_null_text.empty:
                token_lengths = non_null_text.astype(str).str.split().str.len()
                fig = px.histogram(
                    x=token_lengths,
                    nbins=50,
                    title=f"Distribution of Document Lengths (tokens) — Sampled ({min(len(eda_df), len(df)):,} rows)",
                )
                fig.update_layout(
                    xaxis_title="Number of Tokens",
                    yaxis_title="Document Count",
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No text values to analyze.")

    # -----------------------------
    # Dataset preview & download
    # -----------------------------
    with st.expander("👀 Dataset Preview", expanded=True):
        preview_df = df.head(100)
        column_config = {}
        if "text" in preview_df.columns:
            column_config["text"] = st.column_config.TextColumn("text", width=600)
        if "category" in preview_df.columns:
            column_config["category"] = st.column_config.TextColumn("category", width=200)

        st.dataframe(preview_df, hide_index=True, column_config=column_config)

        # Download a small sample for offline inspection
        sample_n = min(500, len(df))
        sample_df = df.sample(sample_n, random_state=random_seed) if sample_n < len(df) else df.copy()
        csv_bytes = sample_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Download Sample CSV ({sample_n:,} rows)",
            data=csv_bytes,
            file_name="compact_sample.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # -----------------------------
    # Enhanced Compact Dataset Builder with Memory Optimization
    # -----------------------------
    def build_optimized_compact_dataset(spark, df, remove_stopwords=True):
        """Build compact dataset with memory optimization and batch processing."""
        try:
            from pyspark.sql import functions as F
            from pyspark.ml.feature import RegexTokenizer, StopWordsRemover

            # Memory optimization: set shuffle partitions based on available memory
            original_partitions = spark.conf.get("spark.sql.shuffle.partitions", "200")
            try:
                import psutil
                memory_gb = psutil.virtual_memory().total / (1024**3)
                if memory_gb < 8:
                    spark.conf.set("spark.sql.shuffle.partitions", "24")
                elif memory_gb < 16:
                    spark.conf.set("spark.sql.shuffle.partitions", "48")
                else:
                    spark.conf.set("spark.sql.shuffle.partitions", "96")
            except Exception:
                spark.conf.set("spark.sql.shuffle.partitions", "48")

            # 0) Locate frozen base or fallback to df
            candidates = [
                Path(r"C:\Users\Saif\Desktop\CSE488\Project\notebook\work\data_clean\base_prothomalo"),
                Path.cwd() / "work" / "data_clean" / "base_prothomalo",
            ]
            df_base, src = None, None
            for p in candidates:
                if p.exists():
                    df_base = spark.read.parquet(str(p))
                    src = f"parquet:{p}"
                    break

            if df_base is None:
                required_cols = [c for c in ["doc_id", "text", "title_text", "category", "timestamp", "source", "uid"] if c in df.columns]
                batch_size = min(10000, len(df) // 4) if len(df) > 40000 else len(df)

                spark_dfs = []
                for i in range(0, len(df), batch_size):
                    batch_df = df[required_cols].iloc[i:i + batch_size]
                    spark_batch = spark.createDataFrame(batch_df)
                    spark_dfs.append(spark_batch)

                if len(spark_dfs) > 1:
                    df_base = spark_dfs[0]
                    for sdf_i in spark_dfs[1:]:
                        df_base = df_base.unionByName(sdf_i)
                else:
                    df_base = spark_dfs[0]
                src = "df (in-memory, batched)"

            st.info(f"Using base source: {src}")

            # 1) Regex-only normalization (no Python UDFs)
            def norm_col(c):
                c = F.regexp_replace(c, r"[\u200C\u200D]", "")             # strip ZWJ/ZWNJ
                c = F.regexp_replace(c, r"[^\p{L}\p{M}\p{Nd}]+", " ")       # keep letters/marks/digits
                c = F.trim(F.regexp_replace(c, r"\s+", " "))
                return c

            dfN = (
                df_base
                .withColumn("text_norm", norm_col(F.col("text")))
                .withColumn("title_norm", norm_col(F.col("title_text")))
                .select("doc_id", "category", "timestamp", "source", "text_norm", "title_norm")
                .cache()
            )
            with st.status("Normalizing text data..."):
                norm_count = dfN.count()
                st.write(f"Normalized {norm_count:,} documents")

            # 2) Tokenize (Bengali block) via JVM transformer
            rtok = RegexTokenizer(
                inputCol="text_norm", outputCol="tokens_base",
                pattern=r"[\u0980-\u09FF]+", gaps=False, toLowercase=False
            )
            with st.status("Tokenizing text..."):
                dfT = rtok.transform(dfN).cache()
                dfN.unpersist()
                token_count = dfT.count()
                st.write(f"Tokenized {token_count:,} documents")

            # 3) Remove Bangla stopwords if requested
            if remove_stopwords:
                LONG_STOPWORDS = """এবং কিন্তু কারণ তাই তাহলে যদি তবে শুধু খুব আরও আরো অথবা কিংবা তবু তবুও বরং নতুবা তথা অর্থাৎ সুতরাং অতএব যদিও যেহেতু সেহেতু ফলে নয় নয় না তো হ্যাঁ আমি আমরা তুমি তোমরা আপনি আপনারা সে তিনি তারা এরা ওরা কেউ কেউই কারো কারও সকল সবাই সমস্ত সব সবই সকলেই আমার আমাদের তোমার তোমাদের আপনার আপনাদের তার তাদের এর ওর এ এই ওই সেই এমন এমনই এমনকি এরকম ওরকম সেরকম এসব ওসব সেসব এটা এটি এগুলো এগুলি ওটা ওটি ওগুলো ওগুলি সেটা সেটি সেগুলো সেগুলি যে যা যার যাকে যাদের যারা যিনি যাঁরা যাঁকে যাঁর যেটা যেটি যেগুলো যেগুলি যাহা যাহার যাহাকে যাহাদের যদি যদিও যখন তখন যত তত যতটা ততটা যতক্ষণ ততক্ষণ যেমন তেমন যেভাবে সেভাবে যেখানে সেখানে এখানে ওখানে কোথায় কোথায় কোথা কোথাও কে কাকে কার কারা কাদের কী কি কেন কেননা কিভাবে কীভাবে কোন কোনো কোনটা কোনটি কোনদিকে কোনখানে কবে কখন কত কতটা কেমন আজ আজকে এখন বর্তমানে এখনও এখনো আগে পূর্বে পরে পরবর্তীতে পরবর্তী পূর্ববর্তী সদ্য সদ্যই আগামী গত গতকাল জন্য কারণে উপর উপরে উপরেই অধীনে অন্তর্গত মধ্যে ভিতরে ভেতরে বাইরে পাশে সামনে পেছনে দিকে পর্যন্ত সঙ্গে সাথে সম্পর্কে সম্বন্ধে বরাবর নিকটে কাছে দূরে ছাড়া ছাড়া বাদে ব্যতীত বিনা মাধ্যমে মধ্য দিয়ে মধ্য দিয়ে প্রতি প্রত্যেক প্রত্যেকে প্রত্যেকটি প্রত্যেকটা প্রায় প্রায় প্রায়ই প্রায়ই প্রায়শই প্রায়শই অধিকাংশ বিভিন্ন নানা নানান একাধিক কয়েক কয়েকটি কয়েকটা কিছু কিছুটা কিছুই কোনো কোনো অনেক অনেকটা অনেকগুলো বেশ বেশি কম অন্তত অন্ততপক্ষে কমপক্ষে ন্যূনতম সর্বোচ্চ মোট মোটেই মোটেও হয় হবে হন হচ্ছেন হচ্ছে হচ্ছি হচ্ছিল হয়েছে হয়নি হব হবেন হতে ছিলেন ছিল ছিলো ছিলাম ছিলে নেই নই থাকবে থাকবেন থাকি থাকে থাকেন থাকতে কর করা করি করে করছেন করেছি করেছেন করেছিল করেছে করতেন করত করতে করল করলেন করলাম করানো করাতে দেওয়া দেয় দেয়া দিয়েছে দিয়েছেন দিয়েছিল দিয়ে নিয়েছে নেয় নেওয়া নিয়ে নিতে পাওয়া পায় পেয়েছে পেয়েছিলেন পেতে পাবে পাবেন বল বলা বলে বলেন বলছেন বলেছেন বলেছিল বলেছিলাম বলবে বলবেন বলি দেখ দেখা দেখে দেখেন দেখেছেন দেখেছিল দেখাল দেখালো দেখিয়ে যাওয়া যায় গেছে যাচ্ছেন যাচ্ছি যাচ্ছিল যেতে আসা আসে এসেছে আসছেন আসছে আসতে থাকা থাকেন থাকি থাকে থেকেও থেকেছে থেকেছেন এমনকি আছে হয়েছে হলো হবেন ছিলেন গেল গেলো হয়েছে হতেই করবে করলেন করছিল করছিলেন করছিলাম বলা হয়েছিল জানাল জানিয়েছে জানিয়েছেন জানায় জানালো জানালেন জানিয়েছিল হচ্ছে হয়েছে জানায় জানায়েছিল জানাতেন দিয়েছিলাম জানাচ্ছে হয়েছে হলেন ডেস্ক সংবাদদাতা প্রতিনিধি বিজ্ঞপ্তি ভিডিও ছবি গ্রাফিক্স অনলাইন সংবাদ লাইভ আলোচনা এডিটর স্টাফ অ আ ই ঈ উ ঊ ঋ এ ঐ ও ঔ ক খ গ ঘ ঙ চ ছ জ ঝ ঞ ট ঠ ড ঢ ণ ত থ দ ধ ন প ফ বা ভ ম য র ল শ ষ স হ ড় ঢ় য় ৎ ং ঃ ঁ ০ ১ ২ ৩ ৪ ৫ ৬ ৭ ৮ ৯"""
                STOPWORDS_BN = sorted(set([w.strip() for w in LONG_STOPWORDS.split() if w.strip()]))

                from pyspark.ml.feature import StopWordsRemover
                with st.status(f"Removing stopwords ({len(STOPWORDS_BN)} words)..."):
                    rem = StopWordsRemover(inputCol="tokens_base", outputCol="tokens_nostop", stopWords=STOPWORDS_BN)
                    dfR = rem.transform(dfT).cache()
                    dfT.unpersist()
                    nostop_count = dfR.count()
                    st.write(f"Processed {nostop_count:,} documents")
            else:
                from pyspark.sql import functions as F  # ensure F exists
                dfR = dfT.withColumn("tokens_nostop", F.col("tokens_base")).cache()
                dfT.unpersist()

            # 4) Domain stoplist & length pruning using SQL higher-order functions
            from pyspark.sql import functions as F
            DOMAIN_STOPLIST = [
                "ডেস্ক", "সংবাদদাতা", "প্রতিনিধি", "বিজ্ঞপ্তি", "ভিডিও", "ছবি",
                "গ্রাফিক্স", "অনলাইন", "সংবাদ", "লাইভ", "আলোচনা", "এডিটর", "স্টাফ"
            ]
            stop_arr = F.array(*[F.lit(w) for w in DOMAIN_STOPLIST])

            MIN_TOK_LEN = 2
            MIN_DOC_TOKS = 3

            with st.status("Final processing and filtering..."):
                dfC = (
                    dfR
                    .withColumn(
                        "tokens_final",
                        F.filter(
                            F.col("tokens_nostop"),
                            lambda x: (F.length(x) >= MIN_TOK_LEN) & (~F.array_contains(stop_arr, x))
                        )
                    )
                    .withColumn("tok_len_final", F.size("tokens_final"))
                    .withColumn("is_amp", (F.locate("/ampstories/", F.col("doc_id")) > 0))
                    .filter(F.col("tok_len_final") >= MIN_DOC_TOKS)
                    .select(
                        "doc_id", "category", "timestamp", "source", "text_norm", "title_norm",
                        "tokens_base", "tokens_final", "tok_len_final", "is_amp"
                    )
                    .cache()
                )
                dfR.unpersist()
                final_count = dfC.count()
                st.write(f"Final dataset: {final_count:,} documents")

            # 5) Persist compact Parquet with memory-optimized settings
            base_out = Path(globals().get("PREPROC_DIR", Path.cwd() / "work" / "data_prep" / "prothomalo_prep"))
            out_dir = base_out / "compact_nostop_udffree"
            out_dir.parent.mkdir(parents=True, exist_ok=True)

            # Determine coalesce based on data size
            num_partitions = min(12, max(2, final_count // 50_000))

            with st.status(f"Writing to {out_dir}..."):
                (
                    dfC
                    .coalesce(num_partitions)
                    .write.mode("overwrite")
                    .option("compression", "snappy")
                    .parquet(str(out_dir))
                )

            # 6) Verify written data
            written_df = spark.read.parquet(str(out_dir))
            written_count = written_df.count()

            # Restore original partition settings
            spark.conf.set("spark.sql.shuffle.partitions", original_partitions)

            # Clear final cache
            dfC.unpersist()

            return True, written_count, out_dir

        except Exception as e:
            # Restore original partition settings on error
            try:
                spark.conf.set("spark.sql.shuffle.partitions", original_partitions)
            except Exception:
                pass
            raise e

    st.subheader("🔧 Enhanced Compact Dataset Builder")

    try:
        artifacts = get_artifact_status()
    except Exception:
        artifacts = {}

    compact_exists = bool(artifacts.get("compact", {}).get("exists", False))

    if compact_exists:
        st.success("✅ Compact dataset already exists.")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Rebuild Compact Dataset", use_container_width=True):
                try:
                    success, count, path = build_optimized_compact_dataset(spark, df, remove_stopwords=True)
                    if success:
                        st.success("✅ Compact dataset rebuilt successfully.")
                        st.info(f"📊 {count:,} documents saved to {path}")
                        st.rerun()
                    else:
                        st.warning("Rebuild completed, but reported no changes.")
                except Exception as e:
                    st.error(f"❌ Rebuild failed: {e}")

        with col2:
            if st.button("📊 Check Dataset Status", use_container_width=True):
                try:
                    base_out = Path(globals().get("PREPROC_DIR", Path.cwd() / "work" / "data_prep" / "prothomalo_prep"))
                    compact_path = base_out / "compact_nostop_udffree"
                    if compact_path.exists():
                        sdf_status = spark.read.parquet(str(compact_path))
                        cnt = sdf_status.count()
                        cols = sdf_status.columns
                        st.info(f"📂 Path: {compact_path}")
                        st.info(f"📊 Rows: {cnt:,} | Columns: {len(cols)}")
                        st.code(f"Columns: {', '.join(cols)}")
                    else:
                        st.warning("Compact dataset path not found.")
                except Exception as e:
                    st.error(f"Error checking status: {e}")
    else:
        st.warning("❌ Compact dataset not found. Build it to enable fast Spark operations.")
        col1, col2 = st.columns([1, 1])
        with col1:
            remove_stopwords = st.checkbox("Remove Stopwords", value=True)
        with col2:
            st.info("💡 Memory optimized with batch processing")

        if st.button("🏗️ Build Compact Dataset", use_container_width=True):
            try:
                success, count, path = build_optimized_compact_dataset(spark, df, remove_stopwords=remove_stopwords)
                if success:
                    st.success("✅ Compact dataset built successfully.")
                    st.info(f"📊 {count:,} documents saved to {path}")
                    st.balloons()
                    st.rerun()
                else:
                    st.warning("Build completed, but no output was produced.")
            except Exception as e:
                st.error(f"❌ Build failed: {e}")
                st.exception(e)

    # =============================
    # 🧪 EDA (Spark + Plotly, Bangla-friendly)
    # =============================
    st.subheader("🧪 EDA (Spark + Plotly, Bangla-friendly tokenizer)")

    # Heavy word cloud is optional (kept outside the expander)
    run_word_cloud = st.checkbox(
        "☁️ Generate Word Cloud (optional, slower — may install extra packages)",
        value=False
    )

    eda_exp = st.expander("📊 Show EDA Analysis", expanded=False)
    with eda_exp:
        try:
            from pyspark.sql import functions as F
            from pyspark.ml.feature import NGram

            # --- Common Plotly layout ---
            common_layout = dict(
                template="plotly_white",
                height=450,
                margin=dict(l=16, r=16, t=70, b=16),
                hovermode="x",
                uniformtext_minsize=10,
                uniformtext_mode="hide",
            )

            # --- Build an EDA Spark DataFrame (sample) ---
            try:
                sdf_eda = load_compact_dataset(spark, float(min(eda_fraction, 1.0)), int(random_seed))
            except Exception:
                sdf_eda = None

            if sdf_eda is None:
                base_cols = [c for c in ["doc_id", "text", "title_text", "category", "timestamp"] if c in df.columns]
                base_cols = base_cols or ["text"]
                pdf_sample = df[base_cols].sample(
                    frac=float(np.clip(eda_fraction, 0.10, 1.0)),
                    random_state=random_seed
                ) if len(df) > 5000 else df[base_cols]
                sdf_eda = spark.createDataFrame(pdf_sample)

            # Choose text column
            text_col = "text" if "text" in sdf_eda.columns else (
                "text_norm" if "text_norm" in sdf_eda.columns else None
            )
            if text_col is None:
                st.info("No text column found for tokenization. Skipping EDA charts.")
                return

            # --- Cached Bangla-friendly tokenizer (regex-only) ---
            @st.cache_resource(show_spinner=False)
            def get_bangla_tokenizer():
                def tokenize(colname: str):
                    c = F.col(colname)
                    c = F.regexp_replace(c, r"[\u200C\u200D]", "")                 # strip ZWJ/ZWNJ
                    c = F.lower(F.regexp_replace(c, r"[^\p{L}\p{M}\p{Nd}]+", " "))  # letters/marks/digits
                    c = F.trim(F.regexp_replace(c, r"\s+", " "))
                    return F.split(c, r"\s+")
                return tokenize

            tokenize = get_bangla_tokenizer()

            # Tokenize -> df_tok
            df_tok = (
                sdf_eda
                .withColumn("tokens", tokenize(text_col))
                .select(*(c for c in ["doc_id"] if c in sdf_eda.columns), "tokens")
                .cache()
            )
            _ = df_tok.count()

            # Preview tokens (first 5, truncated)
            rows = df_tok.select("tokens").limit(5).collect()
            st.caption("**Tokenizer preview (first 5):**")
            for i, r in enumerate(rows, 1):
                toks = r["tokens"][:20] if r["tokens"] else []
                preview = " ".join(toks)
                st.code(f"{i:02d}: {preview}{' …' if r['tokens'] and len(r['tokens']) > 20 else ''}")

            # --- E3: Basic stats + approx vocab ---
            if PLOTTING_AVAILABLE:
                n_docs = df_tok.count()
                vocab = (
                    df_tok
                    .select(F.explode("tokens").alias("tok"))
                    .filter(F.length("tok") >= 2)
                    .agg(F.approx_count_distinct("tok").alias("vocab"))
                    .first()["vocab"]
                )
                kpi = pd.DataFrame(
                    {"Metric": ["Documents (EDA)", "Approx Vocab (EDA)"], "Value": [n_docs, vocab]}
                )
                fig = go.Figure(go.Bar(
                    x=kpi["Metric"],
                    y=kpi["Value"],
                    text=[f"{v:,}" for v in kpi["Value"]],
                    textposition="outside",
                    marker=dict(color=["#1f77b4", "#ff7f0e"], line=dict(color="rgba(0,0,0,0.1)", width=1)),
                    hovertemplate="%{x}<br>Value=%{y:,}<extra></extra>"
                ))
                layout_safe = {k: v for k, v in common_layout.items()
                               if k not in ("title", "uniformtext_minsize", "uniformtext_mode")}
                fig.update_layout(**layout_safe, title_text="Basic Stats", xaxis_title="Metric", yaxis_title="Count")
                ymax = int(kpi["Value"].max())
                fig.update_yaxes(range=[0, max(1, int(ymax * 1.12))])
                fig.update_traces(cliponaxis=False)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Docs (EDA): {n_docs:,} • Approx vocabulary (EDA): {vocab:,}")

            # --- E4: Token-length histogram ---
            if PLOTTING_AVAILABLE:
                df_len = df_tok.select(F.size("tokens").alias("tok_len"))
                p10, p50, p90, p99 = df_len.approxQuantile("tok_len", [0.10, 0.50, 0.90, 0.99], 0.01)
                cap = max(50, int(p99))
                bins = 40
                bin_w = max(1, cap // bins)
                df_hist = (
                    df_len
                    .withColumn("bin", (F.col("tok_len") / bin_w).cast("int") * bin_w)
                    .filter(F.col("bin") <= cap)
                    .groupBy("bin").count()
                    .orderBy("bin")
                )
                pdf_hist = df_hist.toPandas()
                if len(pdf_hist):
                    pdf_hist = pdf_hist.sort_values("bin").copy()
                    b0 = pdf_hist["bin"].astype(int)
                    b1 = (pdf_hist["bin"] + bin_w - 1).astype(int)
                    pdf_hist["bin_label"] = b0.astype(str) + "–" + b1.astype(str)

                    fig = go.Figure(go.Bar(
                        x=pdf_hist["bin_label"],
                        y=pdf_hist["count"],
                        text=[f"{v:,}" for v in pdf_hist["count"]],
                        textposition="outside",
                        marker=dict(color=pdf_hist["count"], colorscale="Viridis",
                                    line=dict(color="rgba(0,0,0,0.1)", width=1)),
                        hovertemplate="Bin=%{x}<br>Docs=%{y:,}<extra></extra>"
                    ))
                    layout_safe = {k: v for k, v in common_layout.items()
                                   if k not in ("title", "xaxis", "uniformtext_minsize", "uniformtext_mode")}
                    fig.update_layout(**layout_safe,
                                      title_text="Histogram: Tokens per Document (EDA)",
                                      xaxis_title="Token count bin",
                                      yaxis_title="Documents")
                    ymax = int(pdf_hist["count"].max())
                    fig.update_yaxes(range=[0, max(1, int(ymax * 1.12))])
                    fig.update_xaxes(tickangle=90, automargin=True)
                    fig.update_traces(cliponaxis=False)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        f"Quantiles (EDA): p10={int(p10)}, p50={int(p50)}, p90={int(p90)}, "
                        f"p99={int(p99)} • cap={cap}, bin_w={bin_w}"
                    )

            # === NEW: Before vs After stopword removal — Top 20 tokens (two plots) ===
            if PLOTTING_AVAILABLE:
                # Stopword set (Bangla) — same as builder, kept local for EDA
                LONG_STOPWORDS = """এবং কিন্তু কারণ তাই তাহলে যদি তবে শুধু খুব আরও আরো অথবা কিংবা তবু তবুও বরং নতুবা তথা অর্থাৎ সুতরাং অতএব যদিও যেহেতু সেহেতু ফলে নয় নয় না তো হ্যাঁ আমি আমরা তুমি তোমরা আপনি আপনারা সে তিনি তারা এরা ওরা কেউ কেউই কারো কারও সকল সবাই সমস্ত সব সবই সকলেই আমার আমাদের তোমার তোমাদের আপনার আপনাদের তার তাদের এর ওর এ এই ওই সেই এমন এমনই এমনকি এরকম ওরকম সেরকম এসব ওসব সেসব এটা এটি এগুলো এগুলি ওটা ওটি ওগুলো ওগুলি সেটা সেটি সেগুলো সেগুলি যে যা যার যাকে যাদের যারা যিনি যাঁরা যাঁকে যাঁর যেটা যেটি যেগুলো যেগুলি যাহা যাহার যাহাকে যাহাদের যদি যদিও যখন তখন যত তত যতটা ততটা যতক্ষণ ততক্ষণ যেমন তেমন যেভাবে সেভাবে যেখানে সেখানে এখানে ওখানে কোথায় কোথায় কোথা কোথাও কে কাকে কার কারা কাদের কী কি কেন কেননা কিভাবে কীভাবে কোন কোনো কোনটা কোনটি কোনদিকে কোনখানে কবে কখন কত কতটা কেমন আজ আজকে এখন বর্তমানে এখনও এখনো আগে পূর্বে পরে পরবর্তীতে পরবর্তী পূর্ববর্তী সদ্য সদ্যই আগামী গত গতকাল জন্য কারণে উপর উপরে উপরেই অধীনে অন্তর্গত মধ্যে ভিতরে ভেতরে বাইরে পাশে সামনে পেছনে দিকে পর্যন্ত সঙ্গে সাথে সম্পর্কে সম্বন্ধে বরাবর নিকটে কাছে দূরে ছাড়া ছাড়া বাদে ব্যতীত বিনা মাধ্যমে মধ্য দিয়ে মধ্য দিয়ে প্রতি প্রত্যেক প্রত্যেকে প্রত্যেকটি প্রত্যেকটা প্রায় প্রায় প্রায়ই প্রায়ই প্রায়শই প্রায়শই অধিকাংশ বিভিন্ন নানা নানান একাধিক কয়েক কয়েকটি কয়েকটা কিছু কিছুটা কিছুই কোনো কোনো অনেক অনেকটা অনেকগুলো বেশ বেশি কম অন্তত অন্ততপক্ষে কমপক্ষে ন্যূনতম সর্বোচ্চ মোট মোটেই মোটেও হয় হবে হন হচ্ছেন হচ্ছে হচ্ছি হচ্ছিল হয়েছে হয়নি হব হবেন হতে ছিলেন ছিল ছিলো ছিলাম ছিলে নেই নই থাকবে থাকবেন থাকি থাকে থাকেন থাকতে কর করা করি করে করছেন করেছি করেছেন করেছিল করেছে করতেন করত করতে করল করলেন করলাম করানো করাতে দেওয়া দেয় দেয়া দিয়েছে দিয়েছেন দিয়েছিল দিয়ে নিয়েছে নেয় নেওয়া নিয়ে নিতে পাওয়া পায় পেয়েছে পেয়েছিলেন পেতে পাবে পাবেন বল বলা বলে বলেন বলছেন বলেছেন বলেছিল বলেছিলাম বলবে বলবেন বলি দেখ দেখা দেখে দেখেন দেখেছেন দেখেছিল দেখাল দেখালো দেখিয়ে যাওয়া যায় গেছে যাচ্ছেন যাচ্ছি যাচ্ছিল যেতে আসা আসে এসেছে আসছেন আসছে আসতে থাকা থাকেন থাকি থাকে থেকেও থেকেছে থেকেছেন এমনকি আছে হয়েছে হলো হবেন ছিলেন গেল গেলো হয়েছে হতেই করবে করলেন করছিল করছিলেন করছিলাম বলা হয়েছিল জানাল জানিয়েছে জানিয়েছেন জানায় জানালো জানালেন জানিয়েছিল হচ্ছে হয়েছে জানায় জানায়েছিল জানাতেন দিয়েছিলাম জানাচ্ছে হয়েছে হলেন ডেস্ক সংবাদদাতা প্রতিনিধি বিজ্ঞপ্তি ভিডিও ছবি গ্রাফিক্স অনলাইন সংবাদ লাইভ আলোচনা এডিটর স্টাফ অ আ ই ঈ উ ঊ ঋ এ ঐ ও ঔ ক খ গ ঘ ঙ চ ছ জ ঝ ঞ ট ঠ ড ঢ ণ ত থ দ ধ ন প ফ বা ভ ম য র ল শ ষ স হ ড় ঢ় য় ৎ ং ঃ ঁ ০ ১ ২ ৩ ৪ ৫ ৬ ৭ ৮ ৯"""
                STOPWORDS_BN = sorted(set([w.strip() for w in LONG_STOPWORDS.split() if w.strip()]))
                stop_arr = F.array(*[F.lit(w) for w in STOPWORDS_BN])

                # BEFORE: top-20 on raw tokens
                pre_top = (
                    df_tok.select(F.explode("tokens").alias("tok"))
                    .filter(F.length("tok") >= 2)
                    .groupBy("tok").count()
                    .orderBy(F.desc("count"))
                    .limit(20)
                ).toPandas()

                # AFTER: remove stopwords inline, then top-20
                df_tok_ns = df_tok.withColumn(
                    "tokens_ns",
                    F.filter(F.col("tokens"), lambda x: (F.length(x) >= 2) & (~F.array_contains(stop_arr, x)))
                )
                post_top = (
                    df_tok_ns.select(F.explode("tokens_ns").alias("tok"))
                    .groupBy("tok").count()
                    .orderBy(F.desc("count"))
                    .limit(20)
                ).toPandas()

                st.subheader("🔍 Before vs After Stopword Removal — Top 20 Tokens")
                c_pre, c_post = st.columns(2)

                if len(pre_top):
                    with c_pre:
                        fig = go.Figure(go.Bar(
                            x=pre_top["tok"], y=pre_top["count"],
                            text=[f"{v:,}" for v in pre_top["count"]],
                            textposition="outside",
                            marker=dict(color=pre_top["count"], colorscale="Blues",
                                        line=dict(color="rgba(0,0,0,0.1)", width=1)),
                            hovertemplate="Token=%{x}<br>Count=%{y:,}<extra></extra>"
                        ))
                        layout_safe = {k: v for k, v in common_layout.items()
                                       if k not in ("title", "uniformtext_minsize", "uniformtext_mode")}
                        fig.update_layout(**layout_safe,
                                          title_text="Top 20 — Before Stopwords",
                                          xaxis_title="Token", yaxis_title="Frequency")
                        ymax = int(pre_top["count"].max())
                        fig.update_yaxes(range=[0, max(1, int(ymax * 1.12))])
                        fig.update_traces(cliponaxis=False)
                        st.plotly_chart(fig, use_container_width=True)

                if len(post_top):
                    with c_post:
                        fig = go.Figure(go.Bar(
                            x=post_top["tok"], y=post_top["count"],
                            text=[f"{v:,}" for v in post_top["count"]],
                            textposition="outside",
                            marker=dict(color=post_top["count"], colorscale="Greens",
                                        line=dict(color="rgba(0,0,0,0.1)", width=1)),
                            hovertemplate="Token=%{x}<br>Count=%{y:,}<extra></extra>"
                        ))
                        layout_safe = {k: v for k, v in common_layout.items()
                                       if k not in ("title", "uniformtext_minsize", "uniformtext_mode")}
                        fig.update_layout(**layout_safe,
                                          title_text="Top 20 — After Stopwords",
                                          xaxis_title="Token", yaxis_title="Frequency")
                        ymax = int(post_top["count"].max())
                        fig.update_yaxes(range=[0, max(1, int(ymax * 1.12))])
                        fig.update_traces(cliponaxis=False)
                        st.plotly_chart(fig, use_container_width=True)

            # --- Time-based charts (if timestamp exists) ---
            if PLOTTING_AVAILABLE and "timestamp" in sdf_eda.columns:
                by_m = (
                    sdf_eda.filter(F.col("timestamp").isNotNull())
                    .withColumn("ym", F.date_format("timestamp", "yyyy-MM"))
                    .groupBy("ym").count().orderBy("ym")
                ).toPandas()
                if len(by_m):
                    by_m["ym_date"] = pd.to_datetime(by_m["ym"], format="%Y-%m")
                    fig = go.Figure(go.Scatter(
                        x=by_m["ym_date"], y=by_m["count"], mode="lines+markers",
                        hovertemplate="%{x|%Y-%m}<br>Docs=%{y:,}<extra></extra>"
                    ))
                    fig.update_layout(**common_layout,
                                      title_text="Documents per Month (EDA sample)",
                                      xaxis_title="Month (YYYY-MM)",
                                      yaxis_title="Documents")
                    st.plotly_chart(fig, use_container_width=True)

                by_d = (
                    sdf_eda.filter(F.col("timestamp").isNotNull())
                    .withColumn("ymd", F.date_format("timestamp", "yyyy-MM-dd"))
                    .groupBy("ymd").count().orderBy("ymd")
                ).toPandas()
                if len(by_d):
                    by_d["ymd_date"] = pd.to_datetime(by_d["ymd"], format="%Y-%m-%d")
                    fig = go.Figure(go.Scatter(
                        x=by_d["ymd_date"], y=by_d["count"], mode="lines+markers",
                        hovertemplate="%{x|%Y-%m-%d}<br>Docs=%{y:,}<extra></extra>"
                    ))
                    fig.update_layout(**common_layout,
                                      title_text="Documents per Day (EDA sample)",
                                      xaxis_title="Day (YYYY-MM-DD)",
                                      yaxis_title="Documents")
                    st.plotly_chart(fig, use_container_width=True)

            # --- Category × Month heatmap ---
            if PLOTTING_AVAILABLE and all(c in sdf_eda.columns for c in ["category", "timestamp"]):
                TOPK = 12
                top_cats = (
                    sdf_eda.groupBy("category").count()
                    .orderBy(F.desc("count")).limit(TOPK)
                    .toPandas()["category"].tolist()
                )
                by_cat_month = (
                    sdf_eda
                    .filter(F.col("timestamp").isNotNull() & F.col("category").isin(top_cats))
                    .withColumn("ym", F.date_format("timestamp", "yyyy-MM"))
                    .groupBy("ym", "category").count()
                ).toPandas()
                if len(by_cat_month):
                    by_cat_month["ym_date"] = pd.to_datetime(by_cat_month["ym"], format="%Y-%m")
                    cat_order = (
                        by_cat_month.groupby("category")["count"].sum()
                        .sort_values(ascending=False).index.tolist()
                    )
                    month_order = sorted(by_cat_month["ym_date"].unique())
                    mat = (
                        by_cat_month.pivot_table(
                            index="category", columns="ym_date", values="count", aggfunc="sum", fill_value=0
                        )
                        .reindex(index=cat_order, columns=month_order)
                    )
                    fig = go.Figure(go.Heatmap(
                        z=mat.values,
                        x=[d.strftime("%Y-%m") for d in mat.columns],
                        y=mat.index,
                        colorscale="Blues",
                        colorbar=dict(title="Docs"),
                        hovertemplate="Month=%{x}<br>Category=%{y}<br>Docs=%{z:,}<extra></extra>"
                    ))
                    layout_safe = {k: v for k, v in common_layout.items() if k != "title"}
                    fig.update_layout(**layout_safe,
                                      title_text="Heatmap: Category × Month (EDA sample)",
                                      xaxis_title="Month (YYYY-MM)",
                                      yaxis_title="Category")
                    st.plotly_chart(fig, use_container_width=True)

            # --- Top-20 unigrams/bigrams/trigrams (based on tokenized text) ---
            if PLOTTING_AVAILABLE:
                # Unigrams
                uni = (
                    df_tok.select(F.explode("tokens").alias("tok"))
                    .filter(F.length("tok") >= 2)
                    .groupBy("tok").count()
                    .orderBy(F.desc("count"))
                    .limit(20)
                ).toPandas()
                if len(uni):
                    fig = go.Figure(go.Bar(
                        x=uni["tok"], y=uni["count"],
                        text=[f"{v:,}" for v in uni["count"]],
                        textposition="outside",
                        marker=dict(color=uni["count"], colorscale="Blues",
                                    line=dict(color="rgba(0,0,0,0.1)", width=1)),
                        hovertemplate="Token=%{x}<br>Count=%{y:,}<extra></extra>"
                    ))
                    layout_safe = {k: v for k, v in common_layout.items()
                                   if k not in ("title", "uniformtext_minsize", "uniformtext_mode")}
                    fig.update_layout(**layout_safe,
                                      title_text="Top-20 Unigrams (EDA sample)",
                                      xaxis_title="Token", yaxis_title="Frequency")
                    ymax = int(uni["count"].max())
                    fig.update_yaxes(range=[0, max(1, int(ymax * 1.12))])
                    fig.update_traces(cliponaxis=False)
                    st.plotly_chart(fig, use_container_width=True)

                # Bigrams
                bigrams_df = NGram(n=2, inputCol="tokens", outputCol="bigrams").transform(df_tok)
                bi = (
                    bigrams_df.select(F.explode("bigrams").alias("gram"))
                    .groupBy("gram").count()
                    .orderBy(F.desc("count"))
                    .limit(20)
                ).toPandas().sort_values("count", ascending=True)
                if len(bi):
                    fig = go.Figure(go.Bar(
                        x=bi["count"], y=bi["gram"], orientation="h",
                        text=[f"{v:,}" for v in bi["count"]],
                        textposition="outside",
                        marker=dict(color=bi["count"], colorscale="YlGnBu",
                                    line=dict(color="rgba(0,0,0,0.1)", width=1)),
                        hovertemplate="Bigram=%{y}<br>Count=%{x:,}<extra></extra>"
                    ))
                    layout_safe = {k: v for k, v in common_layout.items()
                                   if k not in ("title", "uniformtext_minsize", "uniformtext_mode")}
                    fig.update_layout(**layout_safe,
                                      title_text="Top-20 Bigrams (EDA sample)",
                                      xaxis_title="Frequency", yaxis_title="Bigram")
                    fig.update_yaxes(autorange="reversed")
                    fig.update_traces(cliponaxis=False)
                    st.plotly_chart(fig, use_container_width=True)

                # Trigrams
                trigrams_df = NGram(n=3, inputCol="tokens", outputCol="trigrams").transform(df_tok)
                tri = (
                    trigrams_df.select(F.explode("trigrams").alias("gram"))
                    .groupBy("gram").count()
                    .orderBy(F.desc("count"))
                    .limit(20)
                ).toPandas().sort_values("count", ascending=True)
                if len(tri):
                    fig = go.Figure(go.Bar(
                        x=tri["count"], y=tri["gram"], orientation="h",
                        text=[f"{v:,}" for v in tri["count"]],
                        textposition="outside",
                        marker=dict(color=tri["count"], colorscale="Cividis",
                                    line=dict(color="rgba(0,0,0,0.1)", width=1)),
                        hovertemplate="Trigram=%{y}<br>Count=%{x:,}<extra></extra>"
                    ))
                    layout_safe = {k: v for k, v in common_layout.items()
                                   if k not in ("title", "uniformtext_minsize", "uniformtext_mode")}
                    fig.update_layout(**layout_safe,
                                      title_text="Top-20 Trigrams (EDA sample)",
                                      xaxis_title="Frequency", yaxis_title="Trigram")
                    fig.update_yaxes(autorange="reversed")
                    fig.update_traces(cliponaxis=False)
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.warning(f"EDA section skipped due to error: {e}")

    # --- Optional Word Cloud (outside expander) ---
    if run_word_cloud:
        try:
            from pyspark.sql import functions as F
            # Reuse df_tok if it exists; else build quickly
            try:
                df_tok  # type: ignore
            except NameError:
                try:
                    sdf_eda = load_compact_dataset(spark, float(min(eda_fraction, 1.0)), int(random_seed))
                except Exception:
                    sdf_eda = None
                if sdf_eda is None:
                    base_cols = [c for c in ["doc_id", "text"] if c in df.columns] or ["text"]
                    pdf_sample = df[base_cols].sample(
                        frac=float(np.clip(eda_fraction, 0.10, 1.0)), random_state=random_seed
                    ) if len(df) > 5000 else df[base_cols]
                    sdf_eda = spark.createDataFrame(pdf_sample)

                @st.cache_resource(show_spinner=False)
                def get_bangla_tokenizer():
                    def tokenize(colname: str):
                        c = F.col(colname)
                        c = F.regexp_replace(c, r"[\u200C\u200D]", "")
                        c = F.lower(F.regexp_replace(c, r"[^\p{L}\p{M}\p{Nd}]+", " "))
                        c = F.trim(F.regexp_replace(c, r"\s+", " "))
                        return F.split(c, r"\s+")
                    return tokenize

                tokenize = get_bangla_tokenizer()
                text_col_wc = "text" if "text" in sdf_eda.columns else "text_norm"
                df_tok = sdf_eda.withColumn("tokens", tokenize(text_col_wc)).select("tokens").cache()
                _ = df_tok.count()

            # Try to render a simple word cloud via pyecharts (optional install)
            import sys, subprocess
            from pathlib import Path as _Path
            try:
                from pyecharts.charts import WordCloud
                from pyecharts import options as opts
                from pyecharts.render import make_snapshot
            except Exception:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyecharts>=2.0.0"], check=False)
                from pyecharts.charts import WordCloud
                from pyecharts import options as opts
                from pyecharts.render import make_snapshot
            try:
                from snapshot_selenium import snapshot as driver
            except Exception:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "snapshot-selenium"], check=False)
                from snapshot_selenium import snapshot as driver

            TOP_N = 1000
            pairs = (
                df_tok.select(F.explode("tokens").alias("tok"))
                .groupBy("tok").count()
                .orderBy(F.desc("count"))
                .limit(TOP_N)
                .rdd.map(lambda r: (r["tok"], int(r["count"])))
                .collect()
            )
            if pairs:
                wc = (
                    WordCloud()
                    .add(series_name="freq", data_pair=pairs)
                    .set_global_opts(
                        title_opts=opts.TitleOpts(title="Word Cloud — Global (EDA)"),
                        tooltip_opts=opts.TooltipOpts(is_show=False),
                    )
                )
                out_dir = _Path.cwd() / "work" / "eda_figs"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_png = out_dir / "wordcloud_global.png"
                make_snapshot(driver, wc.render(), str(out_png))
                st.image(str(out_png), caption=f"Word Cloud — saved to {out_png}", use_column_width=True)
            else:
                st.info("No tokens available to render a word cloud.")
        except Exception as e:
            st.warning(f"Word Cloud generation skipped: {e}")




@st.cache_resource(show_spinner=False)
def get_bangla_font_path() -> Path | None:
    """
    Find a Bangla-capable TTF on the system; if none found,
    download kalpurush.ttf into WORK_DIR/fonts and return that path.
    """
    # 1) Try system fonts
    wanted = ("kalpurush", "solaiman", "bengali", "siyam")
    for fpath in font_manager.findSystemFonts(fontext="ttf"):
        base = os.path.basename(fpath).lower()
        if any(w in base for w in wanted):
            return Path(fpath)

    # 2) Download kalpurush.ttf locally (if not present)
    fonts_dir = WORK_DIR / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    target = fonts_dir / "kalpurush.ttf"
    if not target.exists():
        try:
            url = "https://raw.githubusercontent.com/saifkhancse/prothomalo_article_scraper/main/kalpurush.ttf"
            urllib.request.urlretrieve(url, str(target))
        except Exception as e:
            st.warning(f"Could not download Bangla font automatically: {e}")
            return None
    return target
def show_text_analytics_tab(spark):
    """Show text analytics and n-gram analysis"""
    st.header("📈 Text Analytics")
    
    config = st.session_state.config
    
    # Load compact dataset
    sdf = load_compact_dataset(spark, config['dataset_fraction'], config['random_seed'])
    
    if sdf is None:
        st.error("❌ No compact dataset available. Please build it first.")
        return
    
    # Controls
    st.subheader("🎛️ Analysis Controls")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        n = st.selectbox("N-gram Size", [1, 2, 3], index=1)
    
    with col2:
        top_n = st.slider("Top N Terms", 10, 200, 50)
    
    with col3:
        eda_fraction = st.slider(
        "EDA Fraction",
        min_value=0.01, max_value=0.5, value=float(config['eda_fraction']),
        step=0.01,
        key="ta_eda_fraction")
    with col4:
        recompute = st.checkbox("Force Recompute", False)
    
    # N-gram analysis
    st.subheader("🔤 N-gram Analysis")
    
    if st.button("📊 Analyze N-grams", key="ta_btn_analyze_ngrams") or not recompute:
        with st.status("Analyzing n-grams...") as status:
            try:
                # Sample for analysis
                analysis_sdf = sdf.sample(fraction=eda_fraction, seed=config['random_seed'])

                # Generate n-grams
                status.update(label="Generating n-grams...")
                ngram_transformer = NGram(n=int(n), inputCol="tokens_final", outputCol="ngrams")
                ngram_sdf = ngram_transformer.transform(analysis_sdf)

                # Explode and count n-grams
                status.update(label="Counting n-gram frequencies...")
                ngram_counts = (
                    ngram_sdf
                    .select(explode("ngrams").alias("ngram"))
                    .groupBy("ngram").count()
                    .orderBy(F.desc("count"))
                    .limit(int(top_n))
                )

                # Convert to pandas for visualization + reuse
                ngram_df = ngram_counts.toPandas()

                if not ngram_df.empty:
                    # persist for word cloud reuse
                    st.session_state["ngram_freqs"] = dict(ngram_df.set_index("ngram")["count"])
                    st.session_state["ngram_n"] = int(n)
                    st.session_state["ngram_top_n"] = int(top_n)

                    # Bar chart
                    if PLOTTING_AVAILABLE:
                        fig = px.bar(
                            ngram_df.head(20),
                            x='count', y='ngram',
                            orientation='h',
                            title=f"Top 20 {n}-grams"
                        )
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)

                    # Data table
                    st.dataframe(ngram_df, hide_index=True)

                    # Save results
                    status.update(label="Saving results...")
                    ngram_dir = WORK_DIR / "data_results" / "ngram_analysis"
                    ngram_dir.mkdir(parents=True, exist_ok=True)
                    ngram_file = ngram_dir / f"ngrams_{n}gram_top{top_n}.csv"
                    ngram_df.to_csv(ngram_file, index=False)

                    # Download button
                    csv_data = ngram_df.to_csv(index=False)
                    st.download_button(
                        label=f"📥 Download {n}-gram CSV",
                        data=csv_data,
                        file_name=f"ngrams_{n}gram_top{top_n}.csv",
                        mime="text/csv",
                        key="ta_dl_ngrams_csv"
                    )

                    status.update(label=f"✅ Analysis complete! Saved to {ngram_file}", state="complete")
                else:
                    st.warning("No n-grams found in the dataset.")
            except Exception as e:
                st.error(f"Error during n-gram analysis: {e}")


# === Word cloud (pyecharts -> PNG via snapshot) ===
# === Word cloud (interactive via pyecharts) ===
    st.subheader("☁️ Word Cloud (Interactive) — White Background")



    # Controls
    min_token_len = st.slider("Min token length", 1, 8, 2, step=1, key="ta_wc_minlen")
    min_freq      = st.slider("Min frequency", 1, 100, 2, step=1, key="ta_wc_minfreq")

    ngram_n_for_label = st.session_state.get("ngram_n", None)
    ngram_option_label = f"{ngram_n_for_label}-grams (from last analysis)" if ngram_n_for_label else "N-grams (run analysis first)"
    wc_source = st.radio(
        "Word cloud source",
        options=["Tokens (unigram)", ngram_option_label],
        index=0 if "ngram_freqs" not in st.session_state else 1,
        key="ta_wc_source"
    )

    if st.button("🎨 Generate Word Cloud (Interactive)", key="ta_wc_btn_html"):
        with st.status("Generating interactive word cloud…") as status:
            try:
                # -------- Build frequencies dict --------
                if wc_source.startswith("Tokens"):
                    status.update(label="Collecting tokens…")
                    analysis_sdf = sdf.sample(fraction=min(0.1, eda_fraction), seed=config['random_seed'])
                    tokens_df = analysis_sdf.select(F.explode("tokens_final").alias("token")).toPandas()
                    if tokens_df.empty:
                        st.warning("No tokens found for word cloud generation.")
                        st.stop()

                    s = tokens_df["token"].astype(str).str.strip()
                    s = s[s.str.len() >= int(min_token_len)]
                    freqs = s.value_counts()
                    freqs = freqs[freqs >= int(min_freq)]
                    if freqs.empty:
                        st.warning("No tokens after filters. Try lowering the thresholds.")
                        st.stop()

                    freq_dict = freqs.to_dict()
                    fig_fname = "wordcloud_tokens_final.html"
                else:
                    if "ngram_freqs" not in st.session_state:
                        st.warning("No n-gram analysis found. Run 'Analyze N-grams' first.")
                        st.stop()

                    status.update(label="Using last n-gram analysis…")
                    series = pd.Series(st.session_state["ngram_freqs"], dtype="int64")
                    idx = pd.Index(series.index.astype(str))
                    mask = (idx.str.len() >= int(min_token_len)) & (series >= int(min_freq))
                    filtered = series[mask].sort_values(ascending=False)
                    if filtered.empty:
                        st.warning("No n-grams after filters. Adjust thresholds or re-run analysis.")
                        st.stop()

                    freq_dict = filtered.to_dict()
                    n_for_name = st.session_state.get("ngram_n", "n")
                    fig_fname = f"wordcloud_{n_for_name}gram.html"

                # Convert to (word, freq) pairs (cap to keep rendering snappy)
                pairs = [(str(k), int(v)) for k, v in freq_dict.items()]
                pairs = pairs[:1000]

                # -------- Ensure pyecharts --------
                status.update(label="Preparing pyecharts…")
                try:
                    from pyecharts.charts import WordCloud as ECWordCloud
                    from pyecharts import options as opts
                except Exception:
                    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyecharts>=2.0.0"], check=False)
                    from pyecharts.charts import WordCloud as ECWordCloud
                    from pyecharts import options as opts

                # -------- Embed Bangla font (base64) + white background --------
                status.update(label="Embedding Bangla font & building chart…")
                bn_font = get_bangla_font_path()
                font_css = ""
                font_family = "sans-serif"
                if bn_font and Path(bn_font).exists():
                    font_b64 = base64.b64encode(Path(bn_font).read_bytes()).decode("ascii")
                    font_css = f"""
                    <style>
                    @font-face {{
                        font-family: 'Kalpurush';
                        src: url(data:font/ttf;base64,{font_b64}) format('truetype');
                        font-weight: normal; font-style: normal;
                    }}
                    body {{ background:#ffffff; }}
                    .chart-container {{ background:#ffffff; font-family: 'Kalpurush','Noto Sans Bengali',sans-serif; }}
                    </style>
                    """
                    font_family = "Kalpurush"
                else:
                    font_css = """
                    <style>
                    body { background:#ffffff; }
                    .chart-container { background:#ffffff; font-family: 'Noto Sans Bengali',sans-serif; }
                    </style>
                    """

                wc = (
                    ECWordCloud(init_opts=opts.InitOpts(width="1000px", height="600px", bg_color="#ffffff"))
                    .add(
                        series_name="freq",
                        data_pair=pairs,
                        word_size_range=[12, 80],
                        textstyle_opts=opts.TextStyleOpts(font_family=font_family),
                    )
                    .set_global_opts(
                        title_opts=opts.TitleOpts(title="Word Cloud — EDA"),
                        tooltip_opts=opts.TooltipOpts(is_show=True),
                    )
                )

                # Render as HTML snippet and inject CSS; show interactively
                html_snippet = wc.render_embed()
                full_html = f"<!DOCTYPE html><html><head><meta charset='utf-8'>{font_css}</head>" \
                            f"<body><div class='chart-container'>{html_snippet}</div></body></html>"

                st.components.v1.html(full_html, height=620, scrolling=False)

                # Optional: save the interactive HTML
                status.update(label="Saving interactive HTML…")
                out_dir = WORK_DIR / "eda_figs"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_html = out_dir / fig_fname
                out_html.write_text(full_html, encoding="utf-8")

                status.update(label=f"✅ Saved interactive HTML → {out_html}", state="complete")

            except Exception as e:
                st.error(f"Error generating word cloud: {e}")

from pyspark.ml.linalg import Vectors
import sys, base64, subprocess, pandas as pd
from pathlib import Path
from pyspark.sql import functions as F
import os, sys, base64, subprocess
import pandas as pd
from pathlib import Path
from pyspark.sql import functions as F
from pyspark.ml.feature import NGram
from pyspark.sql import functions as F
from pyspark.sql.functions import explode
def show_document_search_tab(spark):
    """Bangla doc search: Fallback implementation without PySpark ML"""
    import re
    import streamlit as st
    from pyspark.sql import functions as F
    from collections import Counter
    import numpy as np
    import pandas as pd
    
    st.header("🔍 Document Search")
    # st.info("Using basic text matching (PySpark ML libraries not available)")

    # ---- config defaults ----
    cfg = st.session_state.config
    cfg.setdefault('dataset_fraction', 1.0)
    cfg.setdefault('random_seed', 42)
    cfg.setdefault('top_k_search', 20)
    cfg.setdefault('min_cosine', 0.10)

    # Load dataset
    sdf = load_compact_dataset(spark, cfg.get('dataset_fraction', 1.0), cfg.get('random_seed', 42))
    if sdf is None:
        st.error("❌ No compact dataset available. Please build it first.")
        return

    # Ensure required columns exist
    needed = {"doc_id", "tokens_final", "text_norm"}
    missing = needed - set(sdf.columns)
    if missing:
        st.error(f"Compact dataset missing columns: {sorted(missing)}")
        return

    # Bangla tokenization (same as before)
    _bn_nonword = re.compile(r"[^\u0980-\u09FFA-Za-z0-9]+")
    _BN_SUFFIXES = (
        "দেরও","গুলোই","গুলোর","গুলিতে","গুলোরই","গুলা","গুলি","গুলো","টির","তে","তো","টা","টি","তোঁ","দের",
        "দেরই","য়ের","য়েই","য়","রা","র","কে","কো","কেই","খানা","খানি","হতে","গুলির","গুলাতে","দেরকে"
    )
    BN_STOP = {
        "এবং","করে","কেন","কি","কী","কেউ","কিছু","কোন","কোনো","এই","ওই","সেই","তা","তাতে","তার","তাদের",
        "বা","আর","তবে","কিন্তু","তাই","যে","যদি","যেতে","হয়","হবে","ছিল","ছিলেন","থেকে","দিকে","জন্য","তারপর",
        "এক","একটি","দিয়ে","নিয়ে","তিনি","তারা","আমরা","আপনি","আপনার",
        "জানুয়ারি","ফেব্রুয়ারি","মার্চ","এপ্রিল","মে","জুন","জুলাই","আগস্ট","সেপ্টেম্বর","অক্টোবর","নভেম্বর","ডিসেম্বর",
        "সোমবার","মঙ্গলবার","বুধবার","বৃহস্পতিবার","শুক্রবার","শনিবার","রবিবার","আজ","গতকাল","আগামী"
    }

    def tokenize_bn(text):
        """Tokenize Bengali text with suffix stripping and stopword removal"""
        s = (text or "").lower().strip()
        s = _bn_nonword.sub(" ", s)
        toks = [t for t in s.split() if t]
        
        def strip_suffix(w):
            for suf in _BN_SUFFIXES:
                if w.endswith(suf) and len(w) - len(suf) >= 3:
                    return w[:-len(suf)]
            return w
        
        toks = [strip_suffix(t) for t in toks]
        toks = [t for t in toks if t not in BN_STOP and len(t) >= 2]
        return toks

    # ---- Spark stability (Windows, UDF tracebacks, partitions) ----
    try:
        spark.conf.set("spark.python.worker.faulthandler.enabled", "true")
        spark.conf.set("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
        if int(spark.conf.get("spark.sql.shuffle.partitions")) > 64:
            spark.conf.set("spark.sql.shuffle.partitions", "64")
    except Exception:
        pass

    # UI Controls
    c1, c2 = st.columns(2)
    with c1:
        qtext = st.text_area("Search Query", placeholder="Enter Bangla text…", key="fallback_q_text")
        top_k = st.slider("Top K", 5, 100, int(cfg.get('top_k_search', 20)), key="fallback_topk")
        min_matches = st.slider("Min token overlap", 1, 10, 2, key="fallback_minmatch")
    with c2:
        match_threshold = st.slider("Match Score Threshold", 0.0, 1.0, float(cfg.get('min_cosine', 0.1)), step=0.05, key="fallback_threshold")
        use_fuzzy = st.checkbox("Fuzzy matching", False, key="fallback_fuzzy")
        exact_phrase = st.checkbox("Exact phrase matching", False, key="fallback_exact")
    
    # Category filter
    @st.cache_data(show_spinner=False)
    def get_categories(_sdf):
        try:
            return [r[0] for r in (_sdf.select("category")
                                  .where(F.col("category").isNotNull())
                                  .distinct().orderBy("category").collect())]
        except Exception:
            return []
    
    cat_opts = get_categories(sdf)
    cat_sel = st.multiselect("Filter by category (optional)", options=cat_opts, default=[], key="fallback_cats")

    # Search statistics
    if st.checkbox("Show search statistics", False, key="fallback_stats"):
        try:
            total_docs = sdf.count()
            unique_cats = len(cat_opts)
            avg_tokens = sdf.agg(F.avg(F.size("tokens_final")).alias("avg")).collect()[0]["avg"]
            st.info(f"📊 Dataset: {total_docs:,} documents, {unique_cats} categories, avg {avg_tokens:.1f} tokens/doc")
        except Exception:
            pass

    if st.button("🔎 Search", key="fallback_search_btn"):
        if not qtext.strip():
            st.warning("Please enter a search query")
            return
            
        try:
            with st.status("Searching...") as status:
                # Tokenize query
                q_tokens = tokenize_bn(qtext)
                if not q_tokens:
                    st.warning("No valid tokens in query")
                    return
                
                status.update(label=f"Processing query: {len(q_tokens)} tokens")
                
                # Basic filtering
                search_df = sdf
                if cat_sel:
                    search_df = search_df.where(F.col("category").isin(cat_sel))
                    status.update(label=f"Filtered to {len(cat_sel)} categories")
                
                # Different matching strategies
                if exact_phrase:
                    # Exact phrase matching (simple substring search)
                    phrase = " ".join(q_tokens)
                    matched_df = (search_df
                        .where(F.col("text_norm").contains(phrase))
                        .withColumn("match_score", F.lit(1.0))
                        .withColumn("token_matches", F.lit(len(q_tokens)))
                    )
                else:
                    # Token overlap calculation
                    q_tokens_lit = F.array(*[F.lit(t) for t in q_tokens])
                    
                    # Calculate matches with improved scoring
                    matched_df = (search_df
                        .withColumn("token_matches", 
                            F.size(F.array_intersect(F.col("tokens_final"), q_tokens_lit)))
                        .where(F.col("token_matches") >= min_matches)
                        .withColumn("total_tokens", F.size(F.col("tokens_final")))
                        .withColumn("query_tokens", F.lit(len(q_tokens)))
                    )
                    
                    # Improved scoring: combination of precision and recall
                    matched_df = matched_df.withColumn("match_score", 
                        # Harmonic mean of precision and recall (F1-like score)
                        (2.0 * F.col("token_matches")) / 
                        (F.col("total_tokens") + F.col("query_tokens"))
                    ).where(F.col("match_score") >= match_threshold)
                
                # Add fuzzy matching if enabled
                if use_fuzzy and not exact_phrase:
                    status.update(label="Adding fuzzy matches...")
                    # Simple fuzzy: allow 1-character differences in tokens
                    fuzzy_conditions = []
                    for token in q_tokens[:5]:  # Limit to first 5 tokens for performance
                        if len(token) > 3:  # Only for longer tokens
                            # Create variations (simple approach)
                            variations = [
                                token[1:],  # Remove first char
                                token[:-1], # Remove last char
                                token[0] + token[2:] if len(token) > 2 else token,  # Remove second char
                            ]
                            for var in variations:
                                if len(var) >= 2:
                                    fuzzy_conditions.append(F.array_contains("tokens_final", var))
                    
                    if fuzzy_conditions:
                        fuzzy_condition = fuzzy_conditions[0]
                        for cond in fuzzy_conditions[1:]:
                            fuzzy_condition = fuzzy_condition | cond
                        
                        fuzzy_df = (search_df
                            .where(fuzzy_condition)
                            .withColumn("match_score", F.lit(0.5))  # Lower score for fuzzy
                            .withColumn("token_matches", F.lit(1))
                        )
                        
                        # Union with exact matches
                        matched_df = matched_df.unionByName(fuzzy_df, allowMissingColumns=True)
                
                # Add text preview and prepare results
                result_df = (matched_df
                    .select("doc_id", "category", "match_score", "token_matches",
                           F.substring("text_norm", 1, 200).alias("preview"))
                    .dropDuplicates(["doc_id"])  # Remove duplicates from fuzzy matching
                    .orderBy(F.desc("match_score"), F.desc("token_matches"))
                    .limit(top_k)
                )
                
                status.update(label="Converting results...")
                
                # Convert to pandas for display
                try:
                    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
                    results_pdf = result_df.toPandas()
                except Exception:
                    # Fallback if toPandas fails
                    results_list = result_df.collect()
                    if results_list:
                        results_pdf = pd.DataFrame([row.asDict() for row in results_list])
                    else:
                        results_pdf = pd.DataFrame()
                
                if results_pdf.empty:
                    st.warning("No documents found matching your criteria")
                    if q_tokens:
                        st.info(f"Search tokens: {', '.join(q_tokens)}")
                        st.info("Try reducing the minimum token overlap or match threshold")
                    return
                
                status.update(label="Formatting results...")
                
                # Add ranking and clean up
                results_pdf.insert(0, "rank", range(1, len(results_pdf) + 1))
                
                # Clean preview text
                results_pdf['preview'] = results_pdf['preview'].str.replace('\n', ' ').str.replace('\r', '')
                results_pdf['preview'] = results_pdf['preview'].str[:200] + '...'
                
                # Display results
                st.subheader(f"Found {len(results_pdf)} results")
                
                # Show query info
                if st.checkbox("Show query info", False, key="fallback_show_qinfo"):
                    info_box = st.container()
                    with info_box:
                        st.markdown(f"**Original query:** {qtext}")
                        st.markdown(f"**Processed tokens:** {', '.join(q_tokens)}")
                        st.markdown(f"**Search mode:** {'Exact phrase' if exact_phrase else 'Token overlap'}")
                        if use_fuzzy:
                            st.markdown("**Fuzzy matching:** Enabled")

                st.dataframe(
                    results_pdf,
                    hide_index=True,
                    column_config={
                        'rank': st.column_config.NumberColumn('Rank', format="%d"),
                        'doc_id': st.column_config.TextColumn('Doc ID', width=120),
                        'category': st.column_config.TextColumn('Category', width=100),
                        'match_score': st.column_config.NumberColumn('Score', format="%.3f"),
                        'token_matches': st.column_config.NumberColumn('Matches', format="%d"),
                        'preview': st.column_config.TextColumn('Preview', width=400),
                    },
                    use_container_width=True
                )
                
                # Download option
                st.download_button(
                    label="📥 Download Results",
                    data=results_pdf.to_csv(index=False),
                    file_name=f"search_results_{len(results_pdf)}_docs.csv",
                    mime="text/csv",
                    key="fallback_download"
                )
                
                # Show additional statistics
                if len(results_pdf) > 0:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        avg_score = results_pdf['match_score'].mean()
                        st.metric("Avg Score", f"{avg_score:.3f}")
                    with col2:
                        total_matches = results_pdf['token_matches'].sum()
                        st.metric("Total Token Matches", total_matches)
                    with col3:
                        unique_categories = results_pdf['category'].nunique()
                        st.metric("Categories Found", unique_categories)
                
                status.update(label=f"✅ Found {len(results_pdf)} results", state="complete")
                
        except Exception as e:
            st.error(f"Search error: {e}")
            import traceback
            st.code(traceback.format_exc())
# Use this function instead of the original if ML libraries are not available
# show_document_search_tab_fallback(spark)
from pyspark.sql.functions import count, collect_list



def show_clustering_tab(spark):
    """Memory-optimized clustering analysis with robust error handling (KMeans + user-entered PCA/params via a single form submit)"""
    import streamlit as st
    import pandas as pd
    import numpy as np

    from pyspark.sql import functions as F, types as T
    from pyspark.sql.functions import col, count

    st.header("🎯 Clustering Analysis")

    # ---------- Safe Spark conf setter ----------
    def _set_conf(key, value):
        try:
            spark.conf.set(key, value)
            return True
        except Exception:
            return False

    # ---------- Apply memory-lean Spark settings (skip disallowed ones) ----------
    applied = []
    if _set_conf("spark.sql.parquet.enableVectorizedReader", "false"): applied.append("parquet vectorized reader=off")
    if _set_conf("spark.sql.orc.enableVectorizedReader", "false"): applied.append("orc vectorized reader=off")
    if _set_conf("spark.sql.parquet.columnarReaderBatchSize", "1024"): applied.append("parquet batch=1024")
    if _set_conf("spark.sql.orc.columnarReaderBatchSize", "1024"): applied.append("orc batch=1024")
    if _set_conf("spark.sql.execution.arrow.pyspark.enabled", "false"): applied.append("arrow=off")
    if _set_conf("spark.sql.adaptive.enabled", "true"): applied.append("AQE=on")
    if _set_conf("spark.sql.adaptive.coalescePartitions.enabled", "true"): applied.append("AQE coalesce=on")
    if _set_conf("spark.sql.shuffle.partitions", "32"): applied.append("shuffle.partitions=32")
    # NOTE: Don't try to change spark.serializer at runtime (Spark blocks it).

    if applied:
        st.success("✅ Applied Spark optimizations: " + ", ".join(applied))
    else:
        st.info("No Spark optimizations were applied (already set or restricted).")

    # ---------- Imports / availability ----------
    try:
        from pyspark.ml.feature import HashingTF, IDF, Normalizer  # (use sklearn for PCA)
        from pyspark.ml.clustering import KMeans
    except ImportError:
        st.error("❌ PySpark ML libraries not available. Install with: `pip install pyspark[sql,ml]`")
        return

    try:
        from sklearn.decomposition import PCA as SklearnPCA
        from sklearn.metrics import silhouette_score as _silhouette_score
    except Exception:
        SklearnPCA = None
        _silhouette_score = None

    try:
        import plotly.express as px
        import plotly.graph_objects as go
        plotting_available = True
    except ImportError:
        plotting_available = False

    # ---------- Safe vector_to_array wrapper (works across Spark builds) ----------
    try:
        from pyspark.ml.functions import vector_to_array as _v2a
        def v2a_safe(col_):
            return _v2a(col_)
    except Exception:
        v2a_udf = F.udf(lambda v: (v.toArray().tolist() if v is not None else None),
                        T.ArrayType(T.DoubleType()))
        def v2a_safe(col_):
            return v2a_udf(col_)

    # ---------- Helpers to parse free-text inputs ----------
    def _to_int(val, default, min_v=None, max_v=None, name="value"):
        try:
            v = int(val)
            if min_v is not None and v < min_v:
                st.warning(f"{name} raised to minimum {min_v}.")
                v = min_v
            if max_v is not None and v > max_v:
                st.warning(f"{name} lowered to maximum {max_v}.")
                v = max_v
            return v
        except Exception:
            st.warning(f"Invalid {name}; using default {default}.")
            return default

    def _to_float(val, default, min_v=None, max_v=None, name="value"):
        try:
            v = float(val)
            if min_v is not None and v < min_v:
                st.warning(f"{name} raised to minimum {min_v}.")
                v = min_v
            if max_v is not None and v > max_v:
                st.warning(f"{name} lowered to maximum {max_v}.")
                v = max_v
            return v
        except Exception:
            st.warning(f"Invalid {name}; using default {default}.")
            return default

    def _to_bool(val, default, name="value"):
        if isinstance(val, bool):
            return val
        s = str(val).strip().lower()
        if s in {"1", "true", "t", "yes", "y"}:
            return True
        if s in {"0", "false", "f", "no", "n"}:
            return False
        st.warning(f"Invalid {name}; using default {default}.")
        return bool(default)

    def _choice(val, allowed, default, name="value"):
        s = str(val).strip().lower()
        if s in allowed:
            return s
        st.warning(f"Invalid {name} '{val}'; using default '{default}'. Allowed: {', '.join(sorted(allowed))}.")
        return default

    # ---------- UI defaults ----------
    config = st.session_state.get("config", {})
    default_k         = int(config.get('kmeans_k', 8))
    default_hash_dim  = int(config.get('hash_dim', 1024))
    default_min_df    = int(config.get('idf_min_doc_freq', 3))
    default_fraction  = float(config.get('dataset_fraction', 0.1))
    default_seed      = int(config.get('random_seed', 42))

    # ---------- Lightweight memory hint (just a warning, no clamping) ----------
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
        if available_gb < 2:
            st.error(f"⚠️ Low memory detected: {available_gb:.1f} GB available.")
        elif available_gb < 4:
            st.warning(f"⚠️ Memory warning: {available_gb:.1f} GB available.")
        else:
            st.info(f"💾 Available memory: {available_gb:.1f} GB")
    except Exception:
        pass

    # =============== ONE FORM: all inputs are collected and only applied on submit ===============
    with st.form("clustering_form", clear_on_submit=False):
        st.subheader("🧩 Clustering Parameters")
        c1, c2, c3 = st.columns(3)
        with c1:
            k_in           = st.text_input("Number of Clusters (K)", value=str(default_k), key="in_k")
            sample_frac_in = st.text_input("Sample Fraction (0-1)", value="0.01", key="in_sample_frac")
            max_sample_in  = st.text_input("Max Sample Size", value="20000", key="in_max_sample")
        with c2:
            hash_dim_in    = st.text_input("Hash Dimensions", value=str(default_hash_dim), key="in_hash_dim")
            min_df_in      = st.text_input("IDF Min Doc Frequency", value=str(default_min_df), key="in_min_df")
            dist_in        = st.text_input("Distance Measure (euclidean/cosine)", value="euclidean", key="in_dist")
        with c3:
            chunk_in       = st.text_input("Chunk Processing? (true/false)", value="true", key="in_chunk")
            seed_in        = st.text_input("Random Seed", value=str(default_seed), key="in_seed")
            # dataset_fraction is a loader-level knob; keep free-form too (optional)
            dataset_frac_in = st.text_input("Dataset fraction (0-1, optional)", value=str(default_fraction), key="in_dsfrac")

        st.subheader("🧭 PCA / Visualization Settings")
        v1, v2, v3 = st.columns(3)
        with v1:
            pca_dims_in        = st.text_input("Projection (2D or 3D)", value="2D", key="in_pca_dims")
            pca_points_in      = st.text_input("PCA target points", value="1000", key="in_pca_points")
        with v2:
            stratified_in      = st.text_input("Stratified by cluster? (true/false)", value="true", key="in_strat")
            whiten_in          = st.text_input("Whiten 2D components? (true/false)", value="true", key="in_whiten")
        with v3:
            show_metrics_in    = st.text_input("Show PCA analysis metrics? (true/false)", value="true", key="in_metrics")
            show_pwmat_in      = st.text_input("Show centroid distance matrix? (true/false)", value="false", key="in_pwmat")
            export_pca_in      = st.text_input("Enable PCA CSV export? (true/false)", value="false", key="in_export")

        submitted = st.form_submit_button("🚀 Run Analysis")

    # ---------- If form submitted: parse inputs and run pipeline ----------
    if submitted:
        # Parse/sanitize
        k              = _to_int(k_in, default_k, min_v=2, name="K")
        sample_frac    = _to_float(sample_frac_in, 0.01, min_v=1e-6, max_v=1.0, name="sample fraction")
        max_sample     = _to_int(max_sample_in, 20000, min_v=10, name="max sample size")
        hash_dim       = _to_int(hash_dim_in, default_hash_dim, min_v=2, name="hash dimensions")
        idf_min_df     = _to_int(min_df_in, default_min_df, min_v=1, name="IDF min doc freq")
        distance_measure = _choice(dist_in, {"euclidean", "cosine"}, "euclidean", name="distance measure")
        chunk_processing = _to_bool(chunk_in, True, name="chunk processing")
        seed           = _to_int(seed_in, default_seed, name="random seed")
        dataset_frac   = _to_float(dataset_frac_in, default_fraction, min_v=1e-6, max_v=1.0, name="dataset fraction")

        pca_dims_str   = str(pca_dims_in).strip().upper()
        pca_dims       = "3D" if pca_dims_str == "3D" else "2D"
        pca_target_pts = _to_int(pca_points_in, 1000, min_v=10, name="PCA target points")
        stratified     = _to_bool(stratified_in, True, name="stratified")
        whiten         = _to_bool(whiten_in, True, name="whiten 2D")
        show_pca_analysis = _to_bool(show_metrics_in, True, name="show PCA analysis metrics")
        show_pwmat     = _to_bool(show_pwmat_in, False, name="show centroid distance matrix")
        export_pca_csv = _to_bool(export_pca_in, False, name="export PCA CSV")

        # Warnings for potentially heavy settings (no automatic clamping)
        if hash_dim > 262_144:
            st.warning("Very large hash_dim may be memory-intensive. Consider ≤ 262,144 if you hit memory limits.")
        if max_sample > 100_000:
            st.warning("Very large max sample sizes will increase runtime and memory usage.")
        if pca_target_pts > max_sample:
            st.info("PCA target points exceed max sample; PCA will use at most the sampled rows.")

        # ---------- Safe dataset loader ----------
        def safe_load_dataset():
            try:
                sdf_ = load_compact_dataset(spark, dataset_frac, seed)
                if sdf_ is None:
                    return None
                try:
                    total_count = sdf_.count()
                    st.info(f"📊 Loaded dataset: {total_count:,} documents (fraction={dataset_frac})")
                except Exception as e_count:
                    st.warning(f"Dataset count failed (continuing): {e_count}")
                return sdf_
            except Exception as e_load:
                st.error(f"Failed to load dataset: {e_load}")
                return None

        df_tf = df_tfidf = features_df = clustered_df = sampled_sdf = viz_ids_sdf = None

        try:
            with st.status("Running memory-safe clustering analysis…") as status:
                # Step 1: Load
                status.update(label="Loading dataset safely…")
                sdf = safe_load_dataset()
                if sdf is None:
                    st.error("❌ Could not load dataset safely")
                    return

                needed = {"doc_id", "tokens_final"}
                missing = needed - set(sdf.columns)
                if missing:
                    st.error(f"Dataset missing columns: {sorted(missing)}")
                    return

                # Step 2: Sampling
                status.update(label="Applying sampling…")
                if chunk_processing:
                    try:
                        total_docs = sdf.count()
                    except Exception:
                        total_docs = max_sample * 10
                    safe_fraction = min(float(sample_frac), float(max_sample) / max(total_docs, 1))
                    sampled_sdf = (
                        sdf.select("doc_id", "tokens_final")
                           .sample(False, safe_fraction, seed)
                           .limit(int(max_sample))
                           .repartition(4)
                           .cache()
                    )
                else:
                    sampled_sdf = (
                        sdf.select("doc_id", "tokens_final")
                           .sample(False, float(sample_frac), seed)
                           .limit(int(max_sample))
                           .cache()
                    )

                try:
                    n_samples = sampled_sdf.count()
                except Exception as e_sample:
                    st.error(f"Failed to sample data: {e_sample}")
                    st.info("Try reducing sample fraction or max sample size.")
                    return

                status.update(label=f"Sampled {n_samples:,} documents")
                if n_samples < 50:
                    st.error(f"Sample too small: {n_samples} docs. Increase sample settings.")
                    return
                elif n_samples < 200:
                    st.warning(f"Small sample: {n_samples} docs. Results may be noisy.")

                # Step 3: Features (TF-IDF + L2)
                status.update(label="Extracting features…")
                try:
                    htf = HashingTF(inputCol="tokens_final", outputCol="tf_h",
                                    numFeatures=int(hash_dim), binary=False)
                    df_tf = htf.transform(sampled_sdf).select("doc_id", "tf_h").cache()
                    _ = df_tf.count()

                    idf = IDF(inputCol="tf_h", outputCol="tfidf_h", minDocFreq=max(1, int(idf_min_df)))
                    idf_model = idf.fit(df_tf)
                    df_tfidf = idf_model.transform(df_tf).select("doc_id", "tfidf_h").cache()
                    _ = df_tfidf.count()

                    norm = Normalizer(inputCol="tfidf_h", outputCol="vec_l2", p=2.0)
                    features_df = norm.transform(df_tfidf).select("doc_id", "vec_l2").cache()
                    feature_count = features_df.count()
                    status.update(label=f"Extracted features for {feature_count:,} documents")
                except Exception as e_feat:
                    st.error(f"Feature extraction failed: {e_feat}")
                    st.info("Try reducing hash dimensions or sample size.")
                    return

                # Step 4: Clustering
                status.update(label="Running K-means…")
                try:
                    kmeans = KMeans(featuresCol="vec_l2", predictionCol="cluster",
                                    k=int(k), seed=int(seed),
                                    maxIter=10, tol=1e-2, initMode="k-means||")
                    try:
                        kmeans.setDistanceMeasure(distance_measure)
                    except Exception:
                        pass  # On L2-normalized vectors, Euclidean ~ Cosine

                    kmeans_model = kmeans.fit(features_df)
                    clustered_df = kmeans_model.transform(features_df).select("doc_id", "cluster").cache()
                    _ = clustered_df.count()
                    status.update(label=f"Clustered {n_samples:,} documents into {k} clusters")
                except Exception as e_cluster:
                    st.error(f"Clustering failed: {e_cluster}")
                    st.info("Try reducing K or sample size.")
                    return

                # Step 5: Simple analysis
                status.update(label="Analyzing results…")
                try:
                    cluster_stats = (
                        clustered_df.groupBy("cluster")
                                    .agg(count("*").alias("doc_count"))
                                    .orderBy("cluster")
                                    .toPandas()
                    )
                    cluster_stats['percentage'] = (
                        cluster_stats['doc_count'] / cluster_stats['doc_count'].sum() * 100
                    ).round(2)

                    st.subheader(f"📊 Clustering Results (K={k})")
                    if plotting_available:
                        fig = px.bar(cluster_stats, x='cluster', y='doc_count',
                                     title=f"Cluster Sizes (n={n_samples:,})",
                                     labels={'cluster': 'Cluster ID', 'doc_count': 'Documents'})
                        st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(
                        cluster_stats,
                        hide_index=True,
                        column_config={
                            'cluster': st.column_config.NumberColumn('Cluster', format="%d"),
                            'doc_count': st.column_config.NumberColumn('Documents', format="%d"),
                            'percentage': st.column_config.NumberColumn('Percentage', format="%.1f%%")
                        }
                    )

                    largest = int(cluster_stats['doc_count'].max())
                    smallest = int(cluster_stats['doc_count'].min())
                    balance_ratio = (smallest / largest) if largest > 0 else 0.0

                    m1, m2, m3 = st.columns(3)
                    with m1: st.metric("Largest Cluster", largest)
                    with m2: st.metric("Smallest Cluster", smallest)
                    with m3: st.metric("Balance Ratio", f"{balance_ratio:.3f}")
                except Exception as e_ana:
                    st.error(f"Analysis failed: {e_ana}")
                    return

                # Step 6: PCA VISUALIZATION & ANALYSIS (uses choices from the form)
                if plotting_available and SklearnPCA is not None:
                    try:
                        status.update(label="Preparing PCA sample…")
                        seed_local = int(seed)
                        pca_n = max(10, min(int(pca_target_pts), int(n_samples)))

                        # Build a DF of doc_id, cluster for visualization
                        if stratified and not cluster_stats.empty:
                            total = int(cluster_stats['doc_count'].sum())
                            frac = min(1.0, float(pca_n) / float(total if total else 1))
                            fractions = {int(r.cluster): float(frac) for _, r in cluster_stats.iterrows()}
                            viz_ids_sdf = (
                                clustered_df.sampleBy("cluster", fractions=fractions, seed=seed_local)
                                            .limit(int(pca_n))
                                            .cache()
                            )
                        else:
                            viz_ids_sdf = clustered_df.orderBy(F.rand(seed_local)).limit(int(pca_n)).cache()

                        # Join to features and convert vectors → arrays
                        viz_sdf = (
                            viz_ids_sdf.alias("c")
                                       .join(features_df.alias("f"), on="doc_id", how="inner")
                                       .select("doc_id", "c.cluster", v2a_safe(F.col("f.vec_l2")).alias("feature_array"))
                                       .where(F.col("feature_array").isNotNull())
                        )

                        arr_df = viz_sdf.toPandas()
                        try:
                            _ = viz_ids_sdf.unpersist()
                        except Exception:
                            pass

                        if arr_df.empty or len(arr_df) < 10:
                            st.warning("Insufficient data for visualization (need ≥10 points).")
                        else:
                            labels_np = arr_df["cluster"].to_numpy(dtype=int)
                            X = np.stack(arr_df["feature_array"].values, axis=0).astype("float32")

                            # Always compute 2D
                            pca2 = SklearnPCA(n_components=2, whiten=bool(whiten), random_state=seed_local)
                            XY = pca2.fit_transform(X)
                            evr2 = pca2.explained_variance_ratio_
                            st.subheader("📈 PCA Visualization")
                            st.caption(f"2D PCA explained variance: {evr2.sum():.3f} • points: {len(X)}")

                            fig2d = px.scatter(
                                x=XY[:, 0], y=XY[:, 1],
                                color=[str(int(i)) for i in labels_np],
                                labels={"color": "cluster"},
                                title="2D PCA Projection"
                            )
                            fig2d.update_traces(marker=dict(size=6, opacity=0.85, line=dict(width=0.5, color="white")))
                            st.plotly_chart(fig2d, use_container_width=True)

                            # Optional 3D
                            XYZ = None
                            evr3 = None
                            if pca_dims == "3D" and X.shape[0] >= 3:
                                pca3 = SklearnPCA(n_components=3, whiten=False, random_state=seed_local)
                                XYZ = pca3.fit_transform(X)
                                evr3 = pca3.explained_variance_ratio_

                                fig3d = go.Figure(data=[
                                    go.Scatter3d(
                                        x=XYZ[:, 0], y=XYZ[:, 1], z=XYZ[:, 2],
                                        mode="markers",
                                        marker=dict(
                                            size=4, opacity=0.85,
                                            color=labels_np,  # numeric for stable color mapping
                                            colorscale="Viridis", showscale=False
                                        ),
                                        text=[f"cluster={int(c)}" for c in labels_np],
                                        hoverinfo="text"
                                    )
                                ])
                                fig3d.update_layout(
                                    title="3D PCA Projection",
                                    scene=dict(
                                        xaxis_title=f"PC1 ({evr3[0]:.1%})",
                                        yaxis_title=f"PC2 ({evr3[1]:.1%})",
                                        zaxis_title=f"PC3 ({evr3[2]:.1%})",
                                    )
                                )
                                st.plotly_chart(fig3d, use_container_width=True)
                                st.caption(f"3D PCA explained variance: {evr3.sum():.3f}")

                            # ---------- PCA ANALYSIS METRICS ----------
                            if show_pca_analysis:
                                use_3d = (pca_dims == "3D" and XYZ is not None)
                                coords = XYZ if use_3d else XY
                                d = 3 if use_3d else 2

                                unique_clusters = np.unique(labels_np)
                                centers = []
                                sizes = []
                                avg_intra = []
                                std_intra = []
                                for cid in unique_clusters:
                                    mask = (labels_np == cid)
                                    pts = coords[mask]
                                    ctr = pts.mean(axis=0)
                                    dists = np.linalg.norm(pts - ctr, axis=1)
                                    centers.append(ctr)
                                    sizes.append(int(mask.sum()))
                                    avg_intra.append(float(dists.mean()))
                                    std_intra.append(float(dists.std(ddof=0)))
                                centers = np.vstack(centers)  # (C, d)

                                diffs = centers[:, None, :] - centers[None, :, :]
                                pw = np.sqrt((diffs ** 2).sum(axis=2))
                                triu_idx = np.triu_indices(pw.shape[0], k=1)
                                inter_vals = pw[triu_idx]
                                inter_avg = float(inter_vals.mean()) if inter_vals.size else 0.0
                                inter_min = float(inter_vals.min()) if inter_vals.size else 0.0
                                inter_max = float(inter_vals.max()) if inter_vals.size else 0.0

                                sil = None
                                if _silhouette_score is not None and len(unique_clusters) > 1 and len(coords) >= 10:
                                    try:
                                        sil = float(_silhouette_score(coords, labels_np, metric="euclidean"))
                                    except Exception:
                                        sil = None

                                st.subheader("🧪 PCA Space Cluster Analytics")
                                a1, a2, a3, a4 = st.columns(4)
                                with a1: st.metric("Clusters", len(unique_clusters))
                                with a2: st.metric("Avg inter-cluster dist", f"{inter_avg:.3f}")
                                with a3: st.metric("Min inter-cluster dist", f"{inter_min:.3f}")
                                with a4: st.metric("Max inter-cluster dist", f"{inter_max:.3f}")
                                if sil is not None:
                                    st.metric("Silhouette (PCA space)", f"{sil:.3f}")

                                cols = ["cluster", "size", "avg_intra_dist", "std_intra_dist"]
                                if d == 2:
                                    cols += ["PC1_center", "PC2_center"]
                                    rows = [
                                        [int(cid), sizes[i], round(avg_intra[i], 4), round(std_intra[i], 4),
                                         float(centers[i, 0]), float(centers[i, 1])]
                                        for i, cid in enumerate(unique_clusters)
                                    ]
                                else:
                                    cols += ["PC1_center", "PC2_center", "PC3_center"]
                                    rows = [
                                        [int(cid), sizes[i], round(avg_intra[i], 4), round(std_intra[i], 4),
                                         float(centers[i, 0]), float(centers[i, 1]), float(centers[i, 2])]
                                        for i, cid in enumerate(unique_clusters)
                                    ]
                                cluster_analysis_df = pd.DataFrame(rows, columns=cols)
                                st.dataframe(cluster_analysis_df, hide_index=True, use_container_width=True)

                                if show_pwmat:
                                    df_pw = pd.DataFrame(pw, index=[int(c) for c in unique_clusters],
                                                         columns=[int(c) for c in unique_clusters])
                                    st.dataframe(df_pw.style.format("{:.3f}"), use_container_width=True)

                                if export_pca_csv:
                                    export_pts = {"doc_id": arr_df["doc_id"].tolist(),
                                                  "cluster": labels_np,
                                                  "PC1": coords[:, 0], "PC2": coords[:, 1]}
                                    if d == 3:
                                        export_pts["PC3"] = coords[:, 2]
                                    st.download_button(
                                        "📥 Download PCA points",
                                        data=pd.DataFrame(export_pts).to_csv(index=False),
                                        file_name=f"pca_points_{d}d_k{k}_n{coords.shape[0]}.csv",
                                        mime="text/csv",
                                        key="dl_pca_pts"
                                    )
                                    st.download_button(
                                        "📥 Download PCA cluster analytics",
                                        data=cluster_analysis_df.to_csv(index=False),
                                        file_name=f"pca_cluster_analytics_{d}d_k{k}.csv",
                                        mime="text/csv",
                                        key="dl_pca_analytics"
                                    )
                    except Exception as viz_error:
                        st.warning(f"PCA visualization failed (not critical): {viz_error}")
                        import traceback
                        st.code(traceback.format_exc())
                else:
                    if not plotting_available:
                        st.info("Install Plotly to see PCA scatter: `pip install plotly`")
                    if SklearnPCA is None:
                        st.info("Install scikit-learn for PCA: `pip install scikit-learn`")

                # Step 7: Sample docs by cluster (tiny)
                try:
                    st.subheader("📄 Sample Documents by Cluster")
                    samples = []
                    for cid in range(int(k)):
                        try:
                            pdf = (clustered_df.where(col("cluster") == cid)
                                             .select("doc_id", "cluster")
                                             .limit(2)
                                             .toPandas())
                            if not pdf.empty:
                                samples.append(pdf)
                        except Exception:
                            pass
                    if samples:
                        st.dataframe(pd.concat(samples, ignore_index=True), hide_index=True)
                except Exception as e_samp:
                    st.warning(f"Could not generate sample documents: {e_samp}")

                # Downloads
                st.subheader("📥 Downloads")
                csv_data = cluster_stats.to_csv(index=False)
                st.download_button(
                    label="📊 Download Cluster Statistics",
                    data=csv_data,
                    file_name=f"memory_safe_clusters_k{k}_n{n_samples}.csv",
                    mime="text/csv",
                    key="download_cluster_stats"
                )

                status.update(
                    label=f"✅ Memory-safe clustering complete! {k} clusters from {n_samples:,} documents",
                    state="complete"
                )

        except Exception as e:
            st.error(f"Memory-safe clustering failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.subheader("💡 Memory Troubleshooting")
            st.info(
                "1) Reduce 'Max Sample Size'\n"
                "2) Reduce 'Sample Fraction' (e.g., 0.001–0.005)\n"
                "3) Lower 'Hash Dimensions' (e.g., 256–2048)\n"
                "4) Use 'Chunk Processing=true'\n"
                "5) Close other apps to free memory\n"
                "6) Restart Streamlit to clear Spark cache"
            )
        finally:
            # Cleanup (best-effort)
            for df_ in [viz_ids_sdf, sampled_sdf, df_tf, df_tfidf, features_df, clustered_df]:
                try:
                    if df_ is not None:
                        df_.unpersist()
                except Exception:
                    pass





def show_parameter_tuning_tab(spark):
    """Show parameter tuning for ANN (MinHashLSH) with batch processing + reuse of features and one-join-per-(n,h)."""
    import math
    import streamlit as st
    import pandas as pd
    import numpy as np

    # PySpark
    from pyspark.storagelevel import StorageLevel
    from pyspark.sql.functions import (
        col, size, sha2, concat_ws, count, rand, lit, floor, asc, broadcast
    )
    from pyspark.ml.feature import NGram, HashingTF
    from pyspark.ml.feature import MinHashLSH, MinHashLSHModel
    from pyspark.ml import Pipeline, PipelineModel

    # Plotly (optional)
    try:
        import plotly.express as px
        PLOTTING_AVAILABLE = True
    except Exception:
        PLOTTING_AVAILABLE = False
        px = None  # type: ignore

    st.header("⚙️ Parameter Tuning - ANN Evaluation (Fast & Batch-Optimized)")

    # --- Config & defaults ---
    config = st.session_state.get("config", {})
    config.setdefault("hash_dim", 1024)       # -> num_features
    config.setdefault("random_seed", 42)
    config.setdefault("dataset_fraction", 0.1)

    # Load compact dataset
    try:
        sdf = load_compact_dataset(spark, float(config["dataset_fraction"]), int(config["random_seed"]))
    except Exception as e:
        st.error(f"❌ Failed to load compact dataset: {e}")
        return

    if sdf is None:
        st.error("❌ No compact dataset available. Please build it first.")
        return

    # Presence checks
    required_cols = {"doc_id", "tokens_final"}
    missing = required_cols - set(sdf.columns)
    if missing:
        st.error(f"❌ Missing required columns: {sorted(missing)}")
        return

    # ---- UI controls ----
    st.subheader("🎛️ Tuning Configuration")
    c1, c2 = st.columns(2)
    with c1:
        sample_fraction = st.slider("Sample Fraction for Tuning", 0.01, 0.50, 0.10, step=0.01)
        ngram_options = st.multiselect("N-gram Sizes", [1, 2, 3], default=[1, 2, 3])
        h_options = st.multiselect("Hash Tables (H)", list(range(4, 33, 4)), default=[8, 16, 24, 32])
        min_doc_tokens = st.number_input("Min Document Tokens", min_value=1, max_value=1000, value=20, step=1)
    with c2:
        threshold_range = st.slider("Distance Thresholds (1 - Jaccard)", 0.05, 0.30, (0.10, 0.25), step=0.05)
        sample_max = st.number_input("Max Sample Size", min_value=1_000, max_value=200_000, value=20_000, step=1_000)
        # Batch knobs
        batch_target_rows = st.number_input("Batch Size (rows per left batch)", min_value=2_000, max_value=200_000, value=20_000, step=2_000)
        max_pairs_per_threshold = st.number_input("Safety Cap: Max Pairs per Threshold", min_value=50_000, max_value=5_000_000, value=500_000, step=50_000)

    # ---- Helper: build features_bin ONCE per n and reuse for all h/thresholds ----
    def _features_for_n(tuning_sdf, n: int, num_features: int):
        """
        Return DF with columns: [doc_id, features_bin] for a given n-gram n.
        Persisted MEMORY_AND_DISK and materialized.
        """
        pipe = Pipeline(stages=[
            NGram(n=int(n), inputCol="tokens_final", outputCol="ngrams"),
            HashingTF(inputCol="ngrams", outputCol="features_bin", numFeatures=int(num_features), binary=True),
        ])
        model = pipe.fit(tuning_sdf)
        feats = (model.transform(tuning_sdf)
                      .select("doc_id", "features_bin")
                      .persist(StorageLevel.MEMORY_AND_DISK))
        _ = feats.count()
        return feats

    # ---- Helper: fit ONLY the LSH stage for (n,h); reuse features ----
    def _fit_lsh_for_h(features_df, h: int):
        """
        Create and fit an LSH model that reads 'features_bin' (no extra transforms).
        """
        lsh = MinHashLSH(inputCol="features_bin", outputCol="minhash_features", numHashTables=int(h))
        return lsh.fit(features_df)

    # ---- Batched similarity join to control memory ----
    def _batched_pairs(features_df, lsh_model, threshold: float, seed: int, batch_rows: int, safety_cap: int):
        """
        Compute similarity pairs in batches at a single threshold using the model's inputCol.
        Then union, dedup, limit (safety cap). Returns (pairs_sdf, predicted_pairs_count).
        """
        input_col = lsh_model.getInputCol()  # "features_bin"
        base = features_df.select("doc_id", input_col).persist(StorageLevel.MEMORY_AND_DISK)
        _ = base.count()

        total_rows = base.count()
        num_batches = max(1, int(math.ceil(total_rows / float(batch_rows))))

        with_batches = base.withColumn("batch_id", floor(rand(seed=seed) * lit(num_batches)).cast("int")) \
                           .persist(StorageLevel.MEMORY_AND_DISK)
        _ = with_batches.count()

        pairs_accum = None

        for i in range(num_batches):
            left_i = with_batches.where(col("batch_id") == lit(i)).drop("batch_id").persist(StorageLevel.MEMORY_AND_DISK)
            left_cnt = left_i.count()
            if left_cnt == 0:
                left_i.unpersist()
                continue

            try:
                sim_i = (
                    lsh_model.approxSimilarityJoin(
                        left_i, base, float(threshold), "distance"
                    )
                    .select(
                        col("datasetA.doc_id").alias("doc_id_1"),
                        col("datasetB.doc_id").alias("doc_id_2"),
                        col("distance")
                    )
                    .where(col("doc_id_1") < col("doc_id_2"))
                    .persist(StorageLevel.MEMORY_AND_DISK)
                )

                pairs_accum = sim_i if pairs_accum is None else pairs_accum.unionByName(sim_i)
                # Dedup + cap to keep lineage/memory bounded
                pairs_accum = (
                    pairs_accum.dropDuplicates(["doc_id_1", "doc_id_2"])
                               .orderBy(asc("doc_id_1"))
                               .limit(int(safety_cap))
                               .persist(StorageLevel.MEMORY_AND_DISK)
                )
                sim_i.unpersist()

            except Exception as e:
                st.warning(f"Batch {i+1}/{num_batches} similarity join failed: {e}")
            finally:
                left_i.unpersist()

            if pairs_accum is not None and pairs_accum.count() >= int(safety_cap):
                break

        with_batches.unpersist()
        base.unpersist()

        if pairs_accum is None:
            empty = spark.createDataFrame([], schema="doc_id_1 string, doc_id_2 string, distance double")
            return empty, 0

        predicted_pairs_count = pairs_accum.count()
        return pairs_accum, predicted_pairs_count

    # ---- Run tuning ----
    if st.button("🔬 Run Parameter Tuning") and ngram_options and h_options:
        tuning_sdf = None
        duplicate_pairs = None

        try:
            with st.status("Running parameter tuning (fast path)…") as status:
                # Sample data for tuning
                status.update(label="Preparing tuning dataset...")
                tuning_sdf = sdf.select("doc_id", "tokens_final")
                tuning_sdf = tuning_sdf.sample(fraction=float(sample_fraction), seed=int(config["random_seed"]))
                tuning_sdf = tuning_sdf.filter(size(col("tokens_final")) >= int(min_doc_tokens)).persist(StorageLevel.MEMORY_AND_DISK)
                total_count = tuning_sdf.count()

                # Cap sample size
                if total_count > int(sample_max):
                    actual_fraction = float(sample_max) / float(total_count)
                    tuning_sdf = tuning_sdf.sample(fraction=actual_fraction, seed=int(config["random_seed"])) \
                                           .persist(StorageLevel.MEMORY_AND_DISK)
                    total_count = tuning_sdf.count()

                if total_count < 10:
                    st.error(f"Too few documents after sampling/filters: {total_count}")
                    return

                st.info(f"📦 Tuning sample size: {total_count:,}")

                # Weak gold standard: exact-duplicate pairs by text hash
                status.update(label="Building weak gold standard...")
                text_norm_col = concat_ws(" ", col("tokens_final"))
                hash_sdf = tuning_sdf.withColumn("text_hash", sha2(text_norm_col, 256)).persist(StorageLevel.MEMORY_AND_DISK)

                dup_hashes = (
                    hash_sdf.groupBy("text_hash")
                            .agg(count("*").alias("cnt"))
                            .filter(col("cnt") > 1)
                            .select("text_hash")
                            .persist(StorageLevel.MEMORY_AND_DISK)
                )

                dup_docs = (
                    hash_sdf.join(dup_hashes, "text_hash", "inner")
                            .select("doc_id", "text_hash")
                            .persist(StorageLevel.MEMORY_AND_DISK)
                )

                a = dup_docs.alias("a")
                b = dup_docs.alias("b")
                duplicate_pairs = (
                    a.join(b, col("a.text_hash") == col("b.text_hash"))
                     .filter(col("a.doc_id") < col("b.doc_id"))
                     .select(
                         col("a.doc_id").alias("doc_id_1"),
                         col("b.doc_id").alias("doc_id_2")
                     )
                     .dropDuplicates(["doc_id_1", "doc_id_2"])
                     .persist(StorageLevel.MEMORY_AND_DISK)
                )

                gold_pairs_count = duplicate_pairs.count()
                if gold_pairs_count < 10:
                    st.warning(f"Only {gold_pairs_count} weak gold pairs found. Results may not be reliable.")
                else:
                    st.info(f"✅ Found {gold_pairs_count:,} weak gold standard pairs")

                # Free intermediates we no longer need
                dup_hashes.unpersist()
                hash_sdf.unpersist()

                # Prepare thresholds: compute once at MAX threshold, filter for smaller thresholds
                num_thresh_steps = 5
                t_min, t_max = float(threshold_range[0]), float(threshold_range[1])
                thresholds = np.linspace(t_min, t_max, num_thresh_steps)
                thresholds_sorted = sorted(set(float(x) for x in thresholds))
                t_global_max = max(thresholds_sorted)

                results = []
                total_combinations = len(ngram_options) * len(h_options)  # joins count (one per (n,h))
                current = 0

                # Cache features per n and reuse across all h
                features_cache = {}

                for n in sorted(set(int(x) for x in ngram_options)):
                    status.update(label=f"Building features for n={n} …")
                    if n not in features_cache:
                        features_cache[n] = _features_for_n(tuning_sdf, n=int(n), num_features=int(config["hash_dim"]))
                    features_df = features_cache[n]

                    for h in sorted(set(int(x) for x in h_options)):
                        current += 1
                        status.update(label=f"[{current}/{total_combinations}] Fitting LSH & 1 join: n={n}, h={h} …")

                        # Fit only LSH (fast) on re-used features
                        try:
                            lsh_model = _fit_lsh_for_h(features_df, h=int(h))
                        except Exception as e:
                            st.warning(
                                f"LSH fit failed for n={n}, h={h}: {e}\n"
                                "Tip: ensure enough rows and non-empty features."
                            )
                            continue

                        # Do ONE batched join at the largest threshold and cache pairs
                        try:
                            pairs_all_tmax, predicted_pairs_count_tmax = _batched_pairs(
                                features_df=features_df,
                                lsh_model=lsh_model,
                                threshold=float(t_global_max),
                                seed=int(config["random_seed"]),
                                batch_rows=int(batch_target_rows),
                                safety_cap=int(max_pairs_per_threshold),
                            )
                            # Persist once; reuse for all smaller thresholds
                            pairs_all_tmax = pairs_all_tmax.persist(StorageLevel.MEMORY_AND_DISK)
                            _ = pairs_all_tmax.count()
                        except Exception as e:
                            st.warning(f"Similarity join failed for n={n}, h={h} at τ_max={t_global_max:.3f}: {e}")
                            continue

                        # Evaluate metrics by filtering the already materialized pairs by distance
                        for t in thresholds_sorted:
                            try:
                                # distance ≤ t  (approxSimilarityJoin returns 'distance')
                                sim_pairs_t = (
                                    pairs_all_tmax.where(col("distance") <= float(t))
                                                   .persist(StorageLevel.MEMORY_AND_DISK)
                                )
                                predicted_pairs_count = sim_pairs_t.count()

                                if predicted_pairs_count > 0 and gold_pairs_count > 0:
                                    tp_pairs = sim_pairs_t.join(
                                        broadcast(duplicate_pairs),
                                        ["doc_id_1", "doc_id_2"],
                                        "inner"
                                    )
                                    tp_count = tp_pairs.count()
                                    precision = tp_count / predicted_pairs_count if predicted_pairs_count else 0.0
                                    recall = tp_count / gold_pairs_count if gold_pairs_count else 0.0
                                else:
                                    precision, recall, tp_count = 0.0, 0.0, 0

                                f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

                                results.append({
                                    "n": int(n),
                                    "h": int(h),
                                    "threshold": float(t),
                                    "jaccard_sim": float(1.0 - t),
                                    "predicted_pairs": int(predicted_pairs_count),
                                    "gold_pairs": int(gold_pairs_count),
                                    "tp_pairs": int(tp_count),
                                    "precision": float(precision),
                                    "recall": float(recall),
                                    "f1": float(f1),
                                })
                                sim_pairs_t.unpersist()
                            except Exception as e:
                                st.warning(f"Metric eval failed for n={n}, h={h}, τ={t:.3f}: {e}")

                        # Done with this (n,h): release pairs
                        try:
                            pairs_all_tmax.unpersist()
                        except Exception:
                            pass

                # ---- Display results ----
                if results:
                    status.update(label="Processing results...")

                    results_df = pd.DataFrame(results)

                    st.subheader("📊 Tuning Results")
                    st.dataframe(
                        results_df,
                        hide_index=True,
                        column_config={
                            "threshold": st.column_config.NumberColumn("Distance Threshold", format="%.3f"),
                            "jaccard_sim": st.column_config.NumberColumn("Jaccard Similarity", format="%.3f"),
                            "precision": st.column_config.NumberColumn("Precision", format="%.3f"),
                            "recall": st.column_config.NumberColumn("Recall", format="%.3f"),
                            "f1": st.column_config.NumberColumn("F1 Score", format="%.3f"),
                        }
                    )

                    if not results_df.empty:
                        rfill = results_df.fillna(0.0)

                        st.subheader("🏆 Best Configurations")
                        m1, m2, m3 = st.columns(3)

                        with m1:
                            i = int(rfill["precision"].idxmax())
                            bp = rfill.iloc[i]
                            st.metric("Best Precision", f"{bp['precision']:.3f}", f"n={int(bp['n'])}, h={int(bp['h'])}, τ={bp['threshold']:.2f}")

                        with m2:
                            i = int(rfill["recall"].idxmax())
                            br = rfill.iloc[i]
                            st.metric("Best Recall", f"{br['recall']:.3f}", f"n={int(br['n'])}, h={int(br['h'])}, τ={br['threshold']:.2f}")

                        with m3:
                            i = int(rfill["f1"].idxmax())
                            bf = rfill.iloc[i]
                            st.metric("Best F1 Score", f"{bf['f1']:.3f}", f"n={int(bf['n'])}, h={int(bf['h'])}, τ={bf['threshold']:.2f}")

                    # Visualizations
                    if PLOTTING_AVAILABLE and not results_df.empty:
                        st.subheader("📈 Performance Visualizations")
                        try:
                            fig = px.scatter(
                                results_df,
                                x="recall",
                                y="precision",
                                color="f1",
                                size="h",
                                hover_data=["n", "h", "jaccard_sim", "threshold", "predicted_pairs", "tp_pairs"],
                                title="Precision–Recall Trade-off"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not render PR scatter: {e}")

                        try:
                            pivot_f1 = results_df.pivot_table(values="f1", index="n", columns="h", aggfunc="mean")
                            fig = px.imshow(
                                pivot_f1,
                                title="F1 Score Heatmap (N-gram vs Hash Tables)",
                                color_continuous_scale="viridis",
                                aspect="auto"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not render heatmap: {e}")

                    # Downloads
                    st.subheader("📥 Download Results")
                    dl1, dl2 = st.columns(2)
                    with dl1:
                        precision_csv = results_df[["n", "h", "jaccard_sim", "threshold", "precision"]].to_csv(index=False)
                        st.download_button("📥 Download Precision Results", precision_csv, "ann_tuning_precision.csv", "text/csv")
                    with dl2:
                        recall_csv = results_df[["n", "h", "jaccard_sim", "threshold", "recall"]].to_csv(index=False)
                        st.download_button("📥 Download Recall Results", recall_csv, "ann_tuning_recall.csv", "text/csv")

                    status.update(label="✅ Parameter tuning complete!", state="complete")
                else:
                    st.warning("No evaluation results generated. Check your parameter settings.")

        except Exception as e:
            st.error(f"Parameter tuning failed: {e}")
        finally:
            # best-effort cleanup
            try:
                if tuning_sdf is not None: tuning_sdf.unpersist()
                if duplicate_pairs is not None: duplicate_pairs.unpersist()
                # Unpersist any cached features
                for _n, _df in locals().get("features_cache", {}).items():
                    try: _df.unpersist()
                    except Exception: pass
            except Exception:
                pass









from pathlib import Path
def _has_mlh_model(dirpath: Path) -> bool:
    return (dirpath.exists() and (dirpath / "metadata").exists())

def show_final_insights_tab(spark):
    """Show final insights and near-duplicate mining (fast path: feature reuse, single batched join, Spark-side stats)."""
    import streamlit as st
    import pandas as pd
    import numpy as np
    import shutil

    from pyspark.storagelevel import StorageLevel
    from pyspark.sql.functions import (
        col, concat_ws, rand, lit, floor, asc, desc, broadcast, avg, countDistinct
    )
    from pyspark.sql.window import Window
    from pyspark.sql.functions import row_number

    from pyspark.ml.feature import NGram, HashingTF, MinHashLSH
    from pyspark.ml import Pipeline

    # Optional plotting (only for lightweight charts on a sampled pandas subset)
    try:
        import plotly.express as px
        PLOTTING_AVAILABLE = True
    except Exception:
        PLOTTING_AVAILABLE = False
        px = None  # type: ignore

    st.header("💎 Final Insights - Near-Duplicate Mining (Fast)")

    # ---- Config (safe defaults) ----
    config = st.session_state.get("config", {})
    config.setdefault("dataset_fraction", 1.0)
    config.setdefault("random_seed", 42)
    config.setdefault("hash_dim", 262_144)
    config.setdefault("lsh_num_hash_tables", 8)
    config.setdefault("ngram_for_minhash", 2)
    config.setdefault("final_insights_jaccard", 0.80)
    # Fast-path knobs
    config.setdefault("final_batch_rows", 25_000)
    config.setdefault("final_safety_cap", 1_500_000)     # cap on raw pairs before trimming
    config.setdefault("final_per_doc_cap", 100)          # 0 = disable per-doc cap
    config.setdefault("final_pd_rows_cap", 100_000)      # pandas conversion cap for charts/downloads
    st.session_state["config"] = config  # persist back

    # ---- Load dataset ----
    try:
        sdf = load_compact_dataset(spark, float(config["dataset_fraction"]), int(config["random_seed"]))
    except Exception as e:
        st.error(f"❌ Failed to load compact dataset: {e}")
        return
    if sdf is None:
        st.error("❌ No compact dataset available. Please build it first.")
        return

    # ---- Controls ----
    st.subheader("🎛️ Mining Configuration")
    c1, c2, c3 = st.columns(3)
    with c1:
        similarity_threshold = st.slider(
            "Jaccard Similarity Threshold",
            0.60, 0.95,
            float(config["final_insights_jaccard"]),
            step=0.05
        )
        n_for_minhash = st.selectbox("N-gram for MinHash", [1, 2, 3], index=1)
    with c2:
        max_pairs_save = st.selectbox(
            "Max Pairs to Save (top by similarity)",
            [50_000, 100_000, 500_000, 1_000_000, 2_000_000],
            index=2
        )
        per_doc_cap = st.number_input("Max Matches per Doc (cap)", min_value=0, max_value=10_000,
                                      value=int(config["final_per_doc_cap"]), step=10,
                                      help="0 disables; otherwise keeps up to K best (lowest distance) per doc_id_1.")
    with c3:
        batch_rows = st.number_input("Batch Size (rows per left batch)", min_value=5_000, max_value=500_000,
                                     value=int(config["final_batch_rows"]), step=5_000)
        safety_cap = st.number_input("Raw Pairs Safety Cap", min_value=100_000, max_value=20_000_000,
                                     value=int(config["final_safety_cap"]), step=100_000)

    st.markdown(
        "_Speed tips: higher threshold (e.g., ≥0.85) → far fewer candidates. "
        "Increase `Batch Size` if your cluster has memory. Use `Max Matches per Doc` to tame hubs._"
    )

    # ---- Fast helpers ----
    def _build_features_once(_sdf, n: int, num_features: int):
        """
        Build features one time: tokens_final -> ngrams -> features_bin.
        Keep only the columns we actually need later.
        """
        pipe = Pipeline(stages=[
            NGram(n=int(n), inputCol="tokens_final", outputCol="ngrams"),
            HashingTF(inputCol="ngrams", outputCol="features_bin",
                      numFeatures=int(num_features), binary=True),
        ])
        model = pipe.fit(_sdf)
        # Choose a reasonable display text column early
        if "text_norm" in _sdf.columns:
            text_expr = col("text_norm")
        elif "text" in _sdf.columns:
            text_expr = col("text")
        else:
            text_expr = concat_ws(" ", col("tokens_final"))

        keep_cols = ["doc_id", "features_bin"]
        if "category" in _sdf.columns:
            keep_cols.append("category")
        feats = (model.transform(_sdf)
                      .withColumn("text_disp", text_expr)
                      .select(*(keep_cols + ["text_disp"]))
                      .persist(StorageLevel.MEMORY_AND_DISK))
        _ = feats.count()  # materialize
        return feats

    def _fit_lsh_only(features_df, num_hash_tables: int):
        """
        Fit just MinHashLSH on precomputed 'features_bin' (very fast).
        """
        lsh = MinHashLSH(inputCol="features_bin", outputCol="minhash_features",
                         numHashTables=int(num_hash_tables))
        return lsh.fit(features_df.select("doc_id", "features_bin"))

    def _batched_self_join(features_df, lsh_model, distance_threshold: float,
                           seed: int, batch_rows: int, safety_cap: int):
        """
        Single-threshold, batched self-join via approxSimilarityJoin.
        - Work only with [doc_id, features_bin] to reduce shuffle.
        - Randomly shard the left side; join with full right.
        - Early dropDuplicates + LIMIT safety_cap after each batch to bound lineage/memory.
        Returns (pairs_sdf: [doc_id_1, doc_id_2, distance], count).
        """
        from math import ceil

        base = features_df.select("doc_id", "features_bin").persist(StorageLevel.MEMORY_AND_DISK)
        total = base.count()
        if total == 0:
            base.unpersist()
            empty = spark.createDataFrame([], "doc_id_1 string, doc_id_2 string, distance double")
            return empty, 0

        num_batches = max(1, int(ceil(total / float(batch_rows))))
        with_batches = base.withColumn("batch_id", floor(rand(seed=seed) * lit(num_batches)).cast("int")) \
                           .persist(StorageLevel.MEMORY_AND_DISK)
        _ = with_batches.count()

        pairs_accum = None
        for i in range(num_batches):
            left_i = with_batches.where(col("batch_id") == lit(i)).drop("batch_id") \
                                 .persist(StorageLevel.MEMORY_AND_DISK)
            left_cnt = left_i.count()
            if left_cnt == 0:
                left_i.unpersist()
                continue

            try:
                sim_i = (
                    lsh_model.approxSimilarityJoin(
                        left_i, base, float(distance_threshold), "distance"
                    )
                    .select(
                        col("datasetA.doc_id").alias("doc_id_1"),
                        col("datasetB.doc_id").alias("doc_id_2"),
                        col("distance"),
                    )
                    .where(col("doc_id_1") < col("doc_id_2"))
                    .persist(StorageLevel.MEMORY_AND_DISK)
                )

                pairs_accum = sim_i if pairs_accum is None else pairs_accum.unionByName(sim_i)
                # Early dedup + cap
                pairs_accum = (
                    pairs_accum.dropDuplicates(["doc_id_1", "doc_id_2"])
                               .orderBy(asc("doc_id_1"))
                               .limit(int(safety_cap))
                               .persist(StorageLevel.MEMORY_AND_DISK)
                )
                sim_i.unpersist()

            except Exception as e:
                st.warning(f"Batch {i+1}/{num_batches} similarity join failed: {e}")
            finally:
                left_i.unpersist()

            if pairs_accum is not None and pairs_accum.count() >= int(safety_cap):
                break

        with_batches.unpersist()
        base.unpersist()

        if pairs_accum is None:
            empty = spark.createDataFrame([], "doc_id_1 string, doc_id_2 string, distance double")
            return empty, 0

        return pairs_accum, pairs_accum.count()

    # ---- Action ----
    if st.button("💎 Mine Near-Duplicates"):
        feats_df = None
        pairs_raw = None
        enriched = None
        try:
            with st.status("Mining near-duplicates (fast)…") as status:
                # 1) Build features once
                status.update(label=f"Building features (n={int(n_for_minhash)})…")
                feats_df = _build_features_once(
                    sdf, n=int(n_for_minhash), num_features=int(config["hash_dim"])
                )

                # 2) Fit only LSH
                status.update(label=f"Fitting LSH (H={int(config['lsh_num_hash_tables'])})…")
                lsh_model = _fit_lsh_only(feats_df, int(config["lsh_num_hash_tables"]))

                # 3) One batched self-join at your distance threshold
                status.update(label=f"Approx join @ Jaccard ≥ {float(similarity_threshold):.2f}…")
                dist_thr = float(1.0 - float(similarity_threshold))
                pairs_raw, raw_cnt = _batched_self_join(
                    features_df=feats_df,
                    lsh_model=lsh_model,
                    distance_threshold=dist_thr,
                    seed=int(config["random_seed"]),
                    batch_rows=int(batch_rows),
                    safety_cap=int(safety_cap),
                )

                # 4) Score & optional per-doc cap (keep best matches per doc_id_1)
                status.update(label="Scoring and capping…")
                scored = pairs_raw.withColumn("jaccard_sim", 1.0 - col("distance"))

                if per_doc_cap and int(per_doc_cap) > 0:
                    w = Window.partitionBy("doc_id_1").orderBy(asc("distance"))
                    scored = (scored
                              .withColumn("rn", row_number().over(w))
                              .where(col("rn") <= int(per_doc_cap))
                              .drop("rn"))

                # 5) Enrich with text & categories using broadcast joins
                meta = feats_df.select("doc_id", "text_disp", *([ "category" ] if "category" in feats_df.columns else []))
                a = meta.select(col("doc_id").alias("doc_id_1"),
                                col("text_disp").alias("text_1"),
                                *([col("category").alias("category_1")] if "category" in meta.columns else []))
                b = meta.select(col("doc_id").alias("doc_id_2"),
                                col("text_disp").alias("text_2"),
                                *([col("category").alias("category_2")] if "category" in meta.columns else []))

                enriched = (scored
                            .join(broadcast(a), ["doc_id_1"], "left")
                            .join(broadcast(b), ["doc_id_2"], "left"))

                # 6) Keep top-N by similarity for saving/viewing
                total_pairs = enriched.count()
                if total_pairs == 0:
                    st.warning("No near-duplicate pairs found. Try lowering the similarity threshold.")
                    return

                if total_pairs > int(max_pairs_save):
                    st.warning(f"Found {total_pairs:,} pairs, limiting to top {int(max_pairs_save):,} by similarity.")
                    enriched = enriched.orderBy(desc("jaccard_sim")).limit(int(max_pairs_save))
                    final_pairs_count = int(max_pairs_save)
                else:
                    final_pairs_count = total_pairs

                st.success(f"✅ Pairs ready: {final_pairs_count:,}")

                # 7) Spark-side stats (fast, no pandas yet)
                status.update(label="Computing summary stats…")
                unique_docs = (enriched.select("doc_id_1").union(enriched.select("doc_id_2")).distinct().count())
                avg_sim = enriched.agg(avg("jaccard_sim")).first()[0] or 0.0
                total_docs = sdf.count()
                coverage = (unique_docs / total_docs * 100.0) if total_docs > 0 else 0.0

                st.subheader("📊 Near-Duplicate Statistics")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Pairs", f"{final_pairs_count:,}")
                m2.metric("Unique Documents", f"{unique_docs:,}")
                m3.metric("Avg Jaccard", f"{avg_sim:.3f}")
                m4.metric("Duplicate Coverage", f"{coverage:.2f}%")

                # 8) Optional lightweight charts on a bounded pandas sample
                has_categories = "category" in feats_df.columns
                pd_cap = int(config["final_pd_rows_cap"])
                pairs_for_pd = enriched.orderBy(desc("jaccard_sim")).limit(pd_cap)
                pairs_df = pairs_for_pd.toPandas()

                if PLOTTING_AVAILABLE and not pairs_df.empty:
                    # Degree analysis
                    st.subheader("📊 Document Degree (sampled)")
                    deg_series = pd.Series(pairs_df["doc_id_1"].tolist() + pairs_df["doc_id_2"].tolist()).value_counts()
                    try:
                        fig = px.histogram(
                            x=deg_series.values, nbins=50,
                            title=f"Distribution of Near-Duplicate Degrees (Top {pd_cap:,} pairs)"
                        )
                        fig.update_layout(xaxis_title="# Near-Duplicates", yaxis_title="# Documents")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Histogram render failed: {e}")

                    # Category analysis (sampled)
                    if has_categories:
                        if "category_1" in pairs_df.columns and "category_2" in pairs_df.columns:
                            st.subheader("📈 Category Analysis (sampled)")
                            pairs_df["same_category"] = pairs_df["category_1"] == pairs_df["category_2"]
                            same_cat = int(pairs_df["same_category"].sum())
                            cross_cat = int(len(pairs_df) - same_cat)
                            try:
                                fig = px.pie(values=[same_cat, cross_cat],
                                             names=["Same Category", "Cross Category"],
                                             title="Near-Duplicates: Same vs Cross Category")
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Pie render failed: {e}")

                # 9) Save results (Spark write, not pandas)
                if st.checkbox("Save Pair Results", True):
                    status.update(label="Saving results…")
                    try:
                        sim_str = f"sim{int(float(similarity_threshold) * 100)}"
                        pairs_dir = WORK_DIR / f"data_results/dup_pairs_minhash_{sim_str}"
                        if pairs_dir.exists():
                            shutil.rmtree(pairs_dir)
                        enriched.write.mode("overwrite").parquet(str(pairs_dir))
                        st.success(f"✅ Saved {final_pairs_count:,} pairs to {pairs_dir}")
                    except Exception as e:
                        st.warning(f"Save skipped: {e}")

                # 10) Downloads (sampled CSV for quick inspection)
                st.subheader("📥 Download (sampled)")
                csv_sample = pairs_df.to_csv(index=False) if 'pairs_df' in locals() and not pairs_df.empty else ""
                st.download_button(
                    label=f"📥 Download CSV sample (top {pd_cap:,})",
                    data=csv_sample,
                    file_name=f"duplicate_pairs_sample_jaccard{float(similarity_threshold):.2f}.csv",
                    mime="text/csv",
                    disabled=(csv_sample == "")
                )

                status.update(label="✅ Near-duplicate mining complete!", state="complete")

        except Exception as e:
            st.error(f"Near-duplicate mining failed: {e}")
        finally:
            # Cleanup caches
            for _df in [feats_df, pairs_raw, enriched]:
                try:
                    if _df is not None:
                        _df.unpersist()
                except Exception:
                    pass







def show_quality_monitoring_tab(spark):
    """Show quality monitoring and artifact management (robust to missing 'path' keys)."""
    import shutil
    from pathlib import Path

    import streamlit as st
    import pandas as pd

    # pyspark bits we use here
    from pyspark.sql.functions import col, size, avg  # for compact-dataset checks
    from pyspark.sql.types import (
        StructType, StructField, IntegerType, StringType, ArrayType
    )

    st.header("📋 Quality & Monitoring")

    # ---------------------------
    # 📁 Artifact status overview
    # ---------------------------
    st.subheader("📁 Artifact Status")

    try:
        artifacts = get_artifact_status()
    except Exception as e:
        st.error(f"❌ Failed to read artifact status: {e}")
        artifacts = {}

    status_data = []
    for name, info in (artifacts or {}).items():
        # Each artifact entry may be a dict or a simple bool.
        exists = bool(info.get("exists", False)) if isinstance(info, dict) else bool(info)

        # Prefer 'path'; otherwise gracefully fall back to 'dir', 'root', or 'paths' (first one).
        raw_path = None
        if isinstance(info, dict):
            raw_path = (
                info.get("path")
                or info.get("dir")
                or info.get("root")
                or (info.get("paths")[0] if isinstance(info.get("paths"), (list, tuple)) and info["paths"] else None)
            )

        size_str, mod_str = "—", "—"

        try:
            if exists and raw_path:
                p = Path(raw_path)
                if p.exists():
                    if p.is_file():
                        size_str = f"{p.stat().st_size / (1024*1024):.1f} MB"
                        mod_str = pd.Timestamp.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                    elif p.is_dir():
                        # directory size (approximate)
                        total_size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                        size_str = f"{total_size / (1024*1024):.1f} MB"
                        # last modified among files
                        latest_mod = max((f.stat().st_mtime for f in p.rglob("*") if f.is_file()), default=0)
                        if latest_mod:
                            mod_str = pd.Timestamp.fromtimestamp(latest_mod).strftime("%Y-%m-%d %H:%M")
        except Exception:
            # Keep defaults if we cannot stat
            pass

        status_data.append(
            {
                "Artifact": name.replace("_", " ").title(),
                "Status": "✅ Exists" if exists else "❌ Missing",
                "Size": size_str,
                "Last Modified": mod_str,
                "Path": str(raw_path) if raw_path else "—",
            }
        )

    status_df = pd.DataFrame(status_data) if status_data else pd.DataFrame(
        [{"Artifact": "—", "Status": "—", "Size": "—", "Last Modified": "—", "Path": "—"}]
    )

    st.dataframe(
        status_df,
        hide_index=True,
        column_config={"Path": st.column_config.TextColumn("Path", width=360)},
        use_container_width=True,
    )

    # ---------------------------
    # 🔍 Data quality checks
    # ---------------------------
    st.subheader("🔍 Data Quality Checks")

    if st.button("🧪 Run Quality Checks", use_container_width=True):
        with st.status("Running data quality checks...") as status:
            quality_results = {}

            # Dataset checks
            status.update(label="Checking dataset…")
            try:
                df = load_dataset(1.0, 42)  # Full dataset for quality check
                if hasattr(df, "empty") and not df.empty:
                    quality_results["dataset"] = {
                        "total_rows": len(df),
                        "null_text_rows": df["text"].isna().sum() if "text" in df.columns else 0,
                        "empty_text_rows": (df["text"] == "").sum() if "text" in df.columns else 0,
                        "duplicate_doc_ids": df["doc_id"].duplicated().sum() if "doc_id" in df.columns else 0,
                        "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                    }
                else:
                    quality_results["dataset"] = {"error": "Dataset is empty or failed to load"}
            except Exception as e:
                quality_results["dataset"] = {"error": str(e)}

            # Compact dataset checks
            status.update(label="Checking compact dataset…")
            try:
                sdf = load_compact_dataset(spark, 1.0, 42)
                if sdf:
                    compact_count = sdf.count()
                    null_tokens = sdf.filter(col("tokens_final").isNull()).count() if "tokens_final" in sdf.columns else 0
                    empty_tokens = (
                        sdf.filter(size(col("tokens_final")) == 0).count() if "tokens_final" in sdf.columns else 0
                    )
                    avg_tok = (
                        sdf.select(avg(size(col("tokens_final")))).collect()[0][0]
                        if "tokens_final" in sdf.columns
                        else None
                    )

                    quality_results["compact"] = {
                        "total_rows": compact_count,
                        "null_tokens": null_tokens,
                        "empty_tokens": empty_tokens,
                        "avg_tokens": float(avg_tok) if avg_tok is not None else "—",
                    }
                else:
                    quality_results["compact"] = {"error": "Compact dataset not available"}
            except Exception as e:
                quality_results["compact"] = {"error": str(e)}

            # Model checks
            status.update(label="Checking models…")
            models_dir = WORK_DIR / "models"
            if models_dir.exists():
                model_files = list(models_dir.rglob("*"))
                quality_results["models"] = {
                    "total_model_files": len([f for f in model_files if f.is_file()]),
                    "total_model_size_mb": sum(f.stat().st_size for f in model_files if f.is_file())
                    / (1024 * 1024),
                    "idf_models": len(list(models_dir.glob("idf_hash*"))),
                    "lsh_models": len(list(models_dir.glob("*lsh*"))),
                }
            else:
                quality_results["models"] = {"error": "Models directory not found"}

            # Results checks
            status.update(label="Checking results…")
            results_dir = WORK_DIR / "data_results"
            if results_dir.exists():
                csv_files = list(results_dir.rglob("*.csv"))
                parquet_dirs = [d for d in results_dir.rglob("*") if d.is_dir() and any(d.glob("*.parquet"))]
                quality_results["results"] = {
                    "csv_files": len(csv_files),
                    "parquet_directories": len(parquet_dirs),
                    "total_results_size_mb": sum(
                        f.stat().st_size for f in results_dir.rglob("*") if f.is_file()
                    )
                    / (1024 * 1024),
                }
            else:
                quality_results["results"] = {"error": "Results directory not found"}

            status.update(label="Quality checks complete!", state="complete")

            # Pretty display
            for category, results in quality_results.items():
                with st.expander(f"📊 {category.title()} Quality Report", expanded=True):
                    if "error" in results:
                        st.error(f"❌ {results['error']}")
                    else:
                        left, right = st.columns(2)
                        items = list(results.items())
                        midpoint = (len(items) + 1) // 2
                        for k, v in items[:midpoint]:
                            left.metric(k.replace("_", " ").title(), f"{v:.2f}" if isinstance(v, float) else v)
                        for k, v in items[midpoint:]:
                            right.metric(k.replace("_", " ").title(), f"{v:.2f}" if isinstance(v, float) else v)

    # ---------------------------
    # 🧪 Parquet sanity test
    # ---------------------------
    st.subheader("🧪 Parquet Sanity Test")

    if st.button("📝 Test Parquet Read/Write", use_container_width=True):
        with st.status("Testing Parquet operations...") as status:
            try:
                test_dir = WORK_DIR / "parquet_sanity"
                test_file = test_dir / "sanity_test.parquet"  # Spark will write a folder
                if test_file.exists():
                    shutil.rmtree(test_file)
                test_dir.mkdir(parents=True, exist_ok=True)

                status.update(label="Creating test data…")

                test_data = [
                    {
                        "id": i,
                        "text": f"Test document {i} with some বাংলা text",
                        "tokens": [f"token_{j}" for j in range(i % 10 + 1)],
                        "category": f"category_{i % 5}",
                    }
                    for i in range(100)
                ]

                test_schema = StructType(
                    [
                        StructField("id", IntegerType(), True),
                        StructField("text", StringType(), True),
                        StructField("tokens", ArrayType(StringType()), True),
                        StructField("category", StringType(), True),
                    ]
                )

                test_sdf = spark.createDataFrame(test_data, test_schema)

                status.update(label="Writing Parquet…")
                test_sdf.write.mode("overwrite").parquet(str(test_file))

                status.update(label="Reading Parquet…")
                read_sdf = spark.read.parquet(str(test_file))

                original_count = test_sdf.count()
                read_count = read_sdf.count()

                if original_count == read_count == 100:
                    original_schema = set(test_sdf.columns)
                    read_schema = set(read_sdf.columns)
                    if original_schema == read_schema:
                        st.success("✅ Parquet sanity test passed!")
                        st.info(f"✅ Successfully wrote and read {read_count} records")
                        st.dataframe(read_sdf.limit(5).toPandas(), hide_index=True)
                    else:
                        st.error(f"❌ Schema mismatch: {original_schema} vs {read_schema}")
                else:
                    st.error(f"❌ Count mismatch: wrote {original_count}, read {read_count}")

                status.update(label="Cleaning up…", state="complete")
                if test_file.exists():
                    shutil.rmtree(test_file)

            except Exception as e:
                st.error(f"❌ Parquet sanity test failed: {e}")

    # ---------------------------
    # 🗄️ Cache management
    # ---------------------------
    st.subheader("🗄️ Cache Management")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🧹 Clear Streamlit Cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("✅ Streamlit cache cleared!")

    with c2:
        if st.button("🔄 Restart Spark Session", use_container_width=True):
            try:
                if spark:
                    spark.stop()
                st.cache_resource.clear()
                st.success("✅ Spark session will restart on next operation")
            except Exception as e:
                st.error(f"❌ Failed to restart Spark: {e}")

    with c3:
        if st.button("📊 Show Cache Stats", use_container_width=True):
            try:
                st.info("Cache statistics (summary)")
                cache_info = {
                    "Cached Data Functions": "load_dataset, load_compact_dataset",
                    "Cached Resource Functions": "get_spark_session, build_or_load_*",
                    "Note": "Use 'Clear Cache' to force recomputation",
                }
                for k, v in cache_info.items():
                    st.text(f"{k}: {v}")
            except Exception as e:
                st.error(f"❌ Failed to get cache stats: {e}")

    # ---------------------------
    # ⚡ Performance monitoring
    # ---------------------------
    st.subheader("⚡ Performance Monitoring")

    if spark:
        try:
            spark_ui = spark.sparkContext.uiWebUrl
            if spark_ui:
                st.info(f"🔗 Spark UI: {spark_ui}")
        except Exception:
            pass

    try:
        import psutil

        with st.expander("💻 System Resources", expanded=False):
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("CPU Usage", f"{psutil.cpu_percent(interval=1):.1f}%")
            with r2:
                mem = psutil.virtual_memory()
                st.metric("Memory Usage", f"{mem.percent:.1f}%")
            with r3:
                disk = psutil.disk_usage(str(APP_ROOT))
                st.metric("Disk Usage", f"{(disk.used / disk.total) * 100:.1f}%")
    except Exception:
        st.info("Install psutil for system resource monitoring: pip install psutil")

    # ---------------------------
    # 🧹 Cleanup utilities
    # ---------------------------
    st.subheader("🧹 Cleanup Utilities")
    st.warning("⚠️ These actions permanently delete data!")

    u1, u2 = st.columns(2)
    with u1:
        if st.button("🗑️ Clean Temporary Files", type="secondary", use_container_width=True):
            try:
                temp_dirs = [WORK_DIR / "spark_local", WORK_DIR / "checkpoints", WORK_DIR / "parquet_sanity"]
                cleaned = 0
                for d in temp_dirs:
                    if d.exists():
                        for f in d.rglob("*"):
                            if f.is_file():
                                cleaned += f.stat().st_size
                        shutil.rmtree(d)
                        d.mkdir(parents=True, exist_ok=True)
                st.success(f"✅ Cleaned {cleaned / (1024*1024):.1f} MB of temporary files")
            except Exception as e:
                st.error(f"❌ Cleanup failed: {e}")

    with u2:
        if st.button("🗑️ Reset All Artifacts", type="secondary", use_container_width=True):
            if st.checkbox("I understand this will delete all processed data", value=False):
                try:
                    dirs_to_clean = [
                        WORK_DIR / "data_clean",
                        WORK_DIR / "models",
                        WORK_DIR / "data_results",
                        WORK_DIR / "eda_figs",
                    ]
                    for d in dirs_to_clean:
                        if d.exists():
                            shutil.rmtree(d)
                    create_directory_structure()
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.success("✅ All artifacts reset! Please rebuild your models and data.")
                except Exception as e:
                    st.error(f"❌ Reset failed: {e}")

    # ---------------------------
    # 📝 Recent activity log (simple)
    # ---------------------------
    st.subheader("📝 Recent Activity")
    with st.expander("Activity Log", expanded=False):
        log_entries = [
            "App started",
            "System preflight checks completed",
            "User accessed Quality & Monitoring tab",
        ]
        if artifacts.get("compact", {}).get("exists"):
            log_entries.append("Compact dataset found")
        if artifacts.get("idf_models", {}).get("exists"):
            log_entries.append("IDF models found")
        if artifacts.get("lsh_models", {}).get("exists"):
            log_entries.append("LSH models found")
        for entry in reversed(log_entries[-10:]):
            st.text(f"• {entry}")

# =====================================
# MAIN APP EXECUTION
# =====================================

if __name__ == "__main__":
    if not SPARK_AVAILABLE:
        st.error("❌ PySpark is not available. Please install it: `pip install pyspark`")
        st.stop()
    
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application error: {e}")
        st.info("Please check your Python environment and try restarting the app.")
        
        # Show error details in expander
        with st.expander("🐛 Error Details"):
            import traceback
            st.code(traceback.format_exc())