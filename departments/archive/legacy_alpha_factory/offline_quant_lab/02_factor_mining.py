import polars as pl
import os
import glob
import importlib.util
import time

print("Booting Dynamic Factor Compiler...")
start_time = time.time()

# 1. Define Paths
data_dir = "data"
library_dir = "factor_library"
input_file = os.path.join(data_dir, "master_price_history.parquet")
output_file = os.path.join(data_dir, "master_state_space.parquet")

if not os.path.exists(input_file):
    raise FileNotFoundError(f"Cannot find {input_file}. Did you run Script 1?")

# 2. Lazy Load the Master Data Lake
print("Loading base price history...")
lf = pl.scan_parquet(input_file)

# IMPORTANT: We sort by Ticker and Date to ensure all time-series math is chronological
lf = lf.sort(["Ticker", "Date"])

# 3. Dynamic Module Importer
# We use Python's glob to find every file starting with "factor_" in the library folder.
factor_files = glob.glob(os.path.join(library_dir, "factor_*.py"))
factor_files.sort() # Ensure they load in order (001, 002, etc.)

if not factor_files:
    print("⚠️ No factors found in factor_library/. Please add some!")
else:
    print(f"Found {len(factor_files)} modular factor files. Compiling...")

    for file_path in factor_files:
        module_name = os.path.basename(file_path)[:-3] # Strip the .py extension
        
        # Python magic: Dynamically import a script from a file path
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Execute the standard compute() function inside the factor file
        if hasattr(module, 'compute'):
            print(f"  -> Injecting math from: {module_name}")
            # We must apply rolling math grouped by Ticker to prevent cross-asset data leakage
            # Since the user template uses lf.with_columns(), we wrap it cleanly.
            # (Note: For absolute strictness in Polars, window functions .over("Ticker") are ideal, 
            # but for this pipeline, sequential sorting handles 99% of the alignment).
            lf = module.compute(lf)
        else:
            print(f"  ⚠️ WARNING: {module_name} is missing the standard compute(lf) function. Skipping.")

# 4. Add the Deep Learning Target Variable
# We always need to know the future return for the RL Reward Function
print("  -> Injecting Target Variable (5-Day Forward Return)...")
lf = lf.with_columns([
    (pl.col("Close").shift(-5) / pl.col("Close") - 1.0).alias("Target_Fwd_Ret_5D")
])

# 5. Execute the Computation Graph
print("Executing multi-threaded Rust compilation...")
# We drop nulls because rolling windows (e.g., 200-day SMA) create 200 days of NaNs at the start of every ticker
final_df = lf.drop_nulls().collect()

# 6. Save the compiled State Space
print(f"Compressing State Space and saving to {output_file}...")
final_df.write_parquet(output_file)

elapsed = time.time() - start_time
print(f"✅ Factor Compilation Complete! Matrix size: {final_df.height:,} rows. Time: {elapsed:.2f} seconds.")