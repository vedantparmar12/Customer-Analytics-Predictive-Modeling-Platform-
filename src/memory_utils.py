"""
Memory optimization utilities for large dataset processing
"""
import pandas as pd
import numpy as np
import gc
from typing import Dict, Optional
from .logger import get_logger

logger = get_logger(__name__)

class MemoryOptimizer:
    """Utilities for memory-efficient data processing"""
    
    @staticmethod
    def optimize_dtypes(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """Optimize data types to reduce memory usage"""
        initial_memory = df.memory_usage().sum() / 1024**2
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= 0:
                if col_max < 255:
                    df[col] = df[col].astype(np.uint8)
                elif col_max < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif col_max < 4294967295:
                    df[col] = df[col].astype(np.uint32)
            else:
                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
        
        # Optimize float columns
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert object columns with low cardinality to category
        for col in df.select_dtypes(include=['object']).columns:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:
                df[col] = df[col].astype('category')
        
        final_memory = df.memory_usage().sum() / 1024**2
        
        if verbose:
            logger.info(f"Memory usage reduced from {initial_memory:.2f} MB to {final_memory:.2f} MB "
                       f"({(1 - final_memory/initial_memory)*100:.1f}% reduction)")
        
        return df
    
    @staticmethod
    def read_csv_in_chunks(filepath: str, chunksize: int = 10000, 
                          optimize: bool = True, **kwargs) -> pd.DataFrame:
        """Read large CSV files in chunks with memory optimization"""
        chunks = []
        
        with pd.read_csv(filepath, chunksize=chunksize, **kwargs) as reader:
            for i, chunk in enumerate(reader):
                if optimize:
                    chunk = MemoryOptimizer.optimize_dtypes(chunk, verbose=False)
                chunks.append(chunk)
                
                if i % 10 == 0:
                    logger.info(f"Processed {(i+1)*chunksize:,} rows...")
        
        df = pd.concat(chunks, ignore_index=True)
        
        if optimize:
            df = MemoryOptimizer.optimize_dtypes(df)
        
        return df
    
    @staticmethod
    def reduce_memory_usage(df: pd.DataFrame, columns_to_drop: Optional[list] = None,
                           sample_frac: Optional[float] = None) -> pd.DataFrame:
        """General memory reduction techniques"""
        # Drop unnecessary columns
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop, errors='ignore')
            logger.info(f"Dropped {len(columns_to_drop)} columns")
        
        # Sample data if specified
        if sample_frac and sample_frac < 1.0:
            original_size = len(df)
            df = df.sample(frac=sample_frac, random_state=42)
            logger.info(f"Sampled {sample_frac*100:.1f}% of data "
                       f"({original_size:,} -> {len(df):,} rows)")
        
        # Force garbage collection
        gc.collect()
        
        return df
    
    @staticmethod
    def batch_process(df: pd.DataFrame, process_func, batch_size: int = 1000, 
                     **kwargs) -> pd.DataFrame:
        """Process DataFrame in batches to avoid memory spikes"""
        results = []
        n_batches = (len(df) + batch_size - 1) // batch_size
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            result = process_func(batch, **kwargs)
            results.append(result)
            
            if i // batch_size % 10 == 0:
                logger.info(f"Processed batch {i//batch_size + 1}/{n_batches}")
                gc.collect()
        
        return pd.concat(results, ignore_index=True)
    
    @staticmethod
    def get_memory_usage(df: pd.DataFrame) -> Dict[str, float]:
        """Get detailed memory usage statistics"""
        memory_usage = df.memory_usage(deep=True)
        total_mb = memory_usage.sum() / 1024**2
        
        column_memory = {}
        for col in df.columns:
            col_mb = memory_usage[col] / 1024**2
            column_memory[col] = {
                "mb": col_mb,
                "percentage": (col_mb / total_mb) * 100 if total_mb > 0 else 0,
                "dtype": str(df[col].dtype)
            }
        
        return {
            "total_mb": total_mb,
            "columns": column_memory,
            "shape": df.shape,
            "memory_per_row": total_mb / len(df) * 1024 if len(df) > 0 else 0  # KB per row
        }