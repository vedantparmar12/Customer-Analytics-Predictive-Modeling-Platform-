"""
Data validation module for ensuring data quality and integrity
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from .logger import get_logger
from .custom_exception import CustomException

logger = get_logger(__name__)

class DataValidator:
    """Validates data quality and integrity"""
    
    def __init__(self):
        self.validation_results = {}
        
    def validate_schema(self, df: pd.DataFrame, expected_columns: List[str], 
                       table_name: str) -> Tuple[bool, List[str]]:
        """Validate if DataFrame has expected columns"""
        try:
            missing_columns = set(expected_columns) - set(df.columns)
            extra_columns = set(df.columns) - set(expected_columns)
            
            is_valid = len(missing_columns) == 0
            
            if missing_columns:
                logger.warning(f"{table_name}: Missing columns: {missing_columns}")
            if extra_columns:
                logger.info(f"{table_name}: Extra columns found: {extra_columns}")
                
            self.validation_results[f"{table_name}_schema"] = {
                "valid": is_valid,
                "missing_columns": list(missing_columns),
                "extra_columns": list(extra_columns)
            }
            
            return is_valid, list(missing_columns)
            
        except Exception as e:
            logger.error(f"Error validating schema for {table_name}: {e}")
            raise CustomException(f"Schema validation failed for {table_name}", e)
    
    def validate_data_types(self, df: pd.DataFrame, expected_types: Dict[str, str], 
                           table_name: str) -> Tuple[bool, Dict[str, str]]:
        """Validate data types of columns"""
        try:
            type_mismatches = {}
            
            for col, expected_type in expected_types.items():
                if col in df.columns:
                    actual_type = str(df[col].dtype)
                    if expected_type == "datetime":
                        if not pd.api.types.is_datetime64_any_dtype(df[col]):
                            type_mismatches[col] = f"Expected datetime, got {actual_type}"
                    elif expected_type == "numeric":
                        if not pd.api.types.is_numeric_dtype(df[col]):
                            type_mismatches[col] = f"Expected numeric, got {actual_type}"
                    elif expected_type == "string":
                        if not pd.api.types.is_string_dtype(df[col]) and not pd.api.types.is_object_dtype(df[col]):
                            type_mismatches[col] = f"Expected string, got {actual_type}"
            
            is_valid = len(type_mismatches) == 0
            
            if type_mismatches:
                logger.warning(f"{table_name}: Type mismatches: {type_mismatches}")
                
            self.validation_results[f"{table_name}_types"] = {
                "valid": is_valid,
                "mismatches": type_mismatches
            }
            
            return is_valid, type_mismatches
            
        except Exception as e:
            logger.error(f"Error validating types for {table_name}: {e}")
            raise CustomException(f"Type validation failed for {table_name}", e)
    
    def validate_nulls(self, df: pd.DataFrame, non_null_columns: List[str], 
                      table_name: str, threshold: float = 0.05) -> Tuple[bool, Dict[str, float]]:
        """Validate null values in critical columns"""
        try:
            null_issues = {}
            
            for col in non_null_columns:
                if col in df.columns:
                    null_ratio = df[col].isnull().sum() / len(df)
                    if null_ratio > threshold:
                        null_issues[col] = null_ratio
            
            is_valid = len(null_issues) == 0
            
            if null_issues:
                logger.warning(f"{table_name}: High null ratios: {null_issues}")
                
            self.validation_results[f"{table_name}_nulls"] = {
                "valid": is_valid,
                "high_null_columns": null_issues
            }
            
            return is_valid, null_issues
            
        except Exception as e:
            logger.error(f"Error validating nulls for {table_name}: {e}")
            raise CustomException(f"Null validation failed for {table_name}", e)
    
    def validate_value_ranges(self, df: pd.DataFrame, range_checks: Dict[str, Dict], 
                             table_name: str) -> Tuple[bool, Dict[str, str]]:
        """Validate value ranges for numeric columns"""
        try:
            range_issues = {}
            
            for col, checks in range_checks.items():
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    if "min" in checks and df[col].min() < checks["min"]:
                        range_issues[col] = f"Values below minimum {checks['min']}"
                    if "max" in checks and df[col].max() > checks["max"]:
                        range_issues[col] = f"Values above maximum {checks['max']}"
                    if "positive" in checks and checks["positive"] and (df[col] < 0).any():
                        range_issues[col] = "Negative values found"
            
            is_valid = len(range_issues) == 0
            
            if range_issues:
                logger.warning(f"{table_name}: Range issues: {range_issues}")
                
            self.validation_results[f"{table_name}_ranges"] = {
                "valid": is_valid,
                "issues": range_issues
            }
            
            return is_valid, range_issues
            
        except Exception as e:
            logger.error(f"Error validating ranges for {table_name}: {e}")
            raise CustomException(f"Range validation failed for {table_name}", e)
    
    def validate_duplicates(self, df: pd.DataFrame, key_columns: List[str], 
                           table_name: str) -> Tuple[bool, int]:
        """Check for duplicate records"""
        try:
            duplicates = df.duplicated(subset=key_columns, keep=False).sum()
            is_valid = duplicates == 0
            
            if duplicates > 0:
                logger.warning(f"{table_name}: Found {duplicates} duplicate records")
                
            self.validation_results[f"{table_name}_duplicates"] = {
                "valid": is_valid,
                "duplicate_count": duplicates
            }
            
            return is_valid, duplicates
            
        except Exception as e:
            logger.error(f"Error checking duplicates for {table_name}: {e}")
            raise CustomException(f"Duplicate check failed for {table_name}", e)
    
    def validate_relationships(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                              key1: str, key2: str, relationship_name: str) -> Tuple[bool, Dict]:
        """Validate foreign key relationships between tables"""
        try:
            missing_keys = set(df1[key1].unique()) - set(df2[key2].unique())
            orphan_ratio = len(missing_keys) / df1[key1].nunique() if df1[key1].nunique() > 0 else 0
            
            is_valid = orphan_ratio < 0.01  # Allow 1% orphan records
            
            result = {
                "valid": is_valid,
                "orphan_count": len(missing_keys),
                "orphan_ratio": orphan_ratio
            }
            
            if not is_valid:
                logger.warning(f"{relationship_name}: {orphan_ratio:.2%} orphan records")
                
            self.validation_results[relationship_name] = result
            
            return is_valid, result
            
        except Exception as e:
            logger.error(f"Error validating relationship {relationship_name}: {e}")
            raise CustomException(f"Relationship validation failed", e)
    
    def get_validation_summary(self) -> Dict:
        """Get summary of all validation results"""
        total_checks = len(self.validation_results)
        passed_checks = sum(1 for v in self.validation_results.values() if v.get("valid", False))
        
        return {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": total_checks - passed_checks,
            "pass_rate": passed_checks / total_checks if total_checks > 0 else 0,
            "details": self.validation_results
        }


# Validation configurations for e-commerce data
ECOMMERCE_VALIDATIONS = {
    "customers": {
        "required_columns": ["customer_id", "customer_unique_id", "customer_zip_code_prefix"],
        "data_types": {
            "customer_id": "string",
            "customer_unique_id": "string",
            "customer_zip_code_prefix": "string"
        },
        "non_null_columns": ["customer_id", "customer_unique_id"],
        "key_columns": ["customer_id"]
    },
    "orders": {
        "required_columns": ["order_id", "customer_id", "order_status", "order_purchase_timestamp"],
        "data_types": {
            "order_id": "string",
            "customer_id": "string",
            "order_status": "string",
            "order_purchase_timestamp": "datetime"
        },
        "non_null_columns": ["order_id", "customer_id", "order_status"],
        "key_columns": ["order_id"]
    },
    "order_items": {
        "required_columns": ["order_id", "order_item_id", "product_id", "price", "freight_value"],
        "data_types": {
            "order_id": "string",
            "product_id": "string",
            "price": "numeric",
            "freight_value": "numeric"
        },
        "non_null_columns": ["order_id", "product_id", "price"],
        "range_checks": {
            "price": {"min": 0, "positive": True},
            "freight_value": {"min": 0, "positive": True}
        }
    },
    "products": {
        "required_columns": ["product_id", "product_category_name"],
        "data_types": {
            "product_id": "string",
            "product_category_name": "string"
        },
        "non_null_columns": ["product_id"],
        "key_columns": ["product_id"]
    }
}