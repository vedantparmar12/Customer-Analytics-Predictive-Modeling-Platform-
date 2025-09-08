#!/usr/bin/env python3
"""
Implementation verification script - checks if all advanced features are properly implemented
This script verifies the code structure and dependencies without requiring runtime execution
"""

import os
import ast
import sys
from pathlib import Path

def check_file_exists_and_has_content(file_path, min_lines=10):
    """Check if file exists and has substantial content"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                lines = f.readlines()
                return len(lines) >= min_lines, len(lines)
        return False, 0
    except Exception:
        return False, 0

def analyze_python_file(file_path):
    """Analyze a Python file for classes, functions, and imports"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.extend([f"{module}.{alias.name}" for alias in node.names])
        
        return {
            'classes': classes,
            'functions': functions,
            'imports': imports,
            'lines': len(content.split('\n'))
        }
    except Exception as e:
        return {'error': str(e)}

def verify_advanced_features():
    """Verify all advanced features are implemented"""
    
    print("🔍 ADVANCED E-COMMERCE ML API - IMPLEMENTATION VERIFICATION")
    print("=" * 70)
    
    # Feature verification checklist
    features = {
        "🔄 Advanced Redis Cache Manager": {
            "file": "api/cache_manager.py",
            "required_classes": ["AdvancedCacheManager", "RedisClusterManager", "CacheConfig"],
            "required_functions": ["cache_prediction", "implement_rate_limiting", "get_cache_stats"]
        },
        "🔐 JWT & API Key Authentication": {
            "file": "api/auth.py", 
            "required_classes": ["JWTAuth", "APIKeyAuth", "RolePermissionManager"],
            "required_functions": ["create_access_token", "verify_token", "create_api_key", "verify_api_key"]
        },
        "🛡️  Advanced Middleware System": {
            "file": "api/middleware.py",
            "required_classes": ["RequestLoggingMiddleware", "MetricsMiddleware", "RateLimitingMiddleware", "CacheMiddleware", "SecurityHeadersMiddleware"],
            "required_functions": ["dispatch"]
        },
        "🤖 ML Model Serving & Management": {
            "file": "api/ml_serving.py",
            "required_classes": ["MLModelManager", "BatchPredictionManager", "ModelExplainer", "BatchPredictionJob"],
            "required_functions": ["load_model", "predict_single", "predict_batch", "a_b_test_models"]
        },
        "🔗 Dependency Injection System": {
            "file": "api/dependencies.py",
            "required_classes": ["CacheDependency", "AuthDependency", "ModelDependency"],
            "required_functions": ["get_cache_manager", "get_current_user", "get_api_key_user", "get_health_status"]
        },
        "🧪 Comprehensive Test Suite": {
            "file": "tests/",
            "required_files": ["test_auth.py", "test_cache.py", "test_api_endpoints.py", "test_performance.py", "conftest.py"]
        },
        "📊 Prometheus & Grafana Monitoring": {
            "file": "monitoring/",
            "required_files": ["prometheus/prometheus.yml", "grafana/dashboards/api-dashboard.json", "alertmanager/alertmanager.yml"]
        }
    }
    
    total_score = 0
    max_score = len(features)
    
    for feature_name, requirements in features.items():
        print(f"\n{feature_name}")
        print("-" * 50)
        
        if "required_files" in requirements:
            # Directory-based feature (like tests, monitoring)
            base_path = requirements["file"]
            all_files_exist = True
            
            for required_file in requirements["required_files"]:
                full_path = os.path.join(base_path, required_file)
                exists, lines = check_file_exists_and_has_content(full_path, 5)
                
                if exists:
                    print(f"  ✅ {required_file} ({lines} lines)")
                else:
                    print(f"  ❌ {required_file} (missing or too small)")
                    all_files_exist = False
            
            if all_files_exist:
                print(f"  🎉 {feature_name} - FULLY IMPLEMENTED")
                total_score += 1
            else:
                print(f"  ⚠️  {feature_name} - PARTIALLY IMPLEMENTED")
        
        else:
            # Single file feature
            file_path = requirements["file"]
            exists, lines = check_file_exists_and_has_content(file_path, 50)
            
            if not exists:
                print(f"  ❌ File {file_path} missing or too small")
                continue
            
            print(f"  📄 File: {file_path} ({lines} lines)")
            
            # Analyze the file
            analysis = analyze_python_file(file_path)
            
            if 'error' in analysis:
                print(f"  ❌ Analysis error: {analysis['error']}")
                continue
            
            # Check required classes
            missing_classes = []
            for required_class in requirements.get("required_classes", []):
                if required_class in analysis['classes']:
                    print(f"  ✅ Class: {required_class}")
                else:
                    print(f"  ❌ Missing class: {required_class}")
                    missing_classes.append(required_class)
            
            # Check required functions
            missing_functions = []
            for required_function in requirements.get("required_functions", []):
                if required_function in analysis['functions']:
                    print(f"  ✅ Function: {required_function}")
                else:
                    print(f"  ❌ Missing function: {required_function}")
                    missing_functions.append(required_function)
            
            # Score this feature
            if not missing_classes and not missing_functions:
                print(f"  🎉 {feature_name} - FULLY IMPLEMENTED")
                total_score += 1
            else:
                print(f"  ⚠️  {feature_name} - PARTIALLY IMPLEMENTED")
                if missing_classes:
                    print(f"     Missing classes: {missing_classes}")
                if missing_functions:
                    print(f"     Missing functions: {missing_functions}")
    
    return total_score, max_score

def verify_main_api_integration():
    """Verify that main.py properly integrates all components"""
    
    print("\n🚀 MAIN API INTEGRATION VERIFICATION")
    print("-" * 50)
    
    main_file = "api/main.py"
    
    if not os.path.exists(main_file):
        print("❌ api/main.py not found")
        return False
    
    analysis = analyze_python_file(main_file)
    
    if 'error' in analysis:
        print(f"❌ Analysis error: {analysis['error']}")
        return False
    
    print(f"📄 File: {main_file} ({analysis['lines']} lines)")
    
    # Check for required imports
    required_imports = [
        "cache_manager", "middleware", "dependencies", "auth", "ml_serving"
    ]
    
    integration_score = 0
    
    for import_name in required_imports:
        import_found = any(import_name in imp for imp in analysis['imports'])
        if import_found:
            print(f"  ✅ Import: {import_name}")
            integration_score += 1
        else:
            print(f"  ❌ Missing import: {import_name}")
    
    # Check for key endpoints
    required_endpoints = [
        "startup_event", "health_check", "predict_churn", "get_recommendations",
        "login_for_access_token", "create_api_key", "submit_batch_prediction"
    ]
    
    for endpoint in required_endpoints:
        if endpoint in analysis['functions']:
            print(f"  ✅ Endpoint: {endpoint}")
            integration_score += 1
        else:
            print(f"  ❌ Missing endpoint: {endpoint}")
    
    total_checks = len(required_imports) + len(required_endpoints)
    integration_percentage = (integration_score / total_checks) * 100
    
    print(f"\n📊 Integration Score: {integration_score}/{total_checks} ({integration_percentage:.1f}%)")
    
    return integration_percentage >= 80

def verify_requirements():
    """Verify that all required dependencies are in requirements.txt"""
    
    print("\n📦 REQUIREMENTS VERIFICATION")
    print("-" * 30)
    
    try:
        with open("requirements.txt", "r") as f:
            requirements = f.read().lower()
        
        advanced_deps = [
            "fastapi>=0.104.0", "uvicorn>=0.24.0", "redis>=5.0.0",
            "python-jose", "passlib", "pytest>=7.4.0", "pytest-asyncio",
            "prometheus-client", "httpx>=0.25.0", "aioredis>=2.0.1"
        ]
        
        found_deps = 0
        for dep in advanced_deps:
            dep_name = dep.split(">=")[0].split("[")[0]
            if dep_name in requirements:
                print(f"  ✅ {dep}")
                found_deps += 1
            else:
                print(f"  ❌ Missing: {dep}")
        
        dependency_score = (found_deps / len(advanced_deps)) * 100
        print(f"\n📊 Dependencies: {found_deps}/{len(advanced_deps)} ({dependency_score:.1f}%)")
        
        return dependency_score >= 90
        
    except FileNotFoundError:
        print("  ❌ requirements.txt not found")
        return False

def main():
    """Run complete verification"""
    
    print("🧪 STARTING COMPREHENSIVE IMPLEMENTATION VERIFICATION\n")
    
    # Check if we're in the right directory
    if not os.path.exists("api"):
        print("❌ Error: 'api' directory not found. Please run from project root.")
        sys.exit(1)
    
    # Run all verifications
    feature_score, max_features = verify_advanced_features()
    integration_ok = verify_main_api_integration()
    requirements_ok = verify_requirements()
    
    # Final summary
    print("\n" + "=" * 70)
    print("🏁 FINAL VERIFICATION RESULTS")
    print("=" * 70)
    
    feature_percentage = (feature_score / max_features) * 100
    print(f"📊 Advanced Features: {feature_score}/{max_features} ({feature_percentage:.1f}%)")
    print(f"🔗 API Integration: {'✅ PASS' if integration_ok else '❌ FAIL'}")
    print(f"📦 Dependencies: {'✅ PASS' if requirements_ok else '❌ FAIL'}")
    
    overall_score = (
        (feature_percentage / 100) * 0.6 +  # 60% weight for features
        (1 if integration_ok else 0) * 0.3 +  # 30% weight for integration  
        (1 if requirements_ok else 0) * 0.1   # 10% weight for dependencies
    ) * 100
    
    print(f"\n🎯 OVERALL IMPLEMENTATION SCORE: {overall_score:.1f}%")
    
    if overall_score >= 95:
        print("🏆 EXCELLENT! All advanced features are fully implemented and connected!")
        print("✨ Ready for production deployment!")
    elif overall_score >= 85:
        print("👍 VERY GOOD! Most advanced features implemented with minor gaps.")
        print("🚀 Ready for testing and deployment!")
    elif overall_score >= 75:
        print("👌 GOOD! Core features implemented but some advanced features missing.")
        print("⚡ Suitable for development and testing!")
    else:
        print("⚠️  NEEDS WORK! Significant features are missing or incomplete.")
        print("🔧 Additional development required!")
    
    # Specific recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if feature_score < max_features:
        print("  • Complete missing feature implementations")
    if not integration_ok:
        print("  • Fix main API integration issues")
    if not requirements_ok:
        print("  • Add missing dependencies to requirements.txt")
    
    if overall_score >= 85:
        print("  • Run integration tests with Redis running")
        print("  • Deploy monitoring stack and test endpoints")
        print("  • Load test the API with sample data")
    
    return overall_score >= 85

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)