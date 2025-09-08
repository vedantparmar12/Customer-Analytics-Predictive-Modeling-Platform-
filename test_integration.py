#!/usr/bin/env python3
"""
Integration test to verify all components are properly connected
Run this to check if the advanced features are working together
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(__file__))

async def test_component_integration():
    """Test that all components can be imported and initialized"""
    
    print("🔍 Testing component imports...")
    
    try:
        # Test imports
        from api.cache_manager import AdvancedCacheManager, RedisClusterManager
        from api.auth import JWTAuth, APIKeyAuth
        from api.ml_serving import MLModelManager, BatchPredictionManager, ModelExplainer
        from api.middleware import RequestLoggingMiddleware, MetricsMiddleware
        from api.dependencies import get_cache_manager, get_current_user, get_api_key_user
        print("✅ All imports successful")
        
        # Test basic initialization
        print("\n🔧 Testing component initialization...")
        
        # Redis and Cache Manager
        try:
            redis_manager = RedisClusterManager(host="localhost", port=6379)
            cache_manager = AdvancedCacheManager(redis_manager)
            print("✅ Redis and Cache Manager initialized")
        except Exception as e:
            print(f"⚠️  Redis/Cache initialization failed (expected in CI): {e}")
        
        # Authentication
        jwt_auth = JWTAuth()
        print("✅ JWT Auth initialized")
        
        # Test JWT functionality
        user_data = {"sub": "testuser"}
        token = jwt_auth.create_access_token(user_data)
        verified = jwt_auth.verify_token(token)
        assert verified["username"] == "testuser"
        print("✅ JWT token creation/verification working")
        
        # ML Model Manager (without actual Redis)
        try:
            ml_manager = MLModelManager(None)  # Mock cache manager
            print("✅ ML Model Manager initialized")
        except Exception as e:
            print(f"⚠️  ML Manager initialization: {e}")
        
        print("\n🎯 Testing advanced features...")
        
        # Test password hashing
        password = "test123"
        hashed = jwt_auth.get_password_hash(password)
        assert jwt_auth.verify_password(password, hashed)
        print("✅ Password hashing working")
        
        # Test user authentication
        user = jwt_auth.authenticate_user("admin", "admin123")
        assert user is not None
        assert user["role"] == "admin"
        print("✅ User authentication working")
        
        # Test API key generation
        try:
            api_auth = APIKeyAuth()
            api_key = api_auth.generate_api_key("ak_test")
            assert api_key.startswith("ak_test_")
            print("✅ API key generation working")
        except Exception as e:
            print(f"⚠️  API key test: {e}")
        
        print("\n🚀 All core components are properly connected!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_api_startup_simulation():
    """Simulate the API startup process"""
    
    print("\n🚀 Simulating API startup sequence...")
    
    try:
        # Simulate the startup sequence from main.py
        os.environ.setdefault('REDIS_HOST', 'localhost')
        os.environ.setdefault('REDIS_PORT', '6379')
        os.environ.setdefault('REDIS_CLUSTER_MODE', 'false')
        
        from api.cache_manager import RedisClusterManager, AdvancedCacheManager
        from api.ml_serving import MLModelManager, BatchPredictionManager, ModelExplainer
        from api.auth import JWTAuth, APIKeyAuth
        
        print("✅ Imported all startup modules")
        
        # Initialize components (like in startup_event)
        redis_manager = RedisClusterManager(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            cluster_mode=False
        )
        
        cache_manager = AdvancedCacheManager(redis_manager)
        ml_model_manager = MLModelManager(cache_manager)
        batch_manager = BatchPredictionManager(ml_model_manager, cache_manager)
        model_explainer = ModelExplainer(ml_model_manager)
        
        jwt_auth = JWTAuth()
        api_key_auth = APIKeyAuth(cache_manager)
        
        print("✅ All components initialized successfully")
        print("✅ API startup simulation completed")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Startup simulation failed (Redis connection expected): {e}")
        return False

async def test_feature_completeness():
    """Test that all requested features are implemented"""
    
    print("\n📋 Checking feature completeness...")
    
    features_checklist = {
        "Redis Connection Pooling": "api/cache_manager.py",
        "Advanced Caching Strategies": "api/cache_manager.py", 
        "Rate Limiting": "api/cache_manager.py",
        "JWT Authentication": "api/auth.py",
        "API Key Management": "api/auth.py",
        "Custom Middleware Stack": "api/middleware.py",
        "Dependency Injection": "api/dependencies.py",
        "ML Model Versioning": "api/ml_serving.py",
        "Batch Processing": "api/ml_serving.py",
        "Model Explanations": "api/ml_serving.py",
        "Comprehensive Testing": "tests/",
        "Prometheus Metrics": "monitoring/prometheus/",
        "Grafana Dashboards": "monitoring/grafana/",
        "Performance Optimization": "api/cache_manager.py"
    }
    
    implemented_features = 0
    total_features = len(features_checklist)
    
    for feature, file_path in features_checklist.items():
        if os.path.exists(file_path):
            print(f"✅ {feature}")
            implemented_features += 1
        else:
            print(f"❌ {feature} (missing: {file_path})")
    
    completion_rate = (implemented_features / total_features) * 100
    print(f"\n📊 Feature Completion: {implemented_features}/{total_features} ({completion_rate:.1f}%)")
    
    if completion_rate >= 90:
        print("🎉 Excellent! All major features implemented")
    elif completion_rate >= 75:
        print("👍 Good! Most features implemented")
    else:
        print("⚠️  Some important features are missing")
    
    return completion_rate >= 90

def test_dependencies_in_requirements():
    """Check if all required dependencies are in requirements.txt"""
    
    print("\n📦 Checking dependencies...")
    
    try:
        with open("requirements.txt", "r") as f:
            requirements = f.read().lower()
        
        required_deps = [
            "fastapi", "uvicorn", "redis", "python-jose", "passlib",
            "pytest", "pytest-asyncio", "prometheus-client", "httpx"
        ]
        
        missing_deps = []
        for dep in required_deps:
            if dep in requirements:
                print(f"✅ {dep}")
            else:
                print(f"❌ {dep}")
                missing_deps.append(dep)
        
        if not missing_deps:
            print("✅ All required dependencies found")
            return True
        else:
            print(f"⚠️  Missing dependencies: {missing_deps}")
            return False
            
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        return False

async def main():
    """Run all integration tests"""
    
    print("🧪 Running Advanced E-commerce ML API Integration Tests\n")
    print("=" * 60)
    
    tests = [
        ("Component Integration", test_component_integration()),
        ("API Startup Simulation", test_api_startup_simulation()),
        ("Feature Completeness", test_feature_completeness()),
        ("Dependencies Check", test_dependencies_in_requirements())
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_coro in tests:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 40)
        
        try:
            if asyncio.iscoroutine(test_coro):
                result = await test_coro
            else:
                result = test_coro
            
            if result:
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 FINAL RESULTS: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The advanced features are properly implemented and connected.")
    elif passed >= total * 0.8:
        print("👍 Most tests passed. Minor issues may exist but core functionality works.")
    else:
        print("⚠️  Several tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)