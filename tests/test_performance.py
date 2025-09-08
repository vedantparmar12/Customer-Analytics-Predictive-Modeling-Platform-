"""
Performance and load tests
"""

import pytest
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import statistics


class TestPerformance:
    """Performance tests for the API"""
    
    def test_health_check_performance(self, client):
        """Test health check endpoint performance"""
        response_times = []
        
        # Make multiple requests to measure performance
        for _ in range(10):
            start_time = time.time()
            response = client.get("/health")
            end_time = time.time()
            
            assert response.status_code == 200
            response_times.append(end_time - start_time)
        
        # Analyze performance metrics
        avg_time = statistics.mean(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        
        print(f"Health check performance:")
        print(f"  Average: {avg_time:.3f}s")
        print(f"  Max: {max_time:.3f}s") 
        print(f"  Min: {min_time:.3f}s")
        
        # Performance assertions
        assert avg_time < 1.0  # Average should be under 1 second
        assert max_time < 2.0  # No request should take more than 2 seconds
    
    def test_single_prediction_performance(self, client, sample_churn_request):
        """Test single prediction performance"""
        response_times = []
        
        for _ in range(5):
            start_time = time.time()
            response = client.post("/predict/churn", json=sample_churn_request)
            end_time = time.time()
            
            # Skip if models not loaded
            if response.status_code == 503:
                pytest.skip("Models not loaded")
            
            assert response.status_code == 200
            response_times.append(end_time - start_time)
        
        avg_time = statistics.mean(response_times)
        print(f"Single prediction performance: {avg_time:.3f}s average")
        
        # Should be reasonably fast
        assert avg_time < 5.0
    
    def test_batch_prediction_performance(self, client, large_dataset):
        """Test batch prediction performance"""
        batch_request = {
            "customers": large_dataset[:100],  # First 100 customers
            "include_churn": True,
            "include_segmentation": True,
            "include_recommendations": False
        }
        
        start_time = time.time()
        response = client.post("/predict/batch", json=batch_request)
        end_time = time.time()
        
        if response.status_code == 503:
            pytest.skip("Models not loaded")
        
        assert response.status_code == 200
        
        processing_time = end_time - start_time
        predictions_per_second = len(batch_request["customers"]) / processing_time
        
        print(f"Batch prediction performance:")
        print(f"  Total time: {processing_time:.3f}s")
        print(f"  Predictions/second: {predictions_per_second:.1f}")
        
        # Should process at least 10 predictions per second
        assert predictions_per_second > 10
    
    def test_concurrent_requests(self, client, sample_churn_request):
        """Test concurrent request handling"""
        def make_request():
            start_time = time.time()
            response = client.post("/predict/churn", json=sample_churn_request)
            end_time = time.time()
            return response.status_code, end_time - start_time
        
        # Use thread pool for concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(20)]
            results = [future.result() for future in futures]
        
        status_codes = [result[0] for result in results]
        response_times = [result[1] for result in results]
        
        # All requests should complete successfully (or consistently fail)
        unique_status_codes = set(status_codes)
        assert len(unique_status_codes) <= 2  # Should be mostly consistent
        assert 200 in unique_status_codes or 503 in unique_status_codes
        
        avg_concurrent_time = statistics.mean(response_times)
        print(f"Concurrent requests performance: {avg_concurrent_time:.3f}s average")
        
        # Concurrent performance shouldn't degrade too much
        assert avg_concurrent_time < 10.0
    
    @pytest.mark.asyncio
    async def test_async_performance(self, async_client, sample_churn_request):
        """Test async request performance"""
        async def make_async_request():
            start_time = time.time()
            response = await async_client.post("/predict/churn", json=sample_churn_request)
            end_time = time.time()
            return response.status_code, end_time - start_time
        
        # Make concurrent async requests
        tasks = [make_async_request() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        status_codes = [result[0] for result in results]
        response_times = [result[1] for result in results]
        
        # Verify results
        assert all(code in [200, 503] for code in status_codes)
        
        avg_async_time = statistics.mean(response_times)
        print(f"Async requests performance: {avg_async_time:.3f}s average")


@pytest.mark.asyncio 
class TestCachePerformance:
    """Test cache performance"""
    
    async def test_cache_hit_performance(self, cache_manager):
        """Test cache hit performance"""
        if not cache_manager:
            pytest.skip("Cache manager not available")
        
        key = "perf_test_key"
        value = {"test": "data", "numbers": list(range(100))}
        
        # Set value in cache
        await cache_manager.set(key, value)
        
        # Measure cache hit performance
        hit_times = []
        for _ in range(100):
            start_time = time.time()
            result = await cache_manager.get(key)
            end_time = time.time()
            
            assert result == value
            hit_times.append(end_time - start_time)
        
        avg_hit_time = statistics.mean(hit_times)
        print(f"Cache hit performance: {avg_hit_time*1000:.2f}ms average")
        
        # Cache hits should be very fast
        assert avg_hit_time < 0.01  # Under 10ms
    
    async def test_cache_set_performance(self, cache_manager):
        """Test cache set performance"""
        if not cache_manager:
            pytest.skip("Cache manager not available")
        
        set_times = []
        
        for i in range(50):
            key = f"perf_set_key_{i}"
            value = {"index": i, "data": f"value_{i}", "list": list(range(10))}
            
            start_time = time.time()
            success = await cache_manager.set(key, value)
            end_time = time.time()
            
            assert success is True
            set_times.append(end_time - start_time)
        
        avg_set_time = statistics.mean(set_times)
        print(f"Cache set performance: {avg_set_time*1000:.2f}ms average")
        
        # Cache sets should be reasonably fast
        assert avg_set_time < 0.05  # Under 50ms
    
    async def test_bulk_cache_operations(self, cache_manager):
        """Test bulk cache operations performance"""
        if not cache_manager:
            pytest.skip("Cache manager not available")
        
        # Bulk set operations
        keys = [f"bulk_key_{i}" for i in range(100)]
        values = [{"index": i, "data": f"bulk_value_{i}"} for i in range(100)]
        
        start_time = time.time()
        for key, value in zip(keys, values):
            await cache_manager.set(key, value)
        set_end_time = time.time()
        
        # Bulk get operations
        get_start_time = time.time()
        for key in keys:
            result = await cache_manager.get(key)
            assert result is not None
        get_end_time = time.time()
        
        set_time = set_end_time - start_time
        get_time = get_end_time - get_start_time
        
        print(f"Bulk operations performance:")
        print(f"  100 sets: {set_time:.3f}s ({100/set_time:.1f} ops/sec)")
        print(f"  100 gets: {get_time:.3f}s ({100/get_time:.1f} ops/sec)")
        
        # Should handle bulk operations efficiently
        assert set_time < 5.0  # 100 sets in under 5 seconds
        assert get_time < 2.0  # 100 gets in under 2 seconds


class TestMemoryUsage:
    """Test memory usage patterns"""
    
    def test_large_dataset_handling(self, client, large_dataset):
        """Test handling of large datasets"""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process large dataset
        batch_request = {
            "customers": large_dataset,
            "include_churn": False,  # Skip ML to focus on data handling
            "include_segmentation": True,
            "include_recommendations": False
        }
        
        response = client.post("/predict/batch", json=batch_request)
        
        # Get memory after processing
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage for {len(large_dataset)} customers:")
        print(f"  Initial: {initial_memory / 1024 / 1024:.1f} MB")
        print(f"  Final: {final_memory / 1024 / 1024:.1f} MB")
        print(f"  Increase: {memory_increase / 1024 / 1024:.1f} MB")
        
        # Memory increase should be reasonable
        assert memory_increase < 100 * 1024 * 1024  # Under 100MB increase
    
    def test_memory_cleanup(self, client, sample_churn_request):
        """Test that memory is properly cleaned up"""
        import psutil
        import os
        import gc
        
        process = psutil.Process(os.getpid())
        
        # Make many requests to potentially cause memory leaks
        for _ in range(50):
            response = client.post("/predict/churn", json=sample_churn_request)
        
        # Force garbage collection
        gc.collect()
        
        # Memory shouldn't grow excessively
        memory_usage = process.memory_info().rss
        print(f"Memory after 50 requests: {memory_usage / 1024 / 1024:.1f} MB")
        
        # This is a basic check - in production you'd want more sophisticated monitoring
        assert memory_usage < 500 * 1024 * 1024  # Under 500MB


class TestStressTest:
    """Stress tests for the API"""
    
    @pytest.mark.slow
    def test_sustained_load(self, client, sample_churn_request):
        """Test sustained load over time"""
        duration = 30  # 30 seconds
        start_time = time.time()
        request_count = 0
        error_count = 0
        
        while time.time() - start_time < duration:
            try:
                response = client.post("/predict/churn", json=sample_churn_request)
                if response.status_code not in [200, 503]:
                    error_count += 1
                request_count += 1
            except Exception as e:
                error_count += 1
                request_count += 1
            
            # Small delay to avoid overwhelming
            time.sleep(0.1)
        
        actual_duration = time.time() - start_time
        requests_per_second = request_count / actual_duration
        error_rate = error_count / request_count if request_count > 0 else 0
        
        print(f"Sustained load test results:")
        print(f"  Duration: {actual_duration:.1f}s")
        print(f"  Requests: {request_count}")
        print(f"  RPS: {requests_per_second:.1f}")
        print(f"  Error rate: {error_rate:.2%}")
        
        # Should handle reasonable load
        assert requests_per_second > 1  # At least 1 RPS
        assert error_rate < 0.1  # Less than 10% error rate
    
    @pytest.mark.slow
    def test_gradual_load_increase(self, client, sample_churn_request):
        """Test gradual load increase"""
        results = []
        
        for concurrent_users in [1, 2, 5, 10]:
            print(f"Testing with {concurrent_users} concurrent users...")
            
            def make_requests():
                success_count = 0
                error_count = 0
                
                for _ in range(10):  # 10 requests per user
                    try:
                        response = client.post("/predict/churn", json=sample_churn_request)
                        if response.status_code in [200, 503]:
                            success_count += 1
                        else:
                            error_count += 1
                    except:
                        error_count += 1
                    
                    time.sleep(0.1)
                
                return success_count, error_count
            
            # Run concurrent users
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
                futures = [executor.submit(make_requests) for _ in range(concurrent_users)]
                user_results = [future.result() for future in futures]
            
            end_time = time.time()
            
            total_success = sum(result[0] for result in user_results)
            total_errors = sum(result[1] for result in user_results)
            duration = end_time - start_time
            
            results.append({
                "users": concurrent_users,
                "success": total_success,
                "errors": total_errors,
                "duration": duration,
                "rps": (total_success + total_errors) / duration
            })
        
        # Print results
        print("\\nGradual load increase results:")
        for result in results:
            print(f"  {result['users']} users: {result['success']} success, "
                  f"{result['errors']} errors, {result['rps']:.1f} RPS")
        
        # Performance shouldn't degrade too much with increased load
        first_rps = results[0]["rps"]
        last_rps = results[-1]["rps"]
        degradation = (first_rps - last_rps) / first_rps if first_rps > 0 else 0
        
        print(f"Performance degradation: {degradation:.1%}")
        assert degradation < 0.5  # Less than 50% degradation