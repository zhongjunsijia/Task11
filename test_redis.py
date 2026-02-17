from pollution_app.utils.redis_cache import cache
import time

# 测试基本缓存操作
print("Testing Redis cache...")

# 测试1: 设置和获取缓存
print("\nTest 1: Set and Get")
cache_key = "test:key1"
cache_value = "Hello, Redis!"
result = cache.set(cache_key, cache_value, expire=60)
print(f"Set cache result: {result}")

retrieved_value = cache.get(cache_key)
print(f"Retrieved value: {retrieved_value}")
print(f"Test 1 passed: {retrieved_value == cache_value}")

# 测试2: 测试过期时间
print("\nTest 2: Expiration")
cache_key2 = "test:key2"
cache_value2 = "This should expire soon"
result = cache.set(cache_key2, cache_value2, expire=2)
print(f"Set cache with expiration result: {result}")

# 立即获取
retrieved_value2 = cache.get(cache_key2)
print(f"Retrieved value before expiration: {retrieved_value2}")

# 等待过期
print("Waiting for 3 seconds...")
time.sleep(3)

# 再次获取
retrieved_value2_expired = cache.get(cache_key2)
print(f"Retrieved value after expiration: {retrieved_value2_expired}")
print(f"Test 2 passed: {retrieved_value2_expired is None}")

# 测试3: 测试复杂数据类型
print("\nTest 3: Complex Data Types")
cache_key3 = "test:key3"
cache_value3 = {
    "name": "Test User",
    "age": 30,
    "scores": [85, 90, 95],
    "is_active": True
}
result = cache.set(cache_key3, cache_value3, expire=60)
print(f"Set complex data result: {result}")

retrieved_value3 = cache.get(cache_key3)
print(f"Retrieved complex value: {retrieved_value3}")
print(f"Test 3 passed: {retrieved_value3 == cache_value3}")

# 测试4: 测试删除
print("\nTest 4: Delete")
cache_key4 = "test:key4"
cache_value4 = "This will be deleted"
result = cache.set(cache_key4, cache_value4, expire=60)
print(f"Set cache result: {result}")

# 验证存在
print(f"Cache exists before delete: {cache.exists(cache_key4)}")

# 删除
result = cache.delete(cache_key4)
print(f"Delete cache result: {result}")

# 验证不存在
print(f"Cache exists after delete: {cache.exists(cache_key4)}")
print(f"Test 4 passed: {not cache.exists(cache_key4)}")

# 测试5: 测试批量操作
print("\nTest 5: Batch Operations")
mapping = {
    "test:batch1": "Value 1",
    "test:batch2": "Value 2",
    "test:batch3": "Value 3"
}
result = cache.mset(mapping, expire=60)
print(f"Batch set result: {result}")

# 批量获取
keys = ["test:batch1", "test:batch2", "test:batch3"]
values = cache.mget(keys)
print(f"Batch get values: {values}")
print(f"Test 5 passed: {len(values) == 3 and all(v is not None for v in values)}")

# 清理测试数据
print("\nCleaning up test data...")
test_keys = [cache_key, cache_key2, cache_key3, cache_key4] + keys
for key in test_keys:
    cache.delete(key)

print("\nAll tests completed!")
