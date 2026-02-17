import redis
import json
import pickle
from django.conf import settings


class RedisCache:
    """
    Redis缓存管理类
    """
    
    def __init__(self):
        """
        初始化Redis连接
        """
        self.redis_client = redis.Redis(
            host=getattr(settings, 'REDIS_HOST', 'localhost'),
            port=getattr(settings, 'REDIS_PORT', 6379),
            db=getattr(settings, 'REDIS_DB', 0),
            password=getattr(settings, 'REDIS_PASSWORD', None),
            decode_responses=False  # 保持二进制数据
        )
    
    def set(self, key, value, expire=None):
        """
        设置缓存
        :param key: 缓存键
        :param value: 缓存值
        :param expire: 过期时间（秒）
        :return: bool
        """
        try:
            # 序列化复杂数据
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False).encode('utf-8')
            elif not isinstance(value, (bytes, str)):
                value = pickle.dumps(value)
            elif isinstance(value, str):
                value = value.encode('utf-8')
            
            if expire:
                return self.redis_client.setex(key, expire, value)
            else:
                return self.redis_client.set(key, value)
        except Exception as e:
            print(f"Redis set error: {e}")
            return False
    
    def get(self, key, default=None):
        """
        获取缓存
        :param key: 缓存键
        :param default: 默认值
        :return: 缓存值或默认值
        """
        try:
            value = self.redis_client.get(key)
            if value is None:
                return default
            
            # 尝试反序列化
            try:
                # 尝试JSON反序列化
                return json.loads(value.decode('utf-8'))
            except json.JSONDecodeError:
                try:
                    # 尝试pickle反序列化
                    return pickle.loads(value)
                except (pickle.UnpicklingError, TypeError):
                    # 纯字符串
                    return value.decode('utf-8')
        except Exception as e:
            print(f"Redis get error: {e}")
            return default
    
    def delete(self, key):
        """
        删除缓存
        :param key: 缓存键
        :return: bool
        """
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            print(f"Redis delete error: {e}")
            return False
    
    def exists(self, key):
        """
        检查键是否存在
        :param key: 缓存键
        :return: bool
        """
        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            print(f"Redis exists error: {e}")
            return False
    
    def expire(self, key, seconds):
        """
        设置键的过期时间
        :param key: 缓存键
        :param seconds: 过期时间（秒）
        :return: bool
        """
        try:
            return bool(self.redis_client.expire(key, seconds))
        except Exception as e:
            print(f"Redis expire error: {e}")
            return False
    
    def get_ttl(self, key):
        """
        获取键的剩余过期时间
        :param key: 缓存键
        :return: int 剩余时间（秒），-1表示永不过期，-2表示键不存在
        """
        try:
            return self.redis_client.ttl(key)
        except Exception as e:
            print(f"Redis ttl error: {e}")
            return -2
    
    def mset(self, mapping, expire=None):
        """
        批量设置缓存
        :param mapping: 键值对字典
        :param expire: 过期时间（秒）
        :return: bool
        """
        try:
            # 序列化数据
            serialized_mapping = {}
            for k, v in mapping.items():
                if isinstance(v, (dict, list)):
                    v = json.dumps(v, ensure_ascii=False).encode('utf-8')
                elif not isinstance(v, (bytes, str)):
                    v = pickle.dumps(v)
                elif isinstance(v, str):
                    v = v.encode('utf-8')
                serialized_mapping[k] = v
            
            # 执行批量设置
            result = self.redis_client.mset(serialized_mapping)
            
            # 设置过期时间
            if expire:
                for k in mapping.keys():
                    self.redis_client.expire(k, expire)
            
            return result
        except Exception as e:
            print(f"Redis mset error: {e}")
            return False
    
    def mget(self, keys):
        """
        批量获取缓存
        :param keys: 键列表
        :return: 值列表
        """
        try:
            values = self.redis_client.mget(keys)
            result = []
            
            for value in values:
                if value is None:
                    result.append(None)
                    continue
                
                # 尝试反序列化
                try:
                    result.append(json.loads(value.decode('utf-8')))
                except json.JSONDecodeError:
                    try:
                        result.append(pickle.loads(value))
                    except (pickle.UnpicklingError, TypeError):
                        result.append(value.decode('utf-8'))
            
            return result
        except Exception as e:
            print(f"Redis mget error: {e}")
            return [None] * len(keys)
    
    def flushdb(self):
        """
        清空当前数据库
        :return: bool
        """
        try:
            return self.redis_client.flushdb()
        except Exception as e:
            print(f"Redis flushdb error: {e}")
            return False


# 创建全局缓存实例
cache = RedisCache()