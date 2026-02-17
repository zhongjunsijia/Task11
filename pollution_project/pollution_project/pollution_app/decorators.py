from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .utils.jwt_auth import get_user_from_token
from .utils.redis_cache import cache
import hashlib
import time


def jwt_required(func):
    """
    JWT认证装饰器
    :param func: 视图函数
    :return: 装饰后的视图函数
    """
    @csrf_exempt
    def wrapper(request, *args, **kwargs):
        # 从请求头获取Authorization
        auth_header = request.META.get('HTTP_AUTHORIZATION')
        if not auth_header:
            return JsonResponse({'code': 401, 'message': '缺少认证令牌'}, status=401)
        
        # 提取token
        try:
            token = auth_header.split(' ')[1]
        except IndexError:
            return JsonResponse({'code': 401, 'message': '认证令牌格式错误'}, status=401)
        
        # 验证token并获取用户
        user = get_user_from_token(token)
        if not user:
            return JsonResponse({'code': 401, 'message': '认证令牌无效或已过期'}, status=401)
        
        # 将用户对象添加到请求中
        request.user = user
        return func(request, *args, **kwargs)
    
    return wrapper


def jwt_optional(func):
    """
    可选的JWT认证装饰器
    :param func: 视图函数
    :return: 装饰后的视图函数
    """
    @csrf_exempt
    def wrapper(request, *args, **kwargs):
        # 从请求头获取Authorization
        auth_header = request.META.get('HTTP_AUTHORIZATION')
        if auth_header:
            # 提取token
            try:
                token = auth_header.split(' ')[1]
                # 验证token并获取用户
                user = get_user_from_token(token)
                if user:
                    # 将用户对象添加到请求中
                    request.user = user
            except (IndexError, ValueError):
                pass
        
        return func(request, *args, **kwargs)
    
    return wrapper


def cache_page(timeout=3600):
    """
    页面缓存装饰器
    :param timeout: 缓存过期时间（秒）
    :return: 装饰后的视图函数
    """
    def decorator(func):
        def wrapper(request, *args, **kwargs):
            # 生成缓存键
            cache_key = generate_cache_key(request, args, kwargs)
            
            # 尝试从缓存获取
            cached_response = cache.get(cache_key)
            if cached_response:
                return JsonResponse(cached_response)
            
            # 执行视图函数
            response = func(request, *args, **kwargs)
            
            # 如果是JsonResponse，缓存结果
            if isinstance(response, JsonResponse):
                # 提取响应数据
                import json
                response_data = json.loads(response.content.decode('utf-8'))
                # 缓存数据
                cache.set(cache_key, response_data, expire=timeout)
            
            return response
        
        return wrapper
    
    return decorator


def generate_cache_key(request, args, kwargs):
    """
    生成缓存键
    :param request: 请求对象
    :param args: 位置参数
    :param kwargs: 关键字参数
    :return: 缓存键字符串
    """
    # 基础键
    base_key = f"view:{request.path}"
    
    # 添加查询参数
    if request.GET:
        sorted_params = sorted(request.GET.items())
        params_str = "&".join([f"{k}={v}" for k, v in sorted_params])
        base_key += f":{params_str}"
    
    # 添加路径参数
    if args:
        base_key += f":{args}"
    if kwargs:
        sorted_kwargs = sorted(kwargs.items())
        kwargs_str = ":".join([f"{k}={v}" for k, v in sorted_kwargs])
        base_key += f":{kwargs_str}"
    
    # 添加用户ID（如果已登录）
    if hasattr(request, 'user') and request.user.is_authenticated:
        base_key += f":user:{request.user.id}"
    
    # 生成哈希值以确保键长度合理
    hash_obj = hashlib.md5(base_key.encode('utf-8'))
    return f"cache:{hash_obj.hexdigest()}"


def cache_function(timeout=3600):
    """
    函数缓存装饰器
    :param timeout: 缓存过期时间（秒）
    :return: 装饰后的函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = generate_function_cache_key(func, args, kwargs)
            
            # 尝试从缓存获取
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 缓存结果
            cache.set(cache_key, result, expire=timeout)
            
            return result
        
        return wrapper
    
    return decorator


def generate_function_cache_key(func, args, kwargs):
    """
    生成函数缓存键
    :param func: 函数对象
    :param args: 位置参数
    :param kwargs: 关键字参数
    :return: 缓存键字符串
    """
    # 基础键
    base_key = f"function:{func.__module__}:{func.__name__}"
    
    # 添加参数
    if args:
        base_key += f":{args}"
    if kwargs:
        sorted_kwargs = sorted(kwargs.items())
        kwargs_str = ":".join([f"{k}={v}" for k, v in sorted_kwargs])
        base_key += f":{kwargs_str}"
    
    # 生成哈希值以确保键长度合理
    hash_obj = hashlib.md5(base_key.encode('utf-8'))
    return f"cache:{hash_obj.hexdigest()}"


def permission_required(permission_codename):
    """
    权限检查装饰器
    :param permission_codename: 权限代码
    :return: 装饰后的视图函数
    """
    def decorator(func):
        def wrapper(request, *args, **kwargs):
            # 检查用户是否登录
            if not request.user.is_authenticated:
                # 如果是AJAX请求，返回JSON响应
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({'code': 401, 'message': '请先登录'}, status=401)
                # 否则重定向到登录页
                from django.shortcuts import redirect
                return redirect('login')
            
            # 检查用户是否拥有权限
            from .models import user_has_permission
            if not user_has_permission(request.user, permission_codename):
                # 如果是AJAX请求，返回JSON响应
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({'code': 403, 'message': '权限不足'}, status=403)
                # 否则返回403页面
                return JsonResponse({'code': 403, 'message': '权限不足'}, status=403)
            
            return func(request, *args, **kwargs)
        return wrapper
    return decorator


def role_required(role_name):
    """
    角色检查装饰器
    :param role_name: 角色名称
    :return: 装饰后的视图函数
    """
    def decorator(func):
        def wrapper(request, *args, **kwargs):
            # 检查用户是否登录
            if not request.user.is_authenticated:
                # 如果是AJAX请求，返回JSON响应
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({'code': 401, 'message': '请先登录'}, status=401)
                # 否则重定向到登录页
                from django.shortcuts import redirect
                return redirect('login')
            
            # 检查用户是否拥有角色
            from .models import user_has_role
            if not user_has_role(request.user, role_name):
                # 如果是AJAX请求，返回JSON响应
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({'code': 403, 'message': '角色权限不足'}, status=403)
                # 否则返回403页面
                return JsonResponse({'code': 403, 'message': '角色权限不足'}, status=403)
            
            return func(request, *args, **kwargs)
        return wrapper
    return decorator
