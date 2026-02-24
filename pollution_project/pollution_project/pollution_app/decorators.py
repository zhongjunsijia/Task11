from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .utils.jwt_auth import get_user_from_token


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
            
            # 检查用户是否是超级用户或员工，是的话直接通过
            if request.user.is_staff:
                return func(request, *args, **kwargs)
            
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
