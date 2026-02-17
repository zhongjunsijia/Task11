import jwt
import datetime
from django.conf import settings
from django.contrib.auth.models import User

# JWT配置
JWT_SECRET_KEY = settings.SECRET_KEY
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_DELTA = datetime.timedelta(days=7)

def generate_jwt_token(user):
    """
    生成JWT令牌
    :param user: 用户对象
    :return: JWT令牌字符串
    """
    payload = {
        'user_id': user.id,
        'username': user.username,
        'exp': datetime.datetime.utcnow() + JWT_EXPIRATION_DELTA,
        'iat': datetime.datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token):
    """
    验证JWT令牌
    :param token: JWT令牌字符串
    :return: 解码后的payload或None
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        # 令牌过期
        return None
    except jwt.InvalidTokenError:
        # 令牌无效
        return None

def get_user_from_token(token):
    """
    从JWT令牌获取用户对象
    :param token: JWT令牌字符串
    :return: 用户对象或None
    """
    payload = verify_jwt_token(token)
    if payload:
        try:
            user = User.objects.get(id=payload['user_id'])
            return user
        except User.DoesNotExist:
            return None
    return None