import os
import django

# 设置Django环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pollution_project.settings')
django.setup()

from django.contrib.auth.models import User
from pollution_app.models import Role, Permission, UserRole, RolePermission

# 初始化函数
def initialize_system():
    print("开始初始化系统用户和权限...")
    
    # 创建权限
    permissions = [
        {'name': '系统设置', 'codename': 'system_settings'},
        {'name': '用户管理', 'codename': 'user_management'},
        {'name': '角色管理', 'codename': 'role_management'},
        {'name': '模型管理', 'codename': 'model_management'},
        {'name': '批量预测', 'codename': 'batch_prediction'},
        {'name': '预测历史', 'codename': 'prediction_history'},
        {'name': '系统日志', 'codename': 'system_logs'},
        {'name': '系统监控', 'codename': 'system_monitoring'},
    ]
    
    permission_objects = {}
    for perm_data in permissions:
        perm, created = Permission.objects.get_or_create(
            codename=perm_data['codename'],
            defaults={'name': perm_data['name']}
        )
        permission_objects[perm_data['codename']] = perm
        if created:
            print(f"创建权限: {perm_data['name']}")
    
    # 创建角色
    roles = [
        {'name': '超级管理员', 'permissions': list(permission_objects.keys())},
        {'name': '普通用户', 'permissions': []},
    ]
    
    role_objects = {}
    for role_data in roles:
        role, created = Role.objects.get_or_create(name=role_data['name'])
        role_objects[role_data['name']] = role
        if created:
            print(f"创建角色: {role_data['name']}")
        
        # 分配权限
        for perm_codename in role_data['permissions']:
            if perm_codename in permission_objects:
                _, perm_created = RolePermission.objects.get_or_create(
                    role=role,
                    permission=permission_objects[perm_codename]
                )
                if perm_created:
                    print(f"分配权限 {perm_codename} 给角色 {role_data['name']}")
    
    # 创建用户
    users = [
        {'username': 'admin', 'email': 'admin@example.com', 'password': 'admin123', 'role': '超级管理员'},
        {'username': 'user', 'email': 'user@example.com', 'password': 'user123', 'role': '普通用户'},
    ]
    
    for user_data in users:
        user, created = User.objects.get_or_create(
            username=user_data['username'],
            defaults={
                'email': user_data['email'],
                'is_active': True,
                'is_staff': user_data['role'] == '超级管理员',
            }
        )
        if created:
            user.set_password(user_data['password'])
            user.save()
            print(f"创建用户: {user_data['username']}, 密码: {user_data['password']}")
        elif not user.check_password(user_data['password']):
            user.set_password(user_data['password'])
            user.save()
            print(f"更新用户密码: {user_data['username']}, 新密码: {user_data['password']}")
        
        # 分配角色
        if user_data['role'] in role_objects:
            _, role_created = UserRole.objects.get_or_create(
                user=user,
                role=role_objects[user_data['role']]
            )
            if role_created:
                print(f"分配角色 {user_data['role']} 给用户 {user_data['username']}")
    
    print("系统初始化完成!")
    print("\n可用用户:")
    print("- 管理员: 用户名=admin, 密码=admin123")
    print("- 普通用户: 用户名=user, 密码=user123")

if __name__ == '__main__':
    initialize_system()
