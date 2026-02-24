#!/usr/bin/env python3

import os
import sys
import django

# 添加项目路径到Python路径
sys.path.append(os.path.dirname(__file__))

# 设置Django环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pollution_project.pollution_project.settings')
django.setup()

from django.contrib.auth.models import User
from pollution_app.forms import RegisterForm

# 测试用户创建
print("Testing user creation...")

# 准备表单数据
form_data = {
    'username': 'testuser',
    'email': 'test@example.com',
    'password1': 'TestPassword123!',
    'password2': 'TestPassword123!'
}

# 创建表单并验证
form = RegisterForm(form_data)
print(f"Form valid: {form.is_valid()}")

if form.is_valid():
    # 保存用户
    user = form.save()
    print(f"User created successfully: {user.username}")
    
    # 打印所有用户
    users = User.objects.all()
    print(f"\nUsers in database:")
    for u in users:
        print(f"- {u.username} ({u.email})")
else:
    # 打印错误信息
    print(f"Form errors: {form.errors}")

print("\nTest completed.")
