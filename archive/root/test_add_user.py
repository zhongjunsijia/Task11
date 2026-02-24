from django.contrib.auth.models import User
from pollution_app.forms import RegisterForm

# 测试数据
data = {
    'username': 'testuser123',
    'email': 'test123@example.com',
    'password1': 'TestPassword123!',
    'password2': 'TestPassword123!'
}

# 创建表单
form = RegisterForm(data)
print('Form valid:', form.is_valid())
print('Form errors:', form.errors)

if form.is_valid():
    # 保存用户
    user = form.save()
    print('User created:', user.username, user.id)
    
    # 打印所有用户
    print('All users:')
    for u in User.objects.all():
        print(f'  - {u.username} (id: {u.id})')
else:
    print('Form is not valid, cannot create user')
