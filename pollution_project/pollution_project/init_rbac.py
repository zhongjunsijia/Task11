#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
初始化RBAC角色和权限数据
"""

import os
import django

# 设置Django环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pollution_project.settings')
django.setup()

from pollution_app.models import Role, Permission, RolePermission, UserRole
from django.contrib.auth.models import User

def init_rbac_data():
    """
    初始化RBAC数据
    """
    print("开始初始化RBAC数据...")
    
    # 创建默认权限
    permissions = [
        # 系统管理权限
        {'name': '系统设置', 'codename': 'system_settings', 'description': '管理系统设置'}, 
        {'name': '用户管理', 'codename': 'user_management', 'description': '管理用户账号'}, 
        {'name': '角色管理', 'codename': 'role_management', 'description': '管理角色权限'}, 
        
        # 数据管理权限
        {'name': '数据上传', 'codename': 'data_upload', 'description': '上传污染数据'}, 
        {'name': '数据导出', 'codename': 'data_export', 'description': '导出污染数据'}, 
        {'name': '数据编辑', 'codename': 'data_edit', 'description': '编辑污染数据'}, 
        {'name': '数据删除', 'codename': 'data_delete', 'description': '删除污染数据'}, 
        
        # 模型管理权限
        {'name': '模型训练', 'codename': 'model_train', 'description': '训练预测模型'}, 
        {'name': '模型管理', 'codename': 'model_management', 'description': '管理模型版本'}, 
        
        # 预测管理权限
        {'name': '批量预测', 'codename': 'batch_prediction', 'description': '执行批量预测'}, 
        {'name': '预测历史', 'codename': 'prediction_history', 'description': '查看预测历史'}, 
        
        # 监测分析权限
        {'name': '监测数据查看', 'codename': 'monitoring_view', 'description': '查看监测数据'}, 
        {'name': '趋势分析', 'codename': 'trend_analysis', 'description': '分析污染趋势'}, 
    ]
    
    permission_objects = {}
    for perm_data in permissions:
        perm, created = Permission.objects.get_or_create(
            codename=perm_data['codename'],
            defaults={
                'name': perm_data['name'],
                'description': perm_data['description']
            }
        )
        permission_objects[perm_data['codename']] = perm
        if created:
            print(f"创建权限: {perm_data['name']}")
        else:
            print(f"权限已存在: {perm_data['name']}")
    
    # 创建默认角色
    roles = [
        {'name': '超级管理员', 'description': '拥有所有权限', 'permissions': [p['codename'] for p in permissions]},
        {'name': '系统管理员', 'description': '系统管理权限', 'permissions': ['system_settings', 'user_management', 'role_management']},
        {'name': '数据管理员', 'description': '数据管理权限', 'permissions': ['data_upload', 'data_export', 'data_edit', 'data_delete']},
        {'name': '模型管理员', 'description': '模型管理权限', 'permissions': ['model_train', 'model_management']},
        {'name': '预测分析师', 'description': '预测分析权限', 'permissions': ['batch_prediction', 'prediction_history', 'monitoring_view', 'trend_analysis']},
        {'name': '普通用户', 'description': '基础查看权限', 'permissions': ['monitoring_view']},
    ]
    
    for role_data in roles:
        role, created = Role.objects.get_or_create(
            name=role_data['name'],
            defaults={'description': role_data['description']}
        )
        
        if created:
            print(f"创建角色: {role_data['name']}")
        else:
            print(f"角色已存在: {role_data['name']}")
        
        # 分配权限
        for perm_codename in role_data['permissions']:
            if perm_codename in permission_objects:
                perm = permission_objects[perm_codename]
                rp, created = RolePermission.objects.get_or_create(
                    role=role,
                    permission=perm
                )
                if created:
                    print(f"  - 分配权限: {perm.name}")
    
    # 为超级用户分配超级管理员角色
    superusers = User.objects.filter(is_superuser=True)
    for user in superusers:
        admin_role = Role.objects.filter(name='超级管理员').first()
        if admin_role:
            ur, created = UserRole.objects.get_or_create(
                user=user,
                role=admin_role
            )
            if created:
                print(f"为用户 {user.username} 分配超级管理员角色")
    
    print("RBAC数据初始化完成!")

if __name__ == '__main__':
    init_rbac_data()
