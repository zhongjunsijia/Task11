from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from pollution_app import views  # 统一导入views模块

urlpatterns = [
    # 系统默认路由
    # path('admin/', admin.site.urls),

    # 核心功能路由
    path('', views.index, name='index'),  # 首页
    path('monitoring/', views.monitoring_analysis, name='monitoring_analysis'),   # 监测分析
    path('monitoring/pollution-map/', views.pollution_map, name='pollution_map'),  #污染地图
    path('monitoring/sounding/', views.sounding_view, name='sounding'),  # 探空图
    path('monitoring/trend/', views.trend_chart, name='trend_chart'),  # 走势图
    path('monitoring/windrose/', views.windrose, name='windrose'),  # 风玫瑰图
    path('data-management/', views.data_upload, name='data_management'),  # 数据管理主入口（默认显示数据上传）
    path('data-management/history-query/', views.history_query, name='history_query'),  # 历史数据查询独立页面
    path('data-management/export/', views.data_export, name='data_export'),  # 数据导出独立页面
    path('export-data/', views.export_data, name='export_data'), # 数据导出路由
    path('upload/', views.upload_data, name='upload_data'),  # 数据上传
    path('upload/success/', views.upload_success, name='upload_success'),  # 上传成功页
    path('test/', views.test_view, name='test_view'),  # 测试视图
    path('test/user-management/', views.test_user_management, name='test_user_management'),  # 测试用户管理视图
    path('forecast/air-pollution/', views.air_pollution_forecast, name='air_pollution_forecast'),  # 大气污染预测(预报预警)
    path('forecast/evaluation/', views.prediction_evaluation, name='prediction_evaluation'),  # 预测结果评估
    path('data-operation/', views.data_operation, name='data_operation'),  #数据操作主页（查询/列表）
    path('data-add/', views.add_data, name='add_data'),  # 新增数据
    path('data-edit/<int:data_id>/', views.edit_data, name='edit_data'),  # 编辑数据
    path('data-delete/<int:data_id>/', views.delete_data, name='delete_data'),  # 删除数据
    path('data-operation/', views.data_operation, name='data_operation'),
    path('toggle-visibility/<int:data_id>/', views.toggle_data_visibility, name='toggle_data_visibility'),
    path('prediction-evaluation/', views.prediction_evaluation, name='prediction_evaluation'),

    # 用户相关路由
    path('register/', views.register, name='register'),  # 注册
    path('login/', views.user_login, name='login'),  # 登录
    path('logout/', views.user_logout, name='logout'),  # 退出登录
    path('history/', views.view_history, name='view_history'),  # 历史记录
    path('toggle-pollutant/<str:pollutant_code>/', views.toggle_pollutant, name='toggle_pollutant'),  # 污染物切换
    path('profile/', views.view_profile, name='view_profile'),  # 个人信息页
    path('history/', views.view_history, name='view_history'),
    
    # API路由
    path('api/login/', views.api_login, name='api_login'),
    path('api/test-jwt/', views.test_jwt_auth, name='test_jwt_auth'),
    path('api/batch-predict/', views.api_batch_predict, name='api_batch_predict'),
    path('api/user-permissions/', views.api_user_permissions, name='api_user_permissions'),
    
    # 系统管理路由
    path('system/settings/', views.system_settings, name='system_settings'),
    path('system/users/', views.user_management, name='user_management'),
    path('system/users/delete/<int:user_id>/', views.delete_user, name='delete_user'),
    path('system/roles/', views.role_management, name='role_management'),
    path('system/models/', views.model_management_view, name='model_management_view'),
    path('system/batch-prediction/', views.batch_prediction_view, name='batch_prediction_view'),
    path('system/prediction-history/', views.prediction_history_view, name='prediction_history_view'),
    
    # 数据版本控制路由
    path('data-version/management/', views.data_version_management, name='data_version_management'),
    path('data-version/create/', views.create_data_version_view, name='create_data_version_view'),
    path('data-version/view/<int:version_id>/', views.view_version_data, name='view_version_data'),
    path('data-version/activate/<int:version_id>/', views.activate_data_version, name='activate_data_version'),
    path('data-version/delete/<int:version_id>/', views.delete_data_version, name='delete_data_version'),
    
    # 数据版本控制API
    path('api/data-versions/', views.api_data_versions, name='api_data_versions'),
    path('api/version-data/<int:version_id>/', views.api_version_data, name='api_version_data'),
    
    # 预测历史相关路由
    path('prediction-history/export/', views.export_prediction_history, name='export_prediction_history'),
    
    # 预测历史API
    path('api/prediction-history/', views.api_prediction_history, name='api_prediction_history'),
    
    # 系统日志与监控路由
    path('system/logs/', views.system_logs_view, name='system_logs_view'),
    path('system/monitoring/', views.system_monitoring_view, name='system_monitoring_view'),
    
    # 系统日志与监控API
    path('api/system-logs/', views.api_system_logs, name='api_system_logs'),
    path('api/system-status/', views.api_system_status, name='api_system_status'),
    
    # 实时数据API
    path('api/latest-data/', views.api_latest_data, name='api_latest_data'),
    # 贺州市趋势数据API
    path('api/hezhou-trend/', views.api_hezhou_trend_data, name='api_hezhou_trend_data'),
]

# 开发环境下的媒体文件路由配置
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)