from django.contrib import admin
from .models import PollutionData, PredictionResult, DataUpload, UserPollutant, ModelEvaluation

# 注册历史数据模型（新增4种污染物字段）
@admin.register(PollutionData)
class PollutionDataAdmin(admin.ModelAdmin):
    list_display = (
        'date', 'pm25', 'pm10',
        'no2', 'so2', 'o3', 'co',
        'temperature', 'humidity', 'wind_speed'
    )
    list_filter = ('date',)  # 过滤条件
    search_fields = ('date',)  # 搜索字段
    # 详情页字段排序
    fieldsets = (
        ('时间信息', {'fields': ('date',)}),
        ('污染物浓度', {
            'fields': ('pm25', 'pm10', 'no2', 'so2', 'o3', 'co')
        }),
        ('气象数据', {
            'fields': ('temperature', 'humidity', 'wind_speed')
        }),
    )

# 注册预测结果模型（6种污染物预测字段）
@admin.register(PredictionResult)
class PredictionResultAdmin(admin.ModelAdmin):
    list_display = (
        'predict_time', 'target_date',
        'pm25_pred', 'pm10_pred',
        'no2_pred', 'so2_pred', 'o3_pred', 'co_pred',  # 预测字段
        'linear_accuracy', 'nn_accuracy',  # 两种模型的准确率
        'overall_aqi_pred', 'main_pollutant_pred', 'aqi_level_pred',  # AQI字段
        'linear_accuracy', 'nn_accuracy'
    )
    list_filter = ('target_date',)  # 过滤条件
    # 详情页字段分组
    fieldsets = (
        ('预测信息', {
            'fields': ('predict_time', 'target_date', 'linear_accuracy', 'nn_accuracy')
        }),
        ('预测结果', {
            'fields': ('pm25_pred', 'pm10_pred', 'no2_pred', 'so2_pred', 'o3_pred', 'co_pred')
        }),
        ('AQI预测结果', {  # AQI分组
            'fields': ('overall_aqi_pred', 'main_pollutant_pred', 'aqi_level_pred')
        }),
    )

# 注册数据上传记录模型
@admin.register(DataUpload)
class DataUploadAdmin(admin.ModelAdmin):
    list_display = ('file', 'upload_date', 'status', 'records_processed')
    list_filter = ('status', 'upload_date')
    search_fields = ('file__name', 'error_message')
    readonly_fields = ('upload_date',)  # 上传时间只读
    fieldsets = (
        ('文件信息', {'fields': ('file', 'upload_date')}),
        ('处理状态', {
            'fields': ('status', 'records_processed', 'error_message', 'pollution_data')
        }),
    )

@admin.register(ModelEvaluation)
class ModelEvaluationAdmin(admin.ModelAdmin):
    list_display = ('model_name', 'city', 'pollutant', 'test_rmse', 'test_r2', 'training_time')
    list_filter = ('model_name', 'city', 'pollutant')