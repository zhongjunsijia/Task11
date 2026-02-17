from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
from .utils.aqi_calculator import calculate_aqi, get_overall_aqi

class PollutionData(models.Model):
    """历史污染数据模型"""
    date = models.DateTimeField(verbose_name="日期时间")  # 数据采集时间
    pm25 = models.FloatField(verbose_name="PM2.5浓度(μg/m³)")  # 细颗粒物
    pm10 = models.FloatField(verbose_name="PM10浓度(μg/m³)")  # 可吸入颗粒物
    no2 = models.FloatField(verbose_name="NO₂浓度(μg/m³)")  # 二氧化氮
    so2 = models.FloatField(verbose_name="SO₂浓度(μg/m³)")  # 二氧化硫
    o3 = models.FloatField(verbose_name="O₃浓度(μg/m³)")  # 臭氧
    co = models.FloatField(verbose_name="CO浓度(mg/m³)")  # 一氧化碳
    temperature = models.FloatField(verbose_name="温度(℃)")  # 气象因素
    humidity = models.FloatField(verbose_name="湿度(%)")  # 气象因素
    wind_speed = models.FloatField(verbose_name="风速(m/s)")  # 气象因素
    wind_direction = models.FloatField("风向")  # 风向（角度）
    pressure = models.FloatField("气压")  # ：气压
    precipitation = models.FloatField("降水量")  # 降水量
    # 隐藏状态字段
    is_hidden = models.BooleanField("是否隐藏", default=False)

    class Meta:
        verbose_name = "历史污染数据"
        verbose_name_plural = "历史污染数据"

    def __str__(self):
        return f"{self.date.strftime('%Y-%m-%d %H:%M')} 污染数据"

    def get_pollutant_aqi(self, pollutant):
        """获取单污染物的AQI"""
        conc = getattr(self, pollutant, None)
        if conc is None:
            return (None, "无数据", "gray")
        return calculate_aqi(pollutant, conc)

    def get_overall_aqi(self):
        """获取综合AQI"""
        concs = {
            'pm25': self.pm25,
            'pm10': self.pm10,
            'no2': self.no2,
            'so2': self.so2,
            'o3': self.o3,
            'co': self.co
        }
        return get_overall_aqi(concs)

class PredictionResult(models.Model):
    """污染预测结果模型（污染物预测字段）"""
    predict_time = models.DateTimeField(verbose_name="预测时间", default=timezone.now)
    target_date = models.DateTimeField(verbose_name="预测目标日期")  # 预测哪一天的污染
    # 线性回归预测字段
    pm25_pred = models.FloatField(verbose_name="PM2.5预测值（线性）")
    pm10_pred = models.FloatField(verbose_name="PM10预测值（线性）")
    no2_pred = models.FloatField(verbose_name="NO₂预测值（线性）")
    so2_pred = models.FloatField(verbose_name="SO₂预测值（线性）")
    o3_pred = models.FloatField(verbose_name="O₃预测值（线性）")
    co_pred = models.FloatField(verbose_name="CO预测值（线性）")
    # 神经网络预测字段
    pm25_nn_pred = models.FloatField(default=0.0, verbose_name="PM2.5预测值（神经网络）")
    pm10_nn_pred = models.FloatField(default=0.0, verbose_name="PM10预测值（神经网络）")
    no2_nn_pred = models.FloatField(default=0.0, verbose_name="NO₂预测值（神经网络）")
    so2_nn_pred = models.FloatField(default=0.0, verbose_name="SO₂预测值（神经网络）")
    o3_nn_pred = models.FloatField(default=0.0, verbose_name="O₃预测值（神经网络）")
    co_nn_pred = models.FloatField(default=0.0, verbose_name="CO预测值（神经网络）")
    # 后端随机森林预测字段
    pm25_rf_pred = models.FloatField(default=0.0, verbose_name="PM2.5预测值（随机森林）")
    pm10_rf_pred = models.FloatField(default=0.0, verbose_name="PM10预测值（随机森林）")
    no2_rf_pred = models.FloatField(default=0.0, verbose_name="NO₂预测值（随机森林）")
    so2_rf_pred = models.FloatField(default=0.0, verbose_name="SO₂预测值（随机森林）")
    o3_rf_pred = models.FloatField(default=0.0, verbose_name="O₃预测值（随机森林）")
    co_rf_pred = models.FloatField(default=0.0, verbose_name="CO预测值（随机森林）")

    linear_accuracy = models.FloatField(verbose_name="线性回归准确率(%)", null=True, blank=True)
    nn_accuracy = models.FloatField(verbose_name="神经网络准确率(%)", null=True, blank=True)
    rf_accuracy = models.FloatField(verbose_name="随机森林准确率(%)", null=True, blank=True)
    overall_aqi_pred = models.IntegerField(verbose_name="综合AQI预测值", null=True, blank=True)
    main_pollutant_pred = models.CharField(verbose_name="主要污染物预测", max_length=20, null=True, blank=True)
    aqi_level_pred = models.CharField(verbose_name="AQI等级预测", max_length=20, null=True, blank=True)
    linear_rmse = models.JSONField(verbose_name="线性模型RMSE", null=True, blank=True) # 新增
    nn_rmse = models.JSONField(verbose_name="神经网络模型RMSE", null=True, blank=True) # 新增

    class Meta:
        verbose_name = "预测结果"
        verbose_name_plural = "预测结果"

    def __str__(self):
        return f"{self.target_date.strftime('%Y-%m-%d')} 污染预测"

    def get_pollutant_aqi(self, pollutant):
        """获取单污染物预测的AQI（注意字段名是xxx_pred）"""
        conc = getattr(self, f"{pollutant}_pred", None)
        if conc is None:
            return (None, "无数据", "gray")
        return calculate_aqi(pollutant, conc)

    def get_overall_aqi(self):
        """获取综合预测AQI"""
        concs = {
            'pm25': self.pm25_pred,
            'pm10': self.pm10_pred,
            'no2': self.no2_pred,
            'so2': self.so2_pred,
            'o3': self.o3_pred,
            'co': self.co_pred
        }
        return get_overall_aqi(concs)

    # AQI计算逻辑
    def save(self, *args, **kwargs):
        # 保存时自动计算综合AQI
        concs = {
            'pm25': self.pm25_pred,
            'pm10': self.pm10_pred,
            'no2': self.no2_pred,
            'so2': self.so2_pred,
            'o3': self.o3_pred,
            'co': self.co_pred
        }
        overall_aqi, main_pollutant, level, _ = get_overall_aqi(concs)
        self.overall_aqi_pred = overall_aqi
        self.main_pollutant_pred = main_pollutant  # 主要污染物
        self.aqi_level_pred = level  # 污染等级
        super().save(*args, **kwargs)


# 数据上传记录模型
class DataUpload(models.Model):
    file = models.FileField(upload_to='uploads/')
    upload_date = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, default='处理中')
    records_processed = models.IntegerField(default=0)
    error_message = models.TextField(blank=True, null=True)
    pollution_data = models.ForeignKey(
        PollutionData,
        on_delete=models.SET_NULL,
        blank=True,
        null=True,
        verbose_name="关联的污染数据"
    )

    class Meta:
        verbose_name = "数据上传记录"
        verbose_name_plural = "数据上传记录"

    def __str__(self):
        return f"{self.file.name} - {self.upload_date.strftime('%Y-%m-%d %H:%M')}"


# 用户关注的污染物
class UserPollutant(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='followed_pollutants')
    pollutant_code = models.CharField(max_length=20)  # 'pm25', 'pm10', 'no2'等
    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'pollutant_code')  # 确保用户不会重复关注同一污染物
        verbose_name = "用户关注污染物"
        verbose_name_plural = "用户关注污染物"

# 用户查询历史
class SearchHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='search_histories')
    start_date = models.DateField(null=True, blank=True)
    end_date = models.DateField(null=True, blank=True)
    pollutants = models.JSONField()  # 存储查询的污染物列表
    search_time = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "查询历史"
        verbose_name_plural = "查询历史"
        ordering = ['-search_time']  # 按查询时间倒序


class ModelEvaluation(models.Model):
    model_name = models.CharField(max_length=50, verbose_name="模型名称")
    version = models.CharField(max_length=20, verbose_name="模型版本", default="v1.0")
    model_type = models.CharField(max_length=20, verbose_name="模型类型", choices=[
        ('linear', '线性回归'),
        ('ridge', '岭回归'),
        ('lasso', 'Lasso回归'),
        ('rf', '随机森林'),
        ('gb', '梯度提升'),
        ('svr', '支持向量回归'),
        ('mlp', '神经网络')
    ], default='rf')
    city = models.CharField(max_length=50, verbose_name="城市")
    pollutant = models.CharField(max_length=20, verbose_name="污染物")
    hyperparameters = models.JSONField(verbose_name="超参数", default=dict)
    train_mae = models.FloatField(verbose_name="训练集MAE")
    train_rmse = models.FloatField(verbose_name="训练集RMSE")
    train_r2 = models.FloatField(verbose_name="训练集R²")
    test_mae = models.FloatField(verbose_name="测试集MAE")
    test_rmse = models.FloatField(verbose_name="测试集RMSE")
    test_r2 = models.FloatField(verbose_name="测试集R²")
    training_time = models.FloatField(verbose_name="训练时间(秒)")
    is_active = models.BooleanField(verbose_name="是否激活", default=False)
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="记录时间")

    class Meta:
        verbose_name = "模型评估指标"
        verbose_name_plural = "模型评估指标"
        unique_together = ('model_name', 'version', 'pollutant')

    def __str__(self):
        return f"{self.model_name}-v{self.version}-{self.pollutant}"

    def activate(self):
        """激活当前模型版本，同时禁用同模型同污染物的其他版本"""
        # 禁用同模型同污染物的其他版本
        ModelEvaluation.objects.filter(
            model_name=self.model_name,
            pollutant=self.pollutant,
            is_active=True
        ).exclude(id=self.id).update(is_active=False)
        # 激活当前版本
        self.is_active = True
        self.save()


# RBAC权限管理模型
class Role(models.Model):
    """角色模型"""
    name = models.CharField(max_length=50, unique=True, verbose_name="角色名称")
    description = models.TextField(blank=True, verbose_name="角色描述")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新时间")

    class Meta:
        verbose_name = "角色"
        verbose_name_plural = "角色管理"

    def __str__(self):
        return self.name


class Permission(models.Model):
    """权限模型"""
    name = models.CharField(max_length=50, unique=True, verbose_name="权限名称")
    codename = models.CharField(max_length=100, unique=True, verbose_name="权限代码")
    description = models.TextField(blank=True, verbose_name="权限描述")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")

    class Meta:
        verbose_name = "权限"
        verbose_name_plural = "权限管理"

    def __str__(self):
        return self.name


class RolePermission(models.Model):
    """角色权限关联模型"""
    role = models.ForeignKey(Role, on_delete=models.CASCADE, verbose_name="角色")
    permission = models.ForeignKey(Permission, on_delete=models.CASCADE, verbose_name="权限")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")

    class Meta:
        verbose_name = "角色权限"
        verbose_name_plural = "角色权限管理"
        unique_together = ('role', 'permission')

    def __str__(self):
        return f"{self.role.name}-{self.permission.name}"


class UserRole(models.Model):
    """用户角色关联模型"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name="用户")
    role = models.ForeignKey(Role, on_delete=models.CASCADE, verbose_name="角色")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")

    class Meta:
        verbose_name = "用户角色"
        verbose_name_plural = "用户角色管理"
        unique_together = ('user', 'role')

    def __str__(self):
        return f"{self.user.username}-{self.role.name}"


# 为User模型添加角色相关的方法
def get_user_roles(user):
    """获取用户的所有角色"""
    return Role.objects.filter(userrole__user=user)

def get_user_permissions(user):
    """获取用户的所有权限"""
    return Permission.objects.filter(rolepermission__role__userrole__user=user).distinct()

def user_has_permission(user, codename):
    """检查用户是否拥有指定权限"""
    return Permission.objects.filter(
        codename=codename,
        rolepermission__role__userrole__user=user
    ).exists()

def user_has_role(user, role_name):
    """检查用户是否拥有指定角色"""
    return Role.objects.filter(
        name=role_name,
        userrole__user=user
    ).exists()


# 数据版本控制模型
class DataVersion(models.Model):
    """数据版本模型"""
    version_name = models.CharField(max_length=50, verbose_name="版本名称")
    description = models.TextField(blank=True, verbose_name="版本描述")
    data_count = models.IntegerField(verbose_name="数据条数", default=0)
    upload_user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, verbose_name="上传用户")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    is_active = models.BooleanField(verbose_name="是否激活", default=False)

    class Meta:
        verbose_name = "数据版本"
        verbose_name_plural = "数据版本管理"

    def __str__(self):
        return f"{self.version_name} ({self.created_at.strftime('%Y-%m-%d')})"

    def activate(self):
        """激活当前数据版本，同时禁用其他版本"""
        # 禁用其他版本
        DataVersion.objects.filter(is_active=True).exclude(id=self.id).update(is_active=False)
        # 激活当前版本
        self.is_active = True
        self.save()


class PollutionDataVersion(models.Model):
    """污染数据版本关联模型"""
    data = models.ForeignKey(PollutionData, on_delete=models.CASCADE, verbose_name="污染数据")
    version = models.ForeignKey(DataVersion, on_delete=models.CASCADE, verbose_name="数据版本")
    added_at = models.DateTimeField(auto_now_add=True, verbose_name="添加时间")

    class Meta:
        verbose_name = "数据版本关联"
        verbose_name_plural = "数据版本关联管理"
        unique_together = ('data', 'version')

    def __str__(self):
        return f"{self.data.date.strftime('%Y-%m-%d')} - {self.version.version_name}"


# 数据版本控制相关方法
def create_data_version(version_name, description, data_ids, user=None):
    """创建新的数据版本"""
    # 创建版本记录
    version = DataVersion.objects.create(
        version_name=version_name,
        description=description,
        data_count=len(data_ids),
        upload_user=user
    )
    
    # 关联数据
    for data_id in data_ids:
        try:
            data = PollutionData.objects.get(id=data_id)
            PollutionDataVersion.objects.get_or_create(
                data=data,
                version=version
            )
        except PollutionData.DoesNotExist:
            pass
    
    return version


def get_version_data(version_id):
    """获取指定版本的数据"""
    try:
        version = DataVersion.objects.get(id=version_id)
        data_versions = PollutionDataVersion.objects.filter(version=version)
        data_ids = [dv.data.id for dv in data_versions]
        return PollutionData.objects.filter(id__in=data_ids)
    except DataVersion.DoesNotExist:
        return PollutionData.objects.none()


def get_active_version():
    """获取当前激活的数据版本"""
    return DataVersion.objects.filter(is_active=True).first()


# 系统日志模型
class SystemLog(models.Model):
    """系统日志模型"""
    LEVEL_CHOICES = [
        ('info', '信息'),
        ('warning', '警告'),
        ('error', '错误'),
        ('critical', '严重'),
    ]
    
    ACTION_CHOICES = [
        ('login', '登录'),
        ('logout', '退出'),
        ('data_upload', '数据上传'),
        ('data_export', '数据导出'),
        ('model_train', '模型训练'),
        ('model_deploy', '模型部署'),
        ('version_create', '版本创建'),
        ('version_activate', '版本激活'),
        ('system_settings', '系统设置'),
        ('user_management', '用户管理'),
        ('role_management', '角色管理'),
        ('other', '其他'),
    ]
    
    level = models.CharField(max_length=10, choices=LEVEL_CHOICES, default='info', verbose_name="日志级别")
    action = models.CharField(max_length=20, choices=ACTION_CHOICES, default='other', verbose_name="操作类型")
    message = models.TextField(verbose_name="日志内容")
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, verbose_name="操作用户")
    ip_address = models.CharField(max_length=50, blank=True, verbose_name="IP地址")
    user_agent = models.CharField(max_length=255, blank=True, verbose_name="用户代理")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")

    class Meta:
        verbose_name = "系统日志"
        verbose_name_plural = "系统日志管理"
        ordering = ['-created_at']

    def __str__(self):
        return f"[{self.get_level_display()}] {self.message}"


# 系统监控模型
class SystemStatus(models.Model):
    """系统状态模型"""
    cpu_usage = models.FloatField(verbose_name="CPU使用率")
    memory_usage = models.FloatField(verbose_name="内存使用率")
    disk_usage = models.FloatField(verbose_name="磁盘使用率")
    network_in = models.FloatField(verbose_name="网络入流量")
    network_out = models.FloatField(verbose_name="网络出流量")
    active_connections = models.IntegerField(verbose_name="活跃连接数")
    response_time = models.FloatField(verbose_name="响应时间")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="记录时间")

    class Meta:
        verbose_name = "系统状态"
        verbose_name_plural = "系统状态管理"
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.created_at.strftime('%Y-%m-%d %H:%M:%S')} - CPU: {self.cpu_usage}%"


# 日志记录工具函数
def log_action(request, level, action, message):
    """
    记录系统操作日志
    :param request: 请求对象
    :param level: 日志级别
    :param action: 操作类型
    :param message: 日志内容
    :return: 日志对象
    """
    user = request.user if request.user.is_authenticated else None
    ip_address = request.META.get('REMOTE_ADDR', '')
    user_agent = request.META.get('HTTP_USER_AGENT', '')
    
    log = SystemLog.objects.create(
        level=level,
        action=action,
        message=message,
        user=user,
        ip_address=ip_address,
        user_agent=user_agent
    )
    return log


# 城市污染数据模型
class CityPollutionData(models.Model):
    """城市污染数据模型"""
    city = models.CharField(max_length=50, verbose_name="城市名称", unique=True)
    aqi = models.IntegerField(verbose_name="AQI指数")
    pm25 = models.FloatField(verbose_name="PM2.5浓度(μg/m³)")
    pm10 = models.FloatField(verbose_name="PM10浓度(μg/m³)")
    no2 = models.FloatField(verbose_name="NO₂浓度(μg/m³)")
    so2 = models.FloatField(verbose_name="SO₂浓度(μg/m³)")
    o3 = models.FloatField(verbose_name="O₃浓度(μg/m³)")
    co = models.FloatField(verbose_name="CO浓度(mg/m³)")
    quality = models.CharField(max_length=20, verbose_name="空气质量等级")
    update_time = models.DateTimeField(verbose_name="更新时间", auto_now=True)

    class Meta:
        verbose_name = "城市污染数据"
        verbose_name_plural = "城市污染数据"
        ordering = ['aqi']

    def __str__(self):
        return f"{self.city} - AQI: {self.aqi}"

# 系统状态记录函数
def record_system_status():
    """
    记录系统状态
    :return: 系统状态对象
    """
    try:
        import psutil
        import socket
        import time
        
        # 获取CPU使用率
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # 获取内存使用率
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # 获取磁盘使用率
        disk = psutil.disk_usage('/')
        disk_usage = disk.percent
        
        # 获取网络流量
        net_io = psutil.net_io_counters()
        network_in = net_io.bytes_recv / 1024 / 1024  # MB
        network_out = net_io.bytes_sent / 1024 / 1024  # MB
        
        # 获取活跃连接数
        active_connections = len(psutil.net_connections())
        
        # 计算响应时间
        start_time = time.time()
        # 简单的响应时间测试
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        try:
            sock.connect(('127.0.0.1', 8000))
            response_time = (time.time() - start_time) * 1000  # 毫秒
        except:
            response_time = 0
        finally:
            sock.close()
        
        status = SystemStatus.objects.create(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_in=network_in,
            network_out=network_out,
            active_connections=active_connections,
            response_time=response_time
        )
        return status
    except Exception:
        # 如果无法获取系统状态，返回None
        return None