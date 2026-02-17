# 气象污染预测系统

## 项目简介

气象污染预测系统是一个基于Django框架开发的空气质量监测与预测平台，旨在实时监测、精准预测、全面分析空气质量状况，为用户提供专业的污染防控参考。

### 核心功能

- **实时监测**：实时采集并展示各地区空气质量数据，包括PM2.5、PM10、NO₂等多种污染物浓度
- **污染预测**：基于历史数据和气象条件，预测未来7天空气质量变化趋势，提前预警污染风险
- **数据分析**：提供多种可视化分析工具，包括风玫瑰图、走势图和探空图，深入解析污染成因
- **数据管理**：支持数据上传、导出、历史查询等功能，方便用户管理和分析污染数据
- **系统管理**：提供用户管理、角色管理、模型管理等后台管理功能

## 技术栈

- **后端**：Python 3.8+, Django 4.2+
- **前端**：HTML5, CSS3, JavaScript, Bootstrap 5, Chart.js
- **数据库**：SQLite（开发环境）/ MySQL（生产环境）
- **认证**：JWT（JSON Web Token）
- **缓存**：Redis（可选）
- **数据可视化**：Chart.js, Leaflet.js（地图）

## 快速开始

### 环境要求

- Python 3.8 或更高版本
- pip 包管理工具
- Git 版本控制

### 安装步骤

1. **克隆项目**

   ```bash
   git clone https://github.com/yourusername/pollution-prediction-system.git
   cd pollution-prediction-system
   ```

2. **创建虚拟环境**

   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **安装依赖**

   ```bash
   pip install -r requirements.txt
   ```

4. **数据库迁移**

   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

5. **初始化系统数据**

   ```bash
   python create_city_pollution_model.py
   python initialize_users.py
   ```

6. **启动开发服务器**

   ```bash
   python manage.py runserver
   ```

7. **访问系统**

   打开浏览器，访问 http://127.0.0.1:8000/

## 系统用户

系统初始化时会创建以下用户：

### 管理员用户
- **用户名**：admin
- **密码**：admin123
- **角色**：超级管理员（拥有所有系统权限）

### 普通用户
- **用户名**：user
- **密码**：user123
- **角色**：普通用户（无系统管理权限）

## 主要功能模块

### 1. 监测分析
- **污染地图**：展示各地区污染分布情况，支持热力图和监测点两种显示模式
- **污染日历**：以日历形式展示污染数据变化趋势
- **探空图**：展示气象探空数据，分析气象条件对污染的影响
- **走势图**：展示污染物浓度随时间的变化趋势
- **风玫瑰图**：分析风向风速对污染扩散的影响

### 2. 预报预警
- **大气污染预测**：基于多种模型（线性回归、神经网络、随机森林）预测未来空气质量
- **预测结果评估**：评估不同模型的预测准确率和性能指标

### 3. 数据管理
- **数据上传**：支持CSV格式数据上传，批量导入污染数据
- **历史数据查询**：按时间范围和污染物类型查询历史数据
- **数据导出**：支持CSV和Excel格式数据导出

### 4. 系统管理
- **系统设置**：管理系统基本配置和参数
- **用户管理**：管理系统用户，包括创建、编辑、删除用户
- **角色管理**：管理用户角色和权限
- **模型管理**：管理预测模型，包括模型训练、评估和部署
- **批量预测**：批量预测多个时间点的污染数据
- **预测历史**：查看历史预测记录和结果
- **系统日志**：记录系统操作日志，方便故障排查
- **系统监控**：监控系统运行状态和性能指标

## 数据模型

### 核心数据模型

1. **PollutionData**：历史污染数据模型
   - 包含日期时间、PM2.5、PM10、NO₂、SO₂、O₃、CO等污染物浓度
   - 包含温度、湿度、风速等气象因素

2. **PredictionResult**：污染预测结果模型
   - 包含预测时间、目标日期、多种模型的预测结果
   - 包含预测准确率和评估指标

3. **CityPollutionData**：城市污染数据模型
   - 包含城市名称、AQI指数、各项污染物浓度
   - 用于展示全国各地市的空气质量排名

4. **ModelEvaluation**：模型评估指标模型
   - 包含模型名称、版本、类型、评估指标等
   - 用于评估和比较不同模型的性能

## 系统架构

### 前端架构

- **响应式设计**：基于Bootstrap 5实现，适配PC端和移动端
- **数据可视化**：使用Chart.js实现各种图表展示
- **交互体验**：实现实时更新、平滑过渡等交互效果

### 后端架构

- **MVC架构**：基于Django的Model-View-Template架构
- **RESTful API**：提供标准化的API接口，支持前后端分离
- **权限管理**：基于角色的权限控制（RBAC）
- **缓存机制**：支持Redis缓存，提高系统性能

## 预测模型

系统集成了多种污染预测模型：

1. **线性回归模型**：基于历史数据的线性关系预测
2. **神经网络模型**：基于深度学习的非线性关系预测
3. **随机森林模型**：基于集成学习的多因素预测

### 模型评估

系统会自动评估各模型的性能指标，包括：
- MAE（平均绝对误差）
- RMSE（均方根误差）
- R²（决定系数）

## 系统优化

### 性能优化

- **数据库索引**：为常用查询字段添加索引
- **缓存策略**：使用Redis缓存热点数据
- **异步处理**：使用Celery处理耗时任务
- **分页查询**：大数据集采用分页加载

### 安全优化

- **JWT认证**：使用安全的JWT令牌进行身份验证
- **密码加密**：使用Django内置的密码加密机制
- **CSRF防护**：启用Django的CSRF保护
- **XSS防护**：对用户输入进行安全过滤

## 部署建议

### 开发环境

- **数据库**：SQLite（默认）
- **服务器**：Django开发服务器
- **缓存**：无（可选Redis）

### 生产环境

- **数据库**：MySQL 8.0+ 或 PostgreSQL 12+
- **服务器**：Gunicorn + Nginx
- **缓存**：Redis
- **监控**：Prometheus + Grafana（可选）
- **日志**：ELK Stack（可选）

## 常见问题

### 1. 如何添加新的预测模型？

1. 在 `pollution_app/prediction_model.py` 中实现新模型
2. 在 `pollution_app/models.py` 中更新 `ModelEvaluation` 模型
3. 在 `pollution_app/views.py` 中添加模型训练和预测视图

### 2. 如何更新城市污染数据？

1. 编辑 `create_city_pollution_model.py` 文件中的城市数据
2. 运行 `python create_city_pollution_model.py` 重新初始化数据

### 3. 如何修改系统默认用户？

编辑 `initialize_users.py` 文件中的用户数据，然后运行该脚本重新初始化用户。

## 项目结构

```
pollution_project/
├── pollution_project/        # 项目配置目录
│   ├── settings.py           # 项目设置
│   ├── urls.py               # 项目路由
│   └── wsgi.py               # WSGI配置
├── pollution_app/            # 主应用目录
│   ├── templates/            # 模板文件
│   ├── static/               # 静态文件
│   ├── models.py             # 数据模型
│   ├── views.py              # 视图函数
│   ├── forms.py              # 表单
│   ├── prediction_model.py   # 预测模型
│   ├── utils/                # 工具函数
│   └── decorators.py         # 装饰器
├── manage.py                 # 管理脚本
├── requirements.txt          # 依赖文件
├── create_city_pollution_model.py  # 城市数据初始化
├── initialize_users.py       # 用户初始化
└── README.md                 # 项目说明
```
