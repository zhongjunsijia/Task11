def calculate_aqi(pollutant, concentration):
    """
    根据污染物浓度计算AQI值及等级
    :param pollutant: 污染物代码（如'pm25'、'pm10'等）
    :param concentration: 污染物浓度（数值）
    :return: AQI值、等级、颜色
    """
    # AQI分级标准（对应各污染物的浓度限值）
    standards = {
        'pm25': [
            (0, 35, 0, 50),  # 优
            (35.1, 75, 51, 100),  # 良
            (75.1, 115, 101, 150),  # 轻度污染
            (115.1, 150, 151, 200),  # 中度污染
            (150.1, 250, 201, 300),  # 重度污染
            (250.1, float('inf'), 301, 500)  # 严重污染
        ],
        'pm10': [
            (0, 50, 0, 50),
            (50.1, 150, 51, 100),
            (150.1, 250, 101, 150),
            (250.1, 350, 151, 200),
            (350.1, 420, 201, 300),
            (420.1, float('inf'), 301, 500)
        ],
        # 补充no2、so2、o3、co的分级标准（对应前端pollutantConfig中的standards）
        'no2': [
            (0, 40, 0, 50),
            (40.1, 80, 51, 100),
            (80.1, 180, 101, 150),
            (180.1, 280, 151, 200),
            (280.1, 565, 201, 300),
            (565.1, float('inf'), 301, 500)
        ],
        # 其他污染物标准省略，按实际前端配置补充
    }

    if pollutant not in standards:
        return (None, "未知", "gray")

    # 查找浓度对应的区间
    for (c_low, c_high, aqi_low, aqi_high) in standards[pollutant]:
        if c_low <= concentration <= c_high:
            # 计算AQI（线性插值）
            aqi = ((aqi_high - aqi_low) / (c_high - c_low)) * (concentration - c_low) + aqi_low
            aqi = round(aqi)
            # 获取等级和颜色（对应前端pollutantConfig中的standards）
            levels = {
                50: ("优", "green"),
                100: ("良", "yellow"),
                150: ("轻度污染", "orange"),
                200: ("中度污染", "red"),
                300: ("重度污染", "purple"),
                500: ("严重污染", "maroon")
            }
            for max_aqi, (level, color) in levels.items():
                if aqi <= max_aqi:
                    return (aqi, level, color)

    return (None, "未知", "gray")


def get_overall_aqi(pollutant_concentrations):
    """
    根据多种污染物浓度计算综合AQI（取最大值）
    :param pollutant_concentrations: 字典，如{'pm25': 50, 'pm10': 80, ...}
    :return: 综合AQI值、主要污染物、等级、颜色
    """
    aqi_list = []
    for pollutant, conc in pollutant_concentrations.items():
        aqi, level, color = calculate_aqi(pollutant, conc)
        if aqi:
            aqi_list.append((aqi, pollutant, level, color))

    if not aqi_list:
        return (None, None, "未知", "gray")

    # 取最大AQI作为综合AQI
    aqi_list.sort(reverse=True, key=lambda x: x[0])
    overall_aqi, main_pollutant, level, color = aqi_list[0]
    return (overall_aqi, main_pollutant, level, color)