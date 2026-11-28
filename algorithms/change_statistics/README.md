# 模块七 · 变化统计与解释

在仓库根目录运行（Python 3.12）：

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r algorithms/change_statistics/requirements.txt
& .\.venv\Scripts\python.exe algorithms/change_statistics/test_report.py
& .\.venv\Scripts\python.exe algorithms/change_statistics/report.py algorithms/change_statistics/example/request.json --json
```

随附的 8×6 标签图是合成样例。预期有 13 个变化像素、48 个有效像素。面积至少 6 像素的区域有一片，面积为 12，外接框为 [2,1,6,4]。圈选框内 6 个有效像素都发生变化。去掉 `--json` 输出 Markdown。

请求中的 questions 按场景列出指标，inputs 提供同名场景的输入。相对路径以请求文件所在目录为基准。使用本地可信请求。PNG 需要声明 fully_valid 或提供同网格 0/1 valid_mask；GeoTIFF 读取有效像元掩膜。TinyCD 建筑标签为 1，MCI 道路为 1、建筑为 2。

| 指标 | 输入和口径 |
|---|---|
| pixels、area_m2 | 有效区内的变化像素与占比。平方米使用调用方明确提供的 pixel_area_m2 或投影网格像元面积。投影面积精度依赖投影选择。 |
| regions | 八邻域，至少 6 像素。编号按面积降序，同面积按扫描顺序。bbox 为 [左,上,右,下)，centroid_xy 为列、行索引坐标。 |
| comparison | 输入 comparison 对象，包含 before、after 和 region_id（默认 1）。两期原图须与 mask 对齐，按图斑外接框裁出前期、后期、后期加边界三栏 PNG。 |
| roi | 输入 roi.bbox 或 roi.geometry（像素坐标 Polygon/MultiPolygon）。按像元中心选择圈内有效像素。无有效像素时 share 为 null。 |
| vectors | 导出保留的编号图斑。带地理参考时输出 WGS84 GeoJSON；普通 PNG 输出像素坐标 JSON。对照图与边界文件保存在请求目录，文件名独立。 |
| zones | zones 指向 WGS84 GeoJSON FeatureCollection；每个面需 properties.name。输入 mask 须为带投影的 GeoTIFF。按面精确相交，返回各区有效面积、变化面积和占比。这里计算所有变化像元，包括不足 6 像素的变化。重叠分区分别统计。 |
| description | 输入 mci.before、mci.after，要求全有效且对齐的 256×256 RGB 对。需另准备团队 feat/mci-tool 分支的 MCI 权重及 Python 环境。设置 MCI_RUNTIME_DIR 指向含 runtime.py 的 Multi_change/mcp_demo，MCI_PYTHON 指向该环境的解释器。描述针对整对影像。 |

增加标签场景时先在 report.py 的 SCHEMES 登记模型标签字典，再将该场景加入 questions 与 inputs。缺少场景输入返回 missing_mask。
