# 错位检查与受限配准（无权重，CPU）

独立入口：`AlignmentInspector(config=None).inspect(A, B)`、`PairRegistrar(config=None).register(A, B)`。
默认检查使用phase correlation，默认校正使用phase初始化的translation ECC。
`RegistrationConfig`仅由部署方构造；首版仅估计全局平移，不支持旋转、尺度、单应或非刚性形变。

输入严格为原始RGB8、256×256、PNG/JPEG，T1为参考，T2为待移动图；不隐式缩放或应用EXIF方向。
采样矩阵为reference→moving：`out_B_on_A[y,x] = B[y+dy,x+dx]`，输出时使用
`INTER_LINEAR | WARP_INVERSE_MAP`，同时给出moving→reference前向矩阵。

校正只生成派生B，原图保持不变。有效mask为0/1，保守排除双线性采样越界区。
`noop`保留原B像素，mask全1；`unknown/refused`不产生校正图。
纹理不足、重复图案、弱phase峰、过大位移、重叠不足、ECC不收敛或偏离初值都会触发未知/拒绝。
内部相关性采用同一候选有效区比较，仍不是独立TRE或校准置信度。

所有默认门限为未校准的部署检查，真实精度需要独立稳定地物检查点验收。
普通PNG不含可靠地理信息，不报告米或平方米。GeoTIFF、显式NoData和透明输入不在本版范围内。

工具为 `tools/alignment_check_tool.py` 和 `tools/registration_tool.py`。
配准后无效边缘意味着现有“全有效RGB256”检测器前置条件尚未满足，必须先有显式有效区/网格处理策略；本版不偷偷裁剪或缩放接检测器。
详见 `docs/review-tools.md`。

来源：原始组合实现，使用OpenCV4.8现有API，无复制第三方网络或权重。
[phase](https://docs.opencv.org/4.8.0/d7/df3/group__imgproc__motion.html)、
[ECC与逆采样](https://docs.opencv.org/4.8.0/dc/d6b/group__video__track.html)、
[warp约定](https://docs.opencv.org/4.8.0/da/d54/group__imgproc__transform.html)。
