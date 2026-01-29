输入
𝑥
mask 𝑚（作为显式输入）
时间编码 t（hour-of-day, day-of-year，周期正余弦编码）
外生量 e（辐照、温度、历史天气预报等）
随机噪声 z（用于生成多重插补或不确定性）

把这些拼接成通道输入

生成器
Encoder 前向分支（Past）：由若干 卷积提取过去语义。
Encoder 后向分支（Future）：镜像结构，输入序列反转，提取“后文”语义。
Cross-Attention 融合层：把前向/后向 encoder 输出在每一尺度融合（让模型学习前后信息权重）。
Decoder（U-Net）：逐尺度上采样并与 encoder 跳跃连接（skip），输出与原序列长度一致的

判别器
Global Discriminator D global（Conv）：处理整段序列，判断长期一致性和周期性是否合理。
Local Patch Discriminator(s) D local：对所有缺失区（或滑动窗口）裁切短窗口（例如长度 64/128），判别细节真实性
Feature Matching 支持：从 D 的若干中间层抽取特征向量，用 L2 与真实样本特征对齐，作为 generator 的额外损失（能显著稳定训练并提升细节）。

损失函数
g_loss = cfg.lambda_adv * adv_loss + cfg.lambda_rec * rec_loss + cfg.lambda_fm * fm + cfg.lambda_grad * grad_loss
2
adv_loss (对抗损失WGAN-GP)
rec_loss (重建损失)
fm (特征匹配损失)
grad_loss (梯度损失)



生成器在训练时，其实同时干两件事：

🔹 路径 A：插补路径（主任务）

输入：前后上下文 + mask + 其他特征

输出：完整序列𝑋
损失：adv_loss + rec_loss + fm_loss + grad_loss

目标：让缺失段看起来真实、衔接自然

🔹 路径 B：前向预测路径（辅助任务）

输入：仅前段（Past Encoder 输出）

输出：预测的未来序列

损失：pred_loss = |X_{true_future} - X_{pred}|

目标：让模型理解时间趋势和动态变化规律
