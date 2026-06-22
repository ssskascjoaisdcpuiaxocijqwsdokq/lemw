# YECAMRDU 模型数据流结构详细说明

## 🏗️ 模型概述

YECAMRDU = **YECA** (ECA注意力) + **MRFP** (多感受野处理) + **DUET** (线性提取器)
- **YECAMR610**使用聚类线性层处理趋势分量
- **YECAMRDU**使用DUET线性提取器处理趋势分量

## 📊 整体数据流架构

```
输入数据: x_enc (B, L, C)
    ↓
┌─────────────────────────────────────────────────────────────┐
│                  序列分解模块                                │
│  series_decomp: moving_avg → 季节性分量 + 趋势分量          │
└─────────────────────────────────────────────────────────────┘
    ↓ 季节性分量 (B, C, L)           ↓ 趋势分量 (B, L, C)
┌─────────────────────────────────┐ ┌─────────────────────────┐
│   ECA-MRFP融合注意力模块         │ │   DUET线性提取器        │
│  ┌─────────────────────────────┐│ │  ┌───────────────────┐  │
│  │  ECA注意力机制               ││ │  │  RevIN归一化      │  │
│  │  - 通道注意力权重             ││ │  │  时序投影         │  │
│  │  - 1D卷积建模通道相关性       ││ │  │  特征提取网络     │  │
│  └─────────────────────────────┘│ │  │  预测头           │  │
│  ┌─────────────────────────────┐│ │  │  RevIN反归一化    │  │
│  │  MRFP多感受野处理           ││ │  └───────────────────┘  │
│  │  - MultiDWConv1D            ││ │  输出: (B, pred_len, C)│
│  │  - 多尺度深度卷积            ││ └─────────────────────────┘
│  │  - 3,5,7,9,11,13核大小      ││
│  └─────────────────────────────┘│
│  ┌─────────────────────────────┐│
│  │  特征增强网络                ││
│  │  - 1x1卷积 + BatchNorm      ││
│  └─────────────────────────────┘│
└─────────────────────────────────┘
    ↓ (B, C, L)
┌─────────────────────────────────────────────────────────────┐
│                  小波编码器 (多层)                           │
│  AdpWaveletBlock: 逐层小波分解 → 近似系数 + 细节系数         │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│              Transformer编码器                               │
│  多层EncoderLayer: 自注意力 + 前馈网络                      │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│                  小波解码器 (多层)                           │
│  InverseAdpWaveletBlock: 逐层重构信号                       │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│                  输出投影                                    │
│  Linear: seq_len → pred_len                                │
└─────────────────────────────────────────────────────────────┘
    ↓ 季节性预测 (B, pred_len, C)    ↓ 趋势预测 (B, pred_len, C)
┌─────────────────────────────────────────────────────────────┐
│                    最终融合                                  │
│  季节性预测 + 趋势预测 → 最终输出 (B, pred_len, C)           │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 详细数据流分析（以ETTh1为例）

### 输入阶段
```python
输入: x_enc (B, L, C) = (16, 96, 7)
# 16个样本, 96个时间步, 7个特征
```

### 1. 序列分解阶段
```python
x, moving_mean = self.series_decomp(x_enc.permute(0,2,1))
# x_enc.permute(0,2,1): (16, 96, 7) → (16, 7, 96)
# x (季节性): (16, 7, 96)
# moving_mean (趋势): (16, 7, 96) → .permute(0,2,1) → (16, 96, 7)
```

**关键变化:**
- 输入: `(16, 96, 7)`
- 季节性分量: `(16, 7, 96)` - 保持通道×时序格式
- 趋势分量: `(16, 96, 7)` - 转换为时序×通道格式

### 2. ECA-MRFP融合注意力处理（季节性分量）

#### 2.1 ECA注意力机制
```python
# 输入: x (16, 7, 96)
y = self.gap(x)                    # 全局平均池化: (16, 7, 1)
y = y.permute(0, 2, 1)            # 转置: (16, 1, 7)
y = self.conv(y)                  # 1D卷积: (16, 1, 7)
attention_weights = self.sigmoid(y) # 注意力权重: (16, 1, 7)
enhanced_x = x * attention_weights.expand(-1, -1, 96)  # 加权: (16, 7, 96)
output = residual + enhanced_x * scale  # 残差连接: (16, 7, 96)
```

#### 2.2 MRFP多感受野处理
```python
# 输入: attention_out (16, 7, 96)
mrfp_input = attention_out.transpose(1, 2)  # (16, 96, 7)

# MultiDWConv1D处理
x1 = mrfp_input                        # 完整序列: (16, 96, 7)
x2 = mrfp_input[:, 24:72, :]          # 中间部分: (16, 48, 7)
x3 = mrfp_input[:, 36:60, :]          # 核心部分: (16, 24, 7)

# 多尺度卷积处理
x1 = x1.transpose(1, 2)               # (16, 7, 96)
x11, x12 = x1[:, :3, :], x1[:, 3:, :] # 分割通道
x11 = dwconv1(x11)                    # 3x1卷积: (16, 3, 96)
x12 = dwconv2(x12)                    # 5x1卷积: (16, 4, 96)
x1 = cat([x11, x12], dim=1)           # 合并: (16, 7, 96)

# 类似处理x2和x3，然后插值融合
mrfp_out = 0.5*x1 + 0.3*x2_interp + 0.2*x3_interp  # (16, 96, 7)
mrfp_out = mrfp_out.transpose(1, 2)   # (16, 7, 96)
```

#### 2.3 特征增强和融合
```python
enhanced_out = self.feature_enhance(mrfp_out)  # (16, 7, 96)
output = (residual * 0.3 + 
         attention_out * 0.4 + 
         enhanced_out * 0.3)  # (16, 7, 96)
```

### 3. DUET线性提取器处理（趋势分量）

```python
# 输入: moving_mean (16, 96, 7)

# Step 1: RevIN归一化
x_norm = self.revin(moving_mean, 'norm')  # (16, 96, 7)

# Step 2: 通道独立处理
x_reshaped = rearrange(x_norm, 'b l c -> (b c) l 1')
# (16, 96, 7) → (112, 96, 1)  # 16*7=112个独立序列

# Step 3: 时序投影
temporal_features = self.temporal_projection(x_reshaped.squeeze(-1))
# (112, 96) → (112, d_model)  # d_model=256
# Linear(96, 256): 将时序长度映射到模型维度

# Step 4: 特征提取
extracted_features = self.feature_extractor(temporal_features)
# (112, 256) → (112, 512) → (112, 256)
# Linear(256, 512) → GELU → Dropout → Linear(512, 256) → GELU → Dropout

# Step 5: 预测
predictions = self.prediction_head(extracted_features)
# (112, 256) → (112, pred_len)  # pred_len=96
# Linear(256, 96)

# Step 6: 重塑回原始格式
predictions = rearrange(predictions, '(b c) p -> b p c', b=16, c=7)
# (112, 96) → (16, 96, 7)

# Step 7: RevIN反归一化
predictions = self.revin(predictions, 'denorm')  # (16, 96, 7)
```

**关键数据流:**
```
趋势输入: (16, 96, 7)
  ↓ RevIN归一化
归一化后: (16, 96, 7)
  ↓ 通道独立处理 (reshape)
独立序列: (112, 96, 1)  # 每个通道独立处理
  ↓ 时序投影
时序特征: (112, 256)     # 将96个时间步压缩到256维
  ↓ 特征提取网络
提取特征: (112, 256)     # 经过2层MLP
  ↓ 预测头
趋势预测: (112, 96)      # 预测96个未来时间步
  ↓ 重塑格式
重塑后: (16, 96, 7)      # 恢复到批次×时间×通道
  ↓ RevIN反归一化
最终趋势: (16, 96, 7)
```

### 4. 小波编码阶段（季节性分量）

```python
# 输入: x (16, 7, 96)  # 经过ECA-MRFP处理

# 归一化
x_enc = x.permute(0,2,1)  # (16, 96, 7)
means = x_enc.mean(1, keepdim=True)
stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True) + 1e-5)
x_enc = (x_enc - means) / stdev  # 标准化: (16, 96, 7)

x_enc = x_enc.permute(0,2,1)  # (16, 7, 96)

# 多层小波编码 (lifting_levels=3)
# Level 0: (16, 7, 96) → (16, 7, 48) + 细节
# Level 1: (16, 7, 48) → (16, 7, 24) + 细节
# Level 2: (16, 7, 24) → (16, 7, 12) + 细节

for level in encoder_levels:
    x_enc, r, details = level(x_enc)  # 小波分解
    encoded_coefficients.append(details)  # 保存细节系数
```

**数据变化示例 (lifting_levels=3):**
```
输入: (16, 7, 96)
  ↓ Level 0 (小波分解)
近似: (16, 7, 48)  + 细节: (16, 7, 48)
  ↓ Level 1
近似: (16, 7, 24)  + 细节: (16, 7, 24)
  ↓ Level 2
近似: (16, 7, 12)  + 细节: (16, 7, 12)
最终: (16, 7, 12)  # 最小分辨率
```

### 5. Transformer编码阶段

```python
# 输入: x_enc (16, 7, 12)  # 最小分辨率

x_enc = x_enc.permute(0,2,1)  # (16, 12, 7)

# Embedding
enc_out = self.enc_embedding(x_enc, None)
# (16, 12, 7) → (16, 12, d_model)  # d_model=256

# Transformer编码 (e_layers=2)
enc_out, attns = self.encoder(enc_out, attn_mask=None)
# (16, 12, 256) → (16, 12, 256)

# 低秩投影
x_dec = self.lowrank_projection(enc_out)
# (16, 12, 256) → (16, 12, 7)
```

### 6. 小波解码阶段

```python
# 输入: x_dec (16, 12, 7)

x_dec = x_dec.permute(0,2,1)  # (16, 7, 12)

# 多层小波解码 (逆序)
# Level 2: (16, 7, 12) + 细节 → (16, 7, 24)
# Level 1: (16, 7, 24) + 细节 → (16, 7, 48)
# Level 0: (16, 7, 48) + 细节 → (16, 7, 96)

for decoder in decoder_levels:
    details = encoded_coefficients.pop()  # 取出细节系数
    x_dec = decoder(x_dec, details)        # 重构信号
```

**数据变化示例:**
```
输入: (16, 7, 12)
  ↓ Level 2 (逆小波重构) + 细节(16, 7, 12)
重构: (16, 7, 24)
  ↓ Level 1 + 细节(16, 7, 24)
重构: (16, 7, 48)
  ↓ Level 0 + 细节(16, 7, 48)
最终: (16, 7, 96)
```

### 7. 输出投影和融合

```python
# 季节性预测
x_dec = x_dec.permute(0,2,1)  # (16, 96, 7)
dec_out = self.projection(x_dec)  # (16, 96, 7) → (16, pred_len, 7)
# Linear(96, pred_len): 如果pred_len=96则保持不变

# 反归一化
dec_out = dec_out * stdev + means  # (16, pred_len, 7)

# 趋势预测 (DUET提取器输出)
moving_mean_out = self.trend_extractor(moving_mean)  # (16, pred_len, 7)

# 最终融合
final_output = dec_out + moving_mean_out  # (16, pred_len, 7)
```

## 🎯 YECAMRDU vs YECAMR610 关键区别

| 特性 | YECAMR610 | YECAMRDU |
|------|-----------|----------|
| **趋势处理** | K-Means聚类 + ClusteredLinear | DUET线性提取器 |
| **趋势归一化** | RevIN (单独) | RevIN (内置) |
| **趋势预测方式** | 聚类后独立线性层 | 通道独立时序投影+MLP |
| **需要参数** | n_clusters | duet_d_model |
| **处理流程** | 聚类 → 线性映射 | 归一化 → 投影 → 提取 → 预测 |

## 📈 完整数据维度变化总结

### 季节性路径
```
输入: (16, 96, 7)
  ↓ 序列分解
季节性: (16, 7, 96)
  ↓ ECA-MRFP注意力
增强特征: (16, 7, 96)
  ↓ 小波编码 (3层)
最小分辨率: (16, 7, 12)
  ↓ Transformer编码
编码特征: (16, 12, 256)
  ↓ 小波解码 (3层)
重构信号: (16, 7, 96)
  ↓ 投影
季节性预测: (16, 96, 7)  # 或 (16, pred_len, 7)
```

### 趋势路径 (DUET)
```
输入: (16, 96, 7)
  ↓ 序列分解
趋势: (16, 96, 7)
  ↓ RevIN归一化
归一化: (16, 96, 7)
  ↓ 通道独立处理
独立序列: (112, 96, 1)
  ↓ 时序投影
时序特征: (112, 256)
  ↓ 特征提取
提取特征: (112, 256)
  ↓ 预测头
趋势预测: (112, 96)  # 或 (112, pred_len)
  ↓ 重塑
趋势预测: (16, 96, 7)  # 或 (16, pred_len, 7)
  ↓ RevIN反归一化
最终趋势: (16, 96, 7)
```

### 最终融合
```
季节性预测: (16, pred_len, 7)
    +
趋势预测: (16, pred_len, 7)
    =
最终输出: (16, pred_len, 7)
```

## 🔑 核心创新点

1. **DUET线性提取器**: 
   - 通道独立处理，每个特征单独建模
   - 时序投影将时间序列映射到特征空间
   - MLP特征提取网络学习复杂模式
   - 内置RevIN归一化，提升稳定性

2. **双路径并行处理**:
   - 季节性路径：处理复杂周期模式
   - 趋势路径：处理平滑长期趋势
   - 独立建模，最后融合

3. **多尺度特征融合**:
   - ECA注意力：通道级特征增强
   - MRFP：多感受野时序特征提取
   - 小波变换：多分辨率特征分解

这个架构通过DUET线性提取器替代聚类方法，提供了更灵活的趋势预测能力，特别适合需要精确捕捉长期趋势的时序预测任务。






