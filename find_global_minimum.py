#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from io import StringIO

# 设置随机种子以确保结果可重现
np.random.seed(42)

# 1. 加载并准备数据
df = pd.read_csv('rel_energy.csv')
print(f"数据集总大小: {len(df)}")

# 2. 初始化主动学习状态
df['computed'] = False
df['dft_energy_actual'] = df['dft_energy']  # 目标：最小化 dft_energy（越小越好）
df['dft_energy_pred'] = np.nan
df['residual'] = np.nan

# 3. 初始随机采样（5个样本）
initial_samples = df.sample(5, random_state=42)
initial_indices = initial_samples.index.tolist()
print(f"\n初始随机选择的5个构型: {[df.loc[i, 'Confid'] for i in initial_indices]}")

df.loc[initial_indices, 'computed'] = True
df.loc[initial_indices, 'dft_energy_pred'] = df.loc[initial_indices, 'dft_energy_actual']

# 记录DFT计算次数
dft_count = len(initial_indices)

# 4. 主动学习循环 - 目标：寻找 dft_energy 最小的构型
history = []
max_iterations = 50

# 初始最低能量和对应构型（基于 dft_energy）
current_min_energy = df[df['computed']]['dft_energy_actual'].min()
current_min_conf = df[(df['computed']) & (df['dft_energy_actual'] == current_min_energy)]['Confid'].iloc[0]
print(f"初始最低能量: {current_min_energy:.6f} (来自 {current_min_conf})")

for iteration in range(max_iterations):
    # 准备训练数据：已计算DFT的样本
    train_data = df[df['computed']]
    X_train = train_data['gxtb_rel_energy'].values.reshape(-1, 1)
    y_train = train_data['dft_energy_actual'].values  # ←←← 关键：使用 dft_energy

    # 准备预测数据：未计算DFT的样本
    predict_data = df[~df['computed']]
    if len(predict_data) == 0:
        print("所有样本都已计算过DFT!")
        break

    X_predict = predict_data['gxtb_rel_energy'].values.reshape(-1, 1)

    # 训练线性回归模型
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 为所有未计算样本预测 DFT 能量
    y_pred = model.predict(X_predict)
    df.loc[~df['computed'], 'dft_energy_pred'] = y_pred  # ←←← 预测 dft_energy

    # 计算已计算样本的残差（用于不确定性估计）
    if len(train_data) > 1:
        train_pred = model.predict(X_train)
        train_data_indices = train_data.index
        df.loc[train_data_indices, 'residual'] = np.abs(train_data['dft_energy_actual'] - train_pred)

    # 计算未计算样本的不确定性（KNN平均残差）
    from sklearn.neighbors import NearestNeighbors
    if len(train_data) > 0:
        knn = NearestNeighbors(n_neighbors=min(3, len(train_data)))
        knn.fit(X_train)
        distances, indices = knn.kneighbors(X_predict)

        avg_residuals = []
        for neighbor_indices in indices:
            avg_residual = train_data.iloc[neighbor_indices]['residual'].mean()
            avg_residuals.append(avg_residual)

        df.loc[~df['computed'], 'residual'] = avg_residuals

    # 查询策略：专注于寻找最低 dft_energy
    unexplored = df[~df['computed']]

    # 策略1：选择预测 dft_energy 最低的3个（越小越好！）
    exploitation_candidates = unexplored.nsmallest(3, 'dft_energy_pred')

    # 策略2：选择残差最大（最不确定）的1个
    exploration_candidates = unexplored.nlargest(1, 'residual')

    # 合并候选样本
    selected_candidates = pd.concat([exploitation_candidates, exploration_candidates])
    selected_candidates = selected_candidates.drop_duplicates()
    selected_candidates = selected_candidates.head(min(4, len(unexplored)))
    selected_indices = selected_candidates.index.tolist()

    # 执行DFT计算（模拟：获取真实值）
    df.loc[selected_indices, 'computed'] = True
    df.loc[selected_indices, 'dft_energy_pred'] = df.loc[selected_indices, 'dft_energy_actual']

    # 更新DFT计算计数
    dft_count += len(selected_indices)

    # 更新当前最低能量和对应构型（基于 dft_energy）
    new_min_energy = df[df['computed']]['dft_energy_actual'].min()
    new_min_conf = df[(df['computed']) & (df['dft_energy_actual'] == new_min_energy)]['Confid'].iloc[0]

    min_energy_improved = new_min_energy < current_min_energy
    if min_energy_improved:
        current_min_energy = new_min_energy
        current_min_conf = new_min_conf

    # 计算模型性能
    if len(train_data) > 1:
        y_train_pred = model.predict(X_train)
        r2 = r2_score(y_train, y_train_pred)
        rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    else:
        r2 = 0
        rmse = 0

    # 记录迭代统计信息
    history.append({
        'iteration': iteration,
        'dft_count': dft_count,
        'new_selected': [df.loc[i, 'Confid'] for i in selected_indices],
        'min_energy': current_min_energy,
        'min_conf': current_min_conf,
        'min_energy_improved': min_energy_improved,
        'r2_score': r2,
        'rmse': rmse,
        'model_coef': model.coef_[0],
        'model_intercept': model.intercept_
    })

    print(f"\n迭代 {iteration+1}:")
    print(f"  选择的构型: {[df.loc[i, 'Confid'] for i in selected_indices]}")
    print(f"  累计DFT计算: {dft_count}")
    print(f"  当前最低能量: {current_min_energy:.6f} (来自 {current_min_conf})")
    if min_energy_improved:
        print(f"  ✅ 发现了更低的能量!")
    print(f"  模型性能: R²={r2:.3f}, RMSE={rmse:.6f}")
    print(f"  模型参数: y = {model.coef_[0]:.6f} * x + {model.intercept_:.6f}")

    # 停止条件：连续3轮未改进
    no_improvement_rounds = 3
    recent_rounds_without_improvement = 0
    for i in range(max(0, len(history)-no_improvement_rounds), len(history)):
        if not history[i]['min_energy_improved']:
            recent_rounds_without_improvement += 1
        else:
            recent_rounds_without_improvement = 0

    if recent_rounds_without_improvement >= no_improvement_rounds:
        print(f"\n🛑 连续{no_improvement_rounds}轮没有找到更低的能量点，停止迭代。")
        break

    if len(selected_indices) == 0:
        print("\n⚠️ 没有新样本可选择，提前停止。")
        break

# 5. 输出最终结果
print("\n" + "="*50)
print("主动学习模拟完成!")
print(f"总DFT计算次数: {dft_count}")
print(f"找到的最低能量: {current_min_energy:.6f}")
print(f"对应构型: {current_min_conf}")

# 绘制学习曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot([h['dft_count'] for h in history], [h['min_energy'] for h in history], 'o-')
plt.xlabel('DFT计算次数')
plt.ylabel('发现的最低 DFT 能量')
plt.title('最低能量发现进程')
plt.grid(True)

# 标记能量改进的点
improvement_points = [i for i, h in enumerate(history) if h['min_energy_improved']]
if improvement_points:
    improvement_counts = [history[i]['dft_count'] for i in improvement_points]
    improvement_energies = [history[i]['min_energy'] for i in improvement_points]
    plt.scatter(improvement_counts, improvement_energies, color='red', s=50, zorder=5)
    for i, point in enumerate(improvement_points):
        plt.annotate(f"{history[point]['min_conf']}",
                    (history[point]['dft_count'], history[point]['min_energy']),
                    xytext=(5, 5), textcoords='offset points')

plt.subplot(1, 2, 2)
plt.plot([h['iteration']+1 for h in history], [h['rmse'] for h in history], 's-', label='RMSE')
plt.xlabel('迭代次数')
plt.ylabel('RMSE')
plt.title('模型预测误差')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 输出找到的最低能量构型详情
print("\n找到的最低能量构型详情:")
min_energy_row = df[df['Confid'] == current_min_conf].iloc[0]
print(f"  构型ID: {min_energy_row['Confid']}")
print(f"  DFT能量: {min_energy_row['dft_energy_actual']:.6f}")
print(f"  g-xTB相对能量: {min_energy_row['gxtb_rel_energy']:.3f}")
print(f"  是否通过DFT计算: {'是' if min_energy_row['computed'] else '否'}")

# 输出所有已计算构型中能量最低的5个
print("\n所有已计算构型中能量最低的5个:")
computed_lowest = df[df['computed']].nsmallest(5, 'dft_energy_actual')
for _, row in computed_lowest.iterrows():
    print(f"  {row['Confid']}: {row['dft_energy_actual']:.6f} (g-xTB: {row['gxtb_rel_energy']:.3f})")

# 输出迭代历史详情
print("\n迭代历史详情:")
for h in history:
    improvement_flag = "✓" if h['min_energy_improved'] else "✗"
    print(f"迭代 {h['iteration']+1}: DFT计算={h['dft_count']}, "
          f"最低能量={h['min_energy']:.6f} ({h['min_conf']}), "
          f"改进={improvement_flag}")

# 评估最终效果
efficiency = dft_count / len(df) * 100
print(f"\n效率评估: 使用了 {efficiency:.1f}% 的DFT计算找到了最低能量构型 {current_min_conf}")
