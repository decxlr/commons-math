# IncrementalDBSCANClustererV1 与 DBSCANClusterer 聚类结果差异分析

> 生成日期：2026-04-21
> 分析文件：`clustering/IncrementalDBSCANClustererV1.java` vs `clustering/DBSCANClusterer.java`
> 测试文件：`ml/test/DBSCANClustererTest.java`

---

## 一、问题现象

使用相同参数 (`eps=0.5, minPts=5`) 和相同数据集 (`seed=100, count=100, radius=5`)，两个算法的聚类结果不一致：

| 指标 | DBSCANClusterer (原始) | IncrementalDBSCANClustererV1 (增量) |
|------|----------------------|-------------------------------------|
| 簇数量 | 7 | 6 |
| 已聚类点数 | 48 | 54 |
| 噪声点数 | 52 | 46 |

**具体差异：**

1. **簇合并**：原始结果的 Cluster #3（8点）、#5（6点）、#7（3点）在增量版本中被**错误合并**为一个包含17个点的大簇
2. **新簇产生**：增量版本多出了一个由5个点组成的新簇 `(3.6,1.0), (3.9,0.6), (3.2,1.1), (3.9,1.3), (3.7,1.1)`，这些点在原始算法中是噪声
3. **点数不一致**：增量版本将更多点纳入簇中

---

## 二、根本原因：minPts 定义 off-by-one 错误

### 2.1 两个算法对 minPts 的定义不同

**原始 DBSCANClusterer** 的核心点判断：

```java
// DBSCANClusterer.java 第212-215行
private List<T> getNeighbors(final T point, final Collection<T> points) {
    return points.stream()
        .filter(neighbor -> point != neighbor && distance(neighbor, point) <= eps)  // 排除自身
        .collect(Collectors.toList());
}

// 第143行
if (neighbors.size() >= minPts) {  // 邻居数（不含自身）>= minPts → 核心点
```

- `getNeighbors()` 返回的邻居列表**排除自身** (`point != neighbor`)
- 核心点条件：**其他点**（不含自身）在 eps 内的数量 >= minPts
- 即：`minPts = 5` 时，需要 **5个其他点** 在 eps 范围内才是核心点

**IncrementalDBSCANClustererV1** 的核心点判断：

```java
// IncrementalDBSCANClustererV1.java 第157-161行
PointNode(final T point) {
    this.neighbors.add(this);   // 自身始终是邻居！
    this.neighborCount = 1;     // 计数包含自身
}

// 第165-167行
boolean isCore() {
    return neighborCount >= minPts;  // 邻居数（含自身）>= minPts → 核心点
}
```

- `neighborCount` **包含自身**（构造时初始化为1，自身加入 neighbors 集合）
- 核心点条件：**总点数**（含自身）在 eps 内的数量 >= minPts
- 即：`minPts = 5` 时，只需 **4个其他点** 在 eps 范围内就是核心点

### 2.2 差异总结

| 算法 | 核心点条件 (minPts=5) | 实际需要的其他邻居数 |
|------|---------------------|-------------------|
| DBSCANClusterer | neighbors.size() >= 5（不含自身） | >= 5 |
| IncrementalDBSCANClustererV1 | neighborCount >= 5（含自身） | >= 4 |

**增量算法的核心点门槛比原始算法低了 1**，导致更多点被分类为核心点，引发连锁错误。

### 2.3 为什么多出核心点会导致结果差异？

DBSCAN 算法中，核心点具有特殊地位：

1. **核心点定义簇**：只有核心点才能创建和扩展簇
2. **核心点连接簇**：如果两个核心点在 eps 范围内，它们必须属于同一个簇（密度连通性）
3. **核心点吸收边界点**：非核心点如果在一个核心点的 eps 范围内，会被吸收为边界点

当核心点门槛降低后：

- **本应是噪声/边界点的点变成了核心点** → 创建了不该存在的簇
- **本应属于不同簇的核心点之间出现了"桥梁核心点"** → 将本应独立的簇错误合并
- **更多点被核心点吸收** → 噪声点减少，已聚类点增多

---

## 三、具体案例验证

### 3.1 案例1：新增的5点簇

增量结果中 Cluster #1 包含5个点：`(3.6,1.0), (3.9,0.6), (3.2,1.1), (3.9,1.3), (3.7,1.1)`

以点 `(3.6, 1.0)` 为例，计算它与其他4点的距离：

| 目标点 | 欧氏距离 | <= 0.5? |
|--------|---------|---------|
| (3.9, 0.6) | sqrt(0.09+0.16) = 0.50 | 是 |
| (3.2, 1.1) | sqrt(0.16+0.01) = 0.41 | 是 |
| (3.9, 1.3) | sqrt(0.09+0.09) = 0.42 | 是 |
| (3.7, 1.1) | sqrt(0.01+0.01) = 0.14 | 是 |

`(3.6, 1.0)` 在这5个点中有 **4个其他邻居**（加上自身共5个）：

- **增量算法**：neighborCount=5 >= minPts=5 → **核心点** → 创建簇 ✓
- **原始算法**：neighbors.size()=4 < minPts=5 → **不是核心点** → 噪声 ✗

这就是为什么这5个点在增量版本中形成了簇，而在原始版本中是噪声。

### 3.2 案例2：三个簇的错误合并

原始结果的 Cluster #3、#5、#7 在增量版本中被合并为一个17点的大簇。

这三个簇之间存在"桥梁点"，它们在原始算法中因为邻居数不足而**不是核心点**，无法连接不同簇；但在增量算法中，由于核心点门槛降低了1，这些桥梁点**变成了核心点**，从而将三个独立簇错误地连接在一起。

具体来说，位于簇边缘的点（如 `(2.4, 0.6)`, `(2.1, 0.9)` 等）可能恰好有4个其他邻居在 eps 范围内，不足以满足原始算法的5个邻居要求，但满足了增量算法的4个邻居要求（含自身共5个），从而成为核心点，将相邻的簇连通。

---

## 四、受影响的代码位置汇总

所有使用 `neighborCount` 与 `minPts` 比较的地方都受到影响：

| 位置 | 代码 | 影响 |
|------|------|------|
| 第166行 `isCore()` | `neighborCount >= minPts` | 核心点判断门槛低1 |
| 第401行 | `neighbor.neighborCount == minPts` | 新核心点检测门槛低1 |
| 第404行 | `neighbor.neighborCount > minPts` | 旧核心点分类门槛低1 |
| 第518行 | `neighbor.neighborCount == minPts - 1` | 删除时ex-core检测门槛低1 |
| 第522行 | `toDelete.neighborCount + 1 >= minPts` | 删除时核心点判断门槛低1 |

---

## 五、修复方案

### 方案A：移除自身计数（推荐）

让 `neighborCount` 不包含自身，与原始 DBSCANClusterer 保持一致：

```java
// PointNode 构造器修改
PointNode(final T point) {
    this.point = point;
    this.label = LABEL_UNCLASSIFIED;
    // 不再将自身加入 neighbors 集合
    this.neighborCount = 0;     // 不计数自身
}

// linkNeighbors 修改
private void linkNeighbors(final PointNode newNode) {
    for (final PointNode existing : nodes.values()) {
        if (existing == newNode) {
            continue;  // 跳过自身
        }
        if (distance(existing.point, newNode.point) <= eps) {
            newNode.neighbors.add(existing);
            newNode.neighborCount++;
            existing.neighbors.add(newNode);
            existing.neighborCount++;
        }
    }
}

// isCore 不变
boolean isCore() {
    return neighborCount >= minPts;  // 现在不含自身，语义正确
}

// processInsertion 新核心点判断不变
if (neighbor.neighborCount == minPts) {
    newCores.add(neighbor);  // 刚好达到门槛，语义正确
}

// processDeletion ex-core判断不变
if (neighbor.neighborCount == minPts - 1) {
    exCores.add(neighbor);  // 刚好失去核心状态，语义正确
}

// toDelete 核心判断
if (toDelete.neighborCount >= minPts) {  // 不再需要 +1
    exCores.add(toDelete);
}
```

**优点**：语义清晰，与原始算法完全一致，所有比较逻辑无需调整
**缺点**：需要检查所有遍历 `neighbors` 集合的代码（不再包含自身，某些 `for` 循环的逻辑可能受影响）

### 方案B：保留自身计数，调整比较阈值

保留 `neighbors` 包含自身的设计，但调整比较逻辑：

```java
// isCore 修改
boolean isCore() {
    return neighborCount > minPts;  // 等价于 (neighborCount-1) >= minPts
}

// processInsertion 新核心点判断
if (neighbor.neighborCount == minPts + 1) {  // 刚好达到 (neighborCount-1) >= minPts
    newCores.add(neighbor);
} else if (neighbor.neighborCount > minPts + 1) {
    oldCores.add(neighbor);
}

// processDeletion ex-core判断
if (neighbor.neighborCount == minPts) {  // 递减后 (neighborCount-1) == minPts-1
    exCores.add(neighbor);
}

// toDelete 核心判断保持不变
if (toDelete.neighborCount + 1 >= minPts) {  // 这行本身没有bug
```

**优点**：改动较小，不需要修改 `neighbors` 集合的使用
**缺点**：阈值调整不直观，容易出错，代码可读性差

### 推荐方案

**推荐方案A**，因为它从根本上对齐了两个算法的 minPts 语义，代码清晰且不易出错。

---

## 六、修复后的验证方法

修复后，使用以下测试验证两个算法结果一致性：

1. 使用 `simpleIncrementalDBSCANClusterV1Test()` 和 `originalDBSCANTest()` 中的相同参数和数据集
2. 验证增量版本的簇数量、已聚类点数与原始版本一致
3. 验证增量版本中的每个簇的点集与原始版本的某个簇的点集完全对应
4. 注意：边界点的分配在不同算法间可能不同（这是 DBSCAN 的固有歧义性），但核心点的分配必须一致

---

## 七、结论

**IncrementalDBSCANClustererV1 的实现确实存在问题。** 核心原因是 `neighborCount` 包含自身导致 minPts 的实际含义与原始 `DBSCANClusterer` 不一致，使得核心点判定门槛降低了1。这导致：

1. 更多的点被分类为核心点
2. 产生了不应存在的簇（如5点新簇）
3. 将本应独立的簇错误合并（如3个簇合并为1个）
4. 噪声点被错误地纳入簇中

这是一个 **严重的正确性 Bug**，影响所有使用 `IncrementalDBSCANClustererV1` 的聚类结果。
