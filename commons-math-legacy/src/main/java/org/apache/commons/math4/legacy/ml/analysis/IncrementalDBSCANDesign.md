# 增量 DBSCAN 聚类器设计与实现报告

> 生成日期：2026-04-17  
> 参考实现：[incdbscan](https://github.com/vjcitn/incdbscan)（Python）  
> 目标项目：`commons-math-legacy` — `org.apache.commons.math4.legacy.ml.clustering`

---

## 1. 背景与动机

标准 `DBSCANClusterer` 每次调用 `cluster(Collection<T>)` 都会对全量数据重新扫描，时间复杂度为 **O(n²)**（暴力距离计算）或 **O(n log n)**（有索引结构）。当数据以流式方式不断到来、或需要频繁删除过时数据点时，全量重算的开销是不可接受的。

**增量 DBSCAN** 的核心思想：每次只更新被变更点的 ε-邻域，局部修正聚类状态，时间复杂度降为 **O(k)**，其中 k 为局部邻域大小，且 k << n 时效果显著。

---

## 2. incdbscan 方案分析

### 2.1 关键数据结构

| 结构 | 用途 |
|---|---|
| `Object` | 包装数据点；维护 `neighbors`（含自身）、`neighbor_count`、`is_core` 属性 |
| `Objects` (rustworkx 图) | 以图节点表示数据点，以边表示 ε-邻居关系；支持子图查询和连通分量 |
| `LabelHandler` | 维护 `object→label` 和 `label→objects` 双向映射，支持高效的标签批量改名 |
| `NeighborSearcher` | 自适应空间索引（低维用 cKDTree，高维用 sklearn NearestNeighbors） |
| `BFSComponentFinder` | 多源 BFS，用于删除后的裂分检测（仅沿核心点边遍历） |

### 2.2 `neighbor_count` 语义

- 初始为 0；当空间索引返回一组邻居（含自身）后，每找到一个邻居就 +1
- 最终 `neighbor_count` = ε 邻域内的点数（含自身）
- `is_core` = `neighbor_count >= min_pts`（与标准 DBSCAN 定义一致）

### 2.3 插入流程（四种情形）

```
insert(p):
  1. 将 p 加入图，更新所有邻居的 neighbor_count
  2. 区分 new_cores（刚好达到 min_pts 的邻居）和 old_cores（已是核心的邻居）
  3. 若 new_cores 为空：
       if old_cores 非空 → Absorption（吸收到标号最大的簇）
       else              → Noise
  4. 若 new_cores 非空：
       update_seeds = 所有 new_core 的全部核心邻居
       在 update_seeds 的诱导子图中找连通分量
       for each 分量:
         if 无有效标号 → Creation（创建新簇）
         else          → Absorption/Merge（合并到最大标号簇）
       再将每个 new_core 的标号传播给其非核心邻居（边界点）
```

### 2.4 删除流程（裂分检测）

```
delete(p):
  1. 递减所有邻居的 neighbor_count
  2. ex_cores = 刚好失去核心性的点（count 从 min_pts 降到 min_pts-1）
  3. update_seeds = ex_cores 的剩余核心邻居（排除 p）
  4. 按簇分组 update_seeds，对每组：
       BFS 沿核心点边遍历（排除 p）
       若检测到断开 → 较小的子分量获得新标号
  5. 非核心（边界）邻居：重新选择最近的核心邻居的标号，否则变为噪声
```

---

## 3. Java 实现方案

### 3.1 设计决策

| 决策项 | 选择 | 理由 |
|---|---|---|
| 邻居图结构 | `PointNode.neighbors`（HashSet） | 避免引入 rustworkx 外部依赖，Java 标准库即可满足 |
| 空间索引 | 无（线性扫描） | 保持与现有 `DBSCANClusterer` 一致；高维场景可在后续扩展 |
| 标号管理 | 直接存在 `PointNode.label` 中 | 简化实现；`changeLabels()` 遍历 O(n) 可接受 |
| 连通分量（插入） | BFS on `neighbors` 集合 | 子图连通性检测，仅用直接 ε-边 |
| 裂分检测（删除） | 多源 BFS + 分量合并 | 与 `BFSComponentFinder` 逻辑等价 |
| 重复点 | 第二次 `addPoint` 是 no-op | 简化；无需维护 count 字段 |

### 3.2 标号语义

```
LABEL_UNCLASSIFIED = -2   // 刚插入，尚未分类（内部状态）
LABEL_NOISE        = -1   // 噪声点
0, 1, 2, ...              // 有效簇，单调递增，永不复用
```

### 3.3 类结构

```
IncrementalDBSCANClusterer<T extends Clusterable>
  extends Clusterer<T>
  │
  ├── 字段
  │    ├── eps, minPts          // 算法参数
  │    ├── nextLabel            // 下一个可用簇标号
  │    └── nodes: Map<T, PointNode>  // 活跃点元数据
  │
  ├── 内部类 PointNode
  │    ├── point                // 原始数据点
  │    ├── neighbors: Set<PointNode>  // ε-邻居（含自身）
  │    ├── neighborCount        // 邻居数量（含自身）
  │    ├── label                // 当前簇标号
  │    └── isCore()             // neighborCount >= minPts
  │
  ├── 公开 API
  │    ├── cluster(Collection)  // 批量接口（重置状态后逐点插入）
  │    ├── addPoint(T)          // 增量插入
  │    ├── removePoint(T)       // 增量删除
  │    ├── getClusters()        // 获取当前聚类结果
  │    ├── getLabel(T)          // 获取单点标号
  │    ├── reset()              // 清空状态
  │    └── size()               // 活跃点数量
  │
  └── 私有实现
       ├── linkNeighbors()          // 插入时建立邻居链接
       ├── processInsertion()       // 插入后的四种情形处理
       ├── processDeletion()        // 删除后的连通性修复
       ├── findSplitComponents()    // 多源 BFS 裂分检测
       ├── connectedComponentsOf()  // 插入时的子图连通分量
       ├── setLabels() / changeLabels() / propagateAroundNewCores()
       └── mergeComponentMaps() / allMutualNeighbors()
```

### 3.4 核心代码片段说明

#### 插入时区分 new_core / old_core

```java
for (PointNode neighbor : inserted.neighbors) {
    if (neighbor.neighborCount == minPts) {
        newCores.add(neighbor);      // 刚达到核心阈值
    } else if (neighbor.neighborCount > minPts) {
        oldCores.add(neighbor);      // 已经是核心
    }
}
// 插入点自身若成核，也属于 new_core
if (inserted.isCore()) {
    oldCores.remove(inserted);
    newCores.add(inserted);
}
```

#### 删除时的 ex_cores 识别

```java
// 先递减所有邻居的计数（含自身）
for (PointNode neighbor : toDelete.neighbors) {
    neighbor.neighborCount--;
}
// 递减后 neighborCount == minPts-1 → 刚失去核心性
for (PointNode neighbor : toDelete.neighbors) {
    if (neighbor != toDelete && neighbor.neighborCount == minPts - 1) {
        exCores.add(neighbor);
    }
}
// 被删除节点自身若原来是核心也要加入
if (toDelete.neighborCount + 1 >= minPts) {
    exCores.add(toDelete);
}
```

#### 裂分检测（多源 BFS）

```java
// 初始化：每个 seed 属于自己的分量
for (int i = 0; i < seeds.size(); i++) {
    assignment.put(seeds.get(i), i);
    components.put(i, new HashSet<>(Collections.singleton(seeds.get(i))));
}
// BFS（仅遍历核心点，排除被删节点）
while (!queue.isEmpty()) {
    PointNode current = queue.poll();
    for (PointNode neighbor : current.neighbors) {
        if (!visited.contains(neighbor)) {
            // 树边：归属当前 seed 的分量
            assignment.put(neighbor, currentIdx);
            components.get(currentIdx).add(neighbor);
            queue.add(neighbor);
        } else if (assignment.get(neighbor) != currentIdx) {
            // 交叉边：合并两个分量
            mergeComponentMaps(components, assignment, neighborIdx, currentIdx);
        }
    }
}
// 最大分量保留原标号；其余分量获得新标号
```

---

## 4. 与 incdbscan 的对应关系

| incdbscan 模块 | Java 对应位置 |
|---|---|
| `incrementaldbscan.py` (公开 API) | `IncrementalDBSCANClusterer` 公开方法 |
| `_object.py` (Object 类) | `IncrementalDBSCANClusterer.PointNode` 内部类 |
| `_objects.py` (Objects 类) | `linkNeighbors()` + `Map<T,PointNode> nodes` |
| `_inserter.py` (Inserter) | `processInsertion()` |
| `_deleter.py` (Deleter) | `processDeletion()` |
| `_bfscomponentfinder.py` | `findSplitComponents()` + `mergeComponentMaps()` |
| `_labels.py` (LabelHandler) | `setLabels()` / `changeLabels()` / `propagateAroundNewCores()` |
| `_neighbor_searcher.py` | 线性扫描（`linkNeighbors()` 中暴力计算距离） |

---

## 5. 与标准 DBSCANClusterer 的对比

| 维度 | DBSCANClusterer | IncrementalDBSCANClusterer |
|---|---|---|
| 更新模式 | 批量重算 | 增量更新 |
| 时间复杂度（单次更新） | O(n²) | O(k)，k = 邻域大小 |
| 持久化状态 | 无 | 维护邻居图和标号映射 |
| 支持删除 | 否（重算） | 是（`removePoint`） |
| 簇标号稳定性 | 每次重算可能不同 | 单调递增，永不复用 |
| `cluster()` 接口 | 标准批量 | 清空状态后逐点插入，结果等价 |
| 内存占用 | 一次调用后可丢弃 | 需持续持有邻居图 |

---

## 6. 已知限制与后续改进方向

1. **无空间索引**：`linkNeighbors()` 为 O(n) 线性扫描，插入整体为 O(n)。
   可引入 KD-Tree 或球树将其优化至 O(log n) ~ O(k log k)。

2. **无重复点支持**：与 incdbscan 不同，同一点对象（按 `equals`）第二次插入为 no-op。
   如需支持重复，需引入类似 `count` 字段的机制。

3. **`changeLabels()` 为 O(n)**：每次合并操作需遍历全部节点。
   可引入双向映射（label → Set<PointNode>）优化至 O(k)。

4. **并发安全**：当前实现非线程安全，并发场景需外部同步或使用 `ConcurrentHashMap`。

5. **裂分时保留最大分量**：当前策略是保留最大分量的原标号，其余获得新标号。
   更精确的策略应参照原论文根据密度连通性判断。

---

## 7. 文件位置

```
commons-math-legacy/src/main/java/org/apache/commons/math4/legacy/ml/
├── clustering/
│   ├── DBSCANClusterer.java              (原标准实现，不变)
│   └── IncrementalDBSCANClusterer.java   (新增增量实现)
└── analysis/
    └── IncrementalDBSCANDesign.md        (本文档)
```
