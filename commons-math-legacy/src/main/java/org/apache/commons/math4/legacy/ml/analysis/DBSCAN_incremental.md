# DBSCANClusterer 增量改造方案

> 源文件：`clustering/DBSCANClusterer.java` / `Clusterer.java` / `Cluster.java`

---

## 一、当前实现为何只支持全量

`cluster(Collection<T> points)` 中所有状态均是**方法局部变量**：

```java
final List<Cluster<T>> clusters = new ArrayList<>();     // 局部
final Map<Clusterable, PointStatus> visited = new HashMap<>(); // 局部
```

每次调用方法结束后这些状态全部销毁，不存在任何跨调用的记忆。  
此外 `getNeighbors` 的邻域查询范围固定为传入的 `points`，不知道历史数据的存在。

---

## 二、增量化的核心挑战

新点加入时会产生三种影响，处理难度依次递增：

| 影响 | 说明 |
|------|------|
| ① 新点是噪声 | 不影响现有簇，直接标记 NOISE 即可 |
| ② 新点是新簇的核心点 | 在已有点中找不到邻居簇，创建新簇并 BFS 扩展 |
| ③ 新点桥接了多个现有簇 | **最复杂**：新点的邻域同时包含分属不同簇的点，需将这些簇合并 |

另一个隐含问题：**原来被标为 NOISE 的点，可能因为新点的加入变成某个簇的边界点**（border point），需要在扩展阶段处理。

---

## 三、改造方案设计

### 3.1 改造原则

- `cluster(Collection<T>)` **保持原有语义不变**（无状态全量，向后兼容）  
- 新增 `addPoints(Collection<T>)` 方法承载增量逻辑  
- 新增 `reset()` 方法清空增量状态  
- 增量状态通过**实例字段**跨调用保留

### 3.2 新增实例字段（4 个）

```java
/** 增量模式下累积的全部点 */
private final List<T> allPoints = new ArrayList<>();

/** 增量模式下的当前簇列表 */
private final List<Cluster<T>> currentClusters = new ArrayList<>();

/** 增量模式下的点状态表（null=未访问，NOISE，PART_OF_CLUSTER） */
private final Map<Clusterable, PointStatus> visitedState = new HashMap<>();

/** 反向索引：点 → 所属 Cluster（用于快速查找/合并） */
private final Map<Clusterable, Cluster<T>> pointToCluster = new HashMap<>();
```

### 3.3 完整新增代码

```java
// ========== 公共 API ==========

/**
 * 增量聚类入口：只传入新增数据，内部自动与历史数据合并计算邻域。
 *
 * @param newPoints 本批次新增点
 * @return 当前所有簇的快照（含历史数据的聚类结果）
 */
public List<Cluster<T>> addPoints(final Collection<T> newPoints) {
    NullArgumentException.check(newPoints);
    // 先把本批次全部加入 allPoints，使批次内后续点能在邻域查询中看到批次内前面的点
    allPoints.addAll(newPoints);
    for (final T point : newPoints) {
        processPoint(point);
    }
    return Collections.unmodifiableList(currentClusters);
}

/**
 * 清空增量状态，下一次 addPoints 将从零开始。
 */
public void reset() {
    allPoints.clear();
    currentClusters.clear();
    visitedState.clear();
    pointToCluster.clear();
}

// ========== 增量核心逻辑 ==========

/**
 * 处理单个新点：判断是核心点、边界点还是噪声，并执行相应操作。
 */
private void processPoint(final T point) {
    // 已被其他点的扩展阶段处理过（如作为邻居被吸收），跳过
    if (visitedState.containsKey(point)) {
        return;
    }

    final List<T> neighbors = getNeighbors(point, allPoints); // 邻域查全量

    if (neighbors.size() >= minPts) {
        // ---- 核心点分支 ----
        // 找到邻居所属的所有簇，若有多个则合并成一个
        final Cluster<T> target = resolveCluster(neighbors);

        // 将新点自身加入簇
        if (visitedState.get(point) != PointStatus.PART_OF_CLUSTER) {
            target.addPoint(point);
            visitedState.put(point, PointStatus.PART_OF_CLUSTER);
            pointToCluster.put(point, target);
        }

        // BFS 扩展
        expandIncrementally(target, neighbors);
    } else {
        // ---- 非核心点分支 ----
        // 检查邻居中是否有已归簇的点（即新点是某簇的边界点）
        final Cluster<T> adjacent = findAdjacentCluster(neighbors);
        if (adjacent != null) {
            adjacent.addPoint(point);
            visitedState.put(point, PointStatus.PART_OF_CLUSTER);
            pointToCluster.put(point, adjacent);
        } else {
            visitedState.put(point, PointStatus.NOISE);
        }
    }
}

/**
 * 从邻居列表中找到所有被引用的簇，将它们合并成一个后返回。
 * 若没有任何邻居属于已有簇，则创建并注册一个新簇。
 */
private Cluster<T> resolveCluster(final List<T> neighbors) {
    // 用 LinkedHashSet 保持顺序同时去重
    final Set<Cluster<T>> touched = new LinkedHashSet<>();
    for (final T n : neighbors) {
        final Cluster<T> c = pointToCluster.get(n);
        if (c != null) {
            touched.add(c);
        }
    }

    if (touched.isEmpty()) {
        final Cluster<T> fresh = new Cluster<>();
        currentClusters.add(fresh);
        return fresh;
    }

    // 以第一个簇为主，其余簇的点全部迁入
    final Iterator<Cluster<T>> it = touched.iterator();
    final Cluster<T> primary = it.next();
    while (it.hasNext()) {
        absorbCluster(primary, it.next());
    }
    return primary;
}

/**
 * BFS 扩展：从 initialSeeds 开始，将密度可达的点逐步加入 cluster。
 * 与原始 expandCluster 的区别：
 *   1. 邻域查询针对 allPoints（含历史数据）
 *   2. 若扩展到的点已在另一簇中，执行簇合并
 *   3. 原来标记为 NOISE 的点可被提升为边界点
 */
private void expandIncrementally(final Cluster<T> cluster,
                                  final List<T> initialSeeds) {
    List<T> seeds = new ArrayList<>(initialSeeds);
    int index = 0;
    while (index < seeds.size()) {
        final T current = seeds.get(index);
        final PointStatus status = visitedState.get(current);

        if (status == null) {
            // 未访问点：检查是否也是核心点
            final List<T> currentNeighbors = getNeighbors(current, allPoints);
            if (currentNeighbors.size() >= minPts) {
                // 也是核心点，将其邻居并入扩展队列
                seeds = merge(seeds, currentNeighbors);
                // 若 currentNeighbors 中有属于其他簇的点，合并那些簇
                for (final T n : currentNeighbors) {
                    absorbIfDifferentCluster(cluster, n);
                }
            }
            // 将 current 归入本簇
            cluster.addPoint(current);
            visitedState.put(current, PointStatus.PART_OF_CLUSTER);
            pointToCluster.put(current, cluster);

        } else if (status == PointStatus.NOISE) {
            // 原噪声点：被新核心点的扩展覆盖，提升为边界点
            cluster.addPoint(current);
            visitedState.put(current, PointStatus.PART_OF_CLUSTER);
            pointToCluster.put(current, cluster);

        } else {
            // 已归簇：若属于不同簇则合并
            absorbIfDifferentCluster(cluster, current);
        }

        index++;
    }
}

/**
 * 若 point 属于与 target 不同的簇，则将那个簇的全部点迁入 target。
 */
private void absorbIfDifferentCluster(final Cluster<T> target, final T point) {
    final Cluster<T> other = pointToCluster.get(point);
    if (other != null && other != target) {
        absorbCluster(target, other);
    }
}

/**
 * 将 other 中所有点迁移到 target，并从 currentClusters 中移除 other。
 */
private void absorbCluster(final Cluster<T> target, final Cluster<T> other) {
    for (final T p : other.getPoints()) {
        target.addPoint(p);
        pointToCluster.put(p, target);
    }
    currentClusters.remove(other);
}

/**
 * 在 neighbors 中查找第一个已归簇的点，返回其所在 Cluster；若无则返回 null。
 */
private Cluster<T> findAdjacentCluster(final List<T> neighbors) {
    for (final T n : neighbors) {
        final Cluster<T> c = pointToCluster.get(n);
        if (c != null) {
            return c;
        }
    }
    return null;
}
```

> 需在文件头补充 import：
> ```java
> import java.util.Collections;
> import java.util.Iterator;
> import java.util.LinkedHashSet;
> import java.util.Set;
> ```

---

## 四、使用方式对比

### 原有全量方式（不变）

```java
DBSCANClusterer<DoublePoint> clusterer = new DBSCANClusterer<>(0.5, 3);
List<Cluster<DoublePoint>> result = clusterer.cluster(allPoints); // 无状态
```

### 新增量方式

```java
DBSCANClusterer<DoublePoint> clusterer = new DBSCANClusterer<>(0.5, 3);

// 第一批
List<Cluster<DoublePoint>> result1 = clusterer.addPoints(batch1);

// 第二批（自动与 batch1 合并计算邻域）
List<Cluster<DoublePoint>> result2 = clusterer.addPoints(batch2);

// 需要重置时
clusterer.reset();
```

---

## 五、关键行为说明

### 5.1 多批次之间的邻域一致性

`addPoints` 在处理每个点之前，先把本批次**全部点**追加到 `allPoints`：

```
allPoints.addAll(newPoints);   // 先加入
for (T point : newPoints) {    // 再处理
    processPoint(point);
}
```

好处：批次内的点 A 扩展时，能把同批次还未单独处理的点 B 纳入邻域，避免漏算。

### 5.2 噪声点的重新分类

| 时间点 | 某点 p 的状态 | 触发条件 |
|--------|-------------|---------|
| 第1批  | NOISE       | 邻居数 < minPts |
| 第2批新核心点 q 加入 | PART_OF_CLUSTER（提升为边界点） | q 的 BFS 扩展经过 p |

原始批量 DBSCAN 中噪声点被静默丢弃；增量版中噪声点**依然保留在 visitedState**，可被后续批次的核心点"捡回"。

### 5.3 簇合并

当新核心点 q 的邻域同时包含已属于簇 C1 和簇 C2 的点时：

```
resolveCluster()
  → 发现 C1、C2 均被触及
  → 将 C2 的所有点迁入 C1
  → 从 currentClusters 移除 C2
  → 返回 C1 作为目标簇
```

合并后 `pointToCluster` 中 C2 的点全部指向 C1，保证反向索引一致性。

---

## 六、性能特征

| 项目 | 全量 `cluster()` | 增量 `addPoints()` |
|------|----------------|-------------------|
| 每次调用复杂度 | O(n²) | O(k·n)，k 为本批次新点数 |
| 内存占用 | 无状态 | O(n)，累积全部历史点 |
| 需要空间索引 | 否（当前均为线性扫描） | 否（同上，但更值得优化） |

> 增量模式中每个新点调用一次 `getNeighbors`（O(n)），若该点是核心点则其邻居也各调用一次，最坏仍是 O(n) per new point。若数据量大，可在 `allPoints` 上建 KD-tree 或 Ball-tree 将邻域查询降至 O(log n)。

---

*生成日期：2026-04-16*
