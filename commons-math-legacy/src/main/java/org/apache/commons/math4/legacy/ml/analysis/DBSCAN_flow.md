# DBSCANClusterer 执行流程分析

> 源文件：`clustering/DBSCANClusterer.java`

---

## 一、调用入口

```java
List<Cluster<T>> result = dBSCANClusterer.cluster(points);
```

---

## 二、完整方法调用链

```
cluster(points)                              // 主入口，第131行
│
├─ NullArgumentException.check(points)       // 空值校验
│
├─ 初始化
│   ├─ clusters = new ArrayList<>()          // 最终返回的聚类列表
│   └─ visited  = new HashMap<>()            // 记录每个点的状态(NOISE / PART_OF_CLUSTER)
│
└─ for each point in points                  // 遍历全部输入点
    │
    ├─ [跳过] visited.get(point) != null     // 已访问过 → 直接跳过
    │
    ├─ getNeighbors(point, points)           // 第203行：找出 ε 邻域内所有邻居
    │   └─ 对每个 neighbor：
    │       distance(neighbor, point)        // 继承自 Clusterer，调用 DistanceMeasure.compute()
    │       条件：point != neighbor && distance <= eps
    │
    ├─ [核心点分支] neighbors.size() >= minPts
    │   └─ expandCluster(cluster, point, neighbors, points, visited)  // 第165行
    │       │
    │       ├─ cluster.addPoint(point)               // 将核心点加入簇
    │       ├─ visited.put(point, PART_OF_CLUSTER)
    │       │
    │       └─ seeds = new ArrayList<>(neighbors)    // 待扩展队列，初始为核心点的邻居
    │           │
    │           └─ while (index < seeds.size())      // 迭代扩展（非递归，用队列模拟BFS）
    │               │
    │               ├─ current = seeds.get(index)
    │               ├─ pStatus = visited.get(current)
    │               │
    │               ├─ [未访问] pStatus == null
    │               │   ├─ getNeighbors(current, points)   // 再次计算当前点的邻居
    │               │   └─ [也是核心点] currentNeighbors.size() >= minPts
    │               │       └─ merge(seeds, currentNeighbors)  // 第215行：将新邻居追加进队列
    │               │           └─ 用 HashSet 去重后追加到 seeds
    │               │
    │               ├─ [未归簇] pStatus != PART_OF_CLUSTER
    │               │   ├─ visited.put(current, PART_OF_CLUSTER)
    │               │   └─ cluster.addPoint(current)       // 边界点/未访问点加入当前簇
    │               │
    │               └─ index++
    │
    └─ [噪声分支] neighbors.size() < minPts
        └─ visited.put(point, NOISE)                 // 标记为噪声，不加入任何簇
```

---

## 三、各方法说明

| 方法 | 位置 | 作用 |
|------|------|------|
| `cluster(Collection<T>)` | 第131行 | 主入口，遍历所有点，驱动整个算法 |
| `getNeighbors(T, Collection<T>)` | 第203行 | 遍历所有点，找出距离 ≤ eps 的邻居（不含自身） |
| `expandCluster(...)` | 第165行 | BFS 方式扩展一个簇，直到无法再扩展 |
| `merge(List, List)` | 第215行 | 将第二个列表中不重复的元素追加到第一个列表（用 HashSet 去重） |
| `distance(Clusterable, Clusterable)` | 继承自 `Clusterer` | 委托给构造时注入的 `DistanceMeasure.compute()` |

---

## 四、关键数据结构

| 变量 | 类型 | 含义 |
|------|------|------|
| `clusters` | `List<Cluster<T>>` | 最终输出，每个元素是一个簇（不含噪声点） |
| `visited` | `Map<Clusterable, PointStatus>` | 全局访问状态表；`null`=未访问，`NOISE`=噪声，`PART_OF_CLUSTER`=已归簇 |
| `seeds` | `List<T>`（局部） | `expandCluster` 内的 BFS 扩展队列，动态追加 |

---

## 五、算法流程图（文字版）

```
输入: points (全部待聚类点), eps (邻域半径), minPts (最小点数)
输出: clusters (若干 Cluster<T>，噪声点不出现在任何簇中)

FOR 每个未访问点 p:
    N = getNeighbors(p)          ← O(n) 线性扫描
    IF |N| < minPts:
        标记 p 为 NOISE
    ELSE:
        创建新簇 C
        expandCluster(C, p, N):
            将 p 加入 C，标记 PART_OF_CLUSTER
            seeds = N
            FOR 每个 q in seeds (动态增长):
                IF q 未访问:
                    M = getNeighbors(q)      ← O(n)
                    IF |M| >= minPts:
                        seeds ∪= M           ← merge()
                IF q 不属于任何簇:
                    将 q 加入 C，标记 PART_OF_CLUSTER
        将 C 加入 clusters

RETURN clusters
```

---

## 六、时间复杂度

| 环节 | 复杂度 |
|------|--------|
| 外层遍历 | O(n) |
| 每次 `getNeighbors` | O(n)（线性扫描全集） |
| 最坏情况总体 | **O(n²)**（无空间索引） |

> 注意：此实现未使用 R-tree / KD-tree 等空间索引，每次邻域查询均对全集做线性扫描，数据量大时性能会显著下降。

---

## 七、噪声点的处理

- 噪声点在 `visited` 中被标记为 `PointStatus.NOISE`，但**不会被加入任何 `Cluster<T>`**。
- 最终返回的 `clusters` 列表中只包含真正的聚类结果，噪声点被静默丢弃。
- 若需要获取噪声点，需调用方自行对比输入集合与输出簇中的点。

---

*生成日期：2026-04-16*
