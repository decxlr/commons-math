/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.commons.math4.legacy.ml.clustering;

import org.apache.commons.math4.legacy.exception.NotPositiveException;
import org.apache.commons.math4.legacy.exception.NullArgumentException;
import org.apache.commons.math4.legacy.ml.distance.DistanceMeasure;
import org.apache.commons.math4.legacy.ml.distance.EuclideanDistance;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 增量式 DBSCAN（基于密度的带噪声应用空间聚类）算法。
 *
 * <p>与每次调用 {@link #cluster(Collection)} 时都从头重新计算整个聚类的
 * {@link DBSCANClusterer} 不同，此实现维护持久状态，并通过
 * {@link #addPoint(Clusterable)} 和 {@link #removePoint(Clusterable)}
 * 支持高效的增量更新。
 *
 * <p>只有被更改点的 ε-邻域需要重新评估，使得每次单独操作的时间复杂度为 O(k)，
 * 其中 k 是局部邻域的大小，而不是全量扫描的 O(n)。
 *
 * <p><b>插入情况</b>（遵循 Ester 等人 1998 年的增量扩展）：
 * <ul>
 *   <li><b>噪声</b> – 新点没有核心邻居 → 标记为噪声。</li>
 *   <li><b>吸收</b> – 新点有现有的核心邻居，但没有创建新的核心点 → 加入标签值
 *       最高的核心邻居所在的簇。</li>
 *   <li><b>创建</b> – 插入触发了新的核心点，且附近没有现有簇 → 创建新簇。</li>
 *   <li><b>合并</b> – 新核心点桥接了多个现有簇 → 合并为一个簇（最高标签获胜）。</li>
 * </ul>
 *
 * <p><b>删除</b>处理潜在的簇分裂：当删除一个点时，使用沿核心点边的 BFS 遍历
 * 重新评估其先前核心邻居的连通性；断开的子组获得新的簇标签。
 *
 * <p><b>簇标签语义</b>：
 * <ul>
 *   <li>{@link #LABEL_NOISE} ({@code -1}) – 该点被分类为噪声。</li>
 *   <li>任何非负整数 – 该点属于对应的簇。标签单调递增，永不重用。</li>
 * </ul>
 *
 * <p><b>用法</b>：
 * <pre>{@code
 * IncrementalDBSCANClusterer<DoublePoint> clusterer =
 *     new IncrementalDBSCANClusterer<>(2.0, 3);
 *
 * // 增量添加
 * for (DoublePoint p : stream) {
 *     clusterer.addPoint(p);
 * }
 * List<Cluster<DoublePoint>> clusters = clusterer.getClusters();
 *
 * // 删除一个点
 * clusterer.removePoint(somePoint);
 *
 * // 也可用作标准批量聚类器
 * List<Cluster<DoublePoint>> result = clusterer.cluster(allPoints);
 * }</pre>
 *
 * @param <T> 要聚类的点的类型
 * @see DBSCANClusterer
 */
public class IncrementalDBSCANClustererV1<T extends Clusterable> extends Clusterer<T> {

    /** 分配给噪声点的簇标签。 */
    private static final int LABEL_NOISE = -1;

    /**
     * 内部哨兵标签，用于已插入但尚未确定簇分配的点。
     */
    private static final int LABEL_UNCLASSIFIED = -2;

    /** ε-邻域的最大半径。 */
    private final double eps;

    /**
     * 成为核心点所需的 ε 范围内的最小点数（包括点自身）。
     */
    private final int minPts;

    /** 下一个要分配的簇标签；每次创建新簇时递增。 */
    private int nextLabel = 0;

    /**
     * 活跃点，映射到其内部元数据。保留插入顺序以实现确定性迭代。
     */
    private final Map<T, PointNode> nodes = new LinkedHashMap<>();

    // =========================================================================
    // Inner class: PointNode
    // =========================================================================

    /**
     * 内部逐点元数据，在增量更新之间维护。
     *
     * <p>{@link #neighborCount} 计算 ε 距离内的所有点，包括点自身。
     * 当 {@code neighborCount >= minPts} 时，点是<em>核心点</em>。
     *
     * <p>{@link #neighbors} 是实时邻接集（包括 {@code this}）。
     * 保持此集合可避免在每次更新时重新扫描所有点。
     */
    final class PointNode {
        /** 包装的可聚类点。 */
        final T point;

        /**
         * 所有与 {@code point} 距离最多为 ε 的 PointNode（包括 {@code this}）。
         * 在添加或删除点时保持同步。
         */
        final Set<PointNode> neighbors = new HashSet<>();

        /**
         * ε 范围内的点数，包括自身。
         * 不变式：对于没有精确重复的数据集，{@code neighborCount == neighbors.size()}。
         */
        int neighborCount;

        /**
         * 当前簇标签：非负整数表示真实簇，{@link #LABEL_NOISE} 或
         * {@link #LABEL_UNCLASSIFIED}。
         */
        int label;

        PointNode(final T point) {
            this.point = point;
            this.label = LABEL_UNCLASSIFIED;
            this.neighbors.add(this);   // 自身始终是邻居
            this.neighborCount = 1;     // 计数自身
        }

        /** 如果此点至少有 {@code minPts} 个邻居，则返回 {@code true}。 */
        boolean isCore() {
            return neighborCount >= minPts;
        }

        @Override
        public String toString() {
            return "PointNode{label=" + label + ", neighborCount=" + neighborCount + "}";
        }
    }

    // =========================================================================
    // Constructors
    // =========================================================================

    /**
     * 使用欧几里得距离创建新的 IncrementalDBSCANClusterer。
     *
     * @param eps    ε-邻域的最大半径（必须 ≥ 0）
     * @param minPts 形成核心点所需的最小点数（包括自身）（必须 ≥ 0）
     * @throws NotPositiveException 如果 {@code eps < 0.0} 或 {@code minPts < 0}
     */
    public IncrementalDBSCANClustererV1(final double eps, final int minPts) {
        this(eps, minPts, new EuclideanDistance());
    }

    /**
     * 使用自定义距离度量创建新的 IncrementalDBSCANClusterer。
     *
     * @param eps     ε-邻域的最大半径（必须 ≥ 0）
     * @param minPts  形成核心点所需的最小点数（包括自身）（必须 ≥ 0）
     * @param measure 要使用的距离度量
     * @throws NotPositiveException 如果 {@code eps < 0.0} 或 {@code minPts < 0}
     */
    public IncrementalDBSCANClustererV1(final double eps, final int minPts,
                                        final DistanceMeasure measure) {
        super(measure);
        if (eps < 0.0d) {
            throw new NotPositiveException(eps);
        }
        if (minPts < 0) {
            throw new NotPositiveException(minPts);
        }
        this.eps = eps;
        this.minPts = minPts;
    }

    // =========================================================================
    // Public API
    // =========================================================================

    /**
     * 返回 ε-邻域的最大半径。
     *
     * @return ε 值
     */
    public double getEps() {
        return eps;
    }

    /**
     * 返回形成核心点所需的最小点数。
     *
     * @return minPts 值
     */
    public int getMinPts() {
        return minPts;
    }

    /**
     * 执行批量 DBSCAN 聚类，满足 {@link Clusterer} 契约。
     *
     * <p>清除所有现有的增量状态，然后按迭代顺序插入每个点。这等效于先调用
     * {@link #reset()}，然后为每个点调用 {@link #addPoint(Clusterable)}。
     *
     * @param points 要聚类的点（不能为 {@code null}）
     * @return 非噪声簇的列表
     */
    @Override
    public List<Cluster<T>> cluster(final Collection<T> points) {
        NullArgumentException.check(points);
        reset();
        for (final T point : points) {
            addPoint(point);
        }
        return getClusters();
    }

    /**
     * 清除所有增量状态。等效于使用相同参数构造一个新实例。
     */
    public void reset() {
        nodes.clear();
        nextLabel = 0;
    }

    /**
     * 增量插入一个点并更新聚类。
     *
     * <p>如果已经存在与 {@code point} {@code equal} 的点，则此调用为空操作。
     *
     * @param point 要插入的点（不能为 {@code null}）
     */
    public void addPoint(final T point) {
        NullArgumentException.check(point);
        if (nodes.containsKey(point)) {
            return;
        }
        final PointNode newNode = new PointNode(point);
        nodes.put(point, newNode);
        linkNeighbors(newNode);
        processInsertion(newNode);
    }

    public void addPoints(final Collection<T> points) {
        NullArgumentException.check(points);
        for (T point : points) {
            addPoint(point);
        }
    }

    /**
     * 增量删除一个点并更新聚类。
     *
     * <p>如果点不存在，则此调用为空操作。
     *
     * @param point 要删除的点（不能为 {@code null}）
     */
    public void removePoint(final T point) {
        NullArgumentException.check(point);
        final PointNode toDelete = nodes.get(point);
        if (toDelete == null) {
            return;
        }
        processDeletion(toDelete);
        // 在簇记账完成后物理取消节点的链接。
        for (final PointNode neighbor : toDelete.neighbors) {
            if (neighbor != toDelete) {
                neighbor.neighbors.remove(toDelete);
                neighbor.neighborCount--;
            }
        }
        nodes.remove(point);
    }

    /**
     * 返回当前聚类的快照。排除噪声点。
     *
     * @return 簇列表（可能为空）
     */
    public List<Cluster<T>> getClusters() {
        final Map<Integer, Cluster<T>> clusterMap = new LinkedHashMap<>();
        for (final PointNode node : nodes.values()) {
            if (node.label >= 0) {
                clusterMap.computeIfAbsent(node.label, id -> new Cluster<>())
                          .addPoint(node.point);
            }
        }
        return new ArrayList<>(clusterMap.values());
    }

    /**
     * 返回点的当前簇标签。
     *
     * @param point 要查询的点
     * @return 簇标签，如果点是噪声或不存在于此聚类器中，则返回 {@link #LABEL_NOISE}
     */
    public int getLabel(final T point) {
        final PointNode node = nodes.get(point);
        return node != null ? node.label : LABEL_NOISE;
    }

    /**
     * 返回此聚类器当前管理的点数。
     *
     * @return 活跃点数
     */
    public int size() {
        return nodes.size();
    }

    // =========================================================================
    // Neighbor linking
    // =========================================================================

    /**
     * 扫描所有现有节点以查找 {@code newNode} 的 ε-邻居，并创建双向链接，
     * 更新两侧的 {@link PointNode#neighborCount}。
     */
    private void linkNeighbors(final PointNode newNode) {
        for (final PointNode existing : nodes.values()) {
            if (existing == newNode) {
                continue;
            }
            if (distance(existing.point, newNode.point) <= eps) {
                newNode.neighbors.add(existing);
                newNode.neighborCount++;
                existing.neighbors.add(newNode);
                existing.neighborCount++;
            }
        }
    }

    // =========================================================================
    // Insertion logic
    // =========================================================================

    /**
     * 确定 {@code inserted} 的簇分配并更新所有受影响的点。
     *
     * <h3>算法概述</h3>
     * <ol>
     *   <li>将 {@code inserted} 的每个邻居分类为<em>新核心</em>（因为此插入刚刚
     *       达到 {@code minPts}）或<em>旧核心</em>（在此之前已经是核心）。</li>
     *   <li>如果没有新核心：噪声或吸收。</li>
     *   <li>否则：收集<em>更新种子</em>（每个新核心的所有核心邻居），找到诱导子图
     *       的连通分量，并适当创建/合并簇。</li>
     *   <li>将标签从新核心传播到其非核心（边界）邻居。</li>
     * </ol>
     */
    private void processInsertion(final PointNode inserted) {
        final Set<PointNode> newCores = new HashSet<>();
        final Set<PointNode> oldCores = new HashSet<>();

        for (final PointNode neighbor : inserted.neighbors) {
            if (neighbor == inserted) {
                continue;
            }
            if (neighbor.neighborCount == minPts) {
                // 由于此插入刚刚达到核心阈值。
                newCores.add(neighbor);
            } else if (neighbor.neighborCount > minPts) {
                // 在此插入之前已经是核心点。
                oldCores.add(neighbor);
            }
        }

        // 插入的节点本身之前不可能是核心；如果现在是核心，则始终分类为新核心。
        if (inserted.isCore()) {
            oldCores.remove(inserted);
            newCores.add(inserted);
        }

        if (newCores.isEmpty()) {
            // ---- 噪声或吸收 ----
            if (!oldCores.isEmpty()) {
                // 吸收：边界点加入最近创建的簇。
                int maxLabel = LABEL_NOISE;
                for (final PointNode oc : oldCores) {
                    if (oc.label > maxLabel) {
                        maxLabel = oc.label;
                    }
                }
                inserted.label = maxLabel;
            } else {
                // 噪声：完全没有核心邻居。
                inserted.label = LABEL_NOISE;
            }
            return;
        }

        // ---- 创建或合并 ----

        // 更新种子 = 每个新核心的所有核心邻居（包括自身）。
        final Set<PointNode> updateSeeds = new HashSet<>();
        for (final PointNode nc : newCores) {
            for (final PointNode neighbor : nc.neighbors) {
                if (neighbor.isCore()) {
                    updateSeeds.add(neighbor);
                }
            }
        }

        // 找到由 updateSeeds 诱导的子图的连通分量。
        // 如果两个种子是相互的 ε-邻居，则它们直接连接。
        final List<Set<PointNode>> components = connectedComponentsOf(updateSeeds);

        for (final Set<PointNode> component : components) {
            // 收集此分量中的非特殊（真实）簇标签。
            final Set<Integer> effectiveLabels = new HashSet<>();
            for (final PointNode n : component) {
                if (n.label >= 0) {
                    effectiveLabels.add(n.label);
                }
            }

            if (effectiveLabels.isEmpty()) {
                // 创建：不涉及现有簇；创建一个新簇。
                final int newLabel = nextLabel++;
                setLabels(component, newLabel);
            } else {
                // 吸收/合并：将所有参与的簇统一在最高现有标签下。
                final int maxLabel = Collections.max(effectiveLabels);
                for (final int oldLabel : effectiveLabels) {
                    if (oldLabel != maxLabel) {
                        changeLabels(oldLabel, maxLabel);
                    }
                }
                setLabels(component, maxLabel);
            }
        }

        // 将每个新核心的标签传播到其未分类/噪声邻居。
        propagateAroundNewCores(newCores);
    }

    // =========================================================================
    // Deletion logic
    // =========================================================================

    /**
     * 在即将删除 {@code toDelete} 时更新簇分配。
     *
     * <h3>算法概述</h3>
     * <ol>
     *   <li>为 {@code toDelete} 的所有邻居递减 {@link PointNode#neighborCount}
     *       （模拟删除）。</li>
     *   <li>识别<em>前核心点</em>：刚刚失去核心状态的节点。</li>
     *   <li>收集<em>更新种子</em>（前核心点的核心邻居，不包括 {@code toDelete}）
     *       和前核心点的非核心邻居。</li>
     *   <li>对于每个簇组的更新种子，运行 BFS 分裂检测并为任何断开的子组件分配
     *       新标签。</li>
     *   <li>重新评估非核心（边界）邻居：每个要么继承剩余核心邻居的标签，要么成为
     *       噪声。</li>
     * </ol>
     *
     * <p>节点的物理移除在此方法返回后发生在 {@link #removePoint(Clusterable)} 中。
     */
    private void processDeletion(final PointNode toDelete) {
        // 步骤 1：递减邻居计数以模拟删除。
        for (final PointNode neighbor : toDelete.neighbors) {
            neighbor.neighborCount--;
        }

        // 步骤 2：查找前核心点 — 刚刚失去核心状态的节点。
        // 递减后：neighborCount == minPts - 1  ⟺  正好处于阈值。
        // 特殊情况：toDelete 本身 — 如果（递减计数 + 1）>= minPts，则是核心点。
        final Set<PointNode> exCores = new HashSet<>();
        for (final PointNode neighbor : toDelete.neighbors) {
            if (neighbor == toDelete) {
                continue;
            }
            if (neighbor.neighborCount == minPts - 1) {
                exCores.add(neighbor);
            }
        }
        if (toDelete.neighborCount + 1 >= minPts) {
            // toDelete 是核心点；将其添加到 exCores 以进行连通性分析。
            exCores.add(toDelete);
        }

        // 步骤 3：从前核心点收集更新种子和非核心邻居。
        final Set<PointNode> updateSeeds = new HashSet<>();
        final Set<PointNode> nonCoreNeighbors = new HashSet<>();

        for (final PointNode exCore : exCores) {
            for (final PointNode neighbor : exCore.neighbors) {
                if (neighbor == toDelete) {
                    continue;
                }
                if (neighbor.isCore()) {
                    updateSeeds.add(neighbor);
                } else {
                    nonCoreNeighbors.add(neighbor);
                }
            }
        }
        updateSeeds.remove(toDelete);
        nonCoreNeighbors.remove(toDelete);

        // 步骤 4：分裂检测 — 在每个更新种子的簇组内，检查删除 toDelete 是否会
        // 断开核心点图的连接。
        if (!updateSeeds.isEmpty()) {
            final Map<Integer, List<PointNode>> seedsByCluster = new HashMap<>();
            for (final PointNode seed : updateSeeds) {
                seedsByCluster.computeIfAbsent(seed.label, k -> new ArrayList<>()).add(seed);
            }

            for (final List<PointNode> clusterSeeds : seedsByCluster.values()) {
                final List<Set<PointNode>> splitComponents =
                        findSplitComponents(clusterSeeds, toDelete);
                for (final Set<PointNode> component : splitComponents) {
                    final int newLabel = nextLabel++;
                    setLabels(component, newLabel);
                }
            }
        }

        // 步骤 5：重新评估非核心（边界）邻居。
        for (final PointNode border : nonCoreNeighbors) {
            if (border == toDelete) {
                continue;
            }
            int maxLabel = LABEL_NOISE;
            for (final PointNode neighbor : border.neighbors) {
                if (neighbor != toDelete && neighbor.isCore() && neighbor.label > maxLabel) {
                    maxLabel = neighbor.label;
                }
            }
            border.label = maxLabel;
        }
    }

    // =========================================================================
    // BFS split detection
    // =========================================================================

    /**
     * 确定删除 {@code excluded} 后 {@code seeds} 的哪些子组件会断开连接。
     *
     * <p>所有种子属于同一个簇。BFS 遍历仅沿核心点边进行（非核心点不是遍历路径点，
     * 符合密度连通性定义）。最大的组件保留其原始标签；其余的在这里返回以重新标记。
     *
     * @param seeds    核心点更新种子，都在同一个簇中
     * @param excluded 正在删除的节点（在 BFS 期间忽略）
     * @return 要分裂出去的组件（每个都将接收新标签）；可能为空
     */
    private List<Set<PointNode>> findSplitComponents(final List<PointNode> seeds,
                                                      final PointNode excluded) {
        if (seeds.size() <= 1) {
            return Collections.emptyList();
        }
        // 快速路径：如果每对种子都是相互邻居，则无论删除如何，它们都形成单个组件。
        if (allMutualNeighbors(seeds)) {
            return Collections.emptyList();
        }

        // 从所有种子同时进行 BFS。跟踪哪个"原始种子"首先到达每个节点，
        // 并在两个种子相遇时合并组件条目。
        final Map<PointNode, Integer> assignment = new HashMap<>();  // 节点 → 种子索引
        final Map<Integer, Set<PointNode>> components = new HashMap<>();

        for (int i = 0; i < seeds.size(); i++) {
            assignment.put(seeds.get(i), i);
            final Set<PointNode> comp = new HashSet<>();
            comp.add(seeds.get(i));
            components.put(i, comp);
        }

        final Set<PointNode> visited = new HashSet<>(seeds);
        final Deque<PointNode> queue = new ArrayDeque<>(seeds);

        while (!queue.isEmpty()) {
            final PointNode current = queue.poll();
            if (!current.isCore() || current == excluded) {
                continue;
            }
            final int currentIdx = assignment.get(current);

            for (final PointNode neighbor : current.neighbors) {
                if (neighbor == excluded || !neighbor.isCore()) {
                    continue;
                }
                if (!visited.contains(neighbor)) {
                    // 树边：邻居是新发现的。
                    visited.add(neighbor);
                    assignment.put(neighbor, currentIdx);
                    components.get(currentIdx).add(neighbor);
                    queue.add(neighbor);
                } else {
                    // 交叉/后边：两个组件可能刚刚连接。
                    final int neighborIdx = assignment.get(neighbor);
                    if (neighborIdx != currentIdx) {
                        mergeComponentMaps(components, assignment, neighborIdx, currentIdx);
                    }
                }
            }
        }

        if (components.size() <= 1) {
            return Collections.emptyList();
        }

        // 按大小降序排序；保留最大的，返回其余的进行重新标记。
        final List<Set<PointNode>> allComps = new ArrayList<>(components.values());
        allComps.sort((a, b) -> b.size() - a.size());
        allComps.remove(0);  // 保留最大的（保留原始标签）
        return allComps;
    }

    /**
     * 将 {@code fromIdx} 处的组件合并到 {@code intoIdx} 处的组件中，
     * 更新 {@code components} 和 {@code assignment} 映射。
     */
    private void mergeComponentMaps(final Map<Integer, Set<PointNode>> components,
                                     final Map<PointNode, Integer> assignment,
                                     final int fromIdx, final int intoIdx) {
        final Set<PointNode> from = components.remove(fromIdx);
        if (from == null) {
            return;
        }
        final Set<PointNode> into = components.get(intoIdx);
        for (final PointNode node : from) {
            assignment.put(node, intoIdx);
        }
        into.addAll(from);
    }

    /**
     * 如果列表中的每对节点都是相互的 ε-邻居（即，每个都出现在另一个的
     * {@link PointNode#neighbors} 中），则返回 {@code true}。
     */
    private boolean allMutualNeighbors(final List<PointNode> nodeList) {
        for (int i = 0; i < nodeList.size(); i++) {
            for (int j = i + 1; j < nodeList.size(); j++) {
                if (!nodeList.get(i).neighbors.contains(nodeList.get(j))) {
                    return false;
                }
            }
        }
        return true;
    }

    // =========================================================================
    // Connected components (subgraph of update seeds, used during insertion)
    // =========================================================================

    /**
     * 找到由 {@code objects} 诱导的子图的连通分量。
     * 当且仅当两个节点通过相互 ε-邻居的路径连接时，它们才在同一分量中，
     * 所有这些节点都必须是 {@code objects} 的成员。
     *
     * @param objects 要划分为分量的节点集
     * @return 非空分量列表（{@code objects} 的分区）
     */
    private List<Set<PointNode>> connectedComponentsOf(final Set<PointNode> objects) {
        if (objects.size() == 1) {
            return Collections.singletonList(new HashSet<>(objects));
        }

        final Set<PointNode> unvisited = new HashSet<>(objects);
        final List<Set<PointNode>> result = new ArrayList<>();

        while (!unvisited.isEmpty()) {
            final PointNode start = unvisited.iterator().next();
            final Set<PointNode> component = new HashSet<>();
            final Deque<PointNode> queue = new ArrayDeque<>();
            queue.add(start);
            unvisited.remove(start);

            while (!queue.isEmpty()) {
                final PointNode current = queue.poll();
                component.add(current);
                for (final PointNode neighbor : current.neighbors) {
                    if (unvisited.contains(neighbor)) {
                        unvisited.remove(neighbor);
                        queue.add(neighbor);
                    }
                }
            }
            result.add(component);
        }
        return result;
    }

    // =========================================================================
    // Label helpers
    // =========================================================================

    /** 在 {@code nodeSet} 中的每个节点上设置 {@code label}。 */
    private void setLabels(final Set<PointNode> nodeSet, final int label) {
        for (final PointNode node : nodeSet) {
            node.label = label;
        }
    }

    /** 将所有出现的 {@code oldLabel} 重命名为 {@code newLabel}。 */
    private void changeLabels(final int oldLabel, final int newLabel) {
        for (final PointNode node : nodes.values()) {
            if (node.label == oldLabel) {
                node.label = newLabel;
            }
        }
    }

    /**
     * 对于 {@code newCores} 中的每个节点，将该核心的簇标签分配给任何标签当前为
     * {@link #LABEL_NOISE} 或 {@link #LABEL_UNCLASSIFIED} 的邻居节点
     * （即边界点和刚插入的节点）。
     */
    private void propagateAroundNewCores(final Set<PointNode> newCores) {
        for (final PointNode core : newCores) {
            final int label = core.label;
            for (final PointNode neighbor : core.neighbors) {
                if (neighbor.label < 0) {
                    neighbor.label = label;
                }
            }
        }
    }
}
