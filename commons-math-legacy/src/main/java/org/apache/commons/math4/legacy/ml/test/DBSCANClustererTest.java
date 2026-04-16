package org.apache.commons.math4.legacy.ml.test;

import org.apache.commons.math4.legacy.ml.clustering.Cluster;
import org.apache.commons.math4.legacy.ml.clustering.DBSCANClusterer;
import org.apache.commons.math4.legacy.ml.clustering.DoublePoint;
import org.apache.commons.math4.legacy.ml.clustering.IncrementalDBSCANClusterer;

import java.util.ArrayList;
import java.util.List;

/**
 * DBSCAN 聚类算法测试示例
 */
public class DBSCANClustererTest {

    public static void main(String[] args) {
        System.out.println("=== DBSCAN 聚类算法测试 ===\n");

        // 1. 创建测试数据点
        List<DoublePoint> points = new ArrayList<>();
        
        // 第一个簇：中心在 (2, 2) 附近的点
        points.add(new DoublePoint(new double[]{2.0, 2.0}));
        points.add(new DoublePoint(new double[]{2.1, 2.1}));
        points.add(new DoublePoint(new double[]{1.9, 2.0}));
        points.add(new DoublePoint(new double[]{2.0, 1.9}));
        points.add(new DoublePoint(new double[]{2.2, 2.2}));
        points.add(new DoublePoint(new double[]{1.8, 1.8}));
        
        // 第二个簇：中心在 (8, 8) 附近的点
        points.add(new DoublePoint(new double[]{8.0, 8.0}));
        points.add(new DoublePoint(new double[]{8.1, 8.1}));
        points.add(new DoublePoint(new double[]{7.9, 8.0}));
        points.add(new DoublePoint(new double[]{8.0, 7.9}));
        points.add(new DoublePoint(new double[]{8.2, 8.2}));
        points.add(new DoublePoint(new double[]{7.8, 7.8}));
        
        // 噪声点：远离两个簇的点
        points.add(new DoublePoint(new double[]{0.0, 0.0}));
        points.add(new DoublePoint(new double[]{10.0, 10.0}));
        points.add(new DoublePoint(new double[]{5.0, 0.0}));

        System.out.println("测试数据点总数: " + points.size());
        printPoints(points);
        System.out.println();

        // 2. 创建 DBSCAN 聚类器
        // eps=1.0: 邻域半径为 1.0
        // minPts=3: 至少需要 3 个点才能形成簇
        double eps = 1.0;
        int minPts = 3;
        IncrementalDBSCANClusterer<DoublePoint> clusterer = new IncrementalDBSCANClusterer<>(eps, minPts);

        System.out.println("DBSCAN 参数:");
        System.out.println("  - eps (邻域半径): " + clusterer.getEps());
        System.out.println("  - minPts (最小点数): " + clusterer.getMinPts());
        System.out.println();

        // 3. 执行聚类
        List<Cluster<DoublePoint>> clusters = clusterer.cluster(points);

        // 4. 输出聚类结果
        System.out.println("=== 聚类结果 ===");
        System.out.println("发现的簇数量: " + clusters.size());
        System.out.println();

        for (int i = 0; i < clusters.size(); i++) {
            Cluster<DoublePoint> cluster = clusters.get(i);
            System.out.println("簇 #" + (i + 1) + ":");
            System.out.println("  包含点数: " + cluster.getPoints().size());
            System.out.println("  数据点:");
            for (DoublePoint point : cluster.getPoints()) {
                double[] coords = point.getPoint();
                System.out.printf("    (%.1f, %.1f)%n", coords[0], coords[1]);
            }
            System.out.println();
        }

        // 5. 计算未被聚类的噪声点
        int clusteredPoints = clusters.stream()
                .mapToInt(c -> c.getPoints().size())
                .sum();
        int noisePoints = points.size() - clusteredPoints;
        
        System.out.println("=== 统计信息 ===");
        System.out.println("总点数: " + points.size());
        System.out.println("已聚类点数: " + clusteredPoints);
        System.out.println("噪声点数: " + noisePoints);
    }

    /**
     * 打印所有数据点
     */
    private static void printPoints(List<DoublePoint> points) {
        for (int i = 0; i < points.size(); i++) {
            double[] coords = points.get(i).getPoint();
            System.out.printf("  点 %2d: (%.1f, %.1f)%n", i + 1, coords[0], coords[1]);
        }
    }
}
/*
=== DBSCAN 聚类算法测试 ===

测试数据点总数: 15
  点  1: (2.0, 2.0)
  点  2: (2.1, 2.1)
  点  3: (1.9, 2.0)
  点  4: (2.0, 1.9)
  点  5: (2.2, 2.2)
  点  6: (1.8, 1.8)
  点  7: (8.0, 8.0)
  点  8: (8.1, 8.1)
  点  9: (7.9, 8.0)
  点 10: (8.0, 7.9)
  点 11: (8.2, 8.2)
  点 12: (7.8, 7.8)
  点 13: (0.0, 0.0)
  点 14: (10.0, 10.0)
  点 15: (5.0, 0.0)

DBSCAN 参数:
  - eps (邻域半径): 1.0
  - minPts (最小点数): 3

=== 聚类结果 ===
发现的簇数量: 2

簇 #1:
  包含点数: 6
  数据点:
    (2.0, 2.0)
    (2.1, 2.1)
    (1.9, 2.0)
    (2.0, 1.9)
    (2.2, 2.2)
    (1.8, 1.8)

簇 #2:
  包含点数: 6
  数据点:
    (8.0, 8.0)
    (8.1, 8.1)
    (7.9, 8.0)
    (8.0, 7.9)
    (8.2, 8.2)
    (7.8, 7.8)

=== 统计信息 ===
总点数: 15
已聚类点数: 12
噪声点数: 3
*/