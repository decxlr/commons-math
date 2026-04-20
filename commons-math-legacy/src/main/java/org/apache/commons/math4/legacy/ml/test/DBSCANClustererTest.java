package org.apache.commons.math4.legacy.ml.test;

import org.apache.commons.math4.legacy.ml.clustering.Cluster;
import org.apache.commons.math4.legacy.ml.clustering.Clusterable;
import org.apache.commons.math4.legacy.ml.clustering.DBSCANClusterer;
import org.apache.commons.math4.legacy.ml.clustering.DoublePoint;
import org.apache.commons.math4.legacy.ml.clustering.IncrementalDBSCANClusterer;
import org.apache.commons.math4.legacy.ml.clustering.IncrementalDBSCANClustererV1;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * DBSCAN 聚类算法测试示例
 */
public class DBSCANClustererTest {

    private static void executionTimeTest() {
        System.out.println("=== 测试 DBSCAN 聚类算法的运行时间 ===");

        // List<DoublePoint> points = generateSmallDataPoints();
        List<DoublePoint> points = generateDataPoints(1000);

        // 创建 DBSCAN 聚类器
        // eps=1.0: 邻域半径为 1.0
        // minPts=3: 至少需要 3 个点才能形成簇
        double eps = 1.0;
        int minPts = 3;
        System.out.println("=== DBSCAN 聚类算法测试 ===");
        System.out.println("DBSCAN 参数:");
        System.out.println("  - eps (邻域半径): " + eps);
        System.out.println("  - minPts (最小点数): " + minPts);
        System.out.println();

        // 增量 DBSCAN
        System.out.println("=== 增量 DBSCAN 聚类算法测试 ===");
        IncrementalDBSCANClusterer<DoublePoint> incClusterer = new IncrementalDBSCANClusterer<>(eps, minPts);
        long incStart = System.currentTimeMillis();
        List<Cluster<DoublePoint>> clusters = incClusterer.cluster(points);
        long incEnd = System.currentTimeMillis();
        System.out.println("运行时间: " + (incEnd - incStart) + " ms");
        // printClusterResult(clusters, points);
        // printStatistics(clusters);

        // 原始 DBSCAN
        System.out.println("=== 原始 DBSCAN 聚类算法测试 ===");
        DBSCANClusterer<DoublePoint> clusterer = new DBSCANClusterer<>(eps, minPts);
        long originStart = System.currentTimeMillis();
        List<Cluster<DoublePoint>> originClusters = clusterer.cluster(points);
        long originEnd = System.currentTimeMillis();
        System.out.println("运行时间: " + (originEnd - originStart) + " ms");
        // printClusterResult(originClusters, points);
        // printStatistics(clusters);
    }

    private static void incrementalDBSCANTest() {
        System.out.println("=== 测试增量 DBSCAN 聚类算法 ===");
        List<DoublePoint> points = generateDataPoints(1000);


        IncrementalDBSCANClusterer<DoublePoint> clusterer = new IncrementalDBSCANClusterer<>(0.5, 5);
        // IncrementalDBSCANClustererV1<DoublePoint> clusterer = createDBSCANClusterer(IncrementalDBSCANClustererV1.class, 0.5, 5);

        long start = System.currentTimeMillis();
        List<Cluster<DoublePoint>> clusters = clusterer.cluster(points);
        long end = System.currentTimeMillis();
        System.out.println("初始化运行时间: " + (end - start) + " ms");
        printClusterResult(clusters);

        for (int i = 0; i < 20; i++) {
            System.out.printf("------ 第 %d 次 ------%n", i + 1);
            clusterer.addPoints(generateDataPoints(500));
            long incStart = System.currentTimeMillis();
            clusters = clusterer.getClusters();
            long incEnd = System.currentTimeMillis();
            System.out.println("增量运行时间: " + (incEnd - incStart) + " ms");
            long printStart = System.currentTimeMillis();
            printClusterResult(clusters);
            long printEnd = System.currentTimeMillis();
            System.out.println("打印运行时间: " + (printEnd - printStart) + " ms");
        }
    }

    private static void incrementalDBSCANClusterV1Test() {
        int radius = 30;
        long runTime = 0;
        System.out.println("=== 测试增量 DBSCAN 聚类算法 ===");
        List<DoublePoint> points = generateClusterPoints(radius, 1000);


        IncrementalDBSCANClustererV1<DoublePoint> clusterer = new IncrementalDBSCANClustererV1<>(0.5, 5);

        long start = System.currentTimeMillis();
        List<Cluster<DoublePoint>> clusters = clusterer.cluster(points);
        long end = System.currentTimeMillis();
        runTime += (end - start);
        System.out.println("初始化运行时间: " + (end - start) + " ms");
        printClusterResult(clusters);

        for (int i = 0; i < 20; i++) {
            System.out.printf("------ 第 %d 次 ------%n", i + 1);

            long addStart = System.currentTimeMillis();
            clusterer.addPoints(generateClusterPoints(radius, 500));
            long addEnd = System.currentTimeMillis();
            runTime += (addEnd - addStart);
            System.out.println("addPoints运行时间: " + (addEnd - addStart) + " ms");

            // long incStart = System.currentTimeMillis();
            clusters = clusterer.getClusters();
            // long incEnd = System.currentTimeMillis();
            // System.out.println("getClusters运行时间: " + (incEnd - incStart) + " ms");
            // printClusterResult(clusters);

            int clusteredPoints = clusters.stream()
                    .mapToInt(c -> c.getPoints().size())
                    .sum();
            System.out.println("发现的簇数量: " + clusters.size() + "，已聚类点数量: " + clusteredPoints);

        }
        System.out.println("总运行时间: " + runTime + " ms");
    }

    public static void main(String[] args) {
        // executionTimeTest();
        // incrementalDBSCANTest();
        incrementalDBSCANClusterV1Test();
    }

    private static <T extends Clusterable> Object createDBSCANClusterer(Class<?> clazz, double eps, int minPts) {
        System.out.println("使用" + clazz.getName());
        if (IncrementalDBSCANClusterer.class.isAssignableFrom(clazz)) {
            return new IncrementalDBSCANClusterer<>(eps, minPts);
        } else if (IncrementalDBSCANClustererV1.class.isAssignableFrom(clazz)) {
            return new IncrementalDBSCANClustererV1<>(eps, minPts);
        } else {
            throw new IllegalArgumentException("不支持的聚类器类型: " + clazz.getName());
        }
    }

    private static List<DoublePoint> generateSmallDataPoints() {
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
        return points;
    }

    private static List<DoublePoint> generateClusterPoints(double radius, int count) {
        Random random = new Random();
        List<DoublePoint> points = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            double x = random.nextDouble() * radius;
            double y = random.nextDouble() * radius;
            points.add(new DoublePoint(new double[]{x, y}));
        }
        return points;
    }

    private static List<DoublePoint> generateDataPoints(int cnt) {
        // 1. 创建测试数据点
        List<DoublePoint> points = new ArrayList<>();

        // 使用随机数生成器
        Random random = new Random();
        int cnt1 = cnt * 30 / 100;
        int cnt2 = cnt * 30 / 100;
        int cnt3 = cnt * 29 / 100;
        int cnt4 = cnt - cnt1 - cnt2 - cnt3;

        // 第一个簇：中心在 (2, 2) 附近
        for (int i = 0; i < cnt1; i++) {
            double x = 2.0 + random.nextGaussian() * 0.3;
            double y = 2.0 + random.nextGaussian() * 0.3;
            points.add(new DoublePoint(new double[]{x, y}));
        }

        // 第二个簇：中心在 (8, 8) 附近
        for (int i = 0; i < cnt2; i++) {
            double x = 8.0 + random.nextGaussian() * 0.3;
            double y = 8.0 + random.nextGaussian() * 0.3;
            points.add(new DoublePoint(new double[]{x, y}));
        }

        // 第三个簇：中心在 (15, 3) 附近
        for (int i = 0; i < cnt3; i++) {
            double x = 15.0 + random.nextGaussian() * 0.4;
            double y = 3.0 + random.nextGaussian() * 0.4;
            points.add(new DoublePoint(new double[]{x, y}));
        }

        // 噪声点：随机分布在空间中
        for (int i = 0; i < cnt4; i++) {
            double x = random.nextDouble() * 20.0;
            double y = random.nextDouble() * 20.0;
            points.add(new DoublePoint(new double[]{x, y}));
        }

        // System.out.println("测试数据点总数: " + points.size());
        // printPoints(points);
        // System.out.println();
        return points;
    }

    private static void printClusterResult(List<Cluster<DoublePoint>> clusters) {
        System.out.println("=== 聚类结果 ===");
        System.out.println("发现的簇数量: " + clusters.size());
        System.out.println();

        for (int i = 0; i < clusters.size(); i++) {
            Cluster<DoublePoint> cluster = clusters.get(i);
            System.out.println("簇 #" + (i + 1) + ":");
            System.out.println("  包含点数: " + cluster.getPoints().size());
            // System.out.println("  数据点:");
            // for (DoublePoint point : cluster.getPoints()) {
            //     double[] coords = point.getPoint();
            //     System.out.printf("    (%.1f, %.1f)%n", coords[0], coords[1]);
            // }
            // System.out.println();
        }

        printStatistics(clusters);
    }

    private static void printStatistics(List<Cluster<DoublePoint>> clusters) {
        int clusteredPoints = clusters.stream()
                .mapToInt(c -> c.getPoints().size())
                .sum();
        System.out.println("=== 统计信息 ===");
        System.out.println("簇数量: " + clusters.size());
        System.out.println("已聚类点数: " + clusteredPoints);
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