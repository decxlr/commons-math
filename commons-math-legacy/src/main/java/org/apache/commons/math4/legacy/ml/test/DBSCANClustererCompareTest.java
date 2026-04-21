package org.apache.commons.math4.legacy.ml.test;

import org.apache.commons.math4.legacy.ml.clustering.Cluster;
import org.apache.commons.math4.legacy.ml.clustering.Clusterable;
import org.apache.commons.math4.legacy.ml.clustering.DBSCANClusterer;
import org.apache.commons.math4.legacy.ml.clustering.DoublePoint;
import org.apache.commons.math4.legacy.ml.clustering.IncrementalDBSCANClustererV1;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * DBSCAN 聚类算法测试示例
 */
public class DBSCANClustererCompareTest {

    private static final String FILE_PATH = "E:\\work\\OpenNMS\\apache\\commons-math\\commons-math-legacy\\src\\main\\java\\org\\apache\\commons\\math4\\legacy\\ml\\test";

    private static final int COMPARE_COUNT = 20;
    private static final int COMPARE_RADIUS = 5;
    private static final int COMPARE_SEED = 100;

    private static final double COMPARE_EPS = 1;
    private static final int COMPARE_MIN_PTS = 1;

    private static void simpleIncrementalDBSCANClusterV1Test() throws IOException {
        long runTime = 0;
        System.out.printf("%n=== 测试增量 DBSCAN 聚类算法 ===%n");

        IncrementalDBSCANClustererV1<DoublePoint> clusterer = new IncrementalDBSCANClustererV1<>(COMPARE_EPS, COMPARE_MIN_PTS+1);

        int seed = COMPARE_SEED;
        int radius = COMPARE_RADIUS;
        int count = COMPARE_COUNT;

        int part1 = count * 8 / 10;

        List<DoublePoint> doublePointList = generateClusterPoints(radius, count, seed);

        // exportPointsToFile(doublePointList, FILE_PATH + "/inc_points.txt");

        // 第一轮400
        System.out.println("------ 第 1 次 ------");
        List<DoublePoint> doublePointList1 = doublePointList.subList(0, part1);
        long addStart1 = System.currentTimeMillis();
        clusterer.addPoints(doublePointList1);
        long addEnd1 = System.currentTimeMillis();
        runTime += (addEnd1 - addStart1);
        System.out.println("addPoints运行时间: " + (addEnd1 - addStart1) + " ms");
        List<Cluster<DoublePoint>> clusters1 = clusterer.getClusters();
        int clusteredPoints1 = clusters1.stream()
                .mapToInt(c -> c.getPoints().size())
                .sum();
        System.out.println("发现的簇数量: " + clusters1.size() + "，已聚类点数量: " + clusteredPoints1);

        // 第二轮100
        System.out.println("------ 第 2 次 ------");
        List<DoublePoint> doublePointList2 = doublePointList.subList(part1, count);
        long addStart2 = System.currentTimeMillis();
        clusterer.addPoints(doublePointList2);
        long addEnd2 = System.currentTimeMillis();
        runTime += (addEnd2 - addStart2);
        System.out.println("addPoints运行时间: " + (addEnd2 - addStart2) + " ms");
        List<Cluster<DoublePoint>> clusters2 = clusterer.getClusters();
        int clusteredPoints2 = clusters2.stream()
                .mapToInt(c -> c.getPoints().size())
                .sum();
        System.out.println("发现的簇数量: " + clusters2.size() + "，已聚类点数量: " + clusteredPoints2);


        List<Cluster<DoublePoint>> allClusters = clusterer.getClusters();
        int allPoints = allClusters.stream()
                .mapToInt(c -> c.getPoints().size())
                .sum();
        System.out.println("已聚类簇总数: " + allClusters.size() + "，已聚类点数量: " + allPoints);
        System.out.println("总运行时间: " + runTime + " ms");

        printClusterResult(allClusters);
    }

    private static void originalDBSCANTest() throws IOException {
        System.out.printf("%n=== 测试原始 DBSCAN 聚类算法 ===%n");

        long runTime = 0;
        DBSCANClusterer<DoublePoint> clusterer = new DBSCANClusterer<>(COMPARE_EPS, COMPARE_MIN_PTS);

        int seed = COMPARE_SEED;
        int radius = COMPARE_RADIUS;
        int count = COMPARE_COUNT;

        List<DoublePoint> doublePointList = generateClusterPoints(radius, count, seed);

        // exportPointsToFile(doublePointList, FILE_PATH + "/origin_points.txt");
        exportPointsToFile(doublePointList, FILE_PATH + "/points.txt");

        long start = System.currentTimeMillis();
        List<Cluster<DoublePoint>> clusters = clusterer.cluster(doublePointList);
        long end = System.currentTimeMillis();
        runTime += (end - start);

        int clusteredPoints = clusters.stream()
                .mapToInt(c -> c.getPoints().size())
                .sum();
        System.out.println("已聚类簇总数: " + clusters.size() + "，已聚类点数量: " + clusteredPoints);
        System.out.println("总运行时间: " + runTime + " ms, cluster 时间: " + (end - start) + " ms");

        printClusterResult(clusters);
    }

    public static void main(String[] args) throws IOException {
        originalDBSCANTest();
        simpleIncrementalDBSCANClusterV1Test();
    }

    /**
     * 将数据点导出到CSV文件
     *
     * @param points   要导出的数据点列表
     * @param filePath 输出文件路径
     * @throws IOException 文件写入异常
     */
    private static void exportPointsToFile(List<DoublePoint> points, String filePath) throws IOException {
        Path path = Paths.get(filePath);
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(path.toFile()))) {
            // 写入CSV头部
            writer.write("index,x,y");
            writer.newLine();

            // 写入每个点的坐标
            for (int i = 0; i < points.size(); i++) {
                double[] coords = points.get(i).getPoint();
                writer.write(String.format("%d,%.6f,%.6f", i, coords[0], coords[1]));
                writer.newLine();
            }
        }
        System.out.println("数据已导出到: " + filePath);
        System.out.println("导出点数: " + points.size());
    }


    private static <T extends Clusterable> Object createDBSCANClusterer(Class<?> clazz, double eps, int minPts) {
        System.out.println("使用" + clazz.getName());
        if (DBSCANClusterer.class.isAssignableFrom(clazz)) {
            return new DBSCANClusterer<>(eps, minPts);
        } else if (IncrementalDBSCANClustererV1.class.isAssignableFrom(clazz)) {
            return new IncrementalDBSCANClustererV1<>(eps, minPts);
        } else {
            throw new IllegalArgumentException("不支持的聚类器类型: " + clazz.getName());
        }
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

    private static List<DoublePoint> generateClusterPoints(double radius, int count, int seed) {
        Random random = new Random(seed);
        List<DoublePoint> points = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            double x = random.nextDouble() * radius;
            double y = random.nextDouble() * radius;
            points.add(new DoublePoint(new double[]{x, y}));
        }
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
            System.out.println("  数据点:");
            for (DoublePoint point : cluster.getPoints()) {
                double[] coords = point.getPoint();
                System.out.printf("    (%.1f, %.1f)%n", coords[0], coords[1]);
            }
            System.out.println();
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