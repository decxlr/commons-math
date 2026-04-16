# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Build and test everything
mvn clean verify

# Build the legacy module (install dependencies first)
mvn install -pl commons-math-core,commons-math-legacy-exception,commons-math-legacy-core
mvn clean test -pl commons-math-legacy

# Run a single test class
mvn -Dtest=KMeansPlusPlusClustererTest test -pl commons-math-legacy

# Run a single test method
mvn -Dtest=KMeansPlusPlusClustererTest#testSinglePoint test -pl commons-math-legacy
```

## Module Structure

The project is a multi-module Maven build (version 4.0-SNAPSHOT, Java 8+):

- **commons-math-core** — self-contained core utilities (no legacy dependencies)
- **commons-math-neuralnet**, **commons-math-transform** — standalone functional modules
- **commons-math-legacy-exception**, **commons-math-legacy-core** — modularized legacy pieces
- **commons-math-legacy** — bulk of legacy code including the `ml` package; depends on the two legacy-* modules above
- **commons-math-docs**, **commons-math-examples** — depend on legacy; not part of regular test builds

## ML Package Architecture (`commons-math-legacy/.../ml/`)

Four sub-packages under `org.apache.commons.math4.legacy.ml`:

### `distance/`
`DistanceMeasure` interface (single `compute(double[], double[])` method) with implementations: `EuclideanDistance`, `ManhattanDistance`, `ChebyshevDistance`, `CanberraDistance`, `EarthMoversDistance`.

### `clustering/`
Key abstractions:
- `Clusterable` — interface for n-dimensional points (`getPoint(): double[]`)
- `Clusterer<T extends Clusterable>` — abstract base; takes a `DistanceMeasure` by constructor injection; delegates distance calls to it
- `Cluster<T>` / `CentroidCluster<T>` — containers for clustered points
- `DoublePoint` — concrete `Clusterable` wrapping `double[]`

Concrete algorithms: `KMeansPlusPlusClusterer`, `ElkanKMeansPlusPlusClusterer`, `MiniBatchKMeansClusterer`, `MultiKMeansPlusPlusClusterer`, `FuzzyKMeansClusterer`, `DBSCANClusterer`.

`KMeansPlusPlusClusterer` has an `EmptyClusterStrategy` enum (`LARGEST_VARIANCE`, `LARGEST_POINTS_NUMBER`, `FARTHEST_POINT`, `ERROR`) that controls what happens when a cluster becomes empty during iteration.

### `clustering/evaluation/`
`ClusterEvaluator` interface with implementations `CalinskiHarabasz` and `SumOfClusterVariances`. Used by `MultiKMeansPlusPlusClusterer` to pick the best run.

### Architectural patterns to follow
- New distance measures: implement `DistanceMeasure`, place in `distance/`
- New clustering algorithms: extend `Clusterer<T>`, accept `DistanceMeasure` via constructor, place in `clustering/`
- New evaluation metrics: implement `ClusterEvaluator`, place in `clustering/evaluation/`
- Custom point types: implement `Clusterable`; `DoublePoint` is the reference implementation

## 注意事项

- 你接下来尽量只查看E:\work\OpenNMS\apache\commons-math\commons-math-legacy\src\main\java\org\apache\commons\math4\legacy\ml下的代码,如果必须要了解其他的代码就去查看
- 我之后的问题基本围绕DBSCAN聚类算法
- 你的所有回答使用中文输出
- 你的最终执行结果或者结论输出到一个结果文档中,markdown格式
- 输出文档目录为org/apache/commons/math4/legacy/ml/analysis(E:\work\OpenNMS\apache\commons-math\commons-math-legacy\src\main\java\org\apache\commons\math4\legacy\ml\analysis),你的所有输出文档都放在这个目录下
- 输出尽可能准确详细