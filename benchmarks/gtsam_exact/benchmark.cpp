// External GTSAM reference only; ropt does not link against this executable.
#include <gtsam/config.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/dataset.h>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef GTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR
#error "Build GTSAM with GTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR=ON"
#endif
#ifndef NDEBUG
#error "Build this timing reference in Release mode"
#endif

using namespace gtsam;
namespace fs = std::filesystem;
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

double derivative_probe() {
  double largest = 0.0;
  for (double theta : {0.0, M_PI / 2, -2.1}) {
    const Pose2 a, b(3, 1, theta + .4), z(1, 0, theta);
    BetweenFactor<Pose2> f(0, 1, z, noiseModel::Unit::Create(3));
    Matrix h1, h2;
    const Vector r = f.evaluateError(a, b, h1, h2);
    Vector3 expected(2 * std::cos(theta) + std::sin(theta),
                     -2 * std::sin(theta) + std::cos(theta), .4);
    require((r - expected).norm() < 1e-12, "residual-coordinate probe failed");
    for (int column = 0; column < 6; ++column) {
      Vector3 delta = Vector3::Zero();
      delta(column % 3) = 1e-6;
      const Vector plus = column < 3 ? f.evaluateError(a.retract(delta), b)
                                    : f.evaluateError(a, b.retract(delta));
      const Vector minus = column < 3 ? f.evaluateError(a.retract(-delta), b)
                                     : f.evaluateError(a, b.retract(-delta));
      const Vector analytical = column < 3 ? Vector(h1.col(column))
                                          : Vector(h2.col(column - 3));
      largest = std::max(largest, (analytical - (plus - minus) / 2e-6).cwiseAbs().maxCoeff());
    }
  }
  require(std::isfinite(largest) && largest < 1e-7, "exact Jacobian probe failed");
  return largest;
}

// The public files are checksum-validated by the Python runner. Reject unsupported
// records here too, and reproduce ropt's explicit/component anchor policy.
std::set<Key> anchors(const fs::path& input) {
  std::ifstream file(input);
  require(file.good(), "cannot open input");
  std::map<Key, std::set<Key>> adjacent;
  std::set<Key> fixed, visited;
  for (std::string line; std::getline(file, line);) {
    line = line.substr(0, line.find('#'));
    std::istringstream fields(line);
    std::string tag;
    if (!(fields >> tag)) continue;
    Key a;
    require(bool(fields >> a), "invalid record ID");
    if (tag == "VERTEX2" || tag == "VERTEX_SE2") {
      require(adjacent.emplace(a, std::set<Key>{}).second, "duplicate vertex");
    } else if (tag == "FIX") {
      fixed.insert(a);
    } else if (tag != "EDGE2" && tag != "EDGE_SE2") {
      throw std::runtime_error("unsupported record: " + tag);
    }
  }
  file.clear();
  file.seekg(0);
  for (std::string line; std::getline(file, line);) {
    std::istringstream fields(line.substr(0, line.find('#')));
    std::string tag;
    Key a, b;
    if (!(fields >> tag) || (tag != "EDGE2" && tag != "EDGE_SE2")) continue;
    require(bool(fields >> a >> b) && a != b && adjacent.count(a) && adjacent.count(b),
            "invalid edge endpoints");
    adjacent.at(a).insert(b);
    adjacent.at(b).insert(a);
  }
  for (Key key : fixed) require(adjacent.count(key), "FIX references unknown pose");
  for (const auto& entry : adjacent) {
    require(!entry.second.empty(), "isolated pose");
    if (visited.count(entry.first)) continue;
    std::vector<Key> stack{entry.first};
    bool anchored = false;
    while (!stack.empty()) {
      Key key = stack.back();
      stack.pop_back();
      if (!visited.insert(key).second) continue;
      anchored = anchored || fixed.count(key);
      const auto& neighbors = adjacent.at(key);
      stack.insert(stack.end(), neighbors.begin(), neighbors.end());
    }
    if (!anchored) fixed.insert(entry.first);
  }
  require(!adjacent.empty(), "empty graph");
  return fixed;
}

int main(int argc, char** argv) {
  try {
    require(argc >= 5 && argc <= 7, "usage: gtsam_exact input.g2o NEW_DIRECTORY repeats warmup [qr|cholesky [max-iterations]]");
    const std::string linear_solver = argc >= 6 ? argv[5] : "qr";
    require(linear_solver == "qr" || linear_solver == "cholesky", "unknown linear solver");
    const fs::path input(argv[1]), output(argv[2]);
    require(argv[3][0] != '-' && argv[4][0] != '-', "negative repetition count");
    const size_t repeats = std::stoull(argv[3]), warmup = std::stoull(argv[4]);
    require(repeats > 0 && warmup <= std::numeric_limits<size_t>::max() - repeats,
            "invalid repetition count");
    const double discrepancy = derivative_probe();
    const auto fixed = anchors(input);
    auto loaded = readG2o(input.string(), false);
    auto& graph = *loaded.first;
    const Values initial = *loaded.second;
    for (Key key : fixed) graph.emplace_shared<NonlinearEquality<Pose2>>(key, initial.at<Pose2>(key));
    LevenbergMarquardtParams params;
    params.maxIterations = argc == 7 ? std::stoull(argv[6]) : 100;
    require(params.maxIterations > 0 && (argc < 7 || argv[6][0] != '-'), "invalid iteration limit");
    params.lambdaInitial = 1e-3;
    params.absoluteErrorTol = 1e-12;
    params.relativeErrorTol = 1e-12;
    params.errorTol = 0.0;
    params.setLinearSolverType(linear_solver == "qr" ? "MULTIFRONTAL_QR" : "MULTIFRONTAL_CHOLESKY");
    require(fs::create_directory(output), "output must be a new directory");
    std::ofstream options(output / "options.txt");
    auto* stream = std::cout.rdbuf(options.rdbuf());
    params.print();
    std::cout.rdbuf(stream);
    options << "GTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR=ON\n"
            << "GTSAM_VERSION=" << GTSAM_VERSION_STRING << "\n";
    std::ofstream diagnostic(output / "diagnostics.json");
    diagnostic << std::setprecision(17) << "{\"jacobian_max_difference\":" << discrepancy
               << ",\"version\":\"" << GTSAM_VERSION_STRING
               << "\",\"compiler\":\"" << __VERSION__ << "\",\"exact_between\":true,\"linear_solver\":\""
               << linear_solver << "\"}\n";
    std::ofstream samples(output / "samples.csv");
    samples << "sample,elapsed_ns,initial_cost,final_cost,optimizer_iterations,final_lambda\n"
            << std::setprecision(17);
    const double initial_cost = graph.error(initial);
    for (size_t round = 0; round < warmup + repeats; ++round) {
      const Values state(initial);  // Clone outside the timer, like ropt's initial vector.
      const auto start = std::chrono::steady_clock::now();
      LevenbergMarquardtOptimizer optimizer(graph, state, params);
      const Values result = optimizer.optimize();
      const auto stop = std::chrono::steady_clock::now();
      const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
      const double cost = graph.error(result);
      require(std::isfinite(cost) && cost <= initial_cost, "invalid final cost");
      for (Key key : fixed) {
        const auto& a = initial.at<Pose2>(key);
        const auto& b = result.at<Pose2>(key);
        require(a.x() == b.x() && a.y() == b.y() && a.theta() == b.theta(), "anchor moved");
      }
      if (round < warmup) continue;
      samples << round - warmup << ',' << ns << ',' << initial_cost << ',' << cost << ','
              << optimizer.iterations() << ',' << optimizer.lambda() << '\n';
      if (round == warmup) {
        std::ofstream poses(output / "poses.csv");
        poses << "id,x,y,angle\n" << std::setprecision(17);
        for (const auto& entry : result.extract<Pose2>()) {
          const auto& p = entry.second;
          poses << entry.first << ',' << p.x() << ',' << p.y() << ',' << p.theta() << '\n';
        }
        require(poses.good(), "pose output failed");
      }
    }
    require(samples.good() && diagnostic.good() && options.good(), "output failed");
    std::cout << "Saved " << repeats << " exact-Jacobian native GTSAM solves to " << output << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
