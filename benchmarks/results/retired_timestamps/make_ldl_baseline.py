"""Build an isolated benchmark-only snapshot with the LDL reuse/analytic optimization reversed.

Never modifies the working checkout or creates a Git worktree. The saved patch
restores map-based LDL and removes the explicit analytic SE2 path.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new directory outside checkout")
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--before-retired-timestamps", action="store_true",
                       help="restore fixed-lag reports before retired timestamp snapshots")
    stage.add_argument("--before-covariance-preflight", action="store_true",
                       help="restore covariance query validation after nonlinear optimization")
    stage.add_argument("--before-dynamic-lag", action="store_true",
                       help="restore fixed-lag smoothing before runtime duration changes")
    stage.add_argument("--before-robust-covariance-model", action="store_true",
                       help="restore fixed-lag covariance before loss-model consistency checks")
    stage.add_argument("--before-covariance-snapshot", action="store_true",
                       help="restore fixed-lag updates before pre-expiry covariance queries")
    stage.add_argument("--before-expiry-estimates", action="store_true",
                       help="restore fixed-lag reports before expired-estimate snapshots")
    stage.add_argument("--before-tree-refinement", action="store_true",
                       help="restore separate global QR and editable-tree refresh factorization")
    stage.add_argument("--before-tree-gradient", action="store_true",
                       help="restore Gaussian trees before frozen-factor gradient solves")
    stage.add_argument("--before-cache-refresh", action="store_true",
                       help="restore copying factor cache references before scheduled refresh")
    stage.add_argument("--before-factor-capture", action="store_true",
                       help="restore row-iterator growth in Gaussian factor capture")
    stage.add_argument("--before-inline-ad-inputs", action="store_true",
                       help="restore heap wrappers for all AD callback input arities")
    stage.add_argument("--before-block-jacobian", action="store_true",
                       help="restore flat tangent Jacobian/seed allocation before block splitting")
    stage.add_argument("--before-dense-probe", action="store_true",
                       help="restore tree-only probe variable indexing")
    stage.add_argument("--before-probe-cost", action="store_true",
                       help="restore materialized probe residual/zero vectors")
    stage.add_argument("--before-jet-negation", action="store_true",
                       help="restore generic negation with finite scans")
    stage.add_argument("--before-jet-loops", action="store_true",
                       help="restore indexed owned Jet derivative loops")
    stage.add_argument("--before-jet-buffers", action="store_true",
                       help="restore allocating owned Jet arithmetic")
    stage.add_argument("--before-mixed-probe", action="store_true",
                       help="restore graph/CSR probes for graphs containing automatic differentiation")
    stage.add_argument("--before-direct-probe", action="store_true",
                       help="restore graph registration/CSR assembly for every nonlinear probe")
    stage.add_argument("--before-incidence-cache", action="store_true",
                       help="restore full child/factor incidence enumeration in editable Gaussian updates")
    stage.add_argument("--before-editable-cow", action="store_true",
                       help="restore independent editable Gaussian metadata map copies")
    stage.add_argument("--before-local-clique-csr", action="store_true",
                       help="restore triplet assembly in editable Gaussian cliques")
    stage.add_argument("--before-stationarity-probes", action="store_true",
                       help="restore the editable-tree kernel before nonlinear probes, requested keys and topology reuse")
    stage.add_argument("--before-editable-bayes-tree", action="store_true",
                       help="restore the global-QR selected-factor kernel before separator-message replacement")
    stage.add_argument("--before-selective-relinearization", action="store_true",
                       help="restore the pre-selective nonlinear-factor cache kernel")
    stage.add_argument("--before-cartesian-chart", action="store_true",
                       help="restore Lie-chart-only variables before Cartesian SE2 and mapped QR row reuse")
    stage.add_argument("--before-qr-structure", action="store_true",
                       help="restore comparator sorting, scalar graph projection and fresh QR buffers")
    stage.add_argument("--before-bayes-tree", action="store_true",
                       help="restore the QR-only Gaussian cache before optional Bayes-tree paths")
    stage.add_argument("--before-information-precision", action="store_true",
                       help="restore ordinary information whitening before compensated small-pivot arithmetic")
    stage.add_argument("--before-sparse-qr", action="store_true",
                       help="restore LDL-only sparse GN/LM before ordered sparse QR and explicit GN policy")
    stage.add_argument("--before-real-isam", action="store_true",
                       help="restore absolute-gradient iSAM before optional QR ordering/refinement")
    parser.add_argument("--source-only", action="store_true", help="reconstruct source without compiling")
    stage.add_argument("--before-qr-rebuild", action="store_true",
                       help="restore chronological insertion for full-model QR rebuilds")
    stage.add_argument("--before-anchor-coordinates", action="store_true",
                       help="restore repeated iSAM principal local-coordinate calculations")
    stage.add_argument("--before-qr-columns", action="store_true",
                       help="restore per-index alignment checks for all equal-width QR suffixes")
    stage.add_argument("--before-qr-reductions", action="store_true",
                       help="restore per-entry finite checks and zero-count updates for aligned QR")
    stage.add_argument("--before-update-cursor", action="store_true",
                       help="restore full-column binary lookups for Schur update targets")
    stage.add_argument("--before-qr-norm", action="store_true",
                       help="restore standard hypot for every sparse QR rotation")
    stage.add_argument("--before-symbolic-merge", action="store_true",
                       help="restore individual fill insertions and repeated CSR-row lookups")
    stage.add_argument("--before-qr-aligned", action="store_true",
                       help="restore lower-row copying for aligned sparse QR suffixes")
    stage.add_argument("--before-normal-rebind", action="store_true",
                       help="restore full symbolic preparation for every CSR-pattern change")
    stage.add_argument("--before-ordering-adjacency", action="store_true",
                       help="restore tree-only adjacency and per-vertex neighbor allocation")
    stage.add_argument("--before-whitening-buffers", action="store_true",
                       help="restore allocating matrix whitening products")
    stage.add_argument("--before-qr-inplace", action="store_true",
                       help="restore buffered reconstruction of every upper QR row")
    stage.add_argument("--before-graph-retraction", action="store_true",
                       help="restore per-variable owned vector retraction")
    stage.add_argument("--before-canonical-csr", action="store_true",
                       help="restore triplet-based graph/dense-model CSR assembly")
    stage.add_argument("--before-qr-buffers", action="store_true",
                       help="restore per-input-row QR allocation and sentinel-based union merge")
    stage.add_argument("--before-symbolic-pattern", action="store_true",
                        help="restore zero-valued tree-map symbolic patterns, retaining damping reuse")
    stage.add_argument("--before-refinement", action="store_true",
                        help="restore reused LDL before residual refinement, retaining analytic/block options")
    stage.add_argument("--before-objective", action="store_true",
                       help="restore refined LDL before stable true-loss reductions")
    stage.add_argument("--before-isam", action="store_true",
                       help="restore incremental update graph construction before reuse")
    stage.add_argument("--before-scheduled-refresh", action="store_true",
                       help="restore graph-reusing iSAM before the scheduled refresh fix")
    stage.add_argument("--before-qr-merge", action="store_true",
                       help="restore scheduled iSAM before sorted sparse QR merges")
    stage.add_argument("--before-model-reuse", action="store_true",
                       help="restore sorted-QR iSAM before accepted model/input buffer reuse")
    stage.add_argument("--before-damping-reuse", action="store_true",
                       help="restore accepted-model iSAM before optional damping reuse")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.is_relative_to(ROOT):
        raise ValueError("baseline must be outside checkout")
    out.mkdir()
    for name in ("Cargo.toml", "Cargo.lock", "README.md"):
        shutil.copyfile(ROOT / name, out / name)
    crate = out / "crates/ropt"
    crate.mkdir(parents=True)
    shutil.copyfile(ROOT / "crates/ropt/Cargo.toml", crate / "Cargo.toml")
    shutil.copytree(ROOT / "crates/ropt/src", crate / "src")
    (crate / "examples/support").mkdir(parents=True)
    for name in ("benchmark_pose_graph.rs", "profile_pose_graph.rs", "support/allocation_counter.rs"):
        shutil.copyfile(ROOT / "crates/ropt/examples" / name, crate / "examples" / name)
    if args.before_isam or args.before_scheduled_refresh:
        for name in ("benchmark_isam.rs", "support/isam_stream.rs"):
            shutil.copyfile(ROOT / "crates/ropt/examples" / name, crate / "examples" / name)
    if args.before_qr_merge or args.before_model_reuse or args.before_damping_reuse or args.before_symbolic_pattern or args.before_qr_buffers or args.before_canonical_csr or args.before_graph_retraction or args.before_qr_inplace or args.before_whitening_buffers or args.before_ordering_adjacency or args.before_normal_rebind or args.before_qr_aligned or args.before_symbolic_merge or args.before_qr_norm or args.before_update_cursor or args.before_qr_reductions or args.before_qr_columns or args.before_anchor_coordinates or args.before_qr_rebuild or args.before_real_isam or args.before_sparse_qr or args.before_information_precision or args.before_bayes_tree or args.before_qr_structure or args.before_cartesian_chart:
        for name in ("benchmark_isam_se2.rs", "support/se2_benchmark.rs", "benchmark_isam.rs", "support/isam_stream.rs"):
            shutil.copyfile(ROOT / "crates/ropt/examples" / name, crate / "examples" / name)
    if args.before_symbolic_pattern or args.before_qr_buffers or args.before_canonical_csr or args.before_graph_retraction or args.before_qr_inplace or args.before_whitening_buffers or args.before_ordering_adjacency or args.before_normal_rebind or args.before_qr_aligned or args.before_symbolic_merge or args.before_qr_norm or args.before_update_cursor or args.before_qr_reductions or args.before_qr_columns or args.before_anchor_coordinates or args.before_qr_rebuild or args.before_real_isam or args.before_sparse_qr or args.before_information_precision or args.before_bayes_tree or args.before_qr_structure or args.before_cartesian_chart:
        for name in ("benchmark_warm_isam_se2.rs", "benchmark_restart_isam_se2.rs", "profile_symbolic.rs"):
            shutil.copyfile(ROOT / "crates/ropt/examples" / name, crate / "examples" / name)
    if args.before_qr_buffers or args.before_canonical_csr or args.before_graph_retraction or args.before_qr_inplace or args.before_whitening_buffers or args.before_ordering_adjacency or args.before_normal_rebind or args.before_qr_aligned or args.before_symbolic_merge or args.before_qr_norm or args.before_update_cursor or args.before_qr_reductions or args.before_qr_columns or args.before_anchor_coordinates or args.before_qr_rebuild or args.before_real_isam or args.before_sparse_qr or args.before_information_precision or args.before_bayes_tree or args.before_qr_structure or args.before_cartesian_chart:
        for name in ("validate_isam_se2.rs", "benchmark_cached_isam_se2.rs", "support/isam_se2.rs"):
            shutil.copyfile(ROOT / "crates/ropt/examples" / name, crate / "examples" / name)
    if args.before_cache_refresh or args.before_factor_capture or args.before_inline_ad_inputs or args.before_block_jacobian or args.before_dense_probe or args.before_probe_cost or args.before_jet_negation or args.before_jet_loops or args.before_jet_buffers or args.before_mixed_probe or args.before_direct_probe or args.before_incidence_cache or args.before_editable_cow or args.before_local_clique_csr:
        for name in ("benchmark_selective_se2_isam.rs", "support/real_selective_se2.rs", "support/real_se2_benchmark.rs"):
            shutil.copyfile(ROOT / "crates/ropt/examples" / name, crate / "examples" / name)
    if args.before_cache_refresh or args.before_factor_capture or args.before_inline_ad_inputs or args.before_block_jacobian or args.before_dense_probe or args.before_probe_cost or args.before_jet_negation or args.before_jet_loops or args.before_jet_buffers or args.before_mixed_probe:
        shutil.copyfile(ROOT / "crates/ropt/examples/benchmark_autodiff_probe.rs", crate / "examples/benchmark_autodiff_probe.rs")
        shutil.copyfile(ROOT / "crates/ropt/examples/benchmark_ad_selective_se2_isam.rs", crate / "examples/benchmark_ad_selective_se2_isam.rs")
    if args.before_cache_refresh or args.before_factor_capture or args.before_inline_ad_inputs or args.before_block_jacobian or args.before_dense_probe or args.before_probe_cost or args.before_jet_negation or args.before_jet_loops or args.before_jet_buffers:
        shutil.copyfile(ROOT / "crates/ropt/examples/profile_dynamic_jet.rs", crate / "examples/profile_dynamic_jet.rs")
    if args.before_cache_refresh or args.before_factor_capture or args.before_inline_ad_inputs or args.before_block_jacobian or args.before_dense_probe or args.before_probe_cost or args.before_jet_negation:
        shutil.copyfile(ROOT / "crates/ropt/examples/profile_dynamic_negation.rs", crate / "examples/profile_dynamic_negation.rs")
    if args.before_cache_refresh or args.before_factor_capture:
        shutil.copyfile(ROOT / "crates/ropt/examples/benchmark_factor_cache.rs", crate / "examples/benchmark_factor_cache.rs")
    if args.before_retired_timestamps or args.before_covariance_preflight or args.before_dynamic_lag or args.before_robust_covariance_model or args.before_covariance_snapshot or args.before_expiry_estimates or args.before_tree_refinement or args.before_tree_gradient:
        shutil.copytree(ROOT / "crates/ropt/examples", crate / "examples", dirs_exist_ok=True)
    names = []
    for name, stop_here in (
        ("retired_timestamps.patch", args.before_retired_timestamps),
        ("covariance_preflight.patch", args.before_covariance_preflight),
        ("dynamic_lag.patch", args.before_dynamic_lag),
        ("robust_covariance_model.patch", args.before_robust_covariance_model),
        ("covariance_snapshot.patch", args.before_covariance_snapshot),
        ("expiry_estimates.patch", args.before_expiry_estimates),
        ("tree_refinement.patch", args.before_tree_refinement),
        ("tree_gradient.patch", args.before_tree_gradient),
        ("cache_refresh.patch", args.before_cache_refresh),
        ("factor_capture.patch", args.before_factor_capture),
        ("inline_ad_inputs.patch", args.before_inline_ad_inputs),
        ("block_jacobian.patch", args.before_block_jacobian),
        ("dense_probe.patch", args.before_dense_probe),
        ("probe_cost.patch", args.before_probe_cost),
        ("jet_negation.patch", args.before_jet_negation),
        ("jet_loop_split.patch", args.before_jet_loops),
        ("jet_buffer_reuse.patch", args.before_jet_buffers),
        ("mixed_probe.patch", args.before_mixed_probe),
        ("direct_probe.patch", args.before_direct_probe),
        ("incidence_cache.patch", args.before_incidence_cache),
        ("editable_cow.patch", args.before_editable_cow),
        ("local_clique_csr.patch", args.before_local_clique_csr),
        ("stationarity_probes.patch", args.before_stationarity_probes),
        ("editable_bayes_tree.patch", args.before_editable_bayes_tree),
        ("selective_relinearization.patch", args.before_selective_relinearization),
        ("cartesian_chart.patch", args.before_cartesian_chart),
        ("qr_structure.patch", args.before_qr_structure),
        ("bayes_tree.patch", args.before_bayes_tree),
        ("information_precision.patch", args.before_information_precision),
        ("sparse_qr.patch", args.before_sparse_qr),
        ("real_isam.patch", args.before_real_isam),
        ("qr_rebuild.patch", args.before_qr_rebuild),
        ("anchor_coordinates.patch", args.before_anchor_coordinates),
        ("qr_columns.patch", args.before_qr_columns),
        ("qr_reductions.patch", args.before_qr_reductions),
        ("update_cursor.patch", args.before_update_cursor),
        ("qr_norm.patch", args.before_qr_norm),
        ("symbolic_sorted_merge.patch", args.before_symbolic_merge),
        ("qr_aligned.patch", args.before_qr_aligned),
        ("normal_pattern_rebind.patch", args.before_normal_rebind),
        ("ordering_adjacency.patch", args.before_ordering_adjacency),
        ("whitening_buffers.patch", args.before_whitening_buffers),
        ("qr_inplace.patch", args.before_qr_inplace),
        ("graph_retraction.patch", args.before_graph_retraction),
        ("canonical_csr.patch", args.before_canonical_csr),
        ("qr_merge_buffers.patch", args.before_qr_buffers),
        ("symbolic_pattern.patch", args.before_symbolic_pattern),
        ("isam_damping_reuse.patch", args.before_damping_reuse),
        ("isam_model_reuse.patch", args.before_model_reuse),
        ("isam_qr_merge.patch", args.before_qr_merge),
        ("isam_scheduled_refresh.patch", args.before_scheduled_refresh),
        ("isam_graph_reuse.patch", args.before_isam),
        ("stable_objective.patch", args.before_objective),
        ("ldl_refinement.patch", args.before_refinement),
        ("ldl_reuse.patch", True),
    ):
        names.append(name)
        if stop_here:
            break
    for name in names:
        patch = ROOT / "benchmarks" / name
        subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-R", "-d", str(out), "-i", str(patch)], check=True)
    metadata = {"patches_sha256": {name: hashlib.sha256((ROOT / "benchmarks" / name).read_bytes()).hexdigest()
                                   for name in names},
                "purpose": ("restored fixed-lag reports before retired timestamp snapshots" if args.before_retired_timestamps else
                            "restored covariance query validation after nonlinear optimization" if args.before_covariance_preflight else
                            "restored fixed-lag smoothing before runtime duration changes" if args.before_dynamic_lag else
                            "restored fixed-lag covariance before loss-model consistency checks" if args.before_robust_covariance_model else
                            "restored fixed-lag updates before pre-expiry covariance queries" if args.before_covariance_snapshot else
                            "restored fixed-lag reports before expired-estimate snapshots" if args.before_expiry_estimates else
                            "restored separate global QR and editable-tree refresh factorization" if args.before_tree_refinement else
                            "restored Gaussian trees before frozen-factor gradient solves" if args.before_tree_gradient else
                            "restored factor cache copies before scheduled refresh" if args.before_cache_refresh else
                            "restored row-iterator growth in Gaussian factor capture" if args.before_factor_capture else
                            "restored heap wrappers for AD callback inputs" if args.before_inline_ad_inputs else
                            "restored flat tangent Jacobian/seed allocation" if args.before_block_jacobian else
                            "restored tree-only probe variable indexing" if args.before_dense_probe else
                            "restored materialized probe residual/zero vectors" if args.before_probe_cost else
                            "restored generic negation with finite scans" if args.before_jet_negation else
                            "restored indexed owned Jet derivative loops" if args.before_jet_loops else
                            "restored allocating owned Jet arithmetic" if args.before_jet_buffers else
                            "restored graph/CSR probes for AD/mixed factor recipes" if args.before_mixed_probe else
                            "restored graph/CSR nonlinear stationarity probes" if args.before_direct_probe else
                            "restored full editable-tree incidence enumeration" if args.before_incidence_cache else
                            "restored independent editable-tree metadata copies" if args.before_editable_cow else
                            "restored editable-tree triplet clique assembly" if args.before_local_clique_csr else
                            "restored editable tree before nonlinear stationarity probes and symbolic topology reuse" if args.before_stationarity_probes else
                            "restored selected-factor kernel before editable Gaussian Bayes-tree replacement" if args.before_editable_bayes_tree else
                            "restored kernel before selective nonlinear factor evaluation" if args.before_selective_relinearization else
                            "restored Lie-chart-only variables and unmapped QR rows" if args.before_cartesian_chart else
                            "restored comparator QR grouping, scalar ordering projection and fresh numeric buffers" if args.before_qr_structure else
                            "restored the QR-only Gaussian cache" if args.before_bayes_tree else
                            "restored ordinary information whitening" if args.before_information_precision else
                            "restored sparse solver before ordered QR and explicit nonmonotonic GN" if args.before_sparse_qr else
                            "restored absolute-gradient iSAM before optional QR ordering/refinement" if args.before_real_isam else
                            "restored chronological insertion for full-model QR rebuilds" if args.before_qr_rebuild else
                            "restored repeated iSAM principal local-coordinate calculations" if args.before_anchor_coordinates else
                            "restored per-index alignment checks for all equal-width QR suffixes" if args.before_qr_columns else
                            "restored per-entry finite checks and zero-count updates for aligned QR" if args.before_qr_reductions else
                            "restored full-column binary lookups for Schur update targets" if args.before_update_cursor else
                            "restored standard hypot for every sparse QR rotation" if args.before_qr_norm else
                            "restored individual fill insertions and repeated CSR-row lookups" if args.before_symbolic_merge else
                            "restored copying for aligned sparse QR suffixes" if args.before_qr_aligned else
                            "restored full symbolic preparation for every CSR-pattern change" if args.before_normal_rebind else
                            "restored tree-only adjacency and per-vertex neighbors" if args.before_ordering_adjacency else
                            "restored allocating matrix whitening products" if args.before_whitening_buffers else
                            "restored buffered reconstruction of every upper QR row" if args.before_qr_inplace else
                            "restored per-variable owned vector retraction" if args.before_graph_retraction else
                            "restored triplet-based graph/dense-model CSR assembly" if args.before_canonical_csr else
                            "restored QR before per-row buffer reuse and paired/tail merge" if args.before_qr_buffers else
                            "restored tree-map symbolic patterns before compact structural storage" if args.before_symbolic_pattern else
                            "restored iSAM before adaptive damping reuse" if args.before_damping_reuse else
                            "restored iSAM before accepted-model reuse" if args.before_model_reuse else
                            "restored iSAM before sorted QR merge" if args.before_qr_merge else
                            "restored iSAM before scheduled refresh fix" if args.before_scheduled_refresh else
                            "restored iSAM before graph reuse" if args.before_isam else
                            "restored refined LDL before stable objective reductions" if args.before_objective else
                            "restored reused LDL before residual refinement" if args.before_refinement else
                            "restored original map-based LDL and AD-only pose graph") + "; benchmark-only snapshot"}
    (out / "baseline.json").write_text(json.dumps(metadata, indent=2) + "\n")
    command = ["cargo", "build", "--manifest-path", str(out / "Cargo.toml"), "-p", "ropt",
               "--example", "benchmark_pose_graph", "--example", "profile_pose_graph"]
    if args.before_isam or args.before_scheduled_refresh:
        command += ["--example", "benchmark_isam"]
    if args.before_qr_merge or args.before_model_reuse or args.before_damping_reuse or args.before_symbolic_pattern or args.before_qr_buffers or args.before_canonical_csr or args.before_graph_retraction or args.before_qr_inplace or args.before_whitening_buffers or args.before_ordering_adjacency or args.before_normal_rebind or args.before_qr_aligned or args.before_symbolic_merge or args.before_qr_norm or args.before_update_cursor or args.before_qr_reductions or args.before_qr_columns or args.before_anchor_coordinates or args.before_qr_rebuild or args.before_real_isam or args.before_sparse_qr or args.before_information_precision or args.before_bayes_tree or args.before_qr_structure or args.before_cartesian_chart:
        command += ["--example", "benchmark_isam_se2", "--example", "benchmark_isam"]
    if args.before_symbolic_pattern or args.before_qr_buffers or args.before_canonical_csr or args.before_graph_retraction or args.before_qr_inplace or args.before_whitening_buffers or args.before_ordering_adjacency or args.before_normal_rebind or args.before_qr_aligned or args.before_symbolic_merge or args.before_qr_norm or args.before_update_cursor or args.before_qr_reductions or args.before_qr_columns or args.before_anchor_coordinates or args.before_qr_rebuild or args.before_real_isam or args.before_sparse_qr or args.before_information_precision or args.before_bayes_tree or args.before_qr_structure or args.before_cartesian_chart:
        command += ["--example", "benchmark_warm_isam_se2", "--example", "benchmark_restart_isam_se2", "--example", "profile_symbolic"]
    if args.before_qr_buffers or args.before_canonical_csr or args.before_graph_retraction or args.before_qr_inplace or args.before_whitening_buffers or args.before_ordering_adjacency or args.before_normal_rebind or args.before_qr_aligned or args.before_symbolic_merge or args.before_qr_norm or args.before_update_cursor or args.before_qr_reductions or args.before_qr_columns or args.before_anchor_coordinates or args.before_qr_rebuild or args.before_real_isam or args.before_sparse_qr or args.before_information_precision or args.before_bayes_tree or args.before_qr_structure or args.before_cartesian_chart:
        command += ["--example", "validate_isam_se2", "--example", "benchmark_cached_isam_se2"]
    if args.before_cache_refresh or args.before_factor_capture or args.before_inline_ad_inputs or args.before_block_jacobian or args.before_dense_probe or args.before_probe_cost or args.before_jet_negation or args.before_jet_loops or args.before_jet_buffers or args.before_mixed_probe or args.before_direct_probe or args.before_incidence_cache or args.before_editable_cow or args.before_local_clique_csr:
        command += ["--example", "benchmark_selective_se2_isam"]
    if args.before_cache_refresh or args.before_factor_capture or args.before_inline_ad_inputs or args.before_block_jacobian or args.before_dense_probe or args.before_probe_cost or args.before_jet_negation or args.before_jet_loops or args.before_jet_buffers or args.before_mixed_probe:
        command += ["--example", "benchmark_autodiff_probe", "--example", "benchmark_ad_selective_se2_isam"]
    if args.before_cache_refresh or args.before_factor_capture or args.before_inline_ad_inputs or args.before_block_jacobian or args.before_dense_probe or args.before_probe_cost or args.before_jet_negation or args.before_jet_loops or args.before_jet_buffers:
        command += ["--example", "profile_dynamic_jet"]
    if args.before_cache_refresh or args.before_factor_capture or args.before_inline_ad_inputs or args.before_block_jacobian or args.before_dense_probe or args.before_probe_cost or args.before_jet_negation:
        command += ["--example", "profile_dynamic_negation"]
    if args.before_cache_refresh or args.before_factor_capture:
        command += ["--example", "benchmark_factor_cache"]
    if args.before_retired_timestamps or args.before_covariance_preflight or args.before_dynamic_lag or args.before_robust_covariance_model or args.before_covariance_snapshot or args.before_expiry_estimates or args.before_tree_refinement or args.before_tree_gradient:
        command += ["--example", "benchmark_ad_selective_se2_isam"]
    if not args.source_only:
        subprocess.run(command + ["--release", "--locked", "--offline"], check=True)
    if args.source_only:
        print(f"Baseline source saved under {out}; compilation skipped")
    else:
        print(f"Baseline executables saved under {out}/target/release/examples")


if __name__ == "__main__":
    main()
