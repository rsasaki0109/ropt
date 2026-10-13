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
    if args.before_qr_merge or args.before_model_reuse or args.before_damping_reuse or args.before_symbolic_pattern or args.before_qr_buffers or args.before_canonical_csr or args.before_graph_retraction or args.before_qr_inplace or args.before_whitening_buffers or args.before_ordering_adjacency or args.before_normal_rebind or args.before_qr_aligned or args.before_symbolic_merge or args.before_qr_norm or args.before_update_cursor or args.before_qr_reductions or args.before_qr_columns or args.before_anchor_coordinates or args.before_qr_rebuild or args.before_real_isam:
        for name in ("benchmark_isam_se2.rs", "support/se2_benchmark.rs", "benchmark_isam.rs", "support/isam_stream.rs"):
            shutil.copyfile(ROOT / "crates/ropt/examples" / name, crate / "examples" / name)
    if args.before_symbolic_pattern or args.before_qr_buffers or args.before_canonical_csr or args.before_graph_retraction or args.before_qr_inplace or args.before_whitening_buffers or args.before_ordering_adjacency or args.before_normal_rebind or args.before_qr_aligned or args.before_symbolic_merge or args.before_qr_norm or args.before_update_cursor or args.before_qr_reductions or args.before_qr_columns or args.before_anchor_coordinates or args.before_qr_rebuild or args.before_real_isam:
        for name in ("benchmark_warm_isam_se2.rs", "benchmark_restart_isam_se2.rs", "profile_symbolic.rs"):
            shutil.copyfile(ROOT / "crates/ropt/examples" / name, crate / "examples" / name)
    if args.before_qr_buffers or args.before_canonical_csr or args.before_graph_retraction or args.before_qr_inplace or args.before_whitening_buffers or args.before_ordering_adjacency or args.before_normal_rebind or args.before_qr_aligned or args.before_symbolic_merge or args.before_qr_norm or args.before_update_cursor or args.before_qr_reductions or args.before_qr_columns or args.before_anchor_coordinates or args.before_qr_rebuild or args.before_real_isam:
        for name in ("validate_isam_se2.rs", "benchmark_cached_isam_se2.rs", "support/isam_se2.rs"):
            shutil.copyfile(ROOT / "crates/ropt/examples" / name, crate / "examples" / name)
    names = []
    for name, stop_here in (
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
                "purpose": ("restored absolute-gradient iSAM before optional QR ordering/refinement" if args.before_real_isam else
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
    if args.before_qr_merge or args.before_model_reuse or args.before_damping_reuse or args.before_symbolic_pattern or args.before_qr_buffers or args.before_canonical_csr or args.before_graph_retraction or args.before_qr_inplace or args.before_whitening_buffers or args.before_ordering_adjacency or args.before_normal_rebind or args.before_qr_aligned or args.before_symbolic_merge or args.before_qr_norm or args.before_update_cursor or args.before_qr_reductions or args.before_qr_columns or args.before_anchor_coordinates or args.before_qr_rebuild or args.before_real_isam:
        command += ["--example", "benchmark_isam_se2", "--example", "benchmark_isam"]
    if args.before_symbolic_pattern or args.before_qr_buffers or args.before_canonical_csr or args.before_graph_retraction or args.before_qr_inplace or args.before_whitening_buffers or args.before_ordering_adjacency or args.before_normal_rebind or args.before_qr_aligned or args.before_symbolic_merge or args.before_qr_norm or args.before_update_cursor or args.before_qr_reductions or args.before_qr_columns or args.before_anchor_coordinates or args.before_qr_rebuild or args.before_real_isam:
        command += ["--example", "benchmark_warm_isam_se2", "--example", "benchmark_restart_isam_se2", "--example", "profile_symbolic"]
    if args.before_qr_buffers or args.before_canonical_csr or args.before_graph_retraction or args.before_qr_inplace or args.before_whitening_buffers or args.before_ordering_adjacency or args.before_normal_rebind or args.before_qr_aligned or args.before_symbolic_merge or args.before_qr_norm or args.before_update_cursor or args.before_qr_reductions or args.before_qr_columns or args.before_anchor_coordinates or args.before_qr_rebuild or args.before_real_isam:
        command += ["--example", "validate_isam_se2", "--example", "benchmark_cached_isam_se2"]
    if not args.source_only:
        subprocess.run(command + ["--release", "--locked", "--offline"], check=True)
    if args.source_only:
        print(f"Baseline source saved under {out}; compilation skipped")
    else:
        print(f"Baseline executables saved under {out}/target/release/examples")


if __name__ == "__main__":
    main()
