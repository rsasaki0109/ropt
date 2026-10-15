#[path = "../examples/support/isam_se2.rs"]
mod fixture;

#[test]
fn delayed_non_root_loops_match_autodiff_dense_qr_after_nonlinear_refresh() {
    let report = fixture::validate(25, &mut std::io::sink()).unwrap();
    assert_eq!(report.checkpoints, 1);
    assert_eq!(report.forced, 1);
    assert!(report.interval > 0);
    assert!(report.incremental > 0);
    assert!(report.max_pose_difference < 2e-6);
    assert!(report.max_refreshed_gradient < 1e-6);
    // The fixture must actually expose nonlinear approximation between refreshes.
    assert!(report.max_incremental_gradient > 1e-3);
}
