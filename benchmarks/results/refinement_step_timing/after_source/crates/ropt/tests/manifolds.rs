use nalgebra::{SVector, Vector3};
use ropt::manifold::*;
#[test]
fn so2_and_se2_roundtrip() {
    for angle in [0.0, 1e-10, 0.5, std::f64::consts::PI - 1e-8] {
        let r = SO2::exp(angle);
        assert!((r.log() - angle).abs() < 1e-12);
        assert!(r.compose(r.inverse()).log().abs() < 1e-12);
        let v = Vector3::new(1.0, -0.4, angle);
        let p = SE2::exp(v);
        assert!((p.log() - v).norm() < 1e-9);
        assert!(p.compose(p.inverse()).log().norm() < 1e-9);
        let d = Vector3::new(0.1, 0.2, -0.05);
        assert!((p.local(p.retract(d)) - d).norm() < 1e-9);
    }
}
#[test]
fn so3_and_se3_roundtrip_near_zero_and_pi() {
    for angle in [0.0, 1e-10, 0.5, std::f64::consts::PI - 1e-8] {
        let w = Vector3::new(angle, 0.0, 0.0);
        let r = SO3::exp(w);
        assert!((r.log() - w).norm() < 1e-9);
        assert!(r.compose(r.inverse()).log().norm() < 1e-9);
        let v = Tangent6::new(1.0, -0.4, 0.7, angle, 0.0, 0.0);
        let p = SE3::exp(v);
        assert!((p.log() - v).norm() < 1e-9);
        assert!(p.compose(p.inverse()).log().norm() < 1e-9);
        let d = Tangent6::new(0.1, 0.2, -0.05, 0.03, -0.02, 0.04);
        assert!((p.local(p.retract(d)) - d).norm() < 1e-9);
    }
}
#[test]
fn euclidean_retraction() {
    let x = Euclidean(SVector::<f64, 2>::new(1.0, 2.0));
    let d = SVector::<f64, 2>::new(0.3, -0.4);
    assert!((x.local(x.retract(d)) - d).norm() < 1e-15);
}

#[test]
fn se3_non_axis_aligned_roundtrips() {
    for i in 1..25 {
        let t = i as f64 / 25.0;
        let v = Tangent6::new(3.0 * t, -2.0 * t, 0.7 * t, 0.8 * t, -1.4 * t, 1.9 * t);
        let p = SE3::exp(v);
        assert!((p.log() - v).norm() < 1e-10);
        let q = SE3::exp(Tangent6::new(-0.2, 0.3, 0.8, 0.1, -0.5, 0.2));
        assert!(
            p.compose(q)
                .inverse()
                .compose(q.inverse().compose(p.inverse()).inverse())
                .log()
                .norm()
                < 1e-10
        );
        let d = Tangent6::new(0.01, -0.02, 0.04, 0.03, 0.02, -0.01);
        assert!((p.local(p.retract(d)) - d).norm() < 1e-10);
    }
}
