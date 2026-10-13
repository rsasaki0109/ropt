#![doc = include_str!("../../../README.md")]
//! Dense nonlinear least squares with explicit Jacobians and forward differentiation.
//! Experimental dense/sparse LM, factor graphs and block Schur Bundle Adjustment.
pub mod autodiff;
pub mod camera;
pub mod covariance;
pub mod dense;
pub mod dynamic_autodiff;
pub mod factor;
pub mod fixed_lag;
pub mod g2o;
pub mod incremental_qr;
pub mod isam;
pub mod manifold;
mod manifold_autodiff;
pub mod marginalization;
pub mod navigation;
pub mod noise;
pub mod pose_graph;
pub mod preconditioner;
pub mod schur;
pub mod solver;
pub mod sparse;
pub mod sparse_direct;
pub use nalgebra::{DMatrix, DVector};
pub use solver::*;
