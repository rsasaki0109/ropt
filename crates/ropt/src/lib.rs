#![doc = include_str!("../../../README.md")]
//! Dense nonlinear least squares with explicit Jacobians and forward differentiation.
//! Experimental dense/sparse LM, factor graphs and block Schur Bundle Adjustment.
pub mod autodiff;
pub mod camera;
pub mod factor;
pub mod manifold;
pub mod schur;
pub mod solver;
pub mod sparse;
pub use nalgebra::{DMatrix, DVector};
pub use solver::*;
