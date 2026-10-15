use ropt::{factor::*, noise::NoiseModel, *};

fn evaluate(information: DMatrix<f64>, raw: DVector<f64>) -> (DVector<f64>, DMatrix<f64>) {
    let n = raw.len();
    let mut graph = FactorGraph::new();
    let id = graph
        .add_variable(VariableKind::Euclidean(n), raw.clone())
        .unwrap();
    graph
        .add_analytic_factor_with_options(
            &[id],
            n,
            FactorOptions {
                noise: NoiseModel::from_information(information).unwrap(),
                ..Default::default()
            },
            move |v| Ok((v[0].clone(), vec![DMatrix::identity(n, n)])),
        )
        .unwrap();
    let mut r = DVector::zeros(n);
    let mut j = DMatrix::zeros(n, n);
    graph.evaluate_local(&raw, &mut r, &mut j).unwrap();
    (r, j)
}

#[test]
fn exactly_representable_weak_direction_survives_large_dynamic_range() {
    // Known eigenpair: Omega * [1,-1] = 4 * [1,-1].
    // Power-of-two scaling retains the exact input, unlike a rounded decimal fixture.
    for exponent in [-400, 0, 400] {
        let scale = 2_f64.powi(exponent);
        let a = (2_f64.powi(53) + 4.) * scale;
        let b = 2_f64.powi(53) * scale;
        let (r, j) = evaluate(
            DMatrix::from_row_slice(2, 2, &[a, b, b, a]),
            DVector::from_vec(vec![1., -1.]),
        );
        assert!((0.5 * r.norm_squared() / scale - 4.).abs() < 2e-14);
        let g = j.transpose() * &r;
        assert!((g[0] / scale - 4.).abs() < 2e-14);
        assert!((g[1] / scale + 4.).abs() < 2e-14);
    }
}

#[test]
fn recorded_correlated_information_matches_independent_decimal_energy() {
    let omega = DMatrix::from_row_slice(
        3,
        3,
        &[
            2693538350855.096,
            -157146640359.0913,
            0.,
            -157146640359.0913,
            9168262482.076782,
            0.,
            0.,
            0.,
            636.440966,
        ],
    );
    let raw = DVector::from_vec(vec![
        -0.0008459986500432724,
        -0.01450065877031801,
        -0.00031608221007084497,
    ]);
    // 70-digit Decimal evaluation of the original binary64 inputs, not WᵀW.
    let expected = 0.0012039340794135457;
    let (r, _) = evaluate(omega, raw);
    assert!((0.5 * r.norm_squared() - expected).abs() < 3e-18);
}

#[test]
fn information_does_not_regularize_nonpositive_pivots_or_accept_invalid_input() {
    for entries in [[1., 1., 1., 1.], [1., 2., 2., 1.], [1., 0., 0., 0.]] {
        assert!(matches!(
            NoiseModel::from_information(DMatrix::from_row_slice(2, 2, &entries)),
            Err(Error::InvalidOptions)
        ));
    }
    assert!(matches!(
        NoiseModel::from_information(DMatrix::zeros(2, 3)),
        Err(Error::InvalidDimensions)
    ));
    assert!(matches!(
        NoiseModel::from_information(DMatrix::from_element(2, 2, f64::NAN)),
        Err(Error::NonFinite)
    ));
    assert!(matches!(
        NoiseModel::from_information(DMatrix::from_row_slice(2, 2, &[1., 0., 0.1, 1.])),
        Err(Error::InvalidOptions)
    ));
}
