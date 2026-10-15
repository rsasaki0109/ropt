use ropt::{factor::*, noise::NoiseModel, *};

#[test]
fn mixed_factor_arity_dimensions_and_noise_preserve_dense_sparse_whitening() {
    let mut graph = FactorGraph::new();
    let ids: Vec<_> = [1, 3, 12]
        .into_iter()
        .map(|n| {
            graph
                .add_variable(VariableKind::Euclidean(n), DVector::from_element(n, 0.2))
                .unwrap()
        })
        .collect();
    let specs = [
        (3, vec![2, 0, 1]),
        (1, vec![0]),
        (17, vec![1, 2]),
        (2, vec![0, 2]),
        (3, vec![1]),
    ];
    let mut references = Vec::new();
    for (case, (m, slots)) in specs.into_iter().enumerate() {
        let w = if case == 1 {
            DMatrix::identity(m, m)
        } else {
            DMatrix::from_fn(m, m, |r, c| {
                if r == c {
                    1. + r as f64 * 0.1
                } else {
                    ((r + c + 1) as f64).sin() * 0.05
                }
            })
        };
        let noise = if case == 1 {
            NoiseModel::unit()
        } else {
            NoiseModel::sqrt_information(w.clone()).unwrap()
        };
        let blocks: Vec<_> = slots
            .iter()
            .map(|&slot| {
                DMatrix::from_fn(
                    m,
                    graph.variable_kind(ids[slot]).unwrap().dimensions().1,
                    |r, c| ((r + 2 * c + 1) as f64 * 0.17).cos(),
                )
            })
            .collect();
        let seed = DVector::from_fn(m, |r, _| (r as f64 * 0.7).cos());
        let callback_blocks = blocks.clone();
        let callback_seed = seed.clone();
        graph
            .add_analytic_factor_with_options(
                &slots.iter().map(|&slot| ids[slot]).collect::<Vec<_>>(),
                m,
                FactorOptions {
                    noise,
                    ..Default::default()
                },
                move |values| {
                    let mut r = callback_seed.clone();
                    for (block, value) in callback_blocks.iter().zip(values) {
                        r += block * value;
                    }
                    Ok((r, callback_blocks.clone()))
                },
            )
            .unwrap();
        references.push((m, slots, w, seed, blocks));
    }
    for fixed in [vec![], vec![1], vec![2], vec![0, 1, 2]] {
        for (i, &id) in ids.iter().enumerate() {
            graph.set_fixed(id, fixed.contains(&i)).unwrap();
        }
        let x = graph.initial();
        let (_, n, m) = graph.dimensions();
        let mut expected_r = DVector::zeros(m);
        let mut expected_j = DMatrix::zeros(m, n);
        let mut row = 0;
        for (rows, slots, w, seed, blocks) in &references {
            let mut raw = seed.clone();
            for (&slot, block) in slots.iter().zip(blocks) {
                raw += block * graph.value(&x, ids[slot]).unwrap();
            }
            expected_r.rows_mut(row, *rows).copy_from(&(w * raw));
            for (&slot, block) in slots.iter().zip(blocks) {
                if let Some(t) = graph.tangent_range(ids[slot]).unwrap() {
                    expected_j
                        .view_mut((row, t.start), (*rows, t.len()))
                        .copy_from(&(w * block));
                }
            }
            row += rows;
        }
        let mut r = DVector::zeros(m);
        let mut j = DMatrix::zeros(m, n);
        graph.evaluate_local(&x, &mut r, &mut j).unwrap();
        assert!((&r - expected_r).norm() < 1e-12);
        assert!((&j - expected_j).norm() < 1e-12);
        let (sr, sj) = graph.evaluate_sparse(&x).unwrap();
        assert_eq!(sr, r);
        for c in 0..n {
            let mut basis = DVector::zeros(n);
            basis[c] = 1.;
            assert_eq!(sj.multiply(&basis).unwrap(), j.column(c));
        }
    }
}
