//! Offline nonlinear accuracy checks; no performance claims or timings.
#[path = "support/isam_se2.rs"]
mod fixture;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<_> = std::env::args().skip(1).collect();
    let tree = args.last().is_some_and(|arg| arg == "--bayes-tree");
    if tree {
        args.pop();
    }
    if args.len() > 1 {
        return Err(
            "usage: validate_isam_se2 [frames=100] [--bayes-tree] (multiple of 25, <=300)".into(),
        );
    }
    let frames = args
        .first()
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(100);
    let summary = if tree {
        fixture::validate_observed_with_smoother(
            frames,
            &mut std::io::stdout().lock(),
            &mut (),
            |s| Ok(s.with_bayes_tree()?),
        )?
    } else {
        fixture::validate(frames, &mut std::io::stdout().lock())?
    };
    eprintln!("{summary:#?}");
    Ok(())
}
