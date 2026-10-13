//! Offline nonlinear accuracy checks; no performance claims or timings.
#[path = "support/isam_se2.rs"]
mod fixture;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() > 1 {
        return Err("usage: validate_isam_se2 [frames=100] (multiple of 25, <=300)".into());
    }
    let frames = args
        .first()
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(100);
    let summary = fixture::validate(frames, &mut std::io::stdout().lock())?;
    eprintln!("{summary:#?}");
    Ok(())
}
