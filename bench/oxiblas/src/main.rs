//! Cholesky and LU with partial pivoting of OxiBLAS (pure Rust,
//! oxiblas-lapack 0.2.4), for the comparisons of ulmBLAS with OpenBLAS
//! (bench/potrf.jl and bench/getrf.jl of UlmBLAS.jl): the same sizes, the
//! same thread numbers, the same output.
//!
//!     cargo build --release          (RUSTFLAGS="-C target-cpu=native")
//!     THREADS="1 6" SIZES="100 200" RESULTS=name target/release/oxiblas-bench
//!
//! THREADS: thread numbers (default 1 and the number of logical CPUs),
//! SIZES: n (default 100 200 500 1000 2000 4000), RESULTS: writes
//! name.tsv (default oxiblas.tsv).
//!
//! For each n the fastest of the serial variants (compute_blocked,
//! compute_recursive: OxiBLAS's own choices) on 1 thread, and
//! compute_blocked_par (Rayon, a pool of p threads) on p > 1 threads.  The
//! functions take the matrix by reference and return the factor, so the
//! time includes their copy of A (n^2, small against n^3/3 except for
//! small n).  Time: median of at least 5 samples and at least 3 seconds.
//! Each factorization is checked: the solution of A x = b with b = A x0,
//! residual |A x - b| / (n eps |A| |x|) (max norms), must be < 100.
//! GFLOPS: n^3/3 (Cholesky), 2n^3/3 (LU) per second.

use oxiblas_lapack::cholesky::Cholesky;
use oxiblas_lapack::lu::Lu;
use oxiblas_matrix::Mat;
use std::io::Write;
use std::time::Instant;

/// uniform in [-0.5, 0.5), deterministic (LCG)
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / 9007199254740992.0 - 0.5
    }
}

fn general(n: usize) -> Mat<f64> {
    let mut rng = Rng(n as u64 + 1);
    let mut a: Mat<f64> = Mat::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            a[(i, j)] = rng.next();
        }
    }
    a
}

/// symmetric positive definite: (X + X^T)/2 + n I
fn spd(n: usize) -> Mat<f64> {
    let x = general(n);
    let mut a: Mat<f64> = Mat::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            a[(i, j)] = 0.5 * (x[(i, j)] + x[(j, i)]) + if i == j { n as f64 } else { 0.0 };
        }
    }
    a
}

/// |A x - b| / (n eps |A| |x|) with b = A x0, x the computed solution
fn residual(a: &Mat<f64>, solve: &dyn Fn(&Mat<f64>) -> Mat<f64>) -> f64 {
    let n = a.nrows();
    let mut rng = Rng(7);
    let mut x0: Mat<f64> = Mat::zeros(n, 1);
    for i in 0..n {
        x0[(i, 0)] = rng.next();
    }
    let mut b: Mat<f64> = Mat::zeros(n, 1);
    let mut norm_a: f64 = 0.0;
    for i in 0..n {
        let mut s = 0.0;
        let mut row = 0.0;
        for j in 0..n {
            s += a[(i, j)] * x0[(j, 0)];
            row += a[(i, j)].abs();
        }
        b[(i, 0)] = s;
        norm_a = norm_a.max(row);
    }
    let x = solve(&b);
    let mut r: f64 = 0.0;
    let mut norm_x: f64 = 0.0;
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += a[(i, j)] * x[(j, 0)];
        }
        r = r.max((s - b[(i, 0)]).abs());
        norm_x = norm_x.max(x[(i, 0)].abs());
    }
    r / (n as f64 * f64::EPSILON * norm_a * norm_x)
}

/// median time of f: at least 5 samples and at least 3 seconds
fn measure(f: &dyn Fn()) -> f64 {
    f();
    let mut times = Vec::new();
    let start = Instant::now();
    while times.len() < 5 || start.elapsed().as_secs_f64() < 3.0 {
        let t = Instant::now();
        f();
        times.push(t.elapsed().as_secs_f64());
        if times.len() >= 10000 {
            break;
        }
    }
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

fn ints(var: &str, default: &[usize]) -> Vec<usize> {
    match std::env::var(var) {
        Ok(s) => s.split_whitespace().map(|v| v.parse().unwrap()).collect(),
        Err(_) => default.to_vec(),
    }
}

fn main() {
    let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
    let threads = ints("THREADS", &[1, cpus]);
    let sizes = ints("SIZES", &[100, 200, 500, 1000, 2000, 4000]);
    let results = std::env::var("RESULTS").unwrap_or_else(|_| "oxiblas".into()) + ".tsv";
    println!(
        "OxiBLAS 0.2.4 (oxiblas-lapack), {} logical CPUs, sme2 feature: {}",
        cpus,
        cfg!(feature = "sme2")
    );

    let mut rows: Vec<(&str, usize, usize, f64, f64)> = Vec::new();
    for (op, flops) in [("potrf", 1.0 / 3.0), ("getrf", 2.0 / 3.0)] {
        println!("\nOxiBLAS {}", if op == "potrf" { "Cholesky" } else { "LU" });
        for &p in &threads {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(p).build().unwrap();
            for &n in &sizes {
                let a = if op == "potrf" { spd(n) } else { general(n) };
                let (t, r, variant) = pool.install(|| {
                    let mut best = (f64::INFINITY, "");
                    let variants: &[&str] = if p == 1 {
                        &["blocked", "recursive"]
                    } else {
                        &["blocked_par"]
                    };
                    for &v in variants {
                        let t = if op == "potrf" {
                            measure(&|| {
                                let _ = match v {
                                    "blocked" => Cholesky::compute_blocked(a.as_ref()),
                                    "recursive" => Cholesky::compute_recursive(a.as_ref()),
                                    _ => Cholesky::compute_blocked_par(a.as_ref()),
                                }
                                .unwrap();
                            })
                        } else {
                            measure(&|| {
                                let _ = match v {
                                    "blocked" => Lu::compute_blocked(a.as_ref()),
                                    "recursive" => Lu::compute_recursive(a.as_ref()),
                                    _ => Lu::compute_blocked_par(a.as_ref()),
                                }
                                .unwrap();
                            })
                        };
                        if t < best.0 {
                            best = (t, v);
                        }
                    }
                    let v = best.1;
                    let r = if op == "potrf" {
                        let f = match v {
                            "blocked" => Cholesky::compute_blocked(a.as_ref()),
                            "recursive" => Cholesky::compute_recursive(a.as_ref()),
                            _ => Cholesky::compute_blocked_par(a.as_ref()),
                        }
                        .unwrap();
                        residual(&a, &|b| f.solve(b.as_ref()).unwrap())
                    } else {
                        let f = match v {
                            "blocked" => Lu::compute_blocked(a.as_ref()),
                            "recursive" => Lu::compute_recursive(a.as_ref()),
                            _ => Lu::compute_blocked_par(a.as_ref()),
                        }
                        .unwrap();
                        residual(&a, &|b| f.solve(b.as_ref()).unwrap())
                    };
                    (best.0, r, v)
                });
                let g = flops * (n as f64).powi(3) / t / 1e9;
                println!(
                    "  threads {:2}  n {:5}  {:10.3} ms  {:7.1} GFLOPS  (residual {:.2}, {})",
                    p,
                    n,
                    1e3 * t,
                    g,
                    r,
                    variant
                );
                assert!(r < 100.0, "{} n = {}: residual {}", op, n, r);
                rows.push((op, p, n, t, g));
            }
        }
    }
    let mut out = std::fs::File::create(&results).unwrap();
    writeln!(out, "op\tlibrary\tthreads\tn\tseconds\tgflops").unwrap();
    for (op, p, n, t, g) in rows {
        writeln!(out, "{}\tOxiBLAS\t{}\t{}\t{}\t{}", op, p, n, t, g).unwrap();
    }
}
