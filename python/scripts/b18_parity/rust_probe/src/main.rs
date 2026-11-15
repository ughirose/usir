use std::io::{self, Read};

use nishiki_linalg::hsic::{audit_blocks, median_bandwidth_sq, nhsic, sq_distance_matrix};
use nishiki_linalg::lowdin::{standardize, HsicLowdinOrthogonalizer, DEFAULT_EPS_REG};
use nishiki_linalg::matrix::Matrix;

fn next_tok<'a, I: Iterator<Item = &'a str>>(it: &mut I) -> &'a str {
    it.next().expect("unexpected EOF")
}

fn read_matrix<'a, I: Iterator<Item = &'a str>>(it: &mut I, rows: usize, cols: usize) -> Matrix {
    let mut data = Vec::with_capacity(rows * cols);
    for _ in 0..(rows * cols) {
        let v: f64 = next_tok(it).parse().expect("valid f64");
        data.push(v);
    }
    Matrix::from_vec(rows, cols, data).expect("matrix shape")
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).expect("read stdin");
    let mut it = input.split_whitespace();

    while let Some(cmd) = it.next() {
        match cmd {
            "NHSIC" => {
                let n: usize = next_tok(&mut it).parse().unwrap();
                let da: usize = next_tok(&mut it).parse().unwrap();
                let db: usize = next_tok(&mut it).parse().unwrap();
                let xa = read_matrix(&mut it, n, da);
                let xb = read_matrix(&mut it, n, db);
                let d2a = sq_distance_matrix(&xa).unwrap();
                let d2b = sq_distance_matrix(&xb).unwrap();
                let s2a = median_bandwidth_sq(&d2a).unwrap().unwrap_or(0.0);
                let s2b = median_bandwidth_sq(&d2b).unwrap().unwrap_or(0.0);
                let val = nhsic(&xa, &xb).unwrap();
                println!("NHSIC {val:.17e} {s2a:.17e} {s2b:.17e}");
            }
            "LOWDIN" => {
                let n: usize = next_tok(&mut it).parse().unwrap();
                let d: usize = next_tok(&mut it).parse().unwrap();
                let eps: f64 = next_tok(&mut it).parse().unwrap();
                let eps = if eps <= 0.0 { DEFAULT_EPS_REG } else { eps };
                let x = read_matrix(&mut it, n, d);
                let std_x = standardize(&x).unwrap();
                let orth = HsicLowdinOrthogonalizer { eps_reg: eps, gate: 0.25 };
                let fit = orth.fit(&x).unwrap();
                let orth_err = fit.orthogonality_error();
                let dist = fit.distortion();
                let rank = fit.rank(1e-12);
                println!("LOWDIN {orth_err:.17e} {dist:.17e} {rank}");
                print!("Z_STD");
                for v in std_x.z.as_slice() {
                    print!(" {v:.17e}");
                }
                println!();
                print!("EVALS");
                for v in &fit.eigenvalues {
                    print!(" {v:.17e}");
                }
                println!();
                print!("W");
                for v in fit.w.as_slice() {
                    print!(" {v:.17e}");
                }
                println!();
                print!("Z_ORTH");
                for v in fit.z_orth.as_slice() {
                    print!(" {v:.17e}");
                }
                println!();
            }
            "AUDIT" => {
                let n: usize = next_tok(&mut it).parse().unwrap();
                let b: usize = next_tok(&mut it).parse().unwrap();
                let gate: f64 = next_tok(&mut it).parse().unwrap();
                let dims: Vec<usize> = (0..b).map(|_| next_tok(&mut it).parse().unwrap()).collect();
                let blocks: Vec<Matrix> = dims.iter().map(|&d| read_matrix(&mut it, n, d)).collect();
                let res = audit_blocks(&blocks, gate).unwrap();
                print!("AUDIT {:.17e}", res.max_off_diagonal);
                for i in 0..b {
                    for j in (i + 1)..b {
                        print!(" {:.17e}", res.matrix[(i, j)]);
                    }
                }
                println!();
            }
            other => panic!("unknown cmd: {other}"),
        }
    }
}
