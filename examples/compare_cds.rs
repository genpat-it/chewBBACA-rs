use std::path::Path;
use std::collections::HashSet;

fn main() {
    let genome = std::env::args().nth(1).expect("Usage: compare_cds <genome.fasta> <training.trn>");
    let trn = std::env::args().nth(2).expect("Usage: compare_cds <genome.fasta> <training.trn>");
    
    let genome_path = Path::new(&genome);
    let trn_path = Path::new(&trn);
    
    let training = chewcall::prodigal_rs::Training::from_file(trn_path)
        .expect("Failed to load training");
    let rs_output = chewcall::prodigal_rs::predict_genome(genome_path, &training)
        .expect("Rust prodigal failed");
    
    let c_output = std::process::Command::new("/home/IZSNT/a.deruvo/miniconda3/envs/chewbbacca_gpu/bin/prodigal")
        .args(&["-i", &genome, "-t", &trn, "-g", "11", "-d", "/dev/stdout", "-o", "/dev/null", "-q"])
        .output()
        .expect("C prodigal failed");
    
    fn parse_cds(data: &[u8]) -> Vec<(String, String)> {
        let text = String::from_utf8_lossy(data);
        let mut result = Vec::new();
        let mut header = String::new();
        let mut seq = String::new();
        for line in text.lines() {
            if line.starts_with('>') {
                if !seq.is_empty() {
                    result.push((header.clone(), seq.clone()));
                    seq.clear();
                }
                header = line.to_string();
            } else if !line.starts_with('#') && !line.is_empty() {
                seq.push_str(line);
            }
        }
        if !seq.is_empty() {
            result.push((header, seq));
        }
        result
    }
    
    let rs_cds = parse_cds(&rs_output);
    let c_cds = parse_cds(&c_output.stdout);
    
    eprintln!("Rust CDS: {}", rs_cds.len());
    eprintln!("C CDS:    {}", c_cds.len());
    
    let rs_seqs: HashSet<&str> = rs_cds.iter().map(|(_, s)| s.as_str()).collect();
    let c_seqs: HashSet<&str> = c_cds.iter().map(|(_, s)| s.as_str()).collect();
    
    let in_both = rs_seqs.intersection(&c_seqs).count();
    let rs_only = rs_seqs.difference(&c_seqs).count();
    let c_only = c_seqs.difference(&rs_seqs).count();
    
    eprintln!("In both:  {}", in_both);
    eprintln!("Rust only: {}", rs_only);
    eprintln!("C only:    {}", c_only);
    
    if c_only > 0 {
        eprintln!("\nC-only CDS (first 30):");
        let c_only_set: HashSet<&str> = c_seqs.difference(&rs_seqs).cloned().collect();
        let mut count = 0;
        for (h, s) in &c_cds {
            if c_only_set.contains(s.as_str()) && count < 30 {
                eprintln!("  {} (len={})", h, s.len());
                count += 1;
            }
        }
    }
    
    if rs_only > 0 {
        eprintln!("\nRust-only CDS (first 10):");
        let rs_only_set: HashSet<&str> = rs_seqs.difference(&c_seqs).cloned().collect();
        let mut count = 0;
        for (h, s) in &rs_cds {
            if rs_only_set.contains(s.as_str()) && count < 10 {
                eprintln!("  {} (len={})", h, s.len());
                count += 1;
            }
        }
    }
}
