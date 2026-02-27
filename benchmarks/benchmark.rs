use std::collections::HashMap;
use std::sync::Arc;

struct Employee {
    name: String,
    score: i64,
}

/// Validate score is in range [0, 100]
fn validate_score(score: i64) -> Result<i64, String> {
    if score < 0 || score > 100 {
        Err("score out of range".to_string())
    } else {
        Ok(score)
    }
}

/// Classify grade: A(>=90)=0, B(>=70)=1, C(>=50)=2, F(<50)=3
fn classify_grade(score: i64) -> i64 {
    if score >= 90 {
        0
    } else if score >= 70 {
        1
    } else if score >= 50 {
        2
    } else {
        3
    }
}

/// Bonus per grade: A=500, B=300, C=100, F=0
fn compute_bonus(grade: i64) -> i64 {
    match grade {
        0 => 500,
        1 => 300,
        2 => 100,
        _ => 0,
    }
}

/// Create employee, validate, classify, compute bonus+score*10+name_len
fn eval_employee(name: &str, raw_score: i64) -> i64 {
    let nlen = name.len() as i64;
    let emp = Employee {
        name: name.to_string(),
        score: raw_score,
    };
    let _valid = validate_score(emp.score);
    let grade = classify_grade(emp.score);
    let bonus = compute_bonus(grade);
    let weighted = emp.score * 10;
    bonus + weighted + nlen
}

/// Build array of scores, sort, return length
fn roster_stats() -> i64 {
    let mut arr: Vec<i64> = Vec::with_capacity(8);
    arr.push(85);
    arr.push(72);
    arr.push(45);
    arr.push(93);
    arr.push(68);
    arr.sort();
    let _has_72 = arr.contains(&72);
    arr.len() as i64
}

/// Build grade distribution map, return unique grade count
fn grade_distribution(g1: i64, g2: i64, g3: i64, g4: i64, g5: i64) -> i64 {
    let mut m: HashMap<i64, i64> = HashMap::new();
    m.insert(g1, 1);
    m.insert(g2, 1);
    m.insert(g3, 1);
    m.insert(g4, 1);
    m.insert(g5, 1);
    m.len() as i64
}

/// Demonstrate shared ownership: share + 2 clones = 3 refs
fn ownership_demo() -> i64 {
    let emp = Employee {
        name: "Diana".to_string(),
        score: 93,
    };
    let shared = Arc::new(emp);
    let _c1 = Arc::clone(&shared);
    let _c2 = Arc::clone(&shared);
    3
}

/// String operations: total_len + parse
fn string_ops() -> i64 {
    let all = "Alice Bob Charlie Diana Eve";
    let total_len = all.len() as i64;
    let num = "42";
    let parsed: i64 = num.parse().unwrap();
    total_len + parsed
}

fn main() {
    let r1 = eval_employee("Alice", 85);
    let r2 = eval_employee("Bob", 72);
    let r3 = eval_employee("Charlie", 45);
    let r4 = eval_employee("Diana", 93);
    let r5 = eval_employee("Eve", 68);

    let t4 = r1 + r2 + r3 + r4 + r5;

    let g1 = classify_grade(85);
    let g2 = classify_grade(72);
    let g3 = classify_grade(45);
    let g4 = classify_grade(93);
    let g5 = classify_grade(68);
    let gdist = grade_distribution(g1, g2, g3, g4, g5);

    let rstats = roster_stats();
    let own = ownership_demo();
    let sops = string_ops();

    let result = t4 + gdist + rstats + own + sops;
    println!("{}", result);
    std::process::exit((result % 256) as i32);
}
