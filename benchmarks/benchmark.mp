module benchmark.employee_eval
exports { @main }
imports { }
digest "0000000000000000"

heap struct TEmployee {
  field name: Str
  field score: i64
}

;; Validate score is in range [0, 100]
fn @validate_score(%score: i64) -> TResult<i64, Str> meta { } {
  bb0:
    %too_low: bool = icmp.slt { lhs=%score, rhs=const.i64 0 }
    cbr %too_low bb3 bb1
  bb1:
    %too_high: bool = icmp.sgt { lhs=%score, rhs=const.i64 100 }
    cbr %too_high bb3 bb2
  bb2:
    %ok: TResult<i64, Str> = enum.new<Ok> { v=%score }
    ret %ok
  bb3:
    %msg: Str = const.Str "score out of range"
    %err: TResult<i64, Str> = enum.new<Err> { e=%msg }
    ret %err
}

;; Classify grade: A(>=90)=0, B(>=70)=1, C(>=50)=2, F(<50)=3
fn @classify_grade(%score: i64) -> i64 meta { } {
  bb0:
    %lt90: bool = icmp.slt { lhs=%score, rhs=const.i64 90 }
    cbr %lt90 bb1 bb4
  bb1:
    %lt70: bool = icmp.slt { lhs=%score, rhs=const.i64 70 }
    cbr %lt70 bb2 bb5
  bb2:
    %lt50: bool = icmp.slt { lhs=%score, rhs=const.i64 50 }
    cbr %lt50 bb3 bb6
  bb3:
    ret const.i64 3
  bb4:
    ret const.i64 0
  bb5:
    ret const.i64 1
  bb6:
    ret const.i64 2
}

;; Bonus per grade: A=500, B=300, C=100, F=0
fn @compute_bonus(%grade: i64) -> i64 meta { } {
  bb0:
    %is_a: bool = icmp.eq { lhs=%grade, rhs=const.i64 0 }
    cbr %is_a bb4 bb1
  bb1:
    %is_b: bool = icmp.eq { lhs=%grade, rhs=const.i64 1 }
    cbr %is_b bb5 bb2
  bb2:
    %is_c: bool = icmp.eq { lhs=%grade, rhs=const.i64 2 }
    cbr %is_c bb6 bb3
  bb3:
    ret const.i64 0
  bb4:
    ret const.i64 500
  bb5:
    ret const.i64 300
  bb6:
    ret const.i64 100
}

;; Create employee, validate, classify, compute bonus+score*10+name_len
fn @eval_employee(%name: Str, %raw_score: i64) -> i64 meta { uses { @validate_score, @classify_grade, @compute_bonus } } {
  bb0:
    %nb: borrow Str = borrow.shared { v=%name }
    %nlen: i64 = str.len { s=%nb }
    br bb1
  bb1:
    %emp: TEmployee = new TEmployee { name=%name, score=%raw_score }
    %valid: TResult<i64, Str> = call @validate_score { score=%raw_score }
    %grade: i64 = call @classify_grade { score=%raw_score }
    %bonus: i64 = call @compute_bonus { grade=%grade }
    %weighted: i64 = i.mul { lhs=%raw_score, rhs=const.i64 10 }
    %sum1: i64 = i.add { lhs=%bonus, rhs=%weighted }
    %total: i64 = i.add { lhs=%sum1, rhs=%nlen }
    ret %total
}

;; Build array of scores, sort, return length
fn @roster_stats() -> i64 meta { } {
  bb0:
    %arr: Array<i64> = arr.new<i64> { cap=const.i64 8 }
    arr.push { arr=%arr, val=const.i64 85 }
    arr.push { arr=%arr, val=const.i64 72 }
    arr.push { arr=%arr, val=const.i64 45 }
    arr.push { arr=%arr, val=const.i64 93 }
    arr.push { arr=%arr, val=const.i64 68 }
    arr.sort { arr=%arr }
    br bb1
  bb1:
    %sb: borrow Array<i64> = borrow.shared { v=%arr }
    %len: i64 = arr.len { arr=%sb }
    %has72: bool = arr.contains { arr=%sb, val=const.i64 72 }
    ret %len
}

;; Build grade distribution map, return unique grade count
fn @grade_distribution(%g1: i64, %g2: i64, %g3: i64, %g4: i64, %g5: i64) -> i64 meta { } {
  bb0:
    %m: Map<i64, i64> = map.new<i64, i64> { }
    map.set { key=%g1, map=%m, val=const.i64 1 }
    map.set { key=%g2, map=%m, val=const.i64 1 }
    map.set { key=%g3, map=%m, val=const.i64 1 }
    map.set { key=%g4, map=%m, val=const.i64 1 }
    map.set { key=%g5, map=%m, val=const.i64 1 }
    br bb1
  bb1:
    %mb: borrow Map<i64, i64> = borrow.shared { v=%m }
    %mlen: i64 = map.len { map=%mb }
    ret %mlen
}

;; Demonstrate shared ownership: share + 2 clones = 3 refs
fn @ownership_demo() -> i64 meta { } {
  bb0:
    %name: Str = const.Str "Diana"
    %score: i64 = const.i64 93
    %emp: TEmployee = new TEmployee { name=%name, score=%score }
    %shared: shared TEmployee = share { v=%emp }
    %c1: shared TEmployee = clone.shared { v=%shared }
    %c2: shared TEmployee = clone.shared { v=%shared }
    ret const.i64 3
}

;; String operations: total_len + parse
fn @string_ops() -> i64 meta { } {
  bb0:
    %all: Str = const.Str "Alice Bob Charlie Diana Eve"
    %ab: borrow Str = borrow.shared { v=%all }
    %total_len: i64 = str.len { s=%ab }
    %num: Str = const.Str "42"
    %numb: borrow Str = borrow.shared { v=%num }
    %parsed: i64 = str.parse_i64 { s=%numb }
    %sum: i64 = i.add { lhs=%total_len, rhs=%parsed }
    ret %sum
}

fn @main() -> i64 meta { uses { @eval_employee, @classify_grade, @roster_stats, @grade_distribution, @ownership_demo, @string_ops } } {
  bb0:
    %n1: Str = const.Str "Alice"
    %r1: i64 = call @eval_employee { name=%n1, raw_score=const.i64 85 }
    %n2: Str = const.Str "Bob"
    %r2: i64 = call @eval_employee { name=%n2, raw_score=const.i64 72 }
    %n3: Str = const.Str "Charlie"
    %r3: i64 = call @eval_employee { name=%n3, raw_score=const.i64 45 }
    %n4: Str = const.Str "Diana"
    %r4: i64 = call @eval_employee { name=%n4, raw_score=const.i64 93 }
    %n5: Str = const.Str "Eve"
    %r5: i64 = call @eval_employee { name=%n5, raw_score=const.i64 68 }
    br bb1
  bb1:
    %t1: i64 = i.add { lhs=%r1, rhs=%r2 }
    %t2: i64 = i.add { lhs=%t1, rhs=%r3 }
    %t3: i64 = i.add { lhs=%t2, rhs=%r4 }
    %t4: i64 = i.add { lhs=%t3, rhs=%r5 }
    %g1: i64 = call @classify_grade { score=const.i64 85 }
    %g2: i64 = call @classify_grade { score=const.i64 72 }
    %g3: i64 = call @classify_grade { score=const.i64 45 }
    %g4: i64 = call @classify_grade { score=const.i64 93 }
    %g5: i64 = call @classify_grade { score=const.i64 68 }
    %gdist: i64 = call @grade_distribution { g1=%g1, g2=%g2, g3=%g3, g4=%g4, g5=%g5 }
    %rstats: i64 = call @roster_stats { }
    %own: i64 = call @ownership_demo { }
    %sops: i64 = call @string_ops { }
    br bb2
  bb2:
    %s1: i64 = i.add { lhs=%t4, rhs=%gdist }
    %s2: i64 = i.add { lhs=%s1, rhs=%rstats }
    %s3: i64 = i.add { lhs=%s2, rhs=%own }
    %final: i64 = i.add { lhs=%s3, rhs=%sops }
    ret %final
}
