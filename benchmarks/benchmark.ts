interface Employee {
    name: string;
    score: number;
}

type Result<T, E> = { ok: true; value: T } | { ok: false; error: E };

/** Validate score is in range [0, 100] */
function validateScore(score: number): Result<number, string> {
    if (score < 0 || score > 100) {
        return { ok: false, error: "score out of range" };
    }
    return { ok: true, value: score };
}

/** Classify grade: A(>=90)=0, B(>=70)=1, C(>=50)=2, F(<50)=3 */
function classifyGrade(score: number): number {
    if (score >= 90) return 0;
    if (score >= 70) return 1;
    if (score >= 50) return 2;
    return 3;
}

/** Bonus per grade: A=500, B=300, C=100, F=0 */
function computeBonus(grade: number): number {
    switch (grade) {
        case 0: return 500;
        case 1: return 300;
        case 2: return 100;
        default: return 0;
    }
}

/** Create employee, validate, classify, compute bonus+score*10+name_len */
function evalEmployee(name: string, rawScore: number): number {
    const nlen = name.length;
    const emp: Employee = { name, score: rawScore };
    const _valid = validateScore(emp.score);
    const grade = classifyGrade(emp.score);
    const bonus = computeBonus(grade);
    const weighted = emp.score * 10;
    return bonus + weighted + nlen;
}

/** Build array of scores, sort, return length */
function rosterStats(): number {
    const arr: number[] = [];
    arr.push(85);
    arr.push(72);
    arr.push(45);
    arr.push(93);
    arr.push(68);
    arr.sort((a, b) => a - b);
    const _has72 = arr.includes(72);
    return arr.length;
}

/** Build grade distribution map, return unique grade count */
function gradeDistribution(g1: number, g2: number, g3: number, g4: number, g5: number): number {
    const m = new Map<number, number>();
    m.set(g1, 1);
    m.set(g2, 1);
    m.set(g3, 1);
    m.set(g4, 1);
    m.set(g5, 1);
    return m.size;
}

/** Demonstrate shared ownership: freeze + spread = 3 refs */
function ownershipDemo(): number {
    const emp: Employee = { name: "Diana", score: 93 };
    const shared = Object.freeze({ ...emp });
    const _c1 = { ...shared };
    const _c2 = { ...shared };
    return 3;
}

/** String operations: total_len + parse */
function stringOps(): number {
    const all = "Alice Bob Charlie Diana Eve";
    const totalLen = all.length;
    const num = "42";
    const parsed = parseInt(num, 10);
    return totalLen + parsed;
}

function main(): number {
    const r1 = evalEmployee("Alice", 85);
    const r2 = evalEmployee("Bob", 72);
    const r3 = evalEmployee("Charlie", 45);
    const r4 = evalEmployee("Diana", 93);
    const r5 = evalEmployee("Eve", 68);

    const t4 = r1 + r2 + r3 + r4 + r5;

    const g1 = classifyGrade(85);
    const g2 = classifyGrade(72);
    const g3 = classifyGrade(45);
    const g4 = classifyGrade(93);
    const g5 = classifyGrade(68);
    const gdist = gradeDistribution(g1, g2, g3, g4, g5);

    const rstats = rosterStats();
    const own = ownershipDemo();
    const sops = stringOps();

    return t4 + gdist + rstats + own + sops;
}

const result = main();
console.log(result);
process.exit(result % 256);
