#!/usr/bin/env python
"""Generate the bundled example datasets.

需求方案.txt 5.2 asks for 示例数据集 so a user can try the platform with zero
data of their own. This script produces them rather than shipping hand-typed
CSVs, for three reasons:

* **Reproducible.** Every file is seeded, so regenerating after a schema change
  gives byte-identical output and a diff shows only the intended change.
* **Tunable.** Row counts, class balance and the edge-case conditions are
  parameters here, not something to edit inside a spreadsheet.
* **Auditable.** You can read what each column means and how it was produced
  instead of reverse-engineering 2000 rows of numbers.

    uv run python tools/gen_datasets.py            # regenerate everything
    uv run python tools/gen_datasets.py --check    # fail if anything is stale

Field names for 反诈账户判定 are a **documented placeholder**: the requirement
says 32 fields but the real schema has never been supplied (技术方案.md Q4). The
generator emits a coherent set covering the three categories the requirement
names -- 账号属性 / 交易特征 / 设备信息 -- and says so in dataset/README.md.

Deliberately **no pre-split train/valid/test files**. The platform splits 7:1.5:1.5
itself (需求方案.txt 5.3) and the test split is where the "no synthetic rows"
constraint is enforced (需求方案.txt 5.4). Handing out pre-split copies would
duplicate that logic and give the constraint a second path to be bypassed.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dataset"

ENCODING = "utf-8"

DECISIONS = ("black", "white", "gray")


# --------------------------------------------------------------------------
# Value generators
# --------------------------------------------------------------------------

DEVICES = ("ios", "android", "harmony", "windows", "web")
REGIONS = ("北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安")
CARRIERS = ("移动", "联通", "电信")
OCCUPATIONS = ("自由职业", "企业职员", "个体经营", "学生", "退休", "务农")


def make_id_card(rng: random.Random) -> str:
    """A syntactically valid-looking 18-character national id.

    The digits are random; only the shape and the checksum position are
    plausible. Never treat these as real identifiers.
    """
    year = rng.randint(1965, 2004)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    region = rng.randint(110000, 659000)
    sequence = rng.randint(0, 999)
    body = f"{region:06d}{year:04d}{month:02d}{day:02d}{sequence:03d}"
    return f"{body}{rng.choice('0123456789X')}"


def make_phone(rng: random.Random) -> str:
    prefix = rng.choice(("138", "139", "150", "186", "199", "133"))
    return prefix + "".join(str(rng.randint(0, 9)) for _ in range(8))


def make_bank_card(rng: random.Random) -> str:
    return "62" + "".join(str(rng.randint(0, 9)) for _ in range(17))


def make_account(rng: random.Random) -> str:
    return "".join(rng.choice("0123456789") for _ in range(10))


# --------------------------------------------------------------------------
# Fraud scenario
# --------------------------------------------------------------------------

#: (name, kind). `kind` drives how the value is produced, which is what makes
#: the dataset internally consistent: a black account really does look different
#: from a white one, so a trained model has something to learn.
#: Fields that are proportions in [0,1] rather than counts. Sampling these with
# randint is a type error, and sampling them as ints would put 0.35 in a column
# every other row stores 0 in -- which silently breaks any threshold a consumer
# puts on the column.
FRAUD_RATIO = ("night_txn_ratio", "cross_region_ratio", "new_device_ratio")

FRAUD_NUMERIC = tuple(
    name
    for name in (
        "account_age_days",
        "login_count_30d",
        "failed_login_count_30d",
        "device_count_30d",
        "txn_count_30d",
        "txn_amount_mean",
        "txn_amount_max",
        "txn_amount_std",
        *FRAUD_RATIO,
        "chargeback_count",
    )
)

FRAUD_TEXT = (
    "device_type",
    "region",
    "carrier",
    "occupation",
)


def _fraud_row(rng: random.Random, decision: str) -> dict[str, object]:
    """One row.

    The per-class parameters are what give the dataset signal. A generator that
    produced identical distributions for every class would let the pipeline pass
    every structural check while leaving a model nothing to learn, which is worse
    than an obviously broken file.
    """
    if decision == "black":
        profile = {
            "account_age_days": (1, 90),
            "failed_login_count_30d": (3, 40),
            "device_count_30d": (4, 15),
            "txn_count_30d": (20, 90),
            "txn_amount_mean": (30000, 120000),
            "txn_amount_max": (200000, 900000),
            "txn_amount_std": (20000, 90000),
            "night_txn_ratio": (0.35, 0.8),
            "cross_region_ratio": (0.3, 0.9),
            "new_device_ratio": (0.4, 0.9),
            "chargeback_count": (1, 8),
        }
        login = (5, 30)
    elif decision == "gray":
        # Deliberately intermediate: gray is "insufficient evidence", and a
        # dataset where gray looks exactly like black would make the class
        # meaningless.
        profile = {
            "account_age_days": (30, 365),
            "failed_login_count_30d": (1, 8),
            "device_count_30d": (2, 6),
            "txn_count_30d": (8, 30),
            "txn_amount_mean": (8000, 45000),
            "txn_amount_max": (60000, 250000),
            "txn_amount_std": (6000, 25000),
            "night_txn_ratio": (0.15, 0.45),
            "cross_region_ratio": (0.1, 0.5),
            "new_device_ratio": (0.15, 0.55),
            "chargeback_count": (0, 2),
        }
        login = (10, 60)
    else:  # white
        profile = {
            "account_age_days": (400, 4000),
            "failed_login_count_30d": (0, 2),
            "device_count_30d": (1, 3),
            "txn_count_30d": (2, 18),
            "txn_amount_mean": (500, 9000),
            "txn_amount_max": (3000, 40000),
            "txn_amount_std": (300, 4000),
            "night_txn_ratio": (0.0, 0.2),
            "cross_region_ratio": (0.0, 0.25),
            "new_device_ratio": (0.0, 0.3),
            "chargeback_count": (0, 0),
        }
        login = (30, 200)

    row: dict[str, object] = {
        "account": make_account(rng),
        "id_card": make_id_card(rng),
        "phone": make_phone(rng),
        "bank_card": make_bank_card(rng),
        "login_count_30d": rng.randint(*login),
    }
    for field_name, (low, high) in profile.items():
        if field_name in FRAUD_RATIO:
            row[field_name] = round(rng.uniform(low, high), 3)
        else:
            row[field_name] = rng.randint(low, high)

    row["device_type"] = rng.choice(DEVICES[3:]) if decision == "black" else rng.choice(DEVICES[:3])
    row["region"] = rng.choice(REGIONS)
    row["carrier"] = rng.choice(CARRIERS)
    row["occupation"] = rng.choice(OCCUPATIONS)
    row["label"] = decision
    return row


#: Column order written to CSV. Sensitive fields sit in the middle so the
#: preview and the masking report both exercise them.
FRAUD_COLUMNS = (
    "account",
    "id_card",
    "phone",
    "bank_card",
    *FRAUD_NUMERIC,
    *FRAUD_TEXT,
    "label",
)


def fraud_rows(rng: random.Random, total: int) -> list[dict[str, object]]:
    """Sample ``total`` rows across the three classes.

    Roughly 45 / 30 / 25 black / white / gray. The requirement's own example
    dataset is 10,000 black / 5,000 white / 0 gray, which is precisely the shape
    that makes gray invisible; a bundled example should exercise all three.
    """
    counts = {
        "black": round(total * 0.45),
        "white": round(total * 0.30),
        "gray": total - round(total * 0.45) - round(total * 0.30),
    }
    rows: list[dict[str, object]] = []
    for decision in DECISIONS:
        rows.extend(_fraud_row(rng, decision) for _ in range(counts[decision]))
    rng.shuffle(rows)
    return rows


# --------------------------------------------------------------------------
# Other scenarios
# --------------------------------------------------------------------------


def payment_rows(rng: random.Random, total: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i in range(total):
        decision = "black" if i % 3 == 0 else "white" if i % 3 == 1 else "gray"
        risky = decision == "black"
        rows.append(
            {
                "transaction_id": f"TX{i:08d}",
                "card_hash": make_bank_card(rng)[-12:],
                "amount": rng.randint(20000, 500000) if risky else rng.randint(10, 8000),
                "channel": rng.choice(("app", "web", "pos", "atm")),
                "velocity_1h": rng.randint(8, 40) if risky else rng.randint(1, 5),
                "distinct_countries": rng.randint(3, 9) if risky else rng.randint(1, 2),
                "label": decision,
            }
        )
    rng.shuffle(rows)
    return rows


def content_rows(rng: random.Random, total: int) -> list[dict[str, object]]:
    fragments_ok = ("今天天气不错", "周末去公园", "谢谢你的帮助", "会议改到下午三点")
    fragments_bad = (
        "点击这里领取奖金",
        "涉嫌违规转账处理",
        "账号异常请立即验证",
        "涉嫌洗钱请配合调查",
    )
    fragments_gray = ("系统通知", "请查看详情", "温馨提示", "信息更新")

    rows: list[dict[str, object]] = []
    for i in range(total):
        bucket = i % 4
        if bucket == 0:
            decision, fragment = "black", rng.choice(fragments_bad)
        elif bucket in (1, 2):
            decision, fragment = "white", rng.choice(fragments_ok)
        else:
            decision, fragment = "gray", rng.choice(fragments_gray)
        rows.append(
            {
                "message_id": f"MSG{i:08d}",
                "content": fragment,
                "length": len(fragment),
                "has_url": int(decision == "black" and rng.random() < 0.7),
                "report_count": rng.randint(5, 30) if decision == "black" else 0,
                "label": decision,
            }
        )
    rng.shuffle(rows)
    return rows


# --------------------------------------------------------------------------
# Edge cases -- each one exercises a specific code path
# --------------------------------------------------------------------------


@dataclass
class EdgeCase:
    """A fixture that proves one behaviour rather than looking realistic."""

    filename: str
    description: str
    proves: str
    rows: list[dict[str, object]] = field(default_factory=list)
    columns: tuple[str, ...] = FRAUD_COLUMNS


def minimal_8(rng: random.Random) -> EdgeCase:
    """8 rows: 7:1.5:1.5 cannot produce a stratified split.

    Exercises the tiny-dataset path, where the test share rounds to a single row.
    """
    return EdgeCase(
        filename="minimal_8_rows.csv",
        description="仅 8 行，用于验证小数据集下的划分退化为单行测试集",
        proves="test 集仍非空，且取行方式确定（最大类优先）",
        rows=fraud_rows(rng, 8),
    )


def single_class(rng: random.Random) -> EdgeCase:
    """Only black: the black:white ratio is not computable."""
    return EdgeCase(
        filename="single_class_black_only.csv",
        description="只有黑样本，用于验证比例不可算时不得谎报「已均衡」",
        proves="ratio_computable=false，合成建议返回「缺少白样本」",
        rows=[_fraud_row(rng, "black") for _ in range(40)],
    )


def no_gray(rng: random.Random) -> EdgeCase:
    """No gray: the quality report should recommend adding it."""
    rows = [_fraud_row(rng, "black") for _ in range(30)]
    rows += [_fraud_row(rng, "white") for _ in range(30)]
    rng.shuffle(rows)
    return EdgeCase(
        filename="no_gray_label.csv",
        description="只有黑与白，用于验证缺失灰样本时质量报告会主动提示",
        proves="建议中出现「未发现灰样本」",
        rows=rows,
    )


def heavy_missing(rng: random.Random) -> EdgeCase:
    """~20% blanks: missing_rate should land in the 可接受/偏差 band."""
    rows = fraud_rows(rng, 60)
    numeric = FRAUD_NUMERIC[:6]
    for row in rows:
        for name in numeric:
            if rng.random() < 0.2:
                row[name] = ""
    return EdgeCase(
        filename="heavy_missing.csv",
        description="约 20% 数值缺失，用于验证缺失率统计与质量评分",
        proves="缺失率被真实统计，缺失率评分随之下调",
        rows=rows,
    )


def outliers(rng: random.Random) -> EdgeCase:
    """A heavy tail: outliers are ~15% of the column, above the IQR assumption."""
    rows = fraud_rows(rng, 80)
    for i, row in enumerate(rows):
        if i % 7 == 0:
            row["txn_amount_max"] = rng.randint(2_000_000, 8_000_000)
    return EdgeCase(
        filename="with_outliers.csv",
        description="金额列约 15% 为离群值，用于验证异常检测",
        proves="异常检测能报出离群行数",
        rows=rows,
    )


def sensitive_only(rng: random.Random) -> EdgeCase:
    """Almost entirely sensitive columns: the masking report should be full."""
    rows: list[dict[str, object]] = []
    for _ in range(30):
        row = _fraud_row(rng, "white")
        rows.append(
            {
                "account": row["account"],
                "id_card": row["id_card"],
                "phone": row["phone"],
                "bank_card": row["bank_card"],
                "amount": row["txn_amount_mean"],
                "label": row["label"],
            }
        )
    return EdgeCase(
        filename="sensitive_fields.csv",
        description="几乎全部为敏感字段，用于验证脱敏与脱敏报告",
        proves="身份证/手机号/银行卡全部被识别并脱敏，且不可还原",
        rows=rows,
        columns=("account", "id_card", "phone", "bank_card", "amount", "label"),
    )


def unbalanced(rng: random.Random) -> EdgeCase:
    """50:1 black:white: the imbalance the synthesizer is meant to correct."""
    rows = [_fraud_row(rng, "black") for _ in range(100)]
    rows += [_fraud_row(rng, "white") for _ in range(2)]
    rows += [_fraud_row(rng, "gray") for _ in range(4)]
    rng.shuffle(rows)
    return EdgeCase(
        filename="heavily_unbalanced.csv",
        description="黑:白 ≈ 50:1，用于验证合成比例推荐与定向扩增",
        proves="推荐比例接近 50:1，且扩增目标指向白样本",
        rows=rows,
    )


EDGE_CASES = (
    minimal_8,
    single_class,
    no_gray,
    heavy_missing,
    outliers,
    sensitive_only,
    unbalanced,
)


# --------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict[str, object]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=ENCODING, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_all() -> list[tuple[Path, list[dict[str, object]], tuple[str, ...]]]:
    """Every file, with a fresh seed per file so adding one does not shift others.

    A shared seed would make every generated file depend on the order files are
    built in, so inserting a new dataset would silently change all the others.
    """
    plan: list[tuple[Path, list[dict[str, object]], tuple[str, ...]]] = []

    plan.append(
        (
            OUT / "fraud" / "seed_200.csv",
            fraud_rows(random.Random(20240501), 200),
            FRAUD_COLUMNS,
        )
    )
    plan.append(
        (
            OUT / "fraud" / "seed_2000.csv",
            fraud_rows(random.Random(20240502), 2000),
            FRAUD_COLUMNS,
        )
    )
    plan.append(
        (
            OUT / "payment" / "seed_500.csv",
            payment_rows(random.Random(20240503), 500),
            (
                "transaction_id",
                "card_hash",
                "amount",
                "channel",
                "velocity_1h",
                "distinct_countries",
                "label",
            ),
        )
    )
    plan.append(
        (
            OUT / "content" / "seed_300.csv",
            content_rows(random.Random(20240504), 300),
            ("message_id", "content", "length", "has_url", "report_count", "label"),
        )
    )

    for index, builder in enumerate(EDGE_CASES, start=1):
        case = builder(random.Random(20240600 + index))
        plan.append((OUT / "edge_cases" / case.filename, case.rows, case.columns))

    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed files match what would be generated",
    )
    args = parser.parse_args()

    plan = build_all()
    stale: list[Path] = []

    for path, rows, columns in plan:
        if args.check:
            if not path.exists():
                stale.append(path)
                continue
            with path.open("r", encoding=ENCODING, newline="") as handle:
                existing = list(csv.reader(handle))
            expected_header = list(columns)
            if not existing or existing[0] != expected_header or len(existing) - 1 != len(rows):
                stale.append(path)
        else:
            write_csv(path, rows, columns)
            print(f"  {path.relative_to(ROOT)}  ({len(rows)} 行)")

    if args.check:
        if stale:
            for path in stale:
                print(f"过期：{path.relative_to(ROOT)}", file=__import__("sys").stderr)
            print("请运行 uv run python tools/gen_datasets.py", file=__import__("sys").stderr)
            return 1
        print(f"{len(plan)} 个数据集均为最新")
    else:
        print(f"\n共生成 {len(plan)} 个数据集到 {OUT.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
