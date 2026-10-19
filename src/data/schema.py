"""Label schemas for CORD and SROIE, and the unified output JSON.

The two datasets are complementary and are trained as two separate models:
CORD covers line items and the totals breakdown but carries no vendor, date
or address; SROIE carries exactly those and no line items. Their label sets
overlap only on the total. The extraction system runs both and merges the
results into one JSON document (see UNIFIED_SCHEMA).
"""
from __future__ import annotations

# --- CORD -------------------------------------------------------------
# All 29 categories present in naver-clova-ix/cord-v2, ordered by training
# frequency. Kept complete so micro-F1 stays comparable with published CORD
# results, which use the full label set.
CORD_CATEGORIES: tuple[str, ...] = (
    "menu.nm", "menu.price", "menu.cnt", "total.total_price",
    "sub_total.subtotal_price", "total.cashprice", "total.changeprice",
    "sub_total.tax_price", "menu.unitprice", "menu.sub.nm",
    "total.menuqty_cnt", "menu.discountprice", "total.creditcardprice",
    "sub_total.service_price", "sub_total.etc", "sub_total.discount_price",
    "menu.sub.cnt", "menu.sub.price", "total.emoneyprice",
    "total.menutype_cnt", "menu.num", "total.total_etc", "menu.etc",
    "menu.sub.unitprice", "menu.vatyn", "sub_total.othersvc_price",
    "void_menu.nm", "void_menu.price", "menu.itemsubtotal",
)

# Categories with no instances in the CORD test split. Their F1 is undefined
# rather than zero; a macro average that includes them silently scores five
# of twenty-nine classes at 0 and understates the model by ~17%. They stay
# in training (dropping them would break comparability with the literature)
# but must be excluded from any macro average and reported separately.
CORD_ABSENT_FROM_TEST: frozenset[str] = frozenset({
    "menu.sub.unitprice", "menu.vatyn", "sub_total.othersvc_price",
    "void_menu.nm", "void_menu.price",
})

# Categories too rare to learn: fewer than 20 training instances.
# menu.itemsubtotal is the pathological case - 1 train, 6 test - so it is
# graded on examples the model had almost no chance to learn from.
CORD_TOO_RARE: frozenset[str] = frozenset({"menu.etc", "menu.itemsubtotal"})

# Fields that actually reach the output JSON and are frequent enough to be
# present at every training-set size. The low-data curve is plotted on these:
# on the full label set, a drop at n=50 confounds "needs more data" with
# "this class never appeared in the sample", which is not a learning result.
CORD_CORE_FIELDS: tuple[str, ...] = (
    "menu.nm", "menu.cnt", "menu.price", "menu.unitprice",
    "sub_total.subtotal_price", "sub_total.tax_price",
    "sub_total.service_price", "sub_total.discount_price",
    "total.total_price", "total.cashprice", "total.changeprice",
)

# --- SROIE ------------------------------------------------------------
SROIE_CATEGORIES: tuple[str, ...] = ("company", "date", "address", "total")
SROIE_CORE_FIELDS: tuple[str, ...] = SROIE_CATEGORIES


def bio_labels(categories: tuple[str, ...]) -> list[str]:
    """BIO label list: O first, then B-/I- per category. Index 0 is O."""
    out = ["O"]
    for c in categories:
        out += [f"B-{c}", f"I-{c}"]
    return out


CORD_LABELS = bio_labels(CORD_CATEGORIES)          # 59 labels
SROIE_LABELS = bio_labels(SROIE_CATEGORIES)        # 9 labels

CORD_LABEL2ID = {l: i for i, l in enumerate(CORD_LABELS)}
CORD_ID2LABEL = {i: l for l, i in CORD_LABEL2ID.items()}
SROIE_LABEL2ID = {l: i for i, l in enumerate(SROIE_LABELS)}
SROIE_ID2LABEL = {i: l for l, i in SROIE_LABEL2ID.items()}

# --- unified output ---------------------------------------------------
# How each model's categories map into the final document. CORD supplies
# everything below the vendor block; SROIE supplies the vendor block and a
# second opinion on the total.
UNIFIED_SCHEMA = {
    "vendor": {
        "name": ["sroie:company"],
        "address": ["sroie:address"],
    },
    "transaction": {
        "date": ["sroie:date"],
    },
    "line_items": {          # grouped by CORD group_id
        "description": ["cord:menu.nm", "cord:menu.sub.nm"],
        "quantity": ["cord:menu.cnt", "cord:menu.sub.cnt"],
        "unit_price": ["cord:menu.unitprice", "cord:menu.sub.unitprice"],
        "total_price": ["cord:menu.price", "cord:menu.sub.price"],
        "discount": ["cord:menu.discountprice"],
    },
    "totals": {
        "subtotal": ["cord:sub_total.subtotal_price"],
        "tax": ["cord:sub_total.tax_price"],
        "service_charge": ["cord:sub_total.service_price"],
        "discount": ["cord:sub_total.discount_price"],
        # Both models predict this; SROIE's is the cross-check.
        "grand_total": ["cord:total.total_price", "sroie:total"],
    },
    "payment": {
        "cash": ["cord:total.cashprice"],
        "card": ["cord:total.creditcardprice"],
        "e_money": ["cord:total.emoneyprice"],
        "change": ["cord:total.changeprice"],
    },
}


def describe() -> str:
    lines = [
        f"CORD : {len(CORD_CATEGORIES)} categories -> {len(CORD_LABELS)} BIO labels",
        f"       {len(CORD_ABSENT_FROM_TEST)} absent from test (excluded from macro)",
        f"       {len(CORD_TOO_RARE)} too rare to learn (<20 train instances)",
        f"       {len(CORD_CORE_FIELDS)} core fields (low-data curve)",
        f"SROIE: {len(SROIE_CATEGORIES)} categories -> {len(SROIE_LABELS)} BIO labels",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
