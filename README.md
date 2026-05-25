def _vertical_strip_check_digit_repairs(raw: str) -> list[str]:
    text = core.clean(raw)
    if not text:
        return []

    out = [text]
    for i in range(max(1, len(text) - 10)):
        part = text[i:i + 12]
        if len(part) != 12:
            continue
        if part[:4].isalpha() and part[4:].isdigit() and part[10] == "1":
            repaired = text[:i] + part[:10] + part[11] + text[i + 12:]
            if repaired not in out:
                out.append(repaired)

    return out


def _iso_correct_owner_confusions(owner: str, base6: str, check_digit: str) -> tuple[str, bool]:
    if len(owner) != 4 or len(base6) != 6 or len(check_digit) != 1:
        return owner, False
    if not base6.isdigit() or not check_digit.isdigit():
        return owner, False
    if core._is_valid_iso6346(f"{owner}{base6}{check_digit}"):
        return owner, False

    choices: list[list[str]] = []
    has_confusable = False
    for ch in owner:
        group = next((grp for grp in LETTER_CONFUSION_GROUPS if ch in grp), None)
        if group is None:
            choices.append([ch])
            continue
        has_confusable = True
        choices.append(sorted(group))

    if not has_confusable:
        return owner, False

    valid: list[str] = []
    for candidate_chars in product(*choices):
        candidate_owner = "".join(candidate_chars)
        if candidate_owner == owner:
            continue
        if core._is_valid_iso6346(f"{candidate_owner}{base6}{check_digit}"):
            valid.append(candidate_owner)

    if len(valid) == 1:
        return valid[0], True
    return owner, False
