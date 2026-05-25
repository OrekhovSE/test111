def _is_wrong_test_item(item: dict[str, Any]) -> bool:
    if item.get("error") or item.get("status_code") != 200:
        return True

    target_length = _item_target_length(item)
    predicted = _normalize_for_target(
        str(item.get("normalized_predicted") or item.get("predicted_text") or ""),
        target_length,
    )
    expected_text = _normalize_for_target(str(item.get("expected_text") or ""), target_length)

    if predicted in {"", "NOTFOUND"}:
        return True
    if expected_text and predicted != expected_text:
        return True

    return False


def _build_requests_table_section(items: list[dict[str, Any]], *, include_type: bool = True) -> str:
    sorted_items = sorted(
        items,
        key=lambda item: (
            0 if _is_wrong_test_item(item) else 1,
            str(item.get("file_name") or ""),
        ),
    )
    row_html = _build_request_rows_html(sorted_items, include_type=include_type)
