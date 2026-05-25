 original_raw = core.clean(raw)
    for raw_variant in _vertical_strip_check_digit_repairs(raw):
        for normalized in core._normalized_candidates(raw_variant):
            quality = core._container_text_quality(normalized)
            final = 0.75 * float(score) + 0.25 * quality
            is_valid = core._is_valid_iso6346(normalized)
            if is_valid:
                final += 0.10
            if raw_variant != original_raw:
                final += 0.08
            if final > best_score:
                best_code = normalized
                best_score = final
                best_valid = is_valid
