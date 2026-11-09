from frontend.math_markdown import normalize_math


def test_inline_parenthesis_delimiters_become_dollars():
    text = r"La energía en reposo es \(E = mc^2\)."
    assert normalize_math(text) == "La energía en reposo es $E = mc^2$."


def test_bracket_delimiters_become_display_math():
    result = normalize_math(r"\[ a^2 + b^2 = c^2 \]")
    assert result.strip() == "$$\na^2 + b^2 = c^2\n$$"


def test_equation_environment_is_wrapped():
    result = normalize_math(r"\begin{equation} x = \frac{-b}{2a} \end{equation}")
    assert "\\begin{equation}" not in result
    assert result.strip() == "$$\nx = \\frac{-b}{2a}\n$$"


def test_align_environment_is_kept_inside_display_math():
    result = normalize_math(r"\begin{align} a &= b \\ c &= d \end{align}")
    assert "\\begin{align}" in result
    assert result.strip().startswith("$$")
    assert result.strip().endswith("$$")


def test_eqnarray_is_converted_to_aligned():
    result = normalize_math(r"\begin{eqnarray} a &=& b \\ c &=& d \end{eqnarray}")
    assert "eqnarray" not in result
    assert "\\begin{aligned}" in result


def test_existing_dollar_math_is_untouched():
    text = r"Ya está: $x^2$ y $$\sum_i x_i$$"
    assert normalize_math(text) == text


def test_code_fences_are_untouched():
    text = "```python\nprint(r'\\(x\\)')\n```"
    assert normalize_math(text) == text


def test_aligned_spacing_is_not_broken():
    text = r"$$\begin{aligned} a &= b \\[2mm] c &= d \end{aligned}$$"
    assert normalize_math(text) == text


def test_escaped_dollars_are_not_math():
    text = r"cuesta \$5 y \$10"
    assert normalize_math(text) == text


def test_plain_text_is_unchanged():
    assert normalize_math("Sin fórmulas.") == "Sin fórmulas."


def test_mixed_delimiters_all_render():
    text = r"Primero \(a\), después \[b\] y al final \(c\)."
    result = normalize_math(text)
    assert "\\(" not in result
    assert "\\[" not in result
    assert "$a$" in result
    assert "$$\nb\n$$" in result
    assert "$c$" in result
