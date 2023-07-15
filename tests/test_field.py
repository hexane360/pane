
import pytest

from pane.field import _split_field_name, rename_field


@pytest.mark.parametrize(('field', 'out'), [
    ('test', ('test',)),
    ('test_snake_Case', ('test', 'snake', 'Case'),),
    ('test-KEBAB-Case', ('test', 'KEBAB', 'Case'),),
    ('TestPascalCase', ('Test', 'Pascal', 'Case'),),
    ('testCamelCase', ('test', 'Camel', 'Case'),),
    ('TEST_SCREAM_CASE', ('TEST', 'SCREAM', 'CASE'),),
    # conservative for now
    ('__test__', ValueError("Unable to interpret field '__test__' for automatic rename")),
])
def test_split_field(field, out):
    if isinstance(out, ValueError):
        with pytest.raises(ValueError, match=out.args[0]):
            _split_field_name(field)
    else:
        assert _split_field_name(field) == out


@pytest.mark.parametrize(('field', 'style', 'out'), [
    ('toPascalCase', 'pascal', 'ToPascalCase'),
    ('To_CAMEL_case', 'camel', 'toCamelCase'),
    ('ToScream_Case', 'scream', 'TO_SCREAM_CASE'),
    ('To_kebab_Case', 'kebab', 'to-kebab-case'),
    ('ToSnakeCase', 'snake', 'to_snake_case'),
])
def test_rename_field(field, style, out):
    assert rename_field(field, style) == out