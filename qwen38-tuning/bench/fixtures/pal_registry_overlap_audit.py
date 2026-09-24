"""Post-run audit of a predeclared requirement missed by the original oracle.

An explicitly selected file/directory must remain authoritative even when its
physical path overlaps a normal user-config directory. Not model feedback.
"""
import json

import pytest
import clink.registry as registry_module
from clink.registry import ClinkRegistry, RegistryLoadError


@pytest.mark.parametrize('kind', ['file', 'directory'])
@pytest.mark.parametrize('user_source', ['legacy', 'current'])
def test_explicit_path_inside_user_directory_remains_fatal(tmp_path, monkeypatch, kind, user_source):
    legacy = tmp_path / 'legacy'
    current = tmp_path / 'current'
    legacy.mkdir(); current.mkdir()
    monkeypatch.setattr(registry_module, 'LEGACY_USER_CONFIG_DIR', legacy)
    monkeypatch.setattr(registry_module, 'USER_CONFIG_DIR', current)
    selected = legacy if user_source == 'legacy' else current
    path = selected / 'unsupported.json'
    path.write_text(json.dumps({
        'name':'unsupported-audit-client', 'command':'never-launched',
        'roles':{'default':{'prompt_path':'systemprompts/clink/default.txt'}},
    }), encoding='utf-8')
    target = path if kind == 'file' else selected
    monkeypatch.setattr(registry_module, 'get_env', lambda name: str(target))
    with pytest.raises(RegistryLoadError, match='not supported'):
        ClinkRegistry()
