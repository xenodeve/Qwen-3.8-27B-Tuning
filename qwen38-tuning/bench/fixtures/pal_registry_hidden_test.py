"""Hidden PAL registry oracle; never expose this file inside model workspaces."""
import json
import logging
from pathlib import Path

import pytest

import clink.registry as registry_module
from clink.registry import ClinkRegistry, RegistryLoadError


def config(name, command='client-bin', *, roles=None):
    return {
        'name': name,
        'command': command,
        'additional_args': [],
        'env': {},
        'roles': roles if roles is not None else {
            'default': {'prompt_path': 'systemprompts/clink/default.txt', 'role_args': []}},
    }


@pytest.fixture
def sources(tmp_path, monkeypatch):
    bundled = Path(registry_module.PROJECT_ROOT) / 'conf' / 'cli_clients'
    explicit = tmp_path / 'explicit'; legacy = tmp_path / 'legacy'; current = tmp_path / 'current'
    explicit.mkdir(); legacy.mkdir(); current.mkdir()
    monkeypatch.setattr(registry_module, 'CONFIG_DIR', bundled)
    monkeypatch.setattr(registry_module, 'LEGACY_USER_CONFIG_DIR', legacy)
    monkeypatch.setattr(registry_module, 'USER_CONFIG_DIR', current)
    monkeypatch.setattr(registry_module, 'get_env', lambda name: None)
    return bundled, explicit, legacy, current


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.mark.parametrize('which', ['legacy', 'current'])
def test_unsupported_user_config_is_warned_skipped_and_valid_clients_survive(sources, caplog, which):
    _, _, legacy, current = sources
    directory = legacy if which == 'legacy' else current
    unsupported = directory / 'not-supported.json'
    write(unsupported, config('not-supported'))
    write(directory / 'opencode.json', config('opencode', which + '-valid'))
    with caplog.at_level(logging.WARNING, logger='clink.registry'):
        loaded = ClinkRegistry()
    messages = [record.getMessage() for record in caplog.records]
    assert loaded.get_client('opencode').executable[-1].endswith(which + '-valid')
    assert any('not-supported' in message and str(unsupported) in message for message in messages)
    if which == 'legacy':
        assert any(str(legacy) in message and 'deprecated' in message for message in messages)


def test_mixed_user_directory_skips_only_unsupported_file(sources):
    _, _, _, current = sources
    write(current / 'bad-name.json', config('bad-name'))
    write(current / 'cursor.json', config('cursor', 'cursor-user'))
    loaded = ClinkRegistry()
    assert loaded.get_client('cursor').executable[-1].endswith('cursor-user')
    assert 'bad-name' not in [name.lower() for name in loaded.list_clients()]


def test_unsupported_bundled_configuration_remains_fatal(sources, monkeypatch, tmp_path):
    bundled = tmp_path / 'bundled'; bundled.mkdir()
    write(bundled / 'bad.json', config('not-supported'))
    monkeypatch.setattr(registry_module, 'CONFIG_DIR', bundled)
    with pytest.raises(RegistryLoadError, match='not supported'):
        ClinkRegistry()


@pytest.mark.parametrize('as_directory', [False, True])
def test_unsupported_explicit_configuration_remains_fatal(sources, monkeypatch, as_directory):
    _, explicit, _, _ = sources
    target = explicit / 'bad.json' if as_directory else explicit.with_suffix('.json')
    write(target, config('not-supported'))
    monkeypatch.setattr(registry_module, 'get_env', lambda name: str(explicit if as_directory else target))
    with pytest.raises(RegistryLoadError, match='not supported'):
        ClinkRegistry()


def test_malformed_user_file_keeps_existing_skip_behavior_but_schema_invalid_remains_fatal(sources):
    _, _, _, current = sources
    (current / 'malformed.json').write_text('{', encoding='utf-8')
    # read_json_file currently returns None for malformed JSON; this task does not
    # broaden into changing that unrelated policy.
    assert ClinkRegistry().list_clients()
    (current / 'malformed.json').unlink()
    write(current / 'schema.json', {'name':'opencode', 'command': {'not':'a string'}})
    with pytest.raises(Exception):
        ClinkRegistry()


def test_supported_config_with_missing_custom_prompt_remains_fatal(sources):
    _, _, _, current = sources
    write(current / 'incomplete.json', config('opencode', roles={
        'custom': {'prompt_path':'missing-prompt.txt', 'role_args':[]}}))
    with pytest.raises(RegistryLoadError, match='Prompt file not found'):
        ClinkRegistry()
