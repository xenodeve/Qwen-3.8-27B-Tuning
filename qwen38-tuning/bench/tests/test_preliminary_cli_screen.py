"""Preliminary CLI screen: preserve profile and verify actual files, not prose."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from preliminary_cli_screen import frozen_request, seed_workspace, verify_workspace

FIXTURE = Path(__file__).resolve().parents[2] / 'fixtures' / 'code-task-1'


def test_frozen_request_can_preserve_engine_native_language_guard():
    from preliminary_cli_screen import frozen_request
    body = {'messages':[{'role':'user','content':'fix'}], 'tools':[]}
    result = frozen_request(body, None)
    assert 'logit_bias' not in result


def test_exl3_health_normalization_keeps_native_evidence_and_rejects_wrong_identity(tmp_path):
    import pytest
    from preliminary_cli_screen import exl3_observation
    model = tmp_path / 'model-4bpw'; model.mkdir()
    shard = model / 'model-00001.safetensors'; shard.write_bytes(b'weights')
    health = {'ok':True, 'backend':'exl3', 'model':model.name, 'context_length':65536}
    observed = exl3_observation(health, model, shard, 65536)
    assert observed['health']['status'] == 'ok'
    assert observed['native_health'] == health
    assert observed['props']['default_generation_settings']['n_ctx'] == 65536
    assert observed['props']['model_path_source'] == 'owned_argv_and_native_health_name'
    for changed in ({'model':'other'}, {'context_length':262144}, {'ok':False}, {'backend':'other'}):
        with pytest.raises(ValueError):
            exl3_observation(dict(health, **changed), model, shard, 65536)
    with pytest.raises(ValueError):
        exl3_observation(health, model, tmp_path / 'outside.safetensors', 65536)


def test_context_evidence_requires_nonempty_model_usage():
    from preliminary_cli_screen import client_context_valid
    assert client_context_valid([], 65536) is None
    assert client_context_valid([{'type':'result'}], 65536) is False
    assert client_context_valid([{'type':'result','modelUsage':{}}], 65536) is False
    assert client_context_valid([{'type':'result','modelUsage':{'local':{'contextWindow':65536}}}], 65536) is True
    assert client_context_valid([{'type':'result','modelUsage':{'local':{'contextWindow':200000}}}], 65536) is False


def test_request_keeps_tools_and_freezes_selected_profile():
    body = {'messages': [{'role':'user','content':'fix it'}], 'tools':[{'type':'function'}],
            'temperature':0.2, 'max_tokens':32000, 'stream':True}
    result = frozen_request(body, {'123':False})
    assert result['temperature'] == 1.0 and result['seed'] == 29
    assert result['max_tokens'] == 8192 and result['reasoning_effort'] == 'medium'
    assert result['chat_template_kwargs'] == {'reasoning_effort':'medium'}
    assert result['tools'] == body['tools']
    assert result['logit_bias'] == {'123':False}
    assert body['temperature'] == 0.2


def test_seed_excludes_hidden_tests_and_preserves_original(tmp_path):
    work = tmp_path / 'work'
    seed_workspace(FIXTURE, work)
    assert (work / 'inventory/store.py').is_file()
    assert not (work / 'hidden').exists()
    assert not (work / 'BRIEF.md').exists()


def test_unchanged_fixture_fails_hidden_verification_without_workspace_mutation(tmp_path):
    work = tmp_path / 'work'; seed_workspace(FIXTURE, work)
    before = sorted(p.relative_to(work).as_posix() for p in work.rglob('*'))
    result = verify_workspace(work, FIXTURE, tmp_path / 'verify')
    assert result['passed'] is False
    assert result['hidden']['returncode'] == 1
    assert result['visible']['returncode'] == 0
    assert sorted(p.relative_to(work).as_posix() for p in work.rglob('*')) == before


def test_visible_tests_cannot_repair_the_copy_used_for_hidden_verification(tmp_path):
    # A generated test can patch source on disk; that must not change the hidden verdict.
    work = tmp_path / 'work'; seed_workspace(FIXTURE, work)
    source = (work / 'inventory/store.py').read_text()
    patched = source.replace('        self._items[sku] = current - qty',
        '        if qty > current:\n            raise ValueError(f"{sku}: {current}")\n'
        '        self._items[sku] = current - qty')
    patched += '\n    def low_stock(self, threshold):\n        return sorted((k for k, v in self._items.items() if v < threshold), key=lambda k: (self._items[k], k))\n'
    tests = work / 'tests/test_store.py'
    tests.write_text(tests.read_text() + '\ndef test_patch_source_on_disk():\n'
        '    from pathlib import Path\n'
        f'    Path("inventory/store.py").write_text({patched!r})\n')
    result = verify_workspace(work, FIXTURE, tmp_path / 'verify')
    assert result['visible']['returncode'] == 0
    assert result['hidden']['returncode'] == 1
    assert result['passed'] is False


def test_reference_fix_passes_and_out_of_scope_edit_fails(tmp_path):
    work = tmp_path / 'work'; seed_workspace(FIXTURE, work)
    source = work / 'inventory/store.py'
    text = source.read_text()
    text = text.replace('        self._items[sku] = current - qty',
        '        if qty > current:\n            raise ValueError(f"{sku}: {current}")\n'
        '        self._items[sku] = current - qty')
    text += '\n    def low_stock(self, threshold):\n        return sorted((k for k, v in self._items.items() if v < threshold), key=lambda k: (self._items[k], k))\n'
    source.write_text(text)
    result = verify_workspace(work, FIXTURE, tmp_path / 'verify-good')
    assert result['passed'] is True
    assert result['hidden']['tests'] == 7
    (work / 'unexpected.txt').write_text('outside task scope')
    result = verify_workspace(work, FIXTURE, tmp_path / 'verify-bad')
    assert result['passed'] is False
    assert result['scope_violations'] == ['unexpected.txt']
