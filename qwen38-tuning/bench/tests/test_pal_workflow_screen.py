"""PAL workflow fixture and parent-only verifier contracts."""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from pal_workflow_screen import verify_pal_workspace


ALLOWED = {'clink/registry.py', 'tests/test_user_config_dir_rename.py'}


def test_parent_verifier_rejects_unchanged_baseline_and_preserves_workspace(tmp_path):
    template = Path('C:/Users/xenod/AppData/Local/Temp/qwen-pal-workflow-jtWZBE/template')
    if not template.is_dir():
        return
    work = tmp_path / 'work'; shutil.copytree(template, work)
    before = (work / 'clink/registry.py').read_bytes()
    result = verify_pal_workspace(
        work, template, 'qwen-pal-test:20260921',
        Path('C:/AI/qwen38-tuning/bench/fixtures/pal_registry_hidden_test.py'),
        tmp_path / 'verify')
    assert result['passed'] is False
    assert result['hidden']['returncode'] == 1
    assert (work / 'clink/registry.py').read_bytes() == before


def test_parent_verifier_requires_model_tests_and_red_then_green_mcp_evidence(tmp_path):
    template = Path('C:/Users/xenod/AppData/Local/Temp/qwen-pal-workflow-jtWZBE/template')
    if not template.is_dir():
        return
    work = tmp_path/'work'; shutil.copytree(template,work)
    # A source-only change is never enough to claim the requested workflow.
    source=work/'clink/registry.py';source.write_text(source.read_text()+'\n# candidate change\n')
    journal=tmp_path/'visible-tests.jsonl';journal.write_text(
        json.dumps({'suite_id':'user-config-focused','passed':True,'stdout_ref':'a','stderr_ref':'b'})+'\n')
    result=verify_pal_workspace(work,template,'qwen-pal-test:20260921',
        Path('C:/AI/qwen38-tuning/bench/fixtures/pal_registry_hidden_test.py'),tmp_path/'verify',
        visible_test_journal=journal)
    assert result['passed'] is False
    assert result['workflow']['tests_file_changed'] is False
    assert result['workflow']['red_then_green'] is False


def test_parent_verifier_rejects_out_of_scope_changes_before_quality_pass(tmp_path):
    template = Path('C:/Users/xenod/AppData/Local/Temp/qwen-pal-workflow-jtWZBE/template')
    if not template.is_dir():
        return
    work = tmp_path / 'work'; shutil.copytree(template, work)
    (work / 'README.md').write_text('unrelated')
    result = verify_pal_workspace(
        work, template, 'qwen-pal-test:20260921',
        Path('C:/AI/qwen38-tuning/bench/fixtures/pal_registry_hidden_test.py'),
        tmp_path / 'verify')
    assert 'README.md' in result['scope_violations']
    assert result['passed'] is False
