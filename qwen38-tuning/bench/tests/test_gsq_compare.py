"""Guard against the historical wrong-artifact and false-depth measurements."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gsq_compare as comparison


def test_gsq_launch_uses_candidate_without_nvfp4_or_external_drafter():
    argv = comparison.llama_argv('gsq', 65536, 'none', '7600,15000')
    assert argv[argv.index('-m') + 1].endswith('Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf')
    assert argv[argv.index('--spec-type') + 1] == 'none'
    assert argv[argv.index('-c') + 1] == '65536'
    assert argv.count('-m') == 1
    assert '-md' not in argv
    assert not any('VERY-LOW' in arg for arg in argv)


def test_dirk_uses_actual_download_not_sharp_on_nvfp4():
    argv = comparison.llama_argv('dirk', 65536, 'none', '7600,15000')
    assert Path(argv[argv.index('-m') + 1]).name == 'Dirk-Qwen3.8-27B-UD-Q6_K.gguf'
    assert '--chat-template-file' not in comparison.apply_template(argv, 'stock')
    assert comparison.artifact_metadata('dirk')['quant'] == 'UD-Q6_K'
    comparison.verify_artifact_digest('dirk', '06601c59c3dd1924c209603b3fc8369531127aa54974e1265a381c850bc37e0d')
    with pytest.raises(ValueError):
        comparison.verify_artifact_digest('dirk', '0' * 64)


def test_explicit_tensor_split_is_frozen_instead_of_following_desktop_vram(monkeypatch, capsys):
    import json
    monkeypatch.setattr(sys, 'argv', ['gsq_compare.py', '--artifact', 'turbo',
                                     '--tensor-split', '8500,15468', '--what-if'])
    monkeypatch.setattr(comparison, 'gpu_snapshot', lambda: [
        '0, uuid0, card0, 12282, 1160, 10838, 616.92',
        '1, uuid1, card1, 16311, 71, 15980, 616.92'])
    comparison.main()
    argv = json.loads(capsys.readouterr().out)['argv']
    assert argv[argv.index('-ts') + 1] == '8500,15468'


def test_unknown_artifact_cannot_silently_become_the_incumbent():
    with pytest.raises(ValueError):
        comparison.llama_argv('misspelled', 65536, 'none', '7600,15000')


def test_q4_candidates_are_distinct_from_q6_and_keep_exact_quant_identity():
    swift = comparison.llama_argv('swift_q4km', 65536, 'mtp-ngram', '9000,15000', 24)
    turbo = comparison.llama_argv('turbo_mtp_q4km', 65536, 'mtp-ngram', '9000,15000', 24)
    assert Path(swift[swift.index('-m') + 1]).name == 'Swift-Qwen3.8-27B-Q4_K_M.gguf'
    assert Path(turbo[turbo.index('-m') + 1]).name.endswith('MAX-MTP-Q4_K_M.gguf')
    assert comparison.artifact_metadata('swift_q4km')['quant'] == 'Q4_K_M'
    assert comparison.artifact_metadata('turbo_mtp_q4km')['quant'] == 'MTP-Q4_K_M'
    assert swift[swift.index('--spec-draft-n-max') + 1] == '3'
    assert turbo[turbo.index('--spec-ngram-mod-n-match') + 1] == '24'
    comparison.verify_artifact_digest('swift_q4km', 'ad5811e291431bd0de1cec0c4004a5eac98daee9850882edac69a823209e88ab')
    with pytest.raises(ValueError):
        comparison.verify_artifact_digest('swift_q4km', '0' * 64)
    comparison.verify_artifact_digest('turbo_mtp_q4km', 'bc7a6cf2bcc78d1190aaf04d1ab1c5cb845b6ff23aa0e7d24fe0d2ea6d3a7c7c')
    with pytest.raises(ValueError):
        comparison.verify_artifact_digest('turbo_mtp_q4km', '0' * 64)
    swift_q6 = comparison.llama_argv('swift',65536,'mtp-ngram','9000,15000',24)
    assert swift[swift.index('-m') + 1] != swift_q6[swift_q6.index('-m') + 1]


def test_swift_launch_selects_the_pinned_q6_artifact_without_fallback():
    argv = comparison.llama_argv('swift', 65536, 'none', '7600,15000')
    model = argv[argv.index('-m') + 1]
    assert model.endswith(r'Swift-Qwen3.8-27B-Q6_K\Swift-Qwen3.8-27B-Q6_K.gguf')
    assert 'VERY-LOW' not in model
    assert argv[argv.index('--alias') + 1] == 'Qwen3.8-27B-swift-none'


def test_turbo_launch_selects_the_pinned_mtp_q6_artifact_without_fallback():
    argv = comparison.llama_argv('turbo', 65536, 'none', '7600,15000')
    model = argv[argv.index('-m') + 1]
    assert model.endswith(
        r'DavidAU-Qwen3.8-27B-TURBO-Q6_K\Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q6_K.gguf')
    assert 'VERY-LOW' not in model
    assert argv[argv.index('--alias') + 1] == 'Qwen3.8-27B-turbo-none'


def test_pinned_candidate_digest_mismatch_fails_loudly():
    comparison.verify_artifact_digest(
        'swift', '7f4de8abd5446c08b0f975a1b38e43d02639f4ae9ecdd2ddfd5c5b0f612bda59')
    comparison.verify_artifact_digest(
        'turbo', 'ac011aabe685edbdf542e49351eb6c76c0e5531408f2507f2235ab10931e23a5')
    with pytest.raises(ValueError):
        comparison.verify_artifact_digest('swift', '0' * 64)


def test_candidate_metadata_keeps_weight_family_and_quant_separate():
    swift = comparison.artifact_metadata('swift')
    assert swift['artifact_family'] == 'UkisAI Swift-Qwen3.8-27B'
    assert swift['quant'] == 'Q6_K'
    assert swift['upstream_revision'] == 'eb0e3a7dc70643c2ab920f5133750d02c20849ff'
    assert comparison.artifact_metadata('turbo')['quant'] == 'MTP-Q6_K'
    with pytest.raises(ValueError):
        comparison.artifact_metadata('not-a-model')


def test_exact_answer_fixture_rejects_explanation_and_accepts_requested_integer():
    assert comparison.exact_text_match(' 120\n', '120') is True
    assert comparison.exact_text_match('The answer is 120.', '120') is False
    assert comparison.exact_text_match('', '120') is False


def test_sharp_changes_only_the_chat_template_in_the_resolved_server_argv():
    stock = comparison.apply_template(
        comparison.llama_argv('nvfp4', 65536, 'none', '7600,15000'), 'stock')
    sharp = comparison.apply_template(
        comparison.llama_argv('nvfp4', 65536, 'none', '7600,15000'), 'sharp')
    assert '--chat-template-file' not in stock
    index = sharp.index('--chat-template-file')
    assert sharp[index + 1].endswith(r'third_party\sharp-template\chat_template.jinja')
    assert sharp[:index] + sharp[index + 2:] == stock


def test_invalid_template_name_fails_instead_of_falling_back_to_stock():
    with pytest.raises(ValueError):
        comparison.apply_template(['llama-server'], 'shrap')


def test_sharp_replaces_an_existing_template_flag_instead_of_duplicating_it():
    argv = ['llama-server', '--chat-template-file', 'current.jinja', '--port', '8080']
    sharp = comparison.apply_template(argv, 'sharp')
    assert sharp.count('--chat-template-file') == 1
    assert 'current.jinja' not in sharp
    assert sharp[sharp.index('--chat-template-file') + 1].endswith(
        r'third_party\sharp-template\chat_template.jinja')


def test_current_preserves_profile_template_and_stock_uses_embedded_template():
    argv = ['llama-server', '--chat-template-file', 'current.jinja', '--port', '8080']
    assert comparison.apply_template(argv, 'current') == argv
    stock = comparison.apply_template(argv, 'stock')
    assert '--chat-template-file' not in stock
    assert stock == ['llama-server', '--port', '8080']


def test_stock_and_sharp_remove_inline_template_override_too():
    argv = ['llama-server', '--chat-template', 'inline-jinja', '--port', '8080']
    stock = comparison.apply_template(argv, 'stock')
    sharp = comparison.apply_template(argv, 'sharp')
    assert '--chat-template' not in stock
    assert '--chat-template' not in sharp
    assert sharp.count('--chat-template-file') == 1


def test_template_selection_does_not_silently_change_requested_effort():
    source = {'system': 's', 'messages': [{'role': 'user', 'content': 'task'}]}
    for effort in ('medium', 'xhigh'):
        body = comparison.replay_request(source, 4096, effort)
        assert body['reasoning_effort'] == effort
        assert body['chat_template_kwargs']['reasoning_effort'] == effort


def test_allocated_context_does_not_become_measured_prompt_depth():
    response = {'usage': {'prompt_tokens': 7915, 'completion_tokens': 12},
                'choices': [{'finish_reason': 'length', 'message': {'content': 'partial'}}]}
    row = comparison.response_record(response, 262144, 2.0)
    assert row['prompt_tokens'] == 7915
    assert row['ctx'] == 262144
    assert row['complete'] is False
    assert row['response'] == response


def test_replay_keeps_tools_and_tool_results_without_replacing_history():
    body = {'system': 'original instructions', 'max_tokens': 12,
            'messages': [{'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 't1', 'name': 'Read', 'input': {'path': 'x'}}]},
                         {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 't1', 'content': 'original file contents'}]}],
            'tools': [{'name': 'Read', 'description': 'Read file', 'input_schema': {'type': 'object'}}]}
    replay = comparison.replay_request(body, 4096)
    assert replay['messages'][0]['content'] == 'original instructions'
    assert replay['messages'][1]['tool_calls'][0]['id'] == 't1'
    assert replay['messages'][2]['content'] == 'original file contents'
    assert replay['tools'][0]['function']['name'] == 'Read'
    assert body['max_tokens'] == 12


def test_common_context_changes_exl3_allocation_without_changing_recipe():
    argv = comparison.exl3_argv(147456, 3)
    assert argv[argv.index('-cs') + 1] == '147456'
    assert argv[argv.index('-cq') + 1] == '4'
    assert '-gs 9,15.5 -ndt 3' in argv[argv.index('--extra') + 1]


def test_stream_records_thinking_and_answer_arrival_separately():
    events = [(0.1, {'choices': [{'delta': {'role': 'assistant'}, 'finish_reason': None}]}),
              (2.0, {'choices': [{'delta': {'reasoning_content': 'Check the tests.'}, 'finish_reason': None}]}),
              (5.0, {'choices': [{'delta': {'content': 'Result'}, 'finish_reason': None}]}),
              (6.0, {'choices': [{'delta': {}, 'finish_reason': 'stop'}],
                     'usage': {'prompt_tokens': 100, 'completion_tokens': 12},
                     'timings': {'prompt_ms': 1800, 'predicted_ms': 4000}})]
    response, phases = comparison.collect_stream(events)
    assert response['choices'][0]['message']['reasoning_content'] == 'Check the tests.'
    assert response['choices'][0]['message']['content'] == 'Result'
    assert phases['first_text_s'] == 2.0
    assert phases['first_answer_s'] == 5.0
    assert phases['thinking_observed_s'] == 3.0
    assert phases['request_wall_s'] == 6.0
    assert response['usage']['completion_tokens'] == 12


def test_absolute_request_deadline_is_not_extended_by_stream_activity():
    comparison.enforce_deadline(1199.0, 1200)
    with pytest.raises(TimeoutError):
        comparison.enforce_deadline(1200.001, 1200)


def test_language_guard_preserves_requests_that_need_chinese():
    pieces = ['hello', '確認', 'ชื่อ', '它是', 'file.py']
    assert comparison.conditional_han_bias(pieces, [{'role': 'user', 'content': 'สรุปเป็นภาษาไทย'}]) == {'1': False, '3': False}
    assert comparison.conditional_han_bias(pieces, [{'role': 'user', 'content': 'แปลคำว่า 確認'}]) == {}
    assert comparison.conditional_han_bias(pieces, [{'role': 'user', 'content': 'translate into Chinese'}]) == {}


def test_warm_task_uses_same_history_without_feeding_back_another_models_answer():
    original = {'system': 'saved system', 'messages': [{'role': 'user', 'content': 'saved history'},
                 {'role': 'assistant', 'content': 'saved answer'}, {'role': 'user', 'content': 'old ask'}]}
    cases = comparison.replay_cases(original, 4096, ['merge_intervals'])
    assert len(cases) == 2
    assert cases[1]['body']['messages'][1]['content'] == 'saved history'
    assert cases[1]['body']['messages'][2]['content'] == 'saved answer'
    assert 'merge_intervals' in cases[1]['body']['messages'][-1]['content']
    assert original['messages'][-1]['content'] == 'old ask'
    assert cases[1]['body']['cache_prompt'] is True


def test_warm_task_replaces_real_user_ask_before_trailing_ambient_system_note():
    original = {'messages': [{'role': 'user', 'content': 'saved history'},
                 {'role': 'assistant', 'content': 'saved answer'},
                 {'role': 'user', 'content': 'Summarize in eight Thai sentences'},
                 {'role': 'system', 'content': 'Ambient server-disconnected note'}]}
    cases = comparison.replay_cases(original, 4096, ['merge_intervals'])
    messages = cases[1]['body']['messages']
    assert messages[-1]['content'] == 'System note: Ambient server-disconnected note'
    assert 'merge_intervals' in messages[-2]['content']
    assert all('Summarize in eight Thai sentences' not in m.get('content', '') for m in messages)


def test_info_level_single_model_offload_evidence_does_not_require_debug_logging():
    log = '0.01.913.869 I load_tensors: offloaded 66/66 layers to GPU\n'
    assert comparison.layer_evidence(log, ['server', '-m', 'one.gguf'])['layers'] == [66, 0]
    partial = '0.01.913.869 I load_tensors: offloaded 65/66 layers to GPU\n'
    assert comparison.layer_evidence(partial, ['server', '-m', 'one.gguf'])['layers'] == [65, 1]
    with pytest.raises(ValueError):
        comparison.layer_evidence(log, ['server', '-m', 'one.gguf', '-md', 'other.gguf'])


def test_gsq_ngram_challenger_keeps_its_own_artifact_and_explicit_window():
    argv = comparison.llama_argv('gsq', 147456, 'mtp-ngram', '7400,15500', ngram_match=24)
    assert argv[argv.index('--spec-type') + 1] == 'draft-mtp,ngram-mod'
    assert argv[argv.index('--spec-ngram-mod-n-match') + 1] == '24'
    assert argv[argv.index('-m') + 1].endswith('Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf')


def test_runtime_window_cannot_silently_differ_from_common_window():
    comparison.require_runtime_context(147456, 147456)
    with pytest.raises(ValueError):
        comparison.require_runtime_context(147456, 65536)
    with pytest.raises(ValueError):
        comparison.require_runtime_context(147456, None)


def test_listener_must_belong_to_spawned_process_tree():
    assert comparison.owned_listener(10, 30, {30:20, 20:10})
    assert comparison.owned_listener(10, 10, {})
    assert not comparison.owned_listener(10, 30, {30:99})
    assert not comparison.owned_listener(10, 30, {30:20, 20:30})


def test_sharp_identity_records_embedded_version_and_pinned_digest():
    identity = comparison.template_metadata('sharp')
    assert identity['embedded_version'] == 'qwen3.8-froggeric-v22.5.0'
    assert identity['sha256'] == 'cdff39fb26b60dc90faa292e726655c6b21f62db497846e02e4c4bbab942a84a'
    assert identity['revision'] == '85461fc118aaf25e7319c7ecf2481f944aac3a32'


def test_swift_identity_preserves_conflicting_upstream_checksum_as_evidence():
    metadata = comparison.artifact_metadata('swift')
    assert metadata['hub_lfs_sha256'] == comparison.EXPECTED_DIGESTS['swift']
    assert metadata['upstream_checksum_file_sha256'].startswith('a429f636')
    assert metadata['checksum_status'] == 'upstream-checksum-file-disagrees-with-hub-lfs'


def test_code_fixture_is_executed_instead_of_scored_from_response_shape(tmp_path):
    passing = comparison.verify_code_answer(
        '```python\ndef add(a, b):\n    return a + b\n```',
        'assert add(2, 3) == 5', 'add', tmp_path)
    failing = comparison.verify_code_answer(
        '```python\ndef add(a, b):\n    return a - b\n```',
        'assert add(2, 3) == 5', 'add', tmp_path)
    assert passing['passed'] is True
    assert failing['passed'] is False
    assert failing['error']
