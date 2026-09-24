"""Parent-only, containerized verification for the isolated PAL registry task."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil

from fixture_test_tool import run_sandboxed_pytest
from recorded_session import digest

_ALLOWED={'clink/registry.py','tests/test_user_config_dir_rename.py'}
_IGNORED_PARTS={'.git','__pycache__','.pytest_cache','.ruff_cache','.codex','.venv'}


def source_inventory(directory: str|Path) -> dict[str,dict[str,object]]:
    root=Path(directory).resolve();result={}
    for current,dirs,files in os.walk(root,topdown=True):
        dirs[:]=sorted(name for name in dirs if name not in _IGNORED_PARTS)
        for name in sorted(files):
            path=Path(current)/name;relative=path.relative_to(root)
            if any(part in _IGNORED_PARTS for part in relative.parts):continue
            if path.is_symlink():raise ValueError('symlink in PAL workflow source')
            result[relative.as_posix()]={'bytes':path.stat().st_size,'sha256':digest(path)}
    return result


def _counts(text: str) -> dict[str,int]:
    result={key:0 for key in ('tests','failures','errors','skipped')}
    labels={'passed':'tests','failed':'failures','error':'errors','errors':'errors','skipped':'skipped'}
    for number,label in re.findall(r'(\d+)\s+(passed|failed|errors?|skipped)',text):
        result[labels[label]]+=int(number)
    result['tests']+=result['failures']+result['errors']+result['skipped']
    return result


def _run(workspace: Path,image: str,arguments:list[str],output:Path,name:str,
         trusted_test:Path|None=None) -> dict[str,object]:
    stdout_path,stderr_path=output/(name+'.stdout'),output/(name+'.stderr')
    execution=run_sandboxed_pytest(workspace,image,arguments,stdout_path,stderr_path,180,
                                   trusted_test=trusted_test)
    text=stdout_path.read_text(encoding='utf-8',errors='replace')
    return {**_counts(text),**execution,'stdout_ref':stdout_path.name,'stderr_ref':stderr_path.name}


def verify_pal_workspace(workspace: str|Path,baseline: str|Path,sandbox_image:str,
                         hidden_test: str|Path,output: str|Path,
                         *, visible_test_journal: str|Path|None=None) -> dict[str,object]:
    workspace,baseline=Path(workspace).resolve(),Path(baseline).resolve();hidden_test=Path(hidden_test).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);before,after=source_inventory(baseline),source_inventory(workspace)
    changes=sorted(key for key in before.keys()|after.keys() if before.get(key)!=after.get(key));violations=sorted(set(changes)-_ALLOWED)
    # The sandbox mounts submissions read-only. Hidden oracle content is mounted
    # only for the parent run and its stdout never returns to the model.
    hidden=_run(workspace,sandbox_image,[str(hidden_test),'-q'],output,'hidden',hidden_test)
    visible=_run(workspace,sandbox_image,['tests/test_user_config_dir_rename.py','-q'],output,'visible')
    workflow={'tests_file_changed':'tests/test_user_config_dir_rename.py' in changes,
              'red_then_green':False,'test_runs':0,'evidence':{}}
    if visible_test_journal is not None:
        journal=Path(visible_test_journal).resolve()
        if journal.is_file():
            runs=[json.loads(line) for line in journal.read_text(encoding='utf-8').splitlines() if line.strip()]
            workflow['test_runs']=len(runs);red_seen=False;red_tests=None
            baseline_source=before.get('clink/registry.py');baseline_tests=before.get('tests/test_user_config_dir_rename.py')
            final_source=after.get('clink/registry.py');final_tests=after.get('tests/test_user_config_dir_rename.py')
            for run in runs:
                if run.get('suite_id')!='user-config-focused' or run.get('timed_out') is True:
                    continue
                observed=run.get('workspace_files') or {}
                source_before=observed.get('clink/registry.py')==baseline_source
                tests_changed=observed.get('tests/test_user_config_dir_rename.py') not in (None,baseline_tests)
                if run.get('passed') is False and source_before and tests_changed:
                    red_seen=True;red_tests=observed.get('tests/test_user_config_dir_rename.py')
                elif (run.get('passed') is True and red_seen and not source_before and tests_changed
                      and observed.get('tests/test_user_config_dir_rename.py')==red_tests==final_tests
                      and observed.get('clink/registry.py')==final_source):
                    workflow['red_then_green']=True
                for field in ('stdout_ref','stderr_ref'):
                    ref=run.get(field)
                    path=journal.parent/ref if isinstance(ref,str) else None
                    if path is not None and path.is_file() and path.parent==journal.parent:
                        workflow['evidence'][ref]={'bytes':path.stat().st_size,'sha256':digest(path)}
            workflow['evidence'][journal.name]={'bytes':journal.stat().st_size,'sha256':digest(journal)}
    passed=(not violations and hidden['returncode']==0 and visible['returncode']==0
            and not hidden['timed_out'] and not visible['timed_out']
            and workflow['tests_file_changed'] and workflow['red_then_green'])
    result={'passed':passed,'returncode':0 if passed else 1,'hidden':hidden,'visible':visible,
            'changed_files':changes,'scope_violations':violations,'workflow':workflow}
    (output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result
