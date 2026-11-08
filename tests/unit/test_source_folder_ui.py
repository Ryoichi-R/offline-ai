"""Execute the real UI methods with a fake DOM and transport, without models."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest

HTML = Path(__file__).resolve().parents[2] / "_internal/web/index.html"
HARNESS = r"""
const fs=require('node:fs'),vm=require('node:vm');
const script=fs.readFileSync(process.argv[1],'utf8').match(/<script>([\s\S]*?)<\/script>/)[1];
const scenario=process.argv[2];
const context={document:{addEventListener(){}},window:{},console};
vm.createContext(context);vm.runInContext(script+'\nglobalThis.fixture={ui,state,api};',context);
const {ui,state,api}=context.fixture;
const element=()=>({value:'',textContent:'',disabled:false,hidden:false,addEventListener(){}});
ui.els=Object.fromEntries(['sourceFolderInput','sourceFolderPickBtn','sourceFolderSaveBtn','sourceFolderActive','sourceFolderMessage'].map(k=>[k,element()]));
const calls=[];
const data={activePath:'/old',savedPath:'/saved',restartRequired:true,available:true,pickerAvailable:true};
api.sourceFolder=async path=>{calls.push(path===undefined?'GET':path);if(scenario==='error')throw new Error('保存できません');return data;};
api.pickSourceFolder=async()=>({path:'/picked',cancelled:scenario==='cancel'});
(async()=>{
 if(scenario==='load-edit')ui.els.sourceFolderInput.value='/typing';
 if(scenario.startsWith('load'))await ui.loadSourceFolder();
 else {
  ui.els.sourceFolderInput.value='/entered';
  if(scenario==='busy')state.sourceFolderBusy=true;
  await ui.changeSourceFolder(scenario==='pick'||scenario==='cancel');
 }
 process.stdout.write(JSON.stringify({calls,input:ui.els.sourceFolderInput.value,active:ui.els.sourceFolderActive.textContent,message:ui.els.sourceFolderMessage.textContent,busy:state.sourceFolderBusy,disabled:ui.els.sourceFolderSaveBtn.disabled}));
})().catch(e=>{console.error(e);process.exitCode=1;});
"""


def run(scenario):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required")
    result = subprocess.run(
        [node, "-e", HARNESS, str(HTML), scenario], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_reload_distinguishes_current_and_saved_folder_and_keeps_typed_edits():
    value = run("load")
    assert value["active"] == "/old" and value["input"] == "/saved"
    assert "再起動" in value["message"]
    assert run("load-edit")["input"] == "/typing"


def test_save_sends_entered_path_and_shows_restart_requirement():
    value = run("save")
    assert value["calls"] == ["/entered"]
    assert "再起動" in value["message"]
    assert not value["busy"] and not value["disabled"]


@pytest.mark.parametrize("scenario,expected", [("pick", "/picked"), ("cancel", "/entered")])
def test_picker_does_not_save_or_replace_input_on_cancel(scenario, expected):
    value = run(scenario)
    assert value["calls"] == [] and value["input"] == expected
    assert not value["busy"] and not value["disabled"]


def test_error_preserves_input_and_restores_controls():
    value = run("error")
    assert value["input"] == "/entered" and value["message"] == "保存できません"
    assert not value["disabled"] and not value["busy"]


def test_busy_action_cannot_duplicate_save_requests():
    assert run("busy")["calls"] == []
