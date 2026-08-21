"""Bounded execution of a masked memory policy with durable transition logs."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Protocol
from pydantic import BaseModel, ConfigDict
from fidmem.actions.environment import EnvironmentTransition, MemoryEnvironment
from fidmem.agent.answerer import AnswerResult, FrozenAnswerer
from fidmem.storage.run_store import RunStore
from fidmem.types import ActionInstance, ActionType, RouterState
class InvalidPolicyActionError(ValueError): pass
class ResumeValidationError(ValueError): pass
class RouterPolicy(Protocol):
    def __call__(self,state:RouterState,legal_actions:tuple[ActionInstance,...])->ActionInstance: ...
class RunResult(BaseModel):
    model_config=ConfigDict(frozen=True)
    transitions:tuple[EnvironmentTransition,...]; answer:AnswerResult; final_state:RouterState; forced_stop:bool
class _AnswerArtifact(BaseModel):
    model_config=ConfigDict(frozen=True)
    run_id:str; state_sha256:str; answer:AnswerResult
def _state_sha256(state:RouterState)->str:
    payload=json.dumps(state.model_dump(mode="json"),ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
class AgentRunner:
    def __init__(self,environment:MemoryEnvironment,policy:RouterPolicy,answerer:FrozenAnswerer,*,run_store:RunStore|None=None,artifact_dir:Path|str|None=None,worker_id:str="agent-runner")->None:
        self.environment=environment;self.policy=policy;self.answerer=answerer;self.run_store=run_store;self.artifact_dir=Path(artifact_dir) if artifact_dir is not None else None;self.worker_id=worker_id
        if run_store is not None and self.artifact_dir is None: raise ValueError("artifact_dir is required with RunStore")
    @staticmethod
    def _key(step:int)->str:return f"transition-{step:03d}"
    def _claim(self,run:str,key:str)->None:
        if self.run_store is not None and not self.run_store.claim(run,key,self.worker_id): raise RuntimeError(f"run item cannot be claimed: {run}/{key}")
    def _complete(self,run:str,key:str,payload:BaseModel)->None:
        if self.run_store is None:return
        assert self.artifact_dir is not None
        target=self.artifact_dir/run/f"{key}.json";target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix(target.suffix+".tmp")
        temp.write_text(payload.model_dump_json(indent=2),encoding="utf-8");temp.replace(target);self.run_store.complete(run,key,str(target))
    def _fail(self,run:str,key:str,error:BaseException)->None:
        if self.run_store is not None:self.run_store.fail(run,key,type(error).__name__,str(error))
    def _load_transition(self,uri:str|None)->EnvironmentTransition:
        if uri is None or not Path(uri).is_file():raise ResumeValidationError("complete transition missing artifact")
        try:return EnvironmentTransition.model_validate_json(Path(uri).read_text(encoding="utf-8"))
        except (OSError,ValueError) as error:raise ResumeValidationError("invalid transition artifact") from error
    def _restore(self,initial:RouterState,run:str)->tuple[list[EnvironmentTransition],RouterState]:
        if self.run_store is None:return [],initial
        state=initial;out=[]
        for step in range(5):
            item=self.run_store.item(run,self._key(step))
            if item is None or item.status!="complete":break
            tr=self._load_transition(item.output_uri)
            if tr.state!=state or len(tr.state.action_history)!=step:raise ResumeValidationError("transition artifact state or index mismatch")
            try: expected=self.environment.replay(tr.state,tr.action,tr.observation)
            except Exception as error: raise ResumeValidationError("persisted transition fails pure replay") from error
            if tr != expected: raise ResumeValidationError("persisted transition differs from pure replay")
            out.append(tr);state=tr.next_state
            if tr.terminal:break
        return out,state
    def _answer(self,state:RouterState,trs:list[EnvironmentTransition],run:str,forced:bool)->RunResult:
        key="answer"
        if self.run_store is not None:
            item=self.run_store.item(run,key)
            if item is not None and item.status=="complete":
                if item.output_uri is None or not Path(item.output_uri).is_file():raise ResumeValidationError("complete answer missing artifact")
                try:
                    artifact=_AnswerArtifact.model_validate_json(Path(item.output_uri).read_text(encoding="utf-8"))
                except (OSError,ValueError) as error:
                    raise ResumeValidationError("legacy or invalid answer artifact envelope") from error
                if artifact.run_id!=run or artifact.state_sha256!=_state_sha256(state):
                    raise ResumeValidationError("answer artifact run or final state mismatch")
                return RunResult(transitions=tuple(trs),answer=artifact.answer,final_state=state,forced_stop=forced)
        self._claim(run,key)
        try:
            answer=self.answerer.answer(state.question,state.options,state.evidence)
            artifact=_AnswerArtifact(run_id=run,state_sha256=_state_sha256(state),answer=answer)
            self._complete(run,key,artifact)
        except BaseException as error:self._fail(run,key,error);raise
        return RunResult(transitions=tuple(trs),answer=answer,final_state=state,forced_stop=forced)
    def run(self,initial_state:RouterState,*,run_id:str)->RunResult:
        trs,state=self._restore(initial_state,run_id);forced=bool(len(trs)==5 and trs[-1].terminal)
        if trs and trs[-1].terminal:return self._answer(state,trs,run_id,forced)
        for step in range(len(trs),5):
            legal=self.environment.valid_actions(state)
            if not legal:raise RuntimeError("non-terminal state has no legal action")
            key=self._key(step);self._claim(run_id,key)
            try:
                if step==4:
                    selected=ActionInstance(ActionType.STOP,None,None)
                    if selected not in legal:raise RuntimeError("fifth transition requires STOP")
                    forced=True
                else:
                    selected=self.policy(state,legal)
                    if selected not in legal:raise InvalidPolicyActionError("policy selected action outside legal tuple")
                tr=self.environment.step(state,selected);self._complete(run_id,key,tr)
            except BaseException as error:self._fail(run_id,key,error);raise
            trs.append(tr);state=tr.next_state
            if tr.terminal:return self._answer(state,trs,run_id,forced)
        raise RuntimeError("runner exceeded transition bound")
