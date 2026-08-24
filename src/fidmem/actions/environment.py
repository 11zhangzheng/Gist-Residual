"""Deterministic hard-masked state transitions for memory acquisition."""
from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from fidmem.costs.tracker import CostRecord
from fidmem.types import ActionInstance, ActionType, EventRecord, EvidenceItem, FidelityLevel, RouterState

class TerminalStateError(RuntimeError): pass
class IllegalActionError(ValueError): pass
class ObservationValidationError(ValueError): pass

class ActionCostTable(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)
    search_gist: float = Field(1, ge=0); search_gist_hit: float = Field(0, ge=0)
    residual: float = Field(2, ge=0); residual_hit: float = Field(0, ge=0)
    context: float = Field(.5, ge=0); context_hit: float = Field(0, ge=0)
    visual_low: float = Field(4, ge=0); visual_high: float = Field(8, ge=0)
    visual_low_question: float = Field(1, ge=0); visual_high_question: float = Field(2, ge=0)
    cache_hit: float = Field(0, ge=0)

CostScope = Literal["search_gist", "residual", "context", "event_observation", "question_verification"]
class OperationMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)
    scope: CostScope
    cache_status: Literal["hit", "miss"]
    amortizable: bool
    input_frames: int = Field(0, ge=0)
    visual_tokens: int = Field(0, ge=0)
    text_tokens: int = Field(0, ge=0)
    cost_record: CostRecord | None = None

    @model_validator(mode="after")
    def nested_cost_record_must_be_valid(self) -> "OperationMetadata":
        if self.cost_record is not None:
            self.cost_record.validate_values()
        return self
class ActionObservation(BaseModel):
    model_config = ConfigDict(frozen=True)
    action_type: ActionType | None = None
    target_event_id: str | None = None
    cache_status: Literal["hit", "miss"] = "miss"
    input_frames: int = Field(0, ge=0)
    context_frontier: tuple[int, int] | None = None
    evidence: tuple[EvidenceItem, ...] = ()
    candidate_event_ids: tuple[str, ...] = ()
    operation_metadata: tuple[OperationMetadata, ...] = ()
class EnvironmentTransition(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)
    state: RouterState; action: ActionInstance; observation: ActionObservation; next_state: RouterState
    step_cost: float = Field(ge=0)
    operation_metadata: tuple[OperationMetadata, ...] = ()
    terminal: bool = False
ActionExecutor = Callable[[ActionInstance, RouterState], ActionObservation]

class MemoryEnvironment:
    def __init__(self, *, events: Sequence[EventRecord], executor: ActionExecutor, costs: ActionCostTable | Mapping[str, float] | None = None) -> None:
        records = tuple(sorted(events, key=lambda e: (e.start_sec,e.end_sec,e.event_id)))
        if len({e.event_id for e in records}) != len(records): raise ValueError("event ids must be unique")
        self._by_id = {e.event_id:e for e in records}; self._ids = tuple(e.event_id for e in records)
        self._executor = executor; self.costs = costs if isinstance(costs, ActionCostTable) else ActionCostTable.model_validate(costs or {})
    @property
    def canonical_events(self) -> tuple[EventRecord, ...]:
        """Return the immutable canonical event table used by action semantics."""
        return tuple(self._by_id[event_id] for event_id in self._ids)
    @property
    def executor(self) -> ActionExecutor:
        """Expose the injected executor for identity attestation, never execution."""
        return self._executor
    @property
    def action_semantics_version(self) -> str:
        return "fidmem.memory-environment/v1"
    @staticmethod
    def _a(kind: ActionType, event: str|None=None, budget: str|None=None) -> ActionInstance: return ActionInstance(kind,event,budget)  # type: ignore[arg-type]
    @staticmethod
    def _terminal(state: RouterState) -> bool: return bool(state.action_history and state.action_history[-1].action_type is ActionType.STOP)
    @staticmethod
    def _seen(state: RouterState, action: ActionInstance) -> bool: return action in state.action_history
    def _event_cost(self,budget:str)->float: return self.costs.visual_low if budget=="low" else self.costs.visual_high
    def _question_cost(self,budget:str)->float: return self.costs.visual_low_question if budget=="low" else self.costs.visual_high_question
    def _upper(self,a:ActionInstance)->float:
        if a.action_type is ActionType.SEARCH_GIST:return self.costs.search_gist
        if a.action_type is ActionType.EXPAND_RESIDUAL:return self.costs.residual
        if a.action_type is ActionType.EXPAND_CONTEXT:return self.costs.context
        if a.action_type is ActionType.VERIFY_VISUAL:return self._event_cost(a.visual_budget or "high")+self._question_cost(a.visual_budget or "high")
        return 0
    def _neighbors(self,event:str,frontier:tuple[int,int])->tuple[str,...]:
        i=self._ids.index(event); left,right=frontier; result=[]
        if left<i: result.append(self._ids[i-left-1])
        if right<len(self._ids)-i-1: result.append(self._ids[i+right+1])
        return tuple(result)
    def valid_actions(self,state:RouterState)->tuple[ActionInstance,...]:
        if self._terminal(state): return ()
        actions=[]
        if not state.candidate_event_ids:
            search=self._a(ActionType.SEARCH_GIST)
            if not self._seen(state,search) and state.remaining_budget>=self._upper(search): actions.append(search)
            return tuple(actions+[self._a(ActionType.STOP)])
        for event in state.candidate_event_ids:
            if event not in self._by_id: raise ObservationValidationError("state contains unknown event")
            for action in (self._a(ActionType.EXPAND_RESIDUAL,event),self._a(ActionType.EXPAND_CONTEXT,event),self._a(ActionType.VERIFY_VISUAL,event,"low"),self._a(ActionType.VERIFY_VISUAL,event,"high")):
                available=bool(self._neighbors(event,state.context_frontiers[event])) if action.action_type is ActionType.EXPAND_CONTEXT else not self._seen(state,action)
                if available and state.remaining_budget>=self._upper(action): actions.append(action)
        return tuple(actions+[self._a(ActionType.STOP)])
    def _expected_scopes(self,a:ActionInstance)->tuple[CostScope,...]:
        return {ActionType.SEARCH_GIST:("search_gist",),ActionType.EXPAND_RESIDUAL:("residual",),ActionType.EXPAND_CONTEXT:("context",),ActionType.VERIFY_VISUAL:("event_observation","question_verification"),ActionType.STOP:()}[a.action_type]
    def _metadata(self,a:ActionInstance,meta:tuple[OperationMetadata,...])->None:
        if tuple(x.scope for x in meta)!=self._expected_scopes(a): raise ObservationValidationError("operation metadata scopes do not match action")
        if a.action_type is ActionType.VERIFY_VISUAL:
            event,question=meta; expected=12 if a.visual_budget=="low" else 32
            if not event.amortizable or question.amortizable: raise ObservationValidationError("invalid amortization scope")
            if (event.cache_status=="hit" and event.input_frames!=0) or (event.cache_status=="miss" and event.input_frames!=expected): raise ObservationValidationError("invalid visual frame count")
            if question.input_frames: raise ObservationValidationError("question verification cannot sample frames")
        elif meta and not meta[0].amortizable: raise ObservationValidationError("nonvisual metadata must be amortizable")
    def _charge(self,a:ActionInstance,meta:tuple[OperationMetadata,...])->float:
        if a.action_type is ActionType.STOP:return 0
        if a.action_type is ActionType.SEARCH_GIST:return self.costs.search_gist_hit if meta[0].cache_status=="hit" else self.costs.search_gist
        if a.action_type is ActionType.EXPAND_RESIDUAL:return self.costs.residual_hit if meta[0].cache_status=="hit" else self.costs.residual
        if a.action_type is ActionType.EXPAND_CONTEXT:return self.costs.context_hit if meta[0].cache_status=="hit" else self.costs.context
        return (0 if meta[0].cache_status=="hit" else self._event_cost(a.visual_budget or "high"))+self._question_cost(a.visual_budget or "high")
    def _validate(self,state:RouterState,a:ActionInstance,o:ActionObservation)->None:
        if o.action_type is not a.action_type or o.target_event_id!=a.event_id: raise ObservationValidationError("observation identity does not match target")
        self._metadata(a,o.operation_metadata)
        if len(set(o.candidate_event_ids))!=len(o.candidate_event_ids) or any(x not in self._by_id for x in o.candidate_event_ids): raise ObservationValidationError("invalid candidate ids")
        if a.action_type is ActionType.STOP:
            if o.evidence or o.candidate_event_ids or o.operation_metadata: raise ObservationValidationError("STOP cannot acquire")
            return
        if a.action_type is ActionType.SEARCH_GIST:
            if o.context_frontier is not None or any(x.event_id not in o.candidate_event_ids or x.fidelity_level is not FidelityLevel.GIST or x.attachments for x in o.evidence): raise ObservationValidationError("SEARCH only introduces candidate Gist")
            return
        if o.candidate_event_ids: raise ObservationValidationError("only SEARCH introduces candidates")
        assert a.event_id is not None
        if a.action_type is ActionType.EXPAND_RESIDUAL:
            if o.context_frontier is not None or any(x.event_id!=a.event_id or x.fidelity_level is not FidelityLevel.RESIDUAL or x.attachments for x in o.evidence): raise ObservationValidationError("Residual must target its event")
            return
        if a.action_type is ActionType.VERIFY_VISUAL:
            if o.context_frontier is not None or any(x.event_id!=a.event_id or x.fidelity_level is not FidelityLevel.VISUAL for x in o.evidence): raise ObservationValidationError("Visual must target its event")
            return
        frontier=state.context_frontiers[a.event_id]; expected=set(self._neighbors(a.event_id,frontier))
        if o.context_frontier!=frontier or {x.event_id for x in o.evidence}!=expected: raise ObservationValidationError("Context evidence must match frontier")
        for x in o.evidence:
            residual=x.fidelity_level is FidelityLevel.RESIDUAL and state.candidate_fidelity_levels.get(x.event_id) in (FidelityLevel.RESIDUAL,FidelityLevel.VISUAL)
            if x.attachments or (x.fidelity_level is not FidelityLevel.GIST and not residual): raise ObservationValidationError("Context leaked high fidelity")
    @staticmethod
    def _max(a:FidelityLevel,b:FidelityLevel)->FidelityLevel: return b if {FidelityLevel.GIST:0,FidelityLevel.RESIDUAL:1,FidelityLevel.VISUAL:2}[b]>{FidelityLevel.GIST:0,FidelityLevel.RESIDUAL:1,FidelityLevel.VISUAL:2}[a] else a
    def _reduce(self,state:RouterState,a:ActionInstance,o:ActionObservation,cost:float)->RouterState:
        ids=list(state.candidate_event_ids); levels=dict(state.candidate_fidelity_levels); fronts=dict(state.context_frontiers)
        if a.action_type is ActionType.SEARCH_GIST:
            for x in o.candidate_event_ids:
                if x not in levels: ids.append(x); levels[x]=FidelityLevel.GIST; fronts[x]=(0,0)
        elif a.action_type is ActionType.EXPAND_CONTEXT:
            assert a.event_id is not None; old=fronts[a.event_id]; new=self._neighbors(a.event_id,old); i=self._ids.index(a.event_id); l,r=old
            if l<i:l+=1
            if r<len(self._ids)-i-1:r+=1
            fronts[a.event_id]=(l,r)
            for x in new:
                if x not in levels:ids.append(x);levels[x]=FidelityLevel.GIST;fronts[x]=(0,0)
        elif a.event_id is not None:
            incoming=FidelityLevel.RESIDUAL if a.action_type is ActionType.EXPAND_RESIDUAL else FidelityLevel.VISUAL
            levels[a.event_id]=self._max(levels[a.event_id],incoming)
        step=len(state.action_history)+1; ev=tuple(x.model_copy(update={"start_sec":self._by_id[x.event_id].start_sec,"acquisition_step":step}) for x in o.evidence)
        return RouterState(question=state.question,options=state.options,evidence=state.evidence+ev,action_history=state.action_history+(a,),remaining_budget=state.remaining_budget-cost,candidate_event_ids=tuple(ids),candidate_fidelity_levels=levels,context_frontiers=fronts,cost_preference=state.cost_preference)
    def replay(self,state:RouterState,a:ActionInstance,o:ActionObservation)->EnvironmentTransition:
        """Purely revalidate a persisted transition; never invokes the executor."""
        if self._terminal(state): raise TerminalStateError("cannot replay terminal")
        if a not in self.valid_actions(state): raise IllegalActionError("replayed action is not legal")
        self._validate(state,a,o); cost=self._charge(a,o.operation_metadata)
        if cost>state.remaining_budget: raise IllegalActionError("replayed action exceeds budget")
        nxt=self._reduce(state,a,o,cost)
        return EnvironmentTransition(state=state,action=a,observation=o,next_state=nxt,step_cost=cost,operation_metadata=o.operation_metadata,terminal=a.action_type is ActionType.STOP)

    def step(self,state:RouterState,a:ActionInstance)->EnvironmentTransition:
        if self._terminal(state): raise TerminalStateError("cannot step terminal")
        if a not in self.valid_actions(state): raise IllegalActionError("action is not legal")
        o=ActionObservation(action_type=ActionType.STOP,target_event_id=None) if a.action_type is ActionType.STOP else self._executor(a,state)
        if not isinstance(o,ActionObservation): raise ObservationValidationError("executor must return ActionObservation")
        if o.action_type is None:
            scopes=self._expected_scopes(a)
            if a.action_type is ActionType.VERIFY_VISUAL:
                frames=0 if o.cache_status=="hit" else (12 if a.visual_budget=="low" else 32)
                metadata=(OperationMetadata(scope="event_observation",cache_status=o.cache_status,amortizable=True,input_frames=frames),OperationMetadata(scope="question_verification",cache_status="miss",amortizable=False))
            else:
                metadata=() if a.action_type is ActionType.STOP else (OperationMetadata(scope=scopes[0],cache_status=o.cache_status,amortizable=True),)
            update={"action_type":a.action_type,"target_event_id":a.event_id,"operation_metadata":metadata}
            if a.action_type is ActionType.EXPAND_CONTEXT and a.event_id is not None:
                frontier=state.context_frontiers[a.event_id]
                update["context_frontier"]=frontier
                update["evidence"]=tuple(EvidenceItem(event_id=x,fidelity_level=FidelityLevel.GIST,content="context",score=0) for x in self._neighbors(a.event_id,frontier))
            o=o.model_copy(update=update)
        self._validate(state,a,o); cost=self._charge(a,o.operation_metadata)
        if cost>state.remaining_budget: raise IllegalActionError("action exceeds budget")
        nxt=self._reduce(state,a,o,cost)
        return EnvironmentTransition(state=state,action=a,observation=o,next_state=nxt,step_cost=cost,operation_metadata=o.operation_metadata,terminal=a.action_type is ActionType.STOP)
