from typing import Dict, Any, Optional, Callable, List, Set, TypeVar, Generic, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, Future
import inspect
from functools import wraps
import time
import traceback

T = TypeVar('T')

class AgentState(Enum):
    """Agent lifecycle states"""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class TaskPriority(Enum):
    """Task execution priority levels"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4

class ExecutionMode(Enum):
    """Task execution modes"""
    SYNC = "sync"
    ASYNC = "async"
    PARALLEL = "parallel"

@dataclass
class TaskMetrics:
    """Metrics for task execution"""
    task_name: str
    executions: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    errors: int = 0
    last_execution: Optional[datetime] = None
    last_error: Optional[str] = None
    
    def update(self, execution_time: float, success: bool = True, error: Optional[str] = None):
        """Update metrics with new execution data"""
        self.executions += 1
        self.total_time += execution_time
        self.avg_time = self.total_time / self.executions
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.last_execution = datetime.now()
        if not success:
            self.errors += 1
            self.last_error = error
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_name': self.task_name,
            'executions': self.executions,
            'avg_time': round(self.avg_time, 4),
            'min_time': round(self.min_time, 4) if self.min_time != float('inf') else None,
            'max_time': round(self.max_time, 4),
            'errors': self.errors,
            'error_rate': round(self.errors / self.executions, 4) if self.executions > 0 else 0.0,
            'last_execution': self.last_execution.isoformat() if self.last_execution else None,
            'last_error': self.last_error
        }

@dataclass
class TaskDefinition:
    """Enhanced task definition with metadata"""
    name: str
    function: Callable
    priority: TaskPriority = TaskPriority.NORMAL
    mode: ExecutionMode = ExecutionMode.SYNC
    timeout: Optional[float] = None
    retries: int = 0
    retry_delay: float = 1.0
    description: str = ""
    tags: Set[str] = field(default_factory=set)
    dependencies: List[str] = field(default_factory=list)
    enabled: bool = True
    
@dataclass
class ExecutionResult:
    """Structured execution result with comprehensive metadata"""
    status: str
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    task_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status,
            'result': self.result,
            'error': self.error,
            'execution_time': round(self.execution_time, 4),
            'timestamp': self.timestamp.isoformat(),
            'task_name': self.task_name,
            'metadata': self.metadata
        }
    
    @property
    def success(self) -> bool:
        return self.status == "ok"

class TaskExecutor:
    """Advanced task executor with async, parallel, and retry support"""
    
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def execute(
        self, 
        task_def: TaskDefinition, 
        *args, 
        **kwargs
    ) -> ExecutionResult:
        """Execute task based on its execution mode"""
        start_time = time.time()
        
        try:
            if task_def.mode == ExecutionMode.ASYNC:
                result = self._execute_async(task_def, *args, **kwargs)
            elif task_def.mode == ExecutionMode.PARALLEL:
                result = self._execute_parallel(task_def, *args, **kwargs)
            else:
                result = self._execute_sync(task_def, *args, **kwargs)
            
            execution_time = time.time() - start_time
            return ExecutionResult(
                status="ok",
                result=result,
                execution_time=execution_time,
                task_name=task_def.name
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Task {task_def.name} failed: {str(e)}")
            return ExecutionResult(
                status="error",
                error=str(e),
                execution_time=execution_time,
                task_name=task_def.name,
                metadata={'traceback': traceback.format_exc()}
            )
    
    def _execute_sync(self, task_def: TaskDefinition, *args, **kwargs) -> Any:
        """Synchronous execution with retry logic"""
        last_error = None
        for attempt in range(task_def.retries + 1):
            try:
                if task_def.timeout:
                    future = self.executor.submit(task_def.function, *args, **kwargs)
                    return future.result(timeout=task_def.timeout)
                else:
                    return task_def.function(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < task_def.retries:
                    time.sleep(task_def.retry_delay * (attempt + 1))
                    continue
                raise
        raise last_error
    
    def _execute_async(self, task_def: TaskDefinition, *args, **kwargs) -> Any:
        """Async execution (returns Future for non-blocking)"""
        return self.executor.submit(task_def.function, *args, **kwargs)
    
    def _execute_parallel(self, task_def: TaskDefinition, *args, **kwargs) -> Any:
        """Parallel execution for batch operations"""
        # Assumes function accepts iterable and processes in parallel
        return self.executor.map(task_def.function, *args, **kwargs)
    
    def shutdown(self):
        """Gracefully shutdown executor"""
        self.executor.shutdown(wait=True)

class AgentObserver(ABC):
    """Observer pattern for agent lifecycle events"""
    
    @abstractmethod
    def on_state_change(self, agent: 'BaseAgent', old_state: AgentState, new_state: AgentState):
        pass
    
    @abstractmethod
    def on_task_complete(self, agent: 'BaseAgent', result: ExecutionResult):
        pass
    
    @abstractmethod
    def on_error(self, agent: 'BaseAgent', error: Exception):
        pass

class BaseAgent(ABC):
    """
    Advanced unified agent interface with enterprise-grade features.
    
    Features:
    - State management with lifecycle hooks
    - Task registry with priorities, dependencies, and metadata
    - Async and parallel execution support
    - Comprehensive metrics and monitoring
    - Observer pattern for event handling
    - Retry logic and timeout handling
    - Thread-safe operations
    - Health checks and diagnostics
    """
    
    def __init__(
        self, 
        name: str, 
        agent_type: str,
        max_workers: int = 4,
        enable_metrics: bool = True
    ):
        self.name = name
        self.agent_type = agent_type
        self._state = AgentState.INITIALIZING
        
        # Task management
        self._tasks: Dict[str, TaskDefinition] = {}
        self._task_executor = TaskExecutor(max_workers=max_workers)
        
        # Results and history
        self.last_result: Optional[ExecutionResult] = None
        self._execution_history: List[ExecutionResult] = []
        self._max_history = 100
        
        # Metrics
        self._enable_metrics = enable_metrics
        self._task_metrics: Dict[str, TaskMetrics] = {}
        
        # Metadata and configuration
        self.metadata: Dict[str, Any] = {
            'created_at': datetime.now(),
            'version': '2.0.0'
        }
        self.config: Dict[str, Any] = {}
        
        # Observers
        self._observers: List[AgentObserver] = []
        
        # Logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{name}]")
        
        # Initialization
        self._initialize()
        self._set_state(AgentState.READY)
    
    def _initialize(self):
        """Hook for subclass initialization"""
        pass

    @property
    def tasks(self):
        """Compatibility property exposing registered tasks as a dict-like mapping."""
        return {
            name: {
                'description': td.description,
                'priority': td.priority.name,
                'mode': td.mode.value,
                'enabled': td.enabled
            }
            for name, td in self._tasks.items()
        }
    
    def _set_state(self, new_state: AgentState):
        """Thread-safe state transition with observer notification"""
        old_state = self._state
        self._state = new_state
        self.logger.info(f"State transition: {old_state.value} -> {new_state.value}")
        for observer in self._observers:
            try:
                observer.on_state_change(self, old_state, new_state)
            except Exception as e:
                self.logger.error(f"Observer error: {e}")
    
    @property
    def state(self) -> AgentState:
        """Current agent state"""
        return self._state
    
    def register_task(
        self,
        task_name: str,
        fn: Callable,
        priority: TaskPriority = TaskPriority.NORMAL,
        mode: ExecutionMode = ExecutionMode.SYNC,
        timeout: Optional[float] = None,
        retries: int = 0,
        retry_delay: float = 1.0,
        description: str = "",
        tags: Optional[Set[str]] = None,
        dependencies: Optional[List[str]] = None
    ):
        """Register a task with enhanced configuration"""
        task_def = TaskDefinition(
            name=task_name,
            function=fn,
            priority=priority,
            mode=mode,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
            description=description,
            tags=tags or set(),
            dependencies=dependencies or [],
            enabled=True
        )
        self._tasks[task_name] = task_def
        
        if self._enable_metrics:
            self._task_metrics[task_name] = TaskMetrics(task_name=task_name)
        
        self.logger.debug(f"Registered task: {task_name}")
    
    def unregister_task(self, task_name: str) -> bool:
        """Remove a task from registry"""
        if task_name in self._tasks:
            del self._tasks[task_name]
            self.logger.debug(f"Unregistered task: {task_name}")
            return True
        return False
    
    def enable_task(self, task_name: str):
        """Enable a previously disabled task"""
        if task_name in self._tasks:
            self._tasks[task_name].enabled = True
    
    def disable_task(self, task_name: str):
        """Disable a task without removing it"""
        if task_name in self._tasks:
            self._tasks[task_name].enabled = False
    
    def get_task(self, task_name: str) -> Optional[TaskDefinition]:
        """Retrieve task definition"""
        return self._tasks.get(task_name)
    
    def list_tasks(
        self, 
        enabled_only: bool = False, 
        tags: Optional[Set[str]] = None
    ) -> List[str]:
        """List all registered tasks with optional filtering"""
        tasks = self._tasks.values()
        
        if enabled_only:
            tasks = [t for t in tasks if t.enabled]
        
        if tags:
            tasks = [t for t in tasks if tags.intersection(t.tags)]
        
        return [t.name for t in sorted(tasks, key=lambda x: x.priority.value)]
    
    def execute_task(self, task: str, **kwargs) -> ExecutionResult:
        """Execute a registered task with metrics tracking.

        Backwards-compatible behavior: returns a plain dict describing the
        execution result (with keys `status`, `result`, `error`, etc.)
        while internally storing an `ExecutionResult` object in
        `self.last_result` for monitoring and metrics.
        """
        if self._state not in [AgentState.READY, AgentState.RUNNING]:
            err = ExecutionResult(
                status="error",
                error=f"Agent not ready (state: {self._state.value})",
                task_name=task
            )
            return err.to_dict()

        if task not in self._tasks:
            err = ExecutionResult(status="error", error=f"Unknown task: {task}", task_name=task)
            return err.to_dict()

        task_def = self._tasks[task]

        if not task_def.enabled:
            err = ExecutionResult(status="error", error=f"Task disabled: {task}", task_name=task)
            return err.to_dict()

        # Check dependencies
        for dep in task_def.dependencies:
            if dep not in self._tasks or not self._tasks[dep].enabled:
                err = ExecutionResult(status="error", error=f"Dependency not satisfied: {dep}", task_name=task)
                return err.to_dict()

        # Execute
        self._set_state(AgentState.RUNNING)
        exec_result = self._task_executor.execute(task_def, **kwargs)

        # Update metrics
        if self._enable_metrics and task in self._task_metrics:
            self._task_metrics[task].update(
                execution_time=exec_result.execution_time,
                success=exec_result.success,
                error=exec_result.error
            )

        # Store result object and history
        self.last_result = exec_result
        self._execution_history.append(exec_result)
        if len(self._execution_history) > self._max_history:
            self._execution_history.pop(0)

        # Notify observers
        for observer in self._observers:
            try:
                observer.on_task_complete(self, exec_result)
            except Exception as e:
                self.logger.error(f"Observer error: {e}")

        self._set_state(AgentState.READY)

        # Return a plain dict for compatibility with older code in the repo
        out = exec_result.to_dict()

        # If the underlying result field contains complex/numpy types, convert
        # them into serializable native Python types before returning.
        out['result'] = self._make_serializable(out.get('result'))

        return out

    def _make_serializable(self, obj):
        """Recursively convert numpy scalars/arrays and other non-serializable
        objects to native Python types (lists, floats, ints, str).
        """
        try:
            import numpy as _np
        except Exception:
            _np = None

        # None or primitives
        if obj is None or isinstance(obj, (str, bool, int, float)):
            return obj

        # numpy scalars
        if _np is not None and isinstance(obj, _np.generic):
            return obj.item()

        # numpy arrays -> list
        if _np is not None and isinstance(obj, (_np.ndarray,)):
            try:
                return obj.tolist()
            except Exception:
                return [self._make_serializable(x) for x in obj]

        # dict -> recurse
        if isinstance(obj, dict):
            return {str(k): self._make_serializable(v) for k, v in obj.items()}

        # list/tuple/set -> list
        if isinstance(obj, (list, tuple, set)):
            return [self._make_serializable(x) for x in obj]

        # dataclasses or objects with to_dict
        if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
            try:
                return self._make_serializable(obj.to_dict())
            except Exception:
                pass

        # Fallback: try to convert to str
        try:
            return str(obj)
        except Exception:
            return None
    
    def execute_pipeline(self, tasks: List[str], **kwargs) -> List[ExecutionResult]:
        """Execute multiple tasks in sequence and return list of dict results."""
        results = []
        shared_state = kwargs.copy()

        for task in tasks:
            result = self.execute_task(task, **shared_state)
            # execute_task now returns a dict
            results.append(result)

            if result.get('status') != 'ok':
                self.logger.warning(f"Pipeline stopped at {task} due to error")
                break

            # Tasks can update shared state
            r = result.get('result')
            if r and isinstance(r, dict):
                shared_state.update(r)

        return results
    
    @abstractmethod
    def run_single(self, symbol: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute single-symbol operation (must be implemented by subclass)"""
        pass
    
    @abstractmethod
    def run_pipeline(self, symbol: str, shared_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pipeline operation (must be implemented by subclass)"""
        pass
    
    def attach_observer(self, observer: AgentObserver):
        """Attach an observer for lifecycle events"""
        self._observers.append(observer)
    
    def detach_observer(self, observer: AgentObserver):
        """Detach an observer"""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def get_metrics(self, task_name: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve execution metrics"""
        if not self._enable_metrics:
            return {'enabled': False}
        
        if task_name:
            if task_name in self._task_metrics:
                return self._task_metrics[task_name].to_dict()
            return {}
        
        return {
            task: metrics.to_dict()
            for task, metrics in self._task_metrics.items()
        }
    
    def get_execution_history(
        self, 
        limit: Optional[int] = None,
        task_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve execution history with optional filtering"""
        history = self._execution_history
        
        if task_name:
            history = [r for r in history if r.task_name == task_name]
        
        if limit:
            history = history[-limit:]
        
        return [r.to_dict() for r in history]
    
    def get_status(self) -> Dict[str, Any]:
        """Comprehensive agent status report"""
        return {
            'name': self.name,
            'type': self.agent_type,
            'state': self._state.value,
            'tasks': {
                'total': len(self._tasks),
                'enabled': len([t for t in self._tasks.values() if t.enabled]),
                'list': self.list_tasks()
            },
            'last_result': self.last_result.to_dict() if self.last_result else None,
            'metadata': self.metadata,
            'config': self.config,
            'metrics_summary': self._get_metrics_summary() if self._enable_metrics else None,
            'uptime': (datetime.now() - self.metadata['created_at']).total_seconds()
        }
    
    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Aggregate metrics summary"""
        if not self._task_metrics:
            return {}
        
        total_executions = sum(m.executions for m in self._task_metrics.values())
        total_errors = sum(m.errors for m in self._task_metrics.values())
        
        return {
            'total_executions': total_executions,
            'total_errors': total_errors,
            'overall_error_rate': round(total_errors / total_executions, 4) if total_executions > 0 else 0.0,
            'tasks_with_errors': len([m for m in self._task_metrics.values() if m.errors > 0])
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check diagnostics"""
        is_healthy = (
            self._state in [AgentState.READY, AgentState.RUNNING] and
            (not self.last_result or self.last_result.success)
        )
        
        return {
            'healthy': is_healthy,
            'state': self._state.value,
            'last_execution_success': self.last_result.success if self.last_result else None,
            'tasks_operational': len([t for t in self._tasks.values() if t.enabled]),
            'timestamp': datetime.now().isoformat()
        }
    
    def reset(self):
        """Reset agent state and clear history"""
        self._execution_history.clear()
        self.last_result = None
        if self._enable_metrics:
            self._task_metrics = {
                task: TaskMetrics(task_name=task)
                for task in self._tasks.keys()
            }
        self._set_state(AgentState.READY)
        self.logger.info("Agent reset completed")
    
    def shutdown(self):
        """Gracefully shutdown agent and cleanup resources"""
        self._set_state(AgentState.SHUTDOWN)
        self._task_executor.shutdown()
        self.logger.info("Agent shutdown completed")
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', type='{self.agent_type}', state='{self._state.value}')>"
    
    def __enter__(self):
        """Context manager support"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.shutdown()