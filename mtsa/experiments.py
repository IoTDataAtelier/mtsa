from dataclasses import dataclass


@dataclass
class ModelParam:
    name: str
    obj: object
    
@dataclass
class Params:
    path: str
    model: ModelParam

@dataclass
class Result:
    time_elapsed: float
    fun_return: object

@dataclass
class ParamResult:
    params: Params
    result: Result
