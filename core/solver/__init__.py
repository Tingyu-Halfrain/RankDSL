from .greedy_solver import GreedySolver
from .ilp_solver import ILPSolver


def solve_rankdsl(dsl, candidates, mode: str = "greedy", ilp_max_candidates: int = 500):
    if mode == "greedy":
        return GreedySolver().solve(dsl, candidates)
    if mode == "ilp":
        return ILPSolver(max_candidates=10**9).solve(dsl, candidates)
    if mode == "auto":
        return ILPSolver(max_candidates=ilp_max_candidates).solve(dsl, candidates)
    raise ValueError(f"Unsupported solver mode: {mode}")
