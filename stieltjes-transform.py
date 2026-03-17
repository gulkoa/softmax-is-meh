# https://github.com/jtaylor-eng/probability-simplex-mappings/blob/master/mappings/stieltjes.py#L33

import torch

#Extend this class to create new simplex mappings

import torch.nn as nn
from abc import ABC, abstractmethod

class ProbabilitySimplexMapping(nn.Module, ABC):
    """Provide uniform interface to use for probabilistic simplex mappings"""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def translate_logits(
        self,
        logits: torch.Tensor,
        dim: int,
        **kwargs,
    ) -> torch.Tensor:
        pass


class StieltjesTransform(ProbabilitySimplexMapping):
    """Stieltjes transform as introduced, using binary search."""
    def __init__(
        self,
        q: float = 1.0,
        num_iter: int = 8,
        eps: float = 1e-9
    ):
        super().__init__()
        self._q = q
        self._num_iter = num_iter
        self._eps = eps

    def _line_search_bs(self, shifted_logits, dim, lb, ub):
        for _ in range(self._num_iter):
            mid = (lb + ub) / 2.0
            
            prob_sum = torch.sum(
                torch.pow((mid - shifted_logits).clamp(min=self._eps), -self._q),
                dim=dim,
                keepdim=True
            ) - 1
            
            lb = torch.where(prob_sum > 0, mid, lb)
            ub = torch.where(prob_sum <= 0, mid, ub)

        return lb, ub

    # def translate_logits(
    #     self,
    #     logits,
    #     dim,
    #     **kwargs,
    # ) -> torch.Tensor:
    #     """Calculates 1 / (lambda_q - x_i)^q"""
        
    #     # logits = torch.clamp(logits, min=-50.0, max=50.0)
        
    #     x_max = torch.max(logits, dim=dim, keepdim=True).values
    #     x_i = logits - x_max

    #     lb = torch.full_like(x_max, self._eps)
    #     ub = torch.full_like(x_max, logits.shape[dim] ** (1.0/ self._q))

    #     lb, ub = self._line_search_bs(
    #         shifted_logits=x_i,
    #         dim=dim,
    #         lb=lb,
    #         ub=ub
    #     )
    #     lambda_1 = (lb + ub) / 2.0
        
    #     # 1 / (lambda_q - x_i)^q
    #     return torch.pow((lambda_1 - x_i).clamp(min=self._eps), -self._q)

    def translate_logits(self, logits, dim, **kwargs) -> torch.Tensor:
        x_max = torch.max(logits, dim=dim, keepdim=True).values
        x_i = logits - x_max
        
        # Initial guess: We know lambda must be at least 1.0 (since x_max = 0)
        # Using 1.1 is a safe, close starting point
        lambd = torch.full_like(x_max, 1.1)
        
        # 3 iterations of Newton-Raphson is usually enough for float16/bfloat16 precision
        for _ in range(3):
            diff = (lambd - x_i).clamp(min=self._eps)
            
            # f(lambda)
            f_val = torch.sum(torch.pow(diff, -self._q), dim=dim, keepdim=True) - 1.0
            # f'(lambda)
            f_deriv = -self._q * torch.sum(torch.pow(diff, -self._q - 1.0), dim=dim, keepdim=True)
            
            # Update
            lambd = lambd - (f_val / f_deriv)

        return torch.pow((lambd - x_i).clamp(min=self._eps), -self._q)