from functools import wraps
from typing import Callable

import numpy as np
import torch
from torch import Tensor


def _has_method(o: object, name: str) -> bool:
    return callable(getattr(o, name, None))


def batch_function(item_ndim: int, unpack_return: bool = False) -> Callable:
    """Decorator that allows to compute item-by-item measures on a batch of items by looping over the batch.

    Args:
        item_ndim: Dimensionality of the items expected by the function. Allows to detect if the function's input is a
            single item, or a batch of items.
        unpack_return: Whether the return value of the function is a tuple of multiple elements that should be unpacked
            before being stacked along the batch dimension.

    Returns:
        Function that accepts batch of items as well as individual items.
    """
    if unpack_return:

        def collate_fn(func_return, is_batch: bool):
            if is_batch:
                # Aggregate multiple values returned by the function over all the items in the batch and create an array
                # for each batch of values
                return tuple(np.array(return_vals) for return_vals in zip(*func_return))
            else:
                # Cast multiple values returned by the function on a single data item to individual numpy arrays
                return tuple(np.array(return_val) for return_val in func_return)

    else:
        collate_fn = np.array

    def batch_function_decorator(func: Callable) -> Callable:
        @wraps(func)
        def _loop_function_on_batch(*args, **kwargs):
            if _has_method(args[0], func.__name__):
                # If `func` is a method, pass over the implicit `self` as first argument
                self_or_empty, args = args[0:1], args[1:]
            else:
                self_or_empty = ()
            data, *args = args
            if not isinstance(data, np.ndarray):
                raise ValueError(
                    f"The first argument provided to '{func.__name__}' was not the input data, as a numpy array, "
                    f"expected by the function. The first argument of the function was rather of type '{type(data)}'."
                )
            collate_fn_kwargs = {}
            if unpack_return:
                collate_fn_kwargs["is_batch"] = bool(data.ndim - item_ndim)
            if data.ndim == item_ndim:  # If the input data is a single item
                result = collate_fn(func(*self_or_empty, data, *args, **kwargs), **collate_fn_kwargs)
            elif data.ndim == (item_ndim + 1):  # If the input data is a batch of items
                result = collate_fn([func(*self_or_empty, item, *args, **kwargs) for item in data], **collate_fn_kwargs)
            else:
                raise RuntimeError(
                    f"Couldn't apply '{func.__name__}', either in batch or one-shot, over the input data. The use of"
                    f"the `batch_function` decorator allows '{func.__name__}' to accept {item_ndim}D (item) or "
                    f"{item_ndim+1}D (batch) input data. The input data passed to '{func.__name__}' was of shape: "
                    f"{data.shape}."
                )
            return result

        return _loop_function_on_batch

    return batch_function_decorator


def auto_cast_data(func: Callable) -> Callable:
    """Decorator to allow functions relying on numpy arrays to accept other input data types.

    Args:
        func: Function for which to automatically convert the first argument to a numpy array.

    Returns:
        Function that accepts input data types other than numpy arrays by converting between them and numpy arrays.
    """
    cast_types = [Tensor]
    dtypes = [np.ndarray, *cast_types]

    @wraps(func)
    def _call_func_with_cast_data(*args, **kwargs):
        if _has_method(args[0], func.__name__):
            # If `func` is a method, pass over the implicit `self` as first argument
            self_or_empty, data, args = args[0:1], args[1], args[2:]
        else:
            self_or_empty, data, args = (), args[0], args[1:]

        dtype = type(data)
        if dtype not in dtypes:
            raise ValueError(
                f"Decorator 'auto_cast_data' used by function '{func.__name__}' does not support casting inputs of "
                f"type '{dtype}' to numpy arrays. Either provide the implementation for casting to numpy arrays "
                f"from '{cast_types}' in 'auto_cast_data' decorator, or manually convert the input of '{func.__name__}'"
                f"to one of the following supported types: {dtypes}."
            )
        if dtype == Tensor:
            data_device = data.device
            data = data.detach().cpu().numpy()
        result = func(*self_or_empty, data, *args, **kwargs)
        if dtype == Tensor:
            result = torch.tensor(result, device=data_device)
        return result

    return _call_func_with_cast_data
