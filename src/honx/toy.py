import functools
import time
import jax
from jax import core, dtypes
from jax.core import ShapedArray
from jax.interpreters import mlir, xla
from jax.interpreters.mlir import ir
from jax.lib import xla_client
import jax.numpy as jnp
from jaxlib.hlo_helpers import custom_call
from honx._toy import get_registrations


def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]


_shift_string = core.Primitive("string_shift")
_shift_string.multiple_results = False
_shift_string.def_impl(functools.partial(xla.apply_primitive, _shift_string))


# ---------- bind capsulated functions ---------- #
def shift_string(x):
    output = _shift_string.bind(x)
    return output


# ---------- lower the ops ---------- #
def _shift_string_lower(ctx, x):
    x_type = ir.RankedTensorType(x.type)
    x_shape = x_type.shape

    size = functools.reduce(lambda x, y: x * y, x_shape)
    out = custom_call(
        b"string_shift",
        out_types=[ir.RankedTensorType.get(x_shape, x_type.element_type)],
        operands=[mlir.ir_constant(size), x],
        operand_layouts=default_layouts((), x_shape),
        result_layouts=default_layouts(x_shape),
    )
    return out


for _name, _value in get_registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="cpu")

mlir.register_lowering(
    _shift_string,
    _shift_string_lower,
    platform="cpu",
)


def _shift_string_abstract(x):
    x_dtype = dtypes.canonicalize_dtype(x.dtype)
    return ShapedArray(x.shape, x_dtype, named_shape=x.named_shape)


_shift_string.def_abstract_eval(_shift_string_abstract)


if __name__ == "__main__":
    d = jnp.array([12, 23, 54, 10, 0, 18], dtype=jnp.int8)
    assert jnp.all(shift_string(d) == jnp.array([23, 54, 10, 0, 18, 12], dtype=jnp.int8))

    @jax.jit
    def shift_n(x, n: int):
        res = jax.lax.fori_loop(0, n, lambda i, x: shift_string(x), x)
        return res

    # Warm-up
    shift_n(d, 100)

    print("Timing for jit-compiled func with fori_loop:")
    start = time.perf_counter()
    print(shift_n(d, 10000))
    print(time.perf_counter() - start)
    print()

    print("Timing for jit-compiled func with python loop:")
    start = time.perf_counter()
    res = d
    for i in range(10000):
        res = shift_n(res, 1)
    print(res)
    print(time.perf_counter() - start)
    print()

    print("Timing for non-jit-compiled func (but C++ capsule exposed through pybind11):")
    start = time.perf_counter()
    res = d
    for i in range(10000):
        res = shift_string(res)
    print(res)
    print(time.perf_counter() - start)


    # todo: implement batching rule
    """
    key = jax.random.PRNGKey(0)
    batch = jax.random.randint(key, (3, 5), 0, 100)
    print("Orig:")
    print(batch)
    print("Batch shifted:")
    print(jax.vmap(shift_string)(batch))
    """


