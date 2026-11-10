#!/usr/bin/env python
"""Shared graph helpers for the dynamic-shape pilots.

`Builder` mirrors `export_onnx.Builder`; `shape_dim` extracts one runtime
extent from a tensor's shape; `arithmetic_shift` is the DXIL signed right shift
used by the int8 requant.
"""
import numpy as np
from onnx import TensorProto, helper, numpy_helper


class Builder:
    def __init__(self):
        self.nodes = []
        self.initializers = []

    def add(self, op, inputs, outputs, **attrs):
        name = attrs.pop("name", "%s_%d" % (op, len(self.nodes)))
        self.nodes.append(helper.make_node(op, inputs, outputs, name=name,
                                           **attrs))

    def tensor(self, name, array):
        self.initializers.append(numpy_helper.from_array(array, name))

    def const(self, name, op, values):
        dtype = {"i64": np.int64, "i32": np.int32, "i8": np.int8,
                 "u8": np.uint8, "fp32": np.float32, "fp64": np.float64}[op]
        self.tensor(name, np.asarray(values, dtype=dtype))


def shape_dim(builder, source, index, name):
    """One extent of `source`, as an int64 [1] tensor named `name`."""
    b = builder
    b.add("Shape", [source], [name + "_shape"])
    b.const(name + "_start", "i64", [index])
    b.const(name + "_end", "i64", [index + 1])
    b.const(name + "_axis", "i64", [0])
    b.const(name + "_step", "i64", [1])
    b.add("Slice", [name + "_shape", name + "_start", name + "_end",
                    name + "_axis", name + "_step"], [name])
    return name


def scalar_const(builder, name, value):
    """A 0-D int64 initializer, which `Range` requires for its inputs."""
    builder.tensor(name, np.asarray(value, dtype=np.int64))


def squeeze_scalar(builder, name, tag=None):
    """Turn an int64 [1] feed into the 0-D tensor `Range` needs."""
    out = tag or name + "_scalar"
    builder.const(out + "_axis", "i64", [0])
    builder.add("Squeeze", [name, out + "_axis"], [out])
    return out


def arithmetic_shift(builder, source, bits, tag):
    """Signed right shift of an int32 tensor, truncating toward -infinity."""
    b = builder
    if bits <= 0:
        return source
    denom = 1 << bits
    b.const(tag + "_add", "i32", [1 << (bits - 1)])
    b.add("Add", [source, tag + "_add"], [tag + "_rounded"])
    b.const(tag + "_denom", "i32", [denom])
    b.const(tag + "_bias", "i32", [denom - 1])
    b.const(tag + "_zero", "i32", [0])
    b.add("Less", [tag + "_rounded", tag + "_zero"], [tag + "_neg"])
    b.add("Where", [tag + "_neg", tag + "_bias", tag + "_zero"],
          [tag + "_adjust"])
    b.add("Sub", [tag + "_rounded", tag + "_adjust"], [tag + "_nudged"])
    b.add("Div", [tag + "_nudged", tag + "_denom"], [tag + "_shifted"])
    return tag + "_shifted"
