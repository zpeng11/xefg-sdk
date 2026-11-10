#!/usr/bin/env python
"""D4: execute the complete XeFG graph as one dynamic ONNX model.

`m9_graph.py` rebuilds every dispatch in-process and translates producer
allocations to consumer views with host `view_bytes`. This module instead
assembles the 38 dynamic pilots into a single graph by resource identity:

  * an edge is wired when the trace says the consumer's SRV is the producer's
    UAV; everything else becomes a graph input fed from the capture;
  * `view_bytes` becomes an in-graph `Reshape`+`Slice` for the documented
    texture2darray/buffer subresource cases;
  * int8/uint8 edges use a bit-preserving `Cast`;
  * the resample f16 pack is an in-graph arithmetic f32->f16 bit chain (ORT has
    no bitcast), feeding both the byte consumers and the f16 decode for d25;
  * the splat's byte allocation is packed to uint32 words for d25.

The model is built once and executed against every supplied capture. Each
node output is compared with its captured allocation, and the M9 gates (47
resource edges / 58 hand-offs, d24 exact, d25 bound, downstream propagation,
d38 exact) are reused from `m9_graph.py`.
"""
# Internal legacy research module; not a supported CLI entry point.
import argparse
import hashlib
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from . import (d25_bound, dynamic_conv, dynamic_flow, dynamic_index,
               dynamic_tail, dynamic_warp_init, export_onnx, m9_graph, recon)


# Model input `xN` -> the dispatch SRV slot it is fed from. The composite is
# handled separately because its inputs carry the slot name (`in_tN`).
SLOTS = {
    "conv": ["t0"],
    "flow conv": ["t0"],
    "upsample": ["t0", "t4"],
    "GEMV": ["t0", "t4"],
    "buffer output": ["t0", "t1"],
    "resample": ["t0", "t1"],
    "downsample": ["t0", "t1"],
    "warp": ["t0", "t1"],
    "splat": ["t0", "t1"],
    "warp + flow init": ["t0", "t2", "t3", "t4", "t5", "t6", "t8"],
}

# Model output name -> UAV slot for the families that do not use `uN`.
OUTPUT_SLOTS = {
    "downsample": {"u0": "ds_u0_out", "u1": "ds_u1_out"},
    "resample": {"u0": "y0"},
}

INT8, UINT8, FLOAT, UINT32 = (TensorProto.INT8, TensorProto.UINT8,
                              TensorProto.FLOAT, TensorProto.UINT32)


class GBuilder:
    """A Builder that prefixes every name, for adapter subgraphs."""

    def __init__(self, prefix):
        self.prefix = prefix
        self.nodes = []
        self.initializers = []
        self.protected = set()

    def protect(self, value):
        """A name that already lives in the assembled graph."""
        self.protected.add(value)
        return value

    def name(self, value):
        if value in self.protected:
            return value
        return self.prefix + value

    def add(self, op, inputs, outputs, **attrs):
        self.nodes.append(helper.make_node(
            op, [self.name(value) for value in inputs],
            [self.name(value) for value in outputs], **attrs))

    def tensor(self, name, array):
        self.initializers.append(numpy_helper.from_array(array, self.name(name)))

    def const(self, name, op, values):
        dtype = {"i32": np.int32, "i64": np.int64, "u8": np.uint8,
                 "i8": np.int8, "u32": np.uint32, "fp32": np.float32,
                 "fp64": np.float64}[op]
        self.tensor(name, np.asarray(values, dtype=dtype))


def _tensor_type(model, name, outputs=False):
    collection = model.graph.output if outputs else model.graph.input
    for value in collection:
        if value.name == name:
            return value.type.tensor_type.elem_type
    for value in model.graph.value_info:
        if value.name == name:
            return value.type.tensor_type.elem_type
    raise KeyError(name)


def _model_sha(model):
    return hashlib.sha256(model.SerializeToString()).hexdigest()


def _input_slots(family, model):
    if family == "final composite":
        mapping = {"x0": "t0"}
        for value in model.graph.input:
            if value.name.startswith("in_"):
                mapping[value.name] = value.name[3:]
        return mapping
    return {"x%d" % index: slot
            for index, slot in enumerate(SLOTS[family])}


def _expected(family, plan):
    if family == "downsample":
        return {"u0": plan["expected0"], "u1": plan["expected1"]}
    if family == "warp + flow init":
        return {slot: plan["expected"][index]
                for index, slot in enumerate(("u0", "u1", "u2", "u3"))}
    return {"u0": plan["expected"]}


def normalized_dispatches(run_dir, frame=1):
    """The frame's dispatch rows with capture ids remapped to nodes 1..38.

    Every other field (resources, key, sha256, tensor file names) is kept as
    captured, so this is only the topology view `Assembler` needs. The original
    id is retained as `capture_dispatch`.
    """
    rows = recon.read_events(run_dir)
    dispatches = dynamic_index.frame_dispatches(run_dir, frame, rows)
    numbers = dynamic_index.normalized_number_map(dispatches)
    normalized = []
    for dispatch in dispatches:
        copy = dict(dispatch)
        copy["capture_dispatch"] = dispatch["dispatch"]
        copy["dispatch"] = numbers[dispatch["dispatch"]]
        normalized.append(copy)
    return normalized


def build_specs(run_dir, frame=1):
    """Plan, build and feed every dispatch of one retained frame.

    Specs are keyed by the frame-relative node id (1..38) so the assembler and
    its graph are identical on every frame. Each plan keeps the original
    capture id as `capture_dispatch`; all tensor reads use the capture row.
    """
    rows = recon.read_events(run_dir)
    dispatches = dynamic_index.frame_dispatches(run_dir, frame, rows)
    numbers = dynamic_index.normalized_number_map(dispatches)
    conv = dynamic_conv._conv_plans(run_dir, frame)
    flow, upsample = dynamic_flow._plans(run_dir, frame)
    warp, composite = dynamic_tail._plans(run_dir, frame)
    resample = dynamic_index._resample_plans(run_dir, frame)
    downsample = dynamic_index._downsample_plans(run_dir, frame)
    gemv = dynamic_index._gemv_plans(run_dir, frame)
    buffer_output = dynamic_index._buffer_plans(run_dir, frame)
    splat = dynamic_index._splat_plan(run_dir, frame)

    specs = {}
    for dispatch in dispatches:
        capture_number = dispatch["dispatch"]
        number = numbers[capture_number]
        family = m9_graph.classify(dispatch,
                                   base=capture_number - number)
        if family == "conv":
            plan = conv[number]
            model = dynamic_conv.build_dynamic_conv(plan)
            feeds = dynamic_conv._feeds(plan)
        elif family == "flow conv":
            plan = flow[number]
            model = dynamic_flow.build_dynamic_flow(plan)
            feeds = dynamic_flow._flow_feeds(plan)
        elif family == "upsample":
            plan = upsample[number]
            model = dynamic_flow.build_dynamic_upsample(plan)
            feeds = dynamic_flow._upsample_feeds(plan)
        elif family == "GEMV":
            plan = gemv
            model = dynamic_index.build_dynamic_gemv(plan)
            feeds = dynamic_index._gemv_feeds(plan)
        elif family == "buffer output":
            plan = buffer_output[number]
            model = dynamic_index.build_dynamic_buffer_output(plan)
            feeds = dynamic_index._buffer_feeds(plan)
        elif family == "resample":
            plan = resample[number]
            model = dynamic_index.build_dynamic_resample(plan)
            feeds = dynamic_index._resample_feeds(plan)
        elif family == "downsample":
            plan = downsample
            model = dynamic_index.build_dynamic_downsample(plan)
            feeds = dynamic_index._downsample_feeds(plan)
        elif family == "warp":
            plan = warp
            model = dynamic_tail.build_dynamic_warp(plan)
            feeds = dynamic_tail._warp_feeds(plan)
        elif family == "splat":
            plan = splat
            model = dynamic_index.build_dynamic_splat(plan)
            feeds = dynamic_index._splat_feeds(plan)
        elif family == "final composite":
            plan = composite
            model = dynamic_tail.build_dynamic_composite(plan)
            feeds = dynamic_tail._composite_feeds(plan)
        else:  # warp + flow init
            plan, error = recon.warp_init_plan(run_dir, dispatch)
            if error:
                raise SystemExit("d%04d: %s" % (number, error))
            plan["capture_dispatch"] = capture_number
            plan["dispatch"] = number
            model = dynamic_warp_init.build_dynamic(plan)
            feeds = dynamic_warp_init.feeds(plan)

        output_names = OUTPUT_SLOTS.get(
            family, {value.name: value.name for value in model.graph.output})
        specs[number] = {
            "dispatch": number,
            "capture_dispatch": capture_number,
            "family": family,
            "plan": plan,
            "model": model,
            "model_sha256": _model_sha(model),
            "feeds": feeds,
            "input_slots": _input_slots(family, model),
            "output_names": output_names,
            "expected": _expected(family, plan),
        }
    if sorted(specs) != list(range(1, 39)):
        raise SystemExit("found dispatches %s" % sorted(specs))
    return specs


def _float_to_f16_bits(g, tag, source):
    """Exact f32 -> f16 bit pattern, without a bitcast.

    The exponent/mantissa are reconstructed from the float value in double
    precision (`Log`/`Pow` with the two boundary corrections), then the same
    integer chain as `recon._f32_to_half_bits` runs on the reconstructed bits.
    The reconstruction is exact for every finite normal float32; zeros,
    subnormals, infinities and NaNs take explicit branches.
    """
    b = g
    b.add("Abs", [source], [tag + "_a"])
    b.add("Cast", [tag + "_a"], [tag + "_d"], to=TensorProto.DOUBLE)
    b.add("Log", [tag + "_d"], [tag + "_log"])
    b.const(tag + "_ln2", "fp64", [float(np.log(2.0))])
    b.add("Div", [tag + "_log", tag + "_ln2"], [tag + "_l2"])
    b.add("Floor", [tag + "_l2"], [tag + "_e0"])
    b.const(tag + "_two_d", "fp64", [2.0])
    b.add("Pow", [tag + "_two_d", tag + "_e0"], [tag + "_p0"])
    b.add("Less", [tag + "_d", tag + "_p0"], [tag + "_lo"])
    b.const(tag + "_one_d", "fp64", [1.0])
    b.add("Sub", [tag + "_e0", tag + "_one_d"], [tag + "_e0m"])
    b.add("Div", [tag + "_p0", tag + "_two_d"], [tag + "_p0h"])
    b.add("Where", [tag + "_lo", tag + "_e0m", tag + "_e0"], [tag + "_e1"])
    b.add("Where", [tag + "_lo", tag + "_p0h", tag + "_p0"], [tag + "_p1"])
    b.add("Mul", [tag + "_p1", tag + "_two_d"], [tag + "_p1x2"])
    b.add("GreaterOrEqual", [tag + "_d", tag + "_p1x2"], [tag + "_hi"])
    b.add("Add", [tag + "_e1", tag + "_one_d"], [tag + "_e1p"])
    b.add("Mul", [tag + "_p1", tag + "_two_d"], [tag + "_p2"])
    b.add("Where", [tag + "_hi", tag + "_e1p", tag + "_e1"], [tag + "_e"])
    b.add("Where", [tag + "_hi", tag + "_p2", tag + "_p1"], [tag + "_p"])
    b.add("Div", [tag + "_d", tag + "_p"], [tag + "_s"])
    b.add("Sub", [tag + "_s", tag + "_one_d"], [tag + "_sm"])
    b.const(tag + "_2p23_d", "fp64", [float(1 << 23)])
    b.add("Mul", [tag + "_sm", tag + "_2p23_d"], [tag + "_mm"])
    b.add("Floor", [tag + "_mm"], [tag + "_mf"])
    b.add("Cast", [tag + "_mf"], [tag + "_mant"], to=TensorProto.INT64)
    b.add("Cast", [tag + "_e"], [tag + "_ei"], to=TensorProto.INT64)
    b.const(tag + "_c127", "i64", [127])
    b.add("Add", [tag + "_ei", tag + "_c127"], [tag + "_eb"])
    b.const(tag + "_c23", "i64", [1 << 23])
    b.add("Mul", [tag + "_eb", tag + "_c23"], [tag + "_esh"])
    b.add("Add", [tag + "_esh", tag + "_mant"], [tag + "_bits_norm"])

    # zero / subnormal / inf / nan branches
    b.const(tag + "_zero_f", "fp32", [0.0])
    b.add("Equal", [tag + "_a", tag + "_zero_f"], [tag + "_zero"])
    b.const(tag + "_min_normal", "fp32", [float(np.float32(1.17549435e-38))])
    b.add("Less", [tag + "_a", tag + "_min_normal"], [tag + "_small"])
    b.add("Not", [tag + "_zero"], [tag + "_nonzero"])
    b.add("And", [tag + "_small", tag + "_nonzero"], [tag + "_subnormal"])
    b.add("Cast", [tag + "_a"], [tag + "_ad"], to=TensorProto.DOUBLE)
    b.const(tag + "_2p149", "fp64", [float(2.0 ** 149)])
    b.add("Mul", [tag + "_ad", tag + "_2p149"], [tag + "_subd"])
    b.add("Cast", [tag + "_subd"], [tag + "_subbits"], to=TensorProto.INT64)
    b.add("IsInf", [tag + "_a"], [tag + "_inf"])
    b.add("IsNaN", [tag + "_a"], [tag + "_nan"])
    b.const(tag + "_inf_bits", "i64", [0x7F800000])
    b.const(tag + "_nan_bits", "i64", [0x7FC00000])
    b.const(tag + "_0", "i64", [0])
    b.add("Where", [tag + "_zero", tag + "_0", tag + "_bits_norm"],
          [tag + "_bits_z"])
    b.add("Where", [tag + "_subnormal", tag + "_subbits", tag + "_bits_z"],
          [tag + "_bits_s"])
    b.add("Where", [tag + "_inf", tag + "_inf_bits", tag + "_bits_s"],
          [tag + "_bits_i"])
    b.add("Where", [tag + "_nan", tag + "_nan_bits", tag + "_bits_i"],
          [tag + "_bits"])
    b.add("Cast", [tag + "_bits"], [tag + "_bits_u"], to=TensorProto.UINT32)

    # sign, including -0 (1/-0 == -inf)
    b.const(tag + "_one_f", "fp32", [1.0])
    b.add("Div", [tag + "_one_f", source], [tag + "_inv"])
    b.const(tag + "_zero_f2", "fp32", [0.0])
    b.add("Less", [tag + "_inv", tag + "_zero_f2"], [tag + "_neg"])
    b.const(tag + "_sign", "u32", [0x80000000])
    b.const(tag + "_0u", "u32", [0])
    b.add("Where", [tag + "_neg", tag + "_sign", tag + "_0u"],
          [tag + "_sgn"])
    b.add("BitwiseOr", [tag + "_sgn", tag + "_bits_u"], [tag + "_f32"])
    return _f32_bits_to_f16(b, tag + "_h", tag + "_f32")


def _f32_bits_to_f16(b, tag, bits):
    """`recon._f32_to_half_bits` on a uint32 tensor holding the f32 pattern."""
    b.const(tag + "_4096", "u32", [4096])
    b.add("Add", [bits, tag + "_4096"], [tag + "_rounded"])
    b.const(tag + "_23", "u32", [23])
    b.add("BitShift", [tag + "_rounded", tag + "_23"], [tag + "_e"],
          direction="RIGHT")
    b.const(tag + "_mask8", "u32", [0xFF])
    b.add("BitwiseAnd", [tag + "_e", tag + "_mask8"], [tag + "_exp"])
    b.const(tag + "_mask23", "u32", [0x7FFFFF])
    b.add("BitwiseAnd", [tag + "_rounded", tag + "_mask23"],
          [tag + "_mant"])
    b.const(tag + "_16", "u32", [16])
    b.add("BitShift", [tag + "_rounded", tag + "_16"], [tag + "_sh16"],
          direction="RIGHT")
    b.const(tag + "_mask16", "u32", [0x8000])
    b.add("BitwiseAnd", [tag + "_sh16", tag + "_mask16"], [tag + "_sign"])

    # normal = ((exp + 16) & 31) << 10 | (mant >> 13)
    b.add("Add", [tag + "_exp", tag + "_16"], [tag + "_exp16"])
    b.const(tag + "_mask5", "u32", [0x1F])
    b.add("BitwiseAnd", [tag + "_exp16", tag + "_mask5"], [tag + "_exp5"])
    b.const(tag + "_10", "u32", [10])
    b.add("BitShift", [tag + "_exp5", tag + "_10"], [tag + "_exp10"],
          direction="LEFT")
    b.const(tag + "_13", "u32", [13])
    b.add("BitShift", [tag + "_mant", tag + "_13"], [tag + "_mant13"],
          direction="RIGHT")
    b.add("BitwiseOr", [tag + "_exp10", tag + "_mant13"], [tag + "_normal0"])
    b.const(tag + "_112", "u32", [112])
    b.add("Greater", [tag + "_exp", tag + "_112"], [tag + "_is_normal"])
    b.const(tag + "_0b", "u32", [0])
    b.add("Where", [tag + "_is_normal", tag + "_normal0", tag + "_0b"],
          [tag + "_normal"])

    # subnormal = ((mant + 8384512) >> ((125 - exp) & 31) + 1) >> 1
    b.const(tag + "_8384512", "u32", [8384512])
    b.add("Add", [tag + "_mant", tag + "_8384512"], [tag + "_sub_n"])
    b.const(tag + "_125", "u32", [125])
    b.add("Sub", [tag + "_125", tag + "_exp"], [tag + "_sub_shift"])
    b.const(tag + "_mask31", "u32", [31])
    b.add("BitwiseAnd", [tag + "_sub_shift", tag + "_mask31"],
          [tag + "_sub_sh"])
    b.add("BitShift", [tag + "_sub_n", tag + "_sub_sh"], [tag + "_sub_a"],
          direction="RIGHT")
    b.const(tag + "_1b", "u32", [1])
    b.add("Add", [tag + "_sub_a", tag + "_1b"], [tag + "_sub_b"])
    b.add("BitShift", [tag + "_sub_b", tag + "_1b"], [tag + "_sub_c"],
          direction="RIGHT")
    b.const(tag + "_102", "u32", [102])
    b.add("Sub", [tag + "_exp", tag + "_102"], [tag + "_sub_ok0"])
    b.const(tag + "_11", "u32", [11])
    b.add("Less", [tag + "_sub_ok0", tag + "_11"], [tag + "_sub_ok"])
    b.add("Where", [tag + "_sub_ok", tag + "_sub_c", tag + "_0b"],
          [tag + "_subnormal"])

    b.const(tag + "_143", "u32", [143])
    b.add("Greater", [tag + "_exp", tag + "_143"], [tag + "_special_ok"])
    b.const(tag + "_32767", "u32", [32767])
    b.add("Where", [tag + "_special_ok", tag + "_32767", tag + "_0b"],
          [tag + "_special"])

    b.add("BitwiseOr", [tag + "_sign", tag + "_normal"], [tag + "_or1"])
    b.add("BitwiseOr", [tag + "_or1", tag + "_subnormal"], [tag + "_or2"])
    b.add("BitwiseOr", [tag + "_or2", tag + "_special"], [tag + "_bits16"])
    return tag + "_bits16"


def _bytes_from_bits(b, tag, bits):
    """uint8 [..., 2] little-endian bytes for a uint16 pattern."""
    b.const(tag + "_mask8", "u32", [0xFF])
    b.add("BitwiseAnd", [bits, tag + "_mask8"], [tag + "_low"])
    b.const(tag + "_8", "u32", [8])
    b.add("BitShift", [bits, tag + "_8"], [tag + "_high0"], direction="RIGHT")
    b.add("BitwiseAnd", [tag + "_high0", tag + "_mask8"], [tag + "_high"])
    b.add("Cast", [tag + "_low"], [tag + "_low8"], to=TensorProto.UINT8)
    b.add("Cast", [tag + "_high"], [tag + "_high8"], to=TensorProto.UINT8)
    b.const(tag + "_ax3", "i64", [3])
    b.add("Unsqueeze", [tag + "_low8", tag + "_ax3"], [tag + "_low_u"])
    b.add("Unsqueeze", [tag + "_high8", tag + "_ax3"], [tag + "_high_u"])
    b.add("Concat", [tag + "_low_u", tag + "_high_u"], [tag + "_bytes4"],
          axis=3)
    b.add("Shape", [bits], [tag + "_shape"])
    b.const(tag + "_i2", "i64", [2])
    b.add("Gather", [tag + "_shape", tag + "_i2"], [tag + "_c"], axis=0)
    b.const(tag + "_two", "i64", [2])
    b.add("Mul", [tag + "_c", tag + "_two"], [tag + "_c2"])
    b.const(tag + "_i0", "i64", [0])
    b.const(tag + "_i1", "i64", [1])
    b.add("Gather", [tag + "_shape", tag + "_i0"], [tag + "_h"], axis=0)
    b.add("Gather", [tag + "_shape", tag + "_i1"], [tag + "_w"], axis=0)
    b.add("Concat", [tag + "_h", tag + "_w", tag + "_c2"],
          [tag + "_bytes_shape"], axis=0)
    b.add("Reshape", [tag + "_bytes4", tag + "_bytes_shape"], [tag + "_bytes"])
    return tag + "_bytes"


def _decoded_float(b, tag, bits):
    b.add("Cast", [bits], [tag + "_u32"], to=TensorProto.UINT32)
    value = export_onnx._half_bits_to_float(b, tag, tag + "_u32")
    return value


def _word_pack_graph(g, source):
    """uint8 bytes -> little-endian uint32 words, into builder `g`."""
    g.add("Shape", [source], ["shape"])
    g.const("i0", "i64", [0])
    g.add("Gather", ["shape", "i0"], ["n"], axis=0)
    g.const("four", "i64", [4])
    g.add("Div", ["n", "four"], ["n4"])
    g.add("Concat", ["n4", "four"], ["words_shape"], axis=0)
    g.add("Reshape", [source, "words_shape"], ["bytes"])
    g.const("one", "i64", [1])
    g.add("Split", ["bytes"], ["b0", "b1", "b2", "b3"], axis=1,
          num_outputs=4)
    for name in ("b0", "b1", "b2", "b3"):
        g.const(name + "_ax", "i64", [1])
        g.add("Squeeze", [name, name + "_ax"], [name + "_s"])
        g.add("Cast", [name + "_s"], [name + "_u"], to=TensorProto.UINT32)
    g.const("s8", "u32", [8])
    g.const("s16", "u32", [16])
    g.const("s24", "u32", [24])
    g.add("BitShift", ["b1_u", "s8"], ["b1_l"], direction="LEFT")
    g.add("BitShift", ["b2_u", "s16"], ["b2_l"], direction="LEFT")
    g.add("BitShift", ["b3_u", "s24"], ["b3_l"], direction="LEFT")
    g.add("BitwiseOr", ["b0_u", "b1_l"], ["o1"])
    g.add("BitwiseOr", ["o1", "b2_l"], ["o2"])
    g.add("BitwiseOr", ["o2", "b3_l"], ["out"])
    return g.name("out")


class Assembler:
    def __init__(self, specs, dispatches):
        self.specs = specs
        self.dispatches = dispatches
        self.edges = m9_graph.edge_index(dispatches)
        self.writer_views, self.reader_views = m9_graph.view_maps(dispatches)
        self.nodes = []
        self.initializers = []
        self.inputs = []
        self.outputs = []
        self.produced = {}
        self.produced_type = {}
        self.produced_rank = {}
        self.resample_bits = {}
        self.resample_pack = {}
        self.resample_f16 = {}
        self.cut_inputs = {}
        self.extra_feeds = {}
        self.shape_feeds = {}
        self.output_specs = []
        self.problems = []

    # -- graph plumbing ----------------------------------------------------
    def _add_input(self, name, elem, scalar=False):
        self.inputs.append(helper.make_tensor_value_info(
            name, elem, [1] if scalar else [None]))

    def _add_input_like(self, name, value):
        """Copy a submodel input's rank but leave every extent symbolic."""
        info = helper.ValueInfoProto()
        info.CopyFrom(value)
        info.name = name
        dims = info.type.tensor_type.shape.dim
        for index in range(len(dims)):
            dims[index].Clear()
            dims[index].dim_param = "%s_D%d" % (name, index)
        self.inputs.append(info)

    def _add_output(self, name, elem, rank=1):
        self.outputs.append(helper.make_tensor_value_info(
            name, elem, ["O%d" % index for index in range(rank)]))

    def _add_output_like(self, name, value):
        """Copy an output's rank, with every extent symbolic."""
        info = helper.ValueInfoProto()
        info.CopyFrom(value)
        info.name = name
        dims = info.type.tensor_type.shape.dim
        for index in range(len(dims)):
            dims[index].Clear()
            dims[index].dim_param = "%s_O%d" % (name, index)
        self.outputs.append(info)

    def _append(self, builder):
        self.nodes.extend(builder.nodes)
        self.initializers.extend(builder.initializers)

    # -- adapters ----------------------------------------------------------
    def _flatten(self, prefix, source):
        g = GBuilder(prefix)
        g.protect(source)
        g.const("flat", "i64", [-1])
        g.add("Reshape", [source, "flat"], ["out"])
        self._append(g)
        return g.name("out")

    def _cast(self, prefix, source, elem):
        g = GBuilder(prefix)
        g.protect(source)
        g.add("Cast", [source], ["out"], to=elem)
        self._append(g)
        return g.name("out")

    def _view_adapter(self, prefix, source, source_elem, producer, consumer):
        if m9_graph._view_shape(producer) == m9_graph._view_shape(consumer):
            return source
        if producer.get("dimension") != consumer.get("dimension"):
            raise ValueError("view dimension %s -> %s"
                             % (producer.get("dimension"),
                                consumer.get("dimension")))
        if producer.get("dimension") == "texture2darray":
            if (producer.get("width"), producer.get("height")) != (
                    consumer.get("width"), consumer.get("height")):
                raise ValueError("texture2darray crop is not supported")
            slice_bytes = m9_graph._slice_bytes(producer)
            if slice_bytes is None:
                raise ValueError("no texel size for texture2darray")
            start = int(consumer.get("first_slice") or 0) * slice_bytes
            end = start + int(consumer.get("array") or 1) * slice_bytes
        elif producer.get("dimension") == "buffer":
            start = int(consumer.get("buffer_offset") or 0)
            size = int(consumer.get("width") or 0) * int(consumer.get("height") or 1)
            end = start + size if size > 0 else 1 << 62
        else:
            raise ValueError("unsupported view translation for %s"
                             % producer.get("dimension"))
        item = {UINT8: 1, INT8: 1, TensorProto.UINT16: 2,
                TensorProto.FLOAT16: 2, UINT32: 4, FLOAT: 4}[source_elem]
        if start % item or end % item:
            raise ValueError("view byte offset is not aligned to the dtype")
        g = GBuilder(prefix)
        g.protect(source)
        g.const("flat", "i64", [-1])
        g.add("Reshape", [source, "flat"], ["flat_t"])
        g.const("start", "i64", [start // item])
        g.const("end", "i64", [end // item])
        g.const("axis", "i64", [0])
        g.const("step", "i64", [1])
        g.add("Slice", ["flat_t", "start", "end", "axis", "step"], ["out"])
        self._append(g)
        return g.name("out")

    def _word_pack(self, prefix, source):
        g = GBuilder(prefix)
        g.protect(source)
        out = _word_pack_graph(g, source)
        self._append(g)
        return out

    def _resample_pack(self, spec):
        """The f16 byte allocation, plus the bits for a later decode."""
        number = spec["dispatch"]
        plan = spec["plan"]
        prefix = "n%04d_pack_" % number
        g = GBuilder(prefix)
        raw = g.protect("n%04d_y0" % number)
        bits = _float_to_f16_bits(g, "bits", raw)
        packed = _bytes_from_bits(g, "bytes", bits)

        # Pad [H, W, 2C] into [rows_alloc, W, cell_bytes].
        g.add("Shape", [packed], ["shape"])
        g.const("i0", "i64", [0])
        g.const("i1", "i64", [1])
        g.const("i2", "i64", [2])
        g.add("Gather", ["shape", "i0"], ["h"], axis=0)
        g.add("Gather", ["shape", "i2"], ["c2"], axis=0)
        rows = g.protect(prefix + "rows_alloc")
        start = g.protect(prefix + "start_row")
        cell = g.protect(prefix + "cell_bytes")
        self._add_input(rows, TensorProto.INT64, scalar=True)
        self._add_input(start, TensorProto.INT64, scalar=True)
        self._add_input(cell, TensorProto.INT64, scalar=True)
        self.extra_feeds[rows] = (number, "rows_alloc")
        self.extra_feeds[start] = (number, "start_row")
        self.extra_feeds[cell] = (number, "cell_bytes")
        g.add("Sub", [rows, start], ["r0"])
        g.add("Sub", ["r0", "h"], ["bottom"])
        g.add("Sub", [cell, "c2"], ["right"])
        g.const("zero", "i64", [0])
        g.add("Concat", [start, "zero", "zero", "bottom", "zero", "right"],
              ["pads"], axis=0)
        g.tensor("pad_value", np.asarray([0], dtype=np.uint8))
        g.add("Pad", [packed, "pads", "pad_value"], ["padded"])
        g.const("flat", "i64", [-1])
        g.add("Reshape", ["padded", "flat"], ["out"])

        # The f16 consumers read the whole allocation as FLOAT16 (padding
        # included). Build it from the same bits, decode to float32 and cast
        # back: the value is exactly representable, so the bits are preserved.
        g.add("Shape", [bits], ["bit_shape"])
        g.add("Gather", ["bit_shape", "i0"], ["bh"], axis=0)
        g.add("Gather", ["bit_shape", "i2"], ["bc"], axis=0)
        g.const("two", "i64", [2])
        g.add("Div", [cell, "two"], ["half_cell"])
        g.add("Sub", ["half_cell", "bc"], ["right_cells"])
        g.add("Sub", [rows, start], ["rbits0"])
        g.add("Sub", ["rbits0", "bh"], ["bottom_bits"])
        g.add("Concat", [start, "zero", "zero", "bottom_bits", "zero",
                         "right_cells"], ["bit_pads"], axis=0)
        g.tensor("bit_zero", np.asarray([0], dtype=np.uint32))
        g.add("Pad", [bits, "bit_pads", "bit_zero"], ["bits_padded"])
        decoded = _decoded_float(g, "f16d", "bits_padded")
        g.add("Cast", [decoded], ["f16_val"], to=TensorProto.FLOAT16)
        g.const("flat2", "i64", [-1])
        g.add("Reshape", ["f16_val", "flat2"], ["f16_out"])
        self._append(g)
        return g.name("out"), g.name(bits), g.name("f16_out")

    # -- assembly ----------------------------------------------------------
    def add_spec(self, spec):
        number = spec["dispatch"]
        model = spec["model"]
        prefix = "n%04d_" % number
        names = {value.name for value in model.graph.input}
        names |= {value.name for value in model.graph.output}
        for value in model.graph.initializer:
            names.add(value.name)
        for node in model.graph.node:
            names.update(value for value in node.input if value)
            names.update(value for value in node.output if value)
        rename = {name: prefix + name for name in names}

        for value in model.graph.input:
            slot = spec["input_slots"].get(value.name)
            elem = value.type.tensor_type.elem_type
            rank = len(value.type.tensor_type.shape.dim)
            target = np.asarray(spec["feeds"][value.name]).shape
            if slot is not None and (number, slot) in self.edges:
                producer = self.edges[(number, slot)]
                rename[value.name] = self._wire(
                    producer, self.reader_views[(number, slot)], elem,
                    number, slot, rank, value.name, target)
            else:
                global_name = prefix + value.name
                self._add_input_like(global_name, value)
                self.cut_inputs[global_name] = (number, value.name)
                rename[value.name] = global_name

        for init in model.graph.initializer:
            copy = onnx.TensorProto()
            copy.CopyFrom(init)
            copy.name = prefix + init.name
            self.initializers.append(copy)
        for node in model.graph.node:
            copy = onnx.NodeProto()
            copy.CopyFrom(node)
            if node.name:
                copy.name = prefix + node.name
            copy.input[:] = [rename.get(value, value) if value else ""
                             for value in node.input]
            copy.output[:] = [rename.get(value, value) for value in node.output]
            self.nodes.append(copy)

        for slot, output_name in spec["output_names"].items():
            elem = _tensor_type(model, output_name, outputs=True)
            raw = prefix + output_name
            if spec["family"] == "resample":
                canonical, bits, f16 = self._resample_pack(spec)
                self.resample_bits[(number, slot)] = bits
                self.resample_pack[(number, slot)] = canonical
                self.resample_f16[(number, slot)] = f16
                elem = UINT8
                self._add_output(canonical, elem, rank=1)
                rank = 1
            else:
                canonical = raw
                info = [value for value in model.graph.output
                        if value.name == output_name][0]
                self._add_output_like(canonical, info)
                rank = len(info.type.tensor_type.shape.dim)
            self.produced[(number, slot)] = canonical
            self.produced_type[(number, slot)] = elem
            self.produced_rank[(number, slot)] = rank
            self.output_specs.append((number, slot, canonical, elem))

    def _wire(self, producer, consumer_view, consumer_elem, number, slot,
              consumer_rank, input_name, target_shape):
        producer_number, producer_slot = producer
        spec = self.specs[producer_number]
        source = self.produced[producer]
        source_elem = self.produced_type[producer]
        prefix = "n%04d_%s_to_%04d_%s_" % (producer_number, producer_slot,
                                           number, slot)
        if len(target_shape) != consumer_rank:
            raise ValueError("d%04d.%s target rank %d != declared %d"
                             % (number, input_name, len(target_shape),
                                consumer_rank))

        source_rank = self.produced_rank.get(producer, 1)
        if m9_graph._view_shape(self.writer_views[producer]) != \
                m9_graph._view_shape(consumer_view):
            source_rank = 1
        source = self._view_adapter(
            prefix + "view_", source, source_elem,
            self.writer_views[producer], consumer_view)

        if spec["family"] == "resample":
            if consumer_elem == FLOAT:
                # d23 -> d25 reads the f16 allocation as float32 planes.
                # d25 consumes plane 0 of the f16 buffer; the producer's
                # logical rect is exactly that plane at every capture.
                g = GBuilder(prefix + "decode_")
                g.protect(self.resample_bits[producer])
                _decoded_float(g, "dec", self.resample_bits[producer])
                g.const("flat", "i64", [-1])
                g.add("Reshape", ["dec_out", "flat"], ["out"])
                self._append(g)
                source = g.name("out")
                source_elem = FLOAT
            elif consumer_elem == TensorProto.FLOAT16:
                source = self.resample_f16[producer]
                source_elem = TensorProto.FLOAT16
            else:
                source = self.resample_pack[producer]
                source_elem = UINT8
            source_rank = 1
        elif spec["family"] == "splat" and consumer_elem == UINT32:
            source = self._word_pack(prefix + "words_", source)
            source_elem = UINT32
            source_rank = 1

        # The consumer's feed array is the authority on its input layout: keep
        # the wire flat and reshape to that runtime shape (a feed, not a
        # baked extent). Byte counts must agree.
        if source_rank != 1:
            source = self._flatten(prefix + "flat_", source)
            source_rank = 1
        if consumer_rank > 1:
            shape_name = prefix + "shape"
            self._add_input(shape_name, TensorProto.INT64, scalar=False)
            self.shape_feeds[shape_name] = (number, input_name)
            g = GBuilder(prefix + "reshape_")
            g.protect(source)
            g.protect(shape_name)
            g.add("Reshape", [source, shape_name], ["out"])
            self._append(g)
            source = g.name("out")
            source_rank = consumer_rank

        if source_elem != consumer_elem:
            if (source_elem in (INT8, UINT8)
                    and consumer_elem in (INT8, UINT8)):
                source = self._cast(prefix + "cast_", source, consumer_elem)
                source_elem = consumer_elem
            else:
                raise ValueError(
                    "d%04d.%s -> d%04d.%s: cannot translate type %d to %d"
                    % (producer_number, producer_slot, number, slot,
                       source_elem, consumer_elem))
        return source

    def build(self):
        for number in sorted(self.specs):
            self.add_spec(self.specs[number])
        graph = helper.make_graph(
            self.nodes, "xefg_m9_dynamic", self.inputs, self.outputs,
            self.initializers)
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 18)])
        model.ir_version = 8
        return model

    def feeds_for(self, specs):
        values = {}
        for global_name, (number, input_name) in self.cut_inputs.items():
            values[global_name] = specs[number]["feeds"][input_name]
        for global_name, (number, key) in self.extra_feeds.items():
            plan = specs[number]["plan"]
            values[global_name] = np.asarray([plan[key]], dtype=np.int64)
        for global_name, (number, input_name) in self.shape_feeds.items():
            shape = np.asarray(specs[number]["feeds"][input_name]).shape
            values[global_name] = np.asarray(shape, dtype=np.int64)
        return values


def compare(run_dir, specs, session, assembler, outputs):
    records = []
    errors = []
    offset = 0
    for number in sorted(specs):
        spec = specs[number]
        slots = sorted(spec["expected"])
        mismatch = 0
        max_diff = 0
        total = 0
        failed = False
        for index, slot in enumerate(slots):
            got = outputs[offset + index]
            expected = spec["expected"][slot]
            try:
                count, largest = m9_graph.diff_count(got, expected)
            except Exception as exc:
                failed = True
                errors.append({"dispatch": number,
                               "error": "d%04d.%s compare: %s"
                                        % (number, slot, exc)})
                continue
            mismatch += count
            max_diff = max(max_diff, largest)
            total += int(np.asarray(expected).size)
        offset += len(slots)
        record = {
            "dispatch": number,
            "node": number,
            "family": spec["family"],
            "bytes": total,
            "mismatch": mismatch,
            "max_diff": max_diff,
            "host": False,
        }
        if failed:
            record["error"] = "comparison failed"
        records.append(record)
    return records, errors


def run(run_dirs, out_dir, frame=1, strict=False, force_structural=False):
    os.makedirs(out_dir, exist_ok=True)
    paths = [os.path.abspath(value) for value in run_dirs]
    all_specs = [build_specs(path, frame) for path in paths]

    # One model per dispatch: every capture must build the same graph.
    problems = []
    for number in sorted(all_specs[0]):
        reference = all_specs[0][number]["model_sha256"]
        for index, specs in enumerate(all_specs[1:], start=1):
            if specs[number]["model_sha256"] != reference:
                problems.append(
                    "d%04d model differs between %s and %s"
                    % (number, os.path.basename(paths[0]),
                       os.path.basename(paths[index])))

    assembler = Assembler(all_specs[0], normalized_dispatches(paths[0], frame))
    model = assembler.build()
    onnx.checker.check_model(model)
    path = os.path.join(out_dir, "xefg_m9_dynamic.onnx")
    onnx.save(model, path)
    model_sha = _model_sha(model)
    onnxruntime = export_onnx._require_onnxruntime()
    options = onnxruntime.SessionOptions()
    options.log_severity_level = 3
    session = onnxruntime.InferenceSession(
        path, sess_options=options, providers=["CPUExecutionProvider"])
    output_names = [entry[2] for entry in assembler.output_specs]

    per_run = []
    for run_dir, specs in zip(paths, all_specs):
        meta = json.load(open(os.path.join(run_dir, "trace", "meta.json"),
                              encoding="utf-8"))
        config = meta.get("configuration") or {}
        resolution = (config.get("width"), config.get("height"))
        content = (d25_bound.STATIC if config.get("static", True)
                   else d25_bound.MOTION)
        table = (m9_graph.M9_MOTION_CALIBRATION
                 if content == d25_bound.MOTION else m9_graph.M9_CALIBRATION)
        calibration = table.get(resolution)
        structural = force_structural or calibration is None
        feeds = assembler.feeds_for(specs)
        outputs = session.run(output_names, feeds)
        records, errors = compare(run_dir, specs, session, assembler, outputs)
        run_problems = list(problems)
        run_problems += m9_graph.result_problems(
            records, errors, "host", assembler.edges, strict, 0,
            None if structural else calibration, structural=structural,
            content=content)
        resources, handoffs = m9_graph.edge_counts(assembler.edges)
        per_run.append({
            "run": run_dir,
            "resolution": list(resolution),
            "content": content,
            "calibrated": calibration is not None,
            "structural": structural,
            "gate": "structural" if structural else "calibrated",
            "edges": resources,
            "handoffs": handoffs,
            "results": records,
            "errors": errors,
            "problems": run_problems,
            "passed": not run_problems,
        })

    report = {
        "model": path,
        "model_sha256": model_sha,
        "frame": frame,
        "nodes": len(model.graph.node),
        "inputs": len(model.graph.input),
        "outputs": len(model.graph.output),
        "edges": m9_graph.edge_counts(assembler.edges)[0],
        "handoffs": m9_graph.edge_counts(assembler.edges)[1],
        "runs": per_run,
        "passed": all(entry["passed"] for entry in per_run),
    }
    report_path = os.path.join(out_dir, "m9_dynamic_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True,
                        help="comma-separated run directories")
    parser.add_argument("--out", required=True)
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--structural", action="store_true",
                        help="use the size-independent gate even when this "
                             "resolution has a pinned calibration")
    args = parser.parse_args()
    report = run(args.runs.split(","), args.out, frame=args.frame,
                 strict=args.strict, force_structural=args.structural)
    for entry in report["runs"]:
        exact = sum(1 for row in entry["results"] if row["mismatch"] == 0)
        print("%s %s  %-10s exact %d/38  edges %d/%d  %s"
              % (os.path.basename(entry["run"]),
                 "x".join(str(value) for value in entry["resolution"]),
                 entry["gate"], exact, entry["edges"], entry["handoffs"],
                 "passed" if entry["passed"] else "FAILED"))
        for problem in entry["problems"]:
            print("  M9 FAILED: %s" % problem)
    print("model %s nodes=%d inputs=%d"
          % (os.path.basename(report["model"]), report["nodes"],
             report["inputs"]))
    print("passed" if report["passed"] else "FAILED")
    return 0 if report["passed"] else 1
