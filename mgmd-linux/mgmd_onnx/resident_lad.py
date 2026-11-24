"""Run the three decoded LAD heads through ORT and concatenate on the user stream."""
import torch


def split_heads(path, batch=1):
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load_from_string(path) if isinstance(path, bytes) else onnx.load(str(path))
    # This exporter concatenates xy, wh and the object/class probabilities on
    # each scale. Export those inputs directly, pruning its four final Concats.
    by_output = {name: node for node in model.graph.node for name in node.output}
    final = by_output[model.graph.output[0].name]
    if final.op_type != "Concat" or len(final.input) != 3:
        raise ValueError("Unrecognized LAD detection head")
    outputs = []
    for scale, flattened in zip((40, 20, 10), final.input):
        reshape = by_output[flattened]
        head = by_output[reshape.input[0]]
        if reshape.op_type != "Reshape" or head.op_type != "Concat" or len(head.input) != 3:
            raise ValueError("Unrecognized LAD head layout")
        for name in head.input:
            outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [batch, 3, scale, scale, 2]))
    needed = {out.name for out in outputs}
    keep = []
    for node in reversed(model.graph.node):
        if needed.intersection(node.output):
            keep.append(node)
            needed.update(node.input)
    del model.graph.node[:]
    model.graph.node.extend(reversed(keep))
    initializers = [value for value in model.graph.initializer if value.name in needed]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    del model.graph.output[:]
    model.graph.output.extend(outputs)
    return model.SerializeToString()


def combine_heads(values):
    batch = values[0].shape[0]
    heads = [torch.cat(values[i:i + 3], dim=-1).reshape(batch, -1, 6) for i in range(0, 9, 3)]
    return torch.cat(heads, dim=1)
