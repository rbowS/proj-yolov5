import onnx_graphsurgeon as gs
import numpy as np
import onnx


# Here we'll register a function to do all the subgraph-replacement heavy-lifting.
# NOTE: Since registered functions are entirely reusable, it may be a good idea to
# refactor them into a separate module so you can use them across all your models.
# 这里写成函数是为了，万一还需要这样的替换操作就可以重复利用了
@gs.Graph.register()
def replace_with_clip(self, inputs, outputs, opname):
    # Disconnect output nodes of all input tensors
    for inp in inputs:
        inp.outputs.clear()

    # Disconnet input nodes of all output tensors
    for out in outputs:
        out.inputs.clear()

    # Insert the new node.
    return self.layer(op=opname, inputs=inputs, outputs=outputs)


# Now we'll do the actual replacement
# 导入onnx模型
graph = gs.import_onnx(onnx.load("workspace/yolov5s.onnx"))

opname = "MYLNODE1"
tmap = graph.tensors()
# You can figure out the input and output tensors using Netron. In our case:
# Inputs: [inp, MIN_VAL, MAX_VAL]
# Outputs: [max_out]
# 子图的需要断开的输入name和子图需要断开的输出name
inputs = [tmap["onnx::Sigmoid_332"], tmap["onnx::Add_349"], tmap["onnx::Mul_463"]]
outputs = [tmap["onnx::Reshape_369"]]

# 断开并替换成新的名叫Clip的 OP
graph.replace_with_clip(inputs, outputs, opname)


opname = "MYLNODE2"
inputs = [tmap["onnx::Sigmoid_375"], tmap["onnx::Add_392"], tmap["onnx::Mul_467"]]
outputs = [tmap["onnx::Reshape_412"]]
# 断开并替换成新的名叫Clip的 OP
graph.replace_with_clip(inputs, outputs, opname)


opname = "MYLNODE3"
inputs = [tmap["onnx::Sigmoid_418"], tmap["onnx::Add_435"], tmap["onnx::Mul_471"]]
outputs = [tmap["onnx::Reshape_455"]]
# 断开并替换成新的名叫Clip的 OP
graph.replace_with_clip(inputs, outputs, opname)


# Remove the now-dangling subgraph.
graph.cleanup().toposort()

# That's it!
onnx.save(gs.export_onnx(graph), "workspace/replaced.onnx")
