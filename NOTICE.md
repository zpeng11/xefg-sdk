# Third-party notices

## Project license

The source in this repository is Copyright (c) 2026 Michael Robles and is
licensed under the MIT License; see [LICENSE](LICENSE). This notice does not
change the MIT grant for this repository's own source.

This repository is the standalone SDK extracted from the `MFG-Unlocker` project
at upstream commit `3d67197`.

## Intel XeSS Frame Generation-derived material

The reconstruction work from which this SDK was extracted studied the multi-frame
model of Intel XeSS Frame Generation. Intel-derived program data, extracted
weights, quantisation constants, the gate look-up table, and ONNX model assets
generated from captured Intel program data are Intel's work. They are not covered
by this repository's MIT License and remain subject to Intel's license terms for
the XeSS SDK.

The upstream `MFG-Unlocker` project states that its `MFGUnlocker.dll` embeds DXIL
and SPIR-V programs derived from Intel XeSS Frame Generation and does not include
Intel's XeSS SDK binaries, model weights, or tensors. This standalone repository
does not contain that DLL, those embedded shader programs, native binaries, Intel
SDK binaries, extracted weights, tensors, LUTs, ONNX models, or capture datasets.
It contains Python SDK source and optional C++/PowerShell capture-tool source,
plus documentation, examples, and tests. The capture tool uses a user-supplied
XeSS SDK at build/run time; no Intel headers or binaries are bundled.

A user who runs the exporter against captures, or who obtains a generated model
package, is responsible for the resulting assets and for reviewing the applicable
Intel terms before redistributing them. This repository does not grant rights to
Intel-derived material and makes no legal conclusion about whether a particular
use or redistribution is permitted.

Intel and XeSS are trademarks of Intel Corporation or its subsidiaries.

## No additional assurance

Nothing in this notice is legal advice or a representation that a generated
package, model, or shader-derived artifact is free of third-party rights. Review
the applicable licenses and terms for your use case.
