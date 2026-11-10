"""Optional ONNX Runtime adapter. The core SDK does not import ORT."""


class OrtExecutor:
    def __init__(self, providers=None, session_options=None):
        self.providers = list(providers) if providers is not None else ["CPUExecutionProvider"]
        self.session_options = session_options

    def load(self, model_path):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("install xefg-sdk[ort] or supply another Executor") from exc
        options = self.session_options
        if options is None:
            options = ort.SessionOptions()
            options.log_severity_level = 3
        return ort.InferenceSession(str(model_path), sess_options=options, providers=self.providers)

    def run(self, handle, inputs, output_names):
        names = list(output_names)
        outputs = handle.run(names, dict(inputs))
        return dict(zip(names, outputs))

    def close(self, handle):
        # Session destruction releases ORT resources after the runtime drops
        # its last reference. There is no explicit ORT close method.
        pass
