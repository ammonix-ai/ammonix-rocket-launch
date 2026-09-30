# Security

Please do not report security issues in public GitHub issues.

Report them privately to licensing@ammonix.ai with "SECURITY" in the subject.
We will acknowledge within 5 business days. This is research software on synthetic data; it
is not intended for production use or for any real launch operation.

The trained models in `layer-4/model.pkl` are Python pickles, and loading a pickle executes
code. They are not shipped: `run_factory.py` builds them on your machine. Load only a
`model.pkl` you built yourself, never one received from someone else.

`ControlRoom/serve.py` binds to 127.0.0.1 and answers only requests addressed to a loopback
name unless you pass `--host` or `--allow-host`. Before exposing it to a network, put it behind
a reverse proxy and cap the analyst with `--max-chats`, since every question is a call to your
LLM server.
