import pytest
from utils import *

server = ServerPreset.tinyllama2()


SHORT_TEXT = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
""".strip()

LONG_TEXT = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
""".strip()

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_ctx = 512
    server.n_slots = 2
    server.n_predict = 128


def test_ctx_shift_enabled():
    # the prompt is 226 tokens
    # the slot context is 512/2 = 256 tokens
    # 96 tokens are generated thanks to shifting the context when it gets full
    global server
    server.enable_ctx_shift = True
    server.start()
    res = server.make_request("POST", "/completion", data={
        "n_predict": 96,
        "prompt": SHORT_TEXT,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == 226
    assert res.body["timings"]["predicted_n"] == 96
    assert res.body["truncated"] is True


@pytest.mark.parametrize("n_predict,n_token_output,truncated", [
    (64, 64, False),
    (-1, 248, True), # 8 tokens prompt + 248 tokens generated = 256 tokens total
])
def test_ctx_shift_disabled_short_prompt(n_predict: int, n_token_output: int, truncated: bool):
    global server
    server.n_predict = -1
    server.start()
    res = server.make_request("POST", "/completion", data={
        "n_predict": n_predict,
        "prompt": "Hi how are you",
    })
    assert res.status_code == 200
    assert res.body["timings"]["predicted_n"] == n_token_output
    assert res.body["truncated"] == truncated


def test_ctx_shift_disabled_long_prompt():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "n_predict": 64,
        "prompt": LONG_TEXT,
    })
    assert res.status_code != 200
    assert "error" in res.body
    assert "exceeds the available context size" in res.body["error"]["message"]

def test_ctx_shift_disabled_stream():
    global server
    server.start()
    res = server.make_stream_request("POST", "/v1/completions", data={
        "n_predict": 256,
        "prompt": "Once",
        "stream": True,
    })
    content = ""
    for data in res:
        choice = data["choices"][0]
        if choice["finish_reason"] == "length":
            assert len(content) > 0
        else:
            assert choice["finish_reason"] is None
            content += choice["text"]


# prompt truncation with --context-shift

TEXT = " ".join(f"Line {i}: the quick brown fox jumps over the lazy dog." for i in range(200))

SLOT_CTX = 256  # n_ctx 512 / 2 slots (see create_server)


def tokenize(text: str) -> list[int]:
    res = server.make_request("POST", "/tokenize", data={"content": text, "add_special": True})
    assert res.status_code == 200
    return res.body["tokens"]


def expected_truncation(tokens: list[int], n_ctx: int, n_keep: int, add_bos: bool) -> list[int]:
    # mirror of launch_slot_with_task() truncation
    n_keep = len(tokens) if n_keep < 0 else n_keep
    if add_bos:
        n_keep += 1
    n_keep = min(n_ctx - 4, n_keep)
    n_left = n_ctx - n_keep
    n_block_size = n_left // 2
    erased_blocks = (len(tokens) - n_keep - n_block_size) // n_block_size
    return tokens[:n_keep] + tokens[n_keep + erased_blocks * n_block_size:]


@pytest.mark.parametrize("n_keep", [0, 1, 20, 100, -1])
def test_long_prompt_truncated_exactly(n_keep: int):
    global server
    server.enable_ctx_shift = True
    server.start()
    tokens = tokenize(TEXT[:3000])
    assert len(tokens) > SLOT_CTX
    add_bos = len(tokens) > len(server.make_request("POST", "/tokenize", data={"content": TEXT[:3000], "add_special": False}).body["tokens"])
    assert add_bos  # the BOS token must survive truncation even with n_keep = 0
    expected = expected_truncation(tokens, SLOT_CTX, n_keep, add_bos)
    assert expected[0] == tokens[0]
    assert len(expected) < SLOT_CTX

    res = server.make_request("POST", "/completion", data={
        "prompt": tokens, "n_keep": n_keep, "n_predict": 16, "temperature": 0.0, "cache_prompt": False,
    })
    assert res.status_code == 200, res.body
    assert res.body["truncated"] is True
    assert res.body["timings"]["prompt_n"] == len(expected)
    assert res.body["timings"]["predicted_n"] == 16

    # the truncated prompt must be exactly the expected tokens: greedy output must match
    ref = server.make_request("POST", "/completion", data={
        "prompt": expected, "n_keep": n_keep, "n_predict": 16, "temperature": 0.0, "cache_prompt": False,
    })
    assert ref.status_code == 200
    assert res.body["content"] == ref.body["content"]


@pytest.mark.parametrize("n_prompt", [250, 253, 254, 255, 256, 257, 300, 1000])
def test_prompt_around_ctx_boundary(n_prompt: int):
    # prompts that end right at / beyond the slot context must still generate the full n_predict
    global server
    server.enable_ctx_shift = True
    server.start()
    tokens = tokenize(TEXT)[:n_prompt]
    res = server.make_request("POST", "/completion", data={
        "prompt": tokens, "n_predict": 32, "temperature": 0.0, "cache_prompt": False,
    })
    assert res.status_code == 200, res.body
    assert res.body["timings"]["predicted_n"] == 32
    assert res.body["truncated"] is (n_prompt + 32 >= SLOT_CTX)


def test_chat_long_history_truncated():
    global server
    server.enable_ctx_shift = True
    server.start()
    res = server.make_request("POST", "/v1/chat/completions", data={
        "messages": [{"role": "user", "content": TEXT[:3000]}], "max_tokens": 16,
    })
    assert res.status_code == 200, res.body
    assert res.body["usage"]["completion_tokens"] == 16


def test_disabled_long_prompt_error_mentions_ctx_shift():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={"prompt": TEXT[:3000], "n_predict": 8})
    assert res.status_code != 200
    assert "enable context shift" in res.body["error"]["message"]


def test_causal_embedding_long_input_truncated():
    # causal model with mean pooling: slot cannot split, memory can shift -> truncated only with commit 3
    global server
    server.n_ctx = 512
    server.n_slots = 1
    server.n_batch = 512
    server.n_ubatch = 512
    server.server_embeddings = True
    server.pooling = "mean"
    server.enable_ctx_shift = True
    server.start()
    res = server.make_request("POST", "/v1/embeddings", data={"input": TEXT})
    assert res.status_code == 200, res.body
    assert len(res.body["data"][0]["embedding"]) > 1


def test_embedding_long_input_truncated():
    global server
    server = ServerPreset.bert_bge_small()
    server.n_ctx = 512
    server.n_slots = 1
    server.n_batch = 512
    server.n_ubatch = 512
    server.enable_ctx_shift = True
    server.start()
    res = server.make_request("POST", "/v1/embeddings", data={"input": TEXT})
    assert res.status_code == 200, res.body
    assert len(res.body["data"][0]["embedding"]) > 1


def test_embedding_long_input_rejected_when_disabled():
    global server
    server = ServerPreset.bert_bge_small()
    server.n_ctx = 512
    server.n_slots = 1
    server.n_batch = 512
    server.n_ubatch = 512
    server.start()
    res = server.make_request("POST", "/v1/embeddings", data={"input": TEXT})
    assert res.status_code != 200
