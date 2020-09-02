# coding=utf-8

import logging
import warnings

from .configuration_utils import PretrainedConfig


logger = logging.getLogger(__name__)

XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "xlnet-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-config.json",
    "xlnet-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-large-cased-config.json",
}


class XLNetConfig(PretrainedConfig):

    model_type = "xlnet"

    def __init__(
        self,
        vocab_size=32000,
        d_model=1024,
        n_layer=24,
        n_head=16,
        d_inner=4096,
        ff_activation="gelu",
        untie_r=True,
        attn_type="bi",
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        dropout=0.1,
        mem_len=None,
        reuse_len=None,
        bi_data=False,
        clamp_len=-1,
        same_length=False,
        summary_type="last",
        summary_use_proj=True,
        summary_activation="tanh",
        summary_last_dropout=0.1,
        start_n_top=5,
        end_n_top=5,
        pad_token_id=5,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs
    ):
        """Constructs XLNetConfig.
        """
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        assert d_model % n_head == 0
        if "d_head" in kwargs:
            assert (
                kwargs["d_head"] == d_model // n_head
            ), f"`d_head` ({kwargs['d_head']}) should be equal to `d_model // n_head` ({d_model // n_head})"
        self.d_head = d_model // n_head
        self.ff_activation = ff_activation
        self.d_inner = d_inner
        self.untie_r = untie_r
        self.attn_type = attn_type

        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        self.dropout = dropout
        self.mem_len = mem_len
        self.reuse_len = reuse_len
        self.bi_data = bi_data
        self.clamp_len = clamp_len
        self.same_length = same_length

        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_last_dropout = summary_last_dropout
        self.start_n_top = start_n_top
        self.end_n_top = end_n_top

        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        if mem_len is None or mem_len == 0:
            warnings.warn(
                "This config doesn't use attention memories, a core feature of XLNet."
                " Consider setting `men_len` to a non-zero value, for example "
                "`xlnet = XLNetLMHeadModel.from_pretrained('xlnet-base-cased'', mem_len=1024)`,"
                " for accurate training performance as well as an order of magnitude faster inference."
                " Starting from version 3.5.0, the default parameter will be 1024, following"
                " the implementation in https://arxiv.org/abs/1906.08237",
                FutureWarning,
            )

    @property
    def max_position_embeddings(self):
        return -1

    @property
    def n_token(self):  # Backward compatibility
        return self.vocab_size

    @n_token.setter
    def n_token(self, value):  # Backward compatibility
        self.vocab_size = value

    @property
    def hidden_size(self):
        return self.d_model

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer
