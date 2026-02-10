import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.modules import LayerNorm, init_bert_params
from unicore.data import data_utils
from .transformer_encoder_with_pair import TransformerEncoderWithPair
from .x_transformer import TransformerWrapper, Encoder
from .dit_transformer import DiT
from .interpolant import FlowMatchingInterpolant
from typing import Dict, Any, List
import numpy as np
import random
#from torchcfm.conditional_flow_matching import *
#from fairseq.modules.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from torchcfm.conditional_flow_matching import *
import roma
logger = logging.getLogger(__name__)
def make_positions(tensor, padding_idx: int, onnx_trace: bool = False):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx

def gaussian(x, mean, std):
    # RBF: exp(-(x - mu)^2 / (2 * sigma^2))
    return torch.exp(-0.5 * (((x - mean) / std) ** 2))

class BondGaussianLayer(nn.Module):

    """
    전략 A: 원자쌍 임베딩으로부터 K개의 (mean, std) 파라미터를 동적으로 예측.
    """
    def __init__(self, type='Gaussian', K=32, vocab_size=31, hidden_dim=31): # Tip: 32의 배수 사용 권장
        super().__init__()
        self.K = K
        
        # 1. 원자 임베딩
        self.atom_embedding = nn.Linear(vocab_size, hidden_dim)
        
        # 2. 파라미터 예측기
        # 입력 차원: hidden_dim (Concat 대신 Sum을 사용할 것이므로 차원이 줄어듦)
        self.param_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, K * 2)
        )
        
        # 3. 분류 헤드
        self.classifier_head = nn.Linear(K, 5)
        
        # [중요] 초기화 전략
        self._init_params()
        self.pred_type = type 
        self.classifier_head_ = nn.Linear(32, 5)
    def _init_params(self):
        # 마지막 레이어의 가중치를 작게 하여 초기 출력을 0에 가깝게 만듦
        nn.init.xavier_uniform_(self.param_predictor[-1].weight, gain=0.01)
        bias = self.param_predictor[-1].bias.data

        # K=32개 커널을 5개 그룹으로 나누어 초기화
        K_per_class = self.K // 5
        for i in range(5):
            if i == 0:  # No bond
                bias[i*K_per_class:(i+1)*K_per_class].uniform_(2.5, 4.0)
            elif i == 1:  # Single bond
                bias[i*K_per_class:(i+1)*K_per_class].uniform_(1.3, 1.7)
            elif i == 2:  # Double bond
                bias[i*K_per_class:(i+1)*K_per_class].uniform_(1.1, 1.5)
            elif i == 3:  # Triple bond
                bias[i*K_per_class:(i+1)*K_per_class].uniform_(1.0, 1.4)
            else:  # Aromatic
                bias[i*K_per_class:(i+1)*K_per_class].uniform_(1.2, 1.6)

        # Std 초기화: 넓은 범위
        bias[self.K:].fill_(1.0)

    def forward(self, x, atom_type_tensor):
        B, N, _ = x.shape
        
        # 1. 임베딩
        atom_emb = self.atom_embedding(atom_type_tensor.float()) # [B, N, H]
        
        # 2. 대칭성 확보 (Symmetry Enforced)
        # Concat 대신 Broadcasting을 이용한 합(Sum) 사용 -> 순서 불변성(Order Invariance)
        atom_i = atom_emb.unsqueeze(2) # [B, N, 1, H]
        atom_j = atom_emb.unsqueeze(1) # [B, 1, N, H]
        edge_feature = atom_i + atom_j # [B, N, N, H] (i+j == j+i)
        
        if self.pred_type == 'Gaussian':
        # 3. 파라미터 예측
            params = self.param_predictor(edge_feature) # [B, N, N, K*2]
            mean_raw, std_raw = params.split(self.K, dim=-1)
            
            # 4. 제약 조건 적용 (Constraints)
            # Mean: 항상 양수여야 함. F.softplus 사용 (또는 0~3 범위 제한을 위해 sigmoid * 3.0 등도 가능)
            mean = F.softplus(mean_raw) 
            # Std: 항상 양수여야 하며, 너무 작아지는 것(0)을 방지 (+ 1e-5)
            std = F.softplus(std_raw) + 1e-5
            
            # 5. 가우시안 적용
            x_expanded = x.unsqueeze(-1).expand(-1, -1, -1, self.K)
            feature = gaussian(x_expanded.float(), mean.float(), std.float())
            
            # 6. 분류
            return self.classifier_head(feature.type_as(self.atom_embedding.weight))

        else:
            feature = torch.cat([edge_feature, x.unsqueeze(-1)], -1)
            return self.classifier_head_(feature)
class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024, auto_expand=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx if padding_idx is not None else 0
        self.register_buffer(
            "weights",
            SinusoidalPositionalEmbedding.get_embedding(
                init_size, embedding_dim, padding_idx
            ),
            persistent=False,
        )
        self.max_positions = int(1e5)
        self.auto_expand = auto_expand
        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Ignore some deprecated keys that were used in older versions
        deprecated_keys = ["weights", "_float_tensor"]
        for key in deprecated_keys:
            if prefix + key in state_dict:
                del state_dict[prefix + key]
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @staticmethod
    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: None
    ):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(
        self,
        input,
        incremental_state= None,
        timestep=None,
        positions=None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        #bspair = torch.onnx.operators.shape_as_tensor(input)
        bspair = torch.tensor(input.shape, device=input.device) 
        bsz, seq_len = bspair[0], bspair[1]
        max_pos = self.padding_idx + 1 + seq_len
        weights = self.weights

        if max_pos > self.weights.size(0):
            # If the input is longer than the number of pre-computed embeddings,
            # compute the extra embeddings on the fly.
            # Only store the expanded embeddings if auto_expand=True.
            # In multithreading environments, mutating the weights of a module
            # may cause trouble. Set auto_expand=False if this happens.
            weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx
            ).to(self.weights)
            if self.auto_expand:
                self.weights = weights

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            if self.onnx_trace:
                return (
                    weights.index_select(index=self.padding_idx + pos, dim=0)
                    .unsqueeze(1)
                    .repeat(bsz, 1, 1)
                )
            return weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = make_positions(
            input, self.padding_idx, onnx_trace=self.onnx_trace
        )
        if self.onnx_trace:
            flat_embeddings = weights.detach().index_select(0, positions.view(-1))
            embedding_shape = torch.cat(
                (bsz.view(1), seq_len.view(1), torch.tensor([-1], dtype=torch.long))
            )
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(
                flat_embeddings, embedding_shape
            )
            return embeddings
        return (
            weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()
        )
        
def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


@register_model("unimol_MAE_padding")
class UniMolMAEPaddingModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="L", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="A",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--emb-dropout",
            type=float,
            metavar="D",
            help="dropout probability for embeddings",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-seq-len", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--post-ln", type=bool, help="use post layernorm or pre layernorm"
        )
        parser.add_argument(
            "--masked-token-loss",
            type=float,
            metavar="D",
            help="mask loss ratio",
        )
        parser.add_argument(
            "--masked-dist-loss",
            type=float,
            metavar="D",
            help="masked distance loss ratio",
        )
        parser.add_argument(
            "--masked-coord-loss",
            type=float,
            metavar="D",
            help="masked coord loss ratio",
        )
        parser.add_argument(
            "--encoder-x-norm-loss",
            type=float,
            metavar="D",
            help="x norm loss ratio",
        )
        parser.add_argument(
            "--encoder-delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio",
        )
        parser.add_argument(
            "--decoder-x-norm-loss",
            type=float,
            metavar="D",
            help="x norm loss ratio",
        )
        parser.add_argument(
            "--decoder-delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio",
        )
        parser.add_argument(
            "--masked-coord-dist-loss",
            type=float,
            metavar="D",
            help="masked coord dist loss ratio",
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="train",
            choices=["train", "infer"],
        )
        parser.add_argument(
            "--encoder-unmasked-tokens-only", action='store_true', help="only input unmasked tokens into encoder"
        )
        parser.add_argument(
            "--encoder-masked-3d-pe", action='store_true', help="only masked #D PE for encoder"
        )
        parser.add_argument(
            "--encoder-apply-pe", action='store_true', help="apply PE for encoder"
        )
        parser.add_argument(
            "--feed-pair-rep-to-decoder", action='store_true', help="feed the pair representations of encoder to decoder"
        )
        parser.add_argument(
            "--decoder-no-pe", action='store_true', help="Don't apply PE for decoder"
        )
        parser.add_argument(
            "--feed-token-rep-to-decoder", action='store_true', help="feed the token representations of encoder to decoder"
        )
        parser.add_argument(
            "--decoder-noise", action='store_true', help="Feed noise or [mask] to decoder"
        )
        parser.add_argument(
            "--random-order", action='store_true', help="Feed noise or [mask] to decoder"
        )

    def __init__(self, args, dictionary):
        super().__init__()
        print('Using modified MAE')
        base_architecture(args)
        self.args = args
        self.encoder_masked_3d_pe = args.encoder_masked_3d_pe
        self.encoder_apply_pe = args.encoder_apply_pe
        self.feed_pair_rep_to_decoder = args.feed_pair_rep_to_decoder
        self.decoder_no_pe = args.decoder_no_pe
        self.feed_token_rep_to_decoder = args.feed_token_rep_to_decoder
        self.decoder_noise = args.decoder_noise
        self.random_order = args.random_order


        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(dictionary), args.encoder_embed_dim, self.padding_idx
        )

        self.PE = None
        self.index = None
        if self.random_order:
            self.init_state()

        self._num_updates = None
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.encoder_delta_pair_repr_norm_loss < 0,
        )
        self.decoder = TransformerEncoderWithPair(
            encoder_layers=args.decoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.decoder_ffn_embed_dim,
            attention_heads=args.decoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.decoder_delta_pair_repr_norm_loss < 0,
        )
        if args.masked_token_loss > 0:
            self.lm_head = MaskLMHead(
                embed_dim=args.encoder_embed_dim,
                output_dim=len(dictionary),
                activation_fn=args.activation_fn,
                weight=None,
            )
            
        self.logvar_proj = NonLinearHead(
            args.encoder_embed_dim, args.encoder_embed_dim, 
            args.activation_fn
        )
        
        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(
            K, args.encoder_attention_heads, args.activation_fn
        )
        self.gbf_proj2 = NonLinearHead(
            K, args.encoder_attention_heads, args.activation_fn
        )
        self.gbf = GaussianLayer(K, n_edge_type)
        self.gbf2 = GaussianLayer(K, 5)
        if args.masked_coord_loss > 0:
            self.pair2coord_proj = NonLinearHead(
                args.decoder_attention_heads, 1, args.activation_fn
            )
        if args.masked_dist_loss > 0:
            self.dist_head = DistanceHead(
                args.decoder_attention_heads, args.activation_fn
            )
        self.classification_heads = nn.ModuleDict()
        self.finetuning_heads = nn.ModuleDict()
        self.apply(init_bert_params)
        self.encoder_unmasked_tokens_only = args.encoder_unmasked_tokens_only
        self.dictionary = dictionary

        self.embed_positions = SinusoidalPositionalEmbedding(
            embedding_dim = args.max_seq_len,
            padding_idx = dictionary.pad(),
            init_size = args.max_seq_len,
        )

        self.mask_idx = dictionary.index("[MASK]")
        self.encoder_attention_heads = args.encoder_attention_heads

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)
    
    @classmethod
    def init_state(self):
        original_state = np.random.get_state()
        np.random.seed(0)
        self.numpy_random_state = np.random.get_state()
        np.random.set_state(original_state)
    
    @classmethod
    def Myenter(self):
        self.original_state = np.random.get_state()
        np.random.set_state(self.numpy_random_state)

    @classmethod
    def Myexit(self):
        self.numpy_random_state = np.random.get_state()
        np.random.set_state(self.original_state)

    def enc(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        src_bond_type,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargs        
    ):
        encoder_src_tokens = src_tokens
        encoder_src_coord = src_coord
        encoder_src_distance = src_distance
        encoder_src_edge_type = src_edge_type

        encoder_padding_mask = encoder_src_tokens.eq(self.padding_idx)
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        x = self.embed_tokens(encoder_src_tokens)

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias
        def get_bond_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf2(dist, et)
            gbf_result = self.gbf_proj2(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias
        graph_attn_bias = get_dist_features(encoder_src_distance, encoder_src_edge_type) + get_bond_features(encoder_src_distance, src_bond_type)

        (
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            encoder_x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0

        if self.feed_token_rep_to_decoder:
            encoder_output_embedding = encoder_rep
        else:
            if encoder_masked_tokens is None:
                encoder_output_embedding = encoder_rep
            else:
                mask_tokens = torch.zeros_like(src_tokens).fill_(self.mask_idx)
                masked_embeddings = self.embed_tokens(mask_tokens)
                encoder_output_embedding = torch.where(encoder_masked_tokens.unsqueeze(-1).expand_as(encoder_rep), masked_embeddings, encoder_rep)
        latent_logvar = self.logvar_proj(encoder_output_embedding)
        
        return encoder_output_embedding,latent_logvar, encoder_x_norm, delta_encoder_pair_rep_norm
    
    def dec(self, encoder_output_embedding, padding_mask, encoder_x_norm=None, delta_encoder_pair_rep_norm=None):
        encoder_masked_tokens=None
        features_only=False
        classification_head_name=None   
             
        if not self.decoder_no_pe:
           encoder_output_embedding = encoder_output_embedding + self.embed_positions(~padding_mask)

        n_node = encoder_output_embedding.size(1)
        # if self.feed_pair_rep_to_decoder:
        #     assert self.decoder_noise is not True
        #     attn_bias = encoder_pair_rep.reshape(-1, n_node, n_node)
        # else:
        #     if not self.decoder_noise:
        #         bsz = padding_mask.size(0)
        #         attn_bias = encoder_output_embedding.new_zeros((bsz*self.encoder_attention_heads, n_node, n_node))
        #     else:
        #         attn_bias = get_dist_features(src_distance, src_edge_type)
        bsz = padding_mask.size(0)
        attn_bias = encoder_output_embedding.new_zeros((bsz*self.encoder_attention_heads, n_node, n_node))
        
        (
            decoder_rep,
            decoder_pair_rep,
            delta_decoder_pair_rep,
            decoder_x_norm,
            delta_decoder_pair_rep_norm,
        ) = self.decoder(encoder_output_embedding, padding_mask=padding_mask, attn_mask=attn_bias)

        decoder_pair_rep[decoder_pair_rep == float("-inf")] = 0

        encoder_distance = None
        encoder_coord = None

        if not features_only:
            if self.args.masked_token_loss > 0:
                #logits = self.lm_head(decoder_rep, encoder_masked_tokens) # padding_mask
                logits = self.lm_head(decoder_rep, ~padding_mask) 
            # if self.args.masked_coord_loss > 0:
            #     coords_emb = src_coord
            #     if padding_mask is not None:
            #         atom_num = (torch.sum(1 - padding_mask.type_as(x), dim=1) - 1).view(
            #             -1, 1, 1, 1
            #         )
            #     else:
            #         atom_num = src_coord.shape[1] - 1
            #     delta_pos = coords_emb.unsqueeze(1) - coords_emb.unsqueeze(2)
            #     attn_probs = self.pair2coord_proj(delta_decoder_pair_rep)
            #     coord_update = delta_pos / atom_num * attn_probs
            #     coord_update = torch.sum(coord_update, dim=2)
            #     encoder_coord = coords_emb + coord_update
            if self.args.masked_dist_loss > 0:
                encoder_distance = self.dist_head(decoder_pair_rep)

        if classification_head_name is not None:
            finetune_input = torch.sum(encoder_output_embedding * ~padding_mask.unsqueeze(-1), 1) / torch.sum(~padding_mask, 1).unsqueeze(-1)
            logits = self.classification_heads[classification_head_name](finetune_input)
        # if self.args.mode == 'infer':
        #     return encoder_rep, encoder_pair_rep
        else:
            return (
                logits,
                encoder_distance,
                None,
                encoder_x_norm,
                decoder_x_norm,
                delta_encoder_pair_rep_norm,
                delta_decoder_pair_rep_norm,
            )            
    
    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargs
    ):
        # print('encoder_masked_tokens:',encoder_masked_tokens)
        # exit()

        if self.random_order and self.index is None:
            self.Myenter()
            self.index = np.random.permutation(512)
            self.Myexit()

        if classification_head_name is not None:
            features_only = True
        # encoder_src_tokens = src_tokens
        # encoder_src_coord = src_coord
        # encoder_src_distance = src_distance
        # encoder_src_edge_type = src_edge_type        
        padding_mask = src_tokens.eq(self.padding_idx)
        latent_emb, latent_logvar, encoder_x_norm, delta_encoder_pair_rep_norm = self.enc(
                src_tokens,
                src_distance,
                src_coord,
                src_edge_type,
                encoder_masked_tokens=None,
                features_only=False,
                classification_head_name=None,
                **kwargs
                )
        # LOG_STD_MAX = 2
        # LOG_STD_MIN = -20 # 너무 작은 분산을 막기 위함
        # latent_logvar = torch.clamp(latent_logvar, LOG_STD_MIN, LOG_STD_MAX)
        
        print(std.max())
        std = torch.exp(0.5 * latent_logvar)
        # eps = torch.randn_like(std)        
        # latent = latent_emb + eps * std
        # kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_emb.pow(2) - latent_logvar.exp())
        # kl_weight = (latent_logvar - latent_emb.pow(2) - latent_logvar.exp()).view(-1,1).shape[0] ** 0.5
        
        q_z_given_x = torch.distributions.Normal(loc=latent_emb, scale=std)
        latent = q_z_given_x.rsample()
        p_z = torch.distributions.Normal(
            loc=torch.zeros_like(latent_emb), 
            scale=torch.ones_like(std)
        )
        kl_divs = torch.distributions.kl.kl_divergence(q_z_given_x, p_z)
        kl_loss = kl_divs[~padding_mask].to(torch.float32).sum()
        #kl_loss = kl_loss * 1000
        #kl_weight = kl_divs[~padding_mask].view(-1,1).shape[0] ** 0.4
        kl_loss = kl_loss * 0.001 
        
        
        (
            logits,
            encoder_distance,
            encoder_coord,
            encoder_x_norm,
            decoder_x_norm,
            delta_encoder_pair_rep_norm,
            delta_decoder_pair_rep_norm,
        ) = self.dec(
            latent,
            padding_mask,
            encoder_x_norm,
            delta_encoder_pair_rep_norm
        )
        
        return (
            logits,
            encoder_distance,
            encoder_coord,
            encoder_x_norm,
            decoder_x_norm,
            delta_encoder_pair_rep_norm,
            delta_decoder_pair_rep_norm,
            kl_loss
        ) 
        # encoder_src_tokens = src_tokens
        # encoder_src_coord = src_coord
        # encoder_src_distance = src_distance
        # encoder_src_edge_type = src_edge_type

        # encoder_padding_mask = encoder_src_tokens.eq(self.padding_idx)
        # padding_mask = src_tokens.eq(self.padding_idx)
        # if not padding_mask.any():
        #     padding_mask = None
        # if not encoder_padding_mask.any():
        #     encoder_padding_mask = None

        # x = self.embed_tokens(encoder_src_tokens)

        # def get_dist_features(dist, et):
        #     n_node = dist.size(-1)
        #     gbf_feature = self.gbf(dist, et)
        #     gbf_result = self.gbf_proj(gbf_feature)
        #     graph_attn_bias = gbf_result
        #     graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
        #     graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
        #     return graph_attn_bias

        # graph_attn_bias = get_dist_features(encoder_src_distance, encoder_src_edge_type)

        # (
        #     encoder_rep,
        #     encoder_pair_rep,
        #     delta_encoder_pair_rep,
        #     encoder_x_norm,
        #     delta_encoder_pair_rep_norm,
        # ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        # encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0

        # if self.feed_token_rep_to_decoder:
        #     encoder_output_embedding = encoder_rep
        # else:
        #     if encoder_masked_tokens is None:
        #         encoder_output_embedding = encoder_rep
        #     else:
        #         mask_tokens = torch.zeros_like(src_tokens).fill_(self.mask_idx)
        #         masked_embeddings = self.embed_tokens(mask_tokens)
        #         encoder_output_embedding = torch.where(encoder_masked_tokens.unsqueeze(-1).expand_as(encoder_rep), masked_embeddings, encoder_rep)


        ####### decoder start
        # if not self.decoder_no_pe:
        #     encoder_output_embedding = encoder_output_embedding + self.embed_positions(src_tokens)

        # n_node = encoder_output_embedding.size(1)
        # if self.feed_pair_rep_to_decoder:
        #     assert self.decoder_noise is not True
        #     attn_bias = encoder_pair_rep.reshape(-1, n_node, n_node)
        # else:
        #     if not self.decoder_noise:
        #         bsz = src_tokens.size(0)
        #         attn_bias = encoder_output_embedding.new_zeros((bsz*self.encoder_attention_heads, n_node, n_node))
        #     else:
        #         attn_bias = get_dist_features(src_distance, src_edge_type)
        # (
        #     decoder_rep,
        #     decoder_pair_rep,
        #     delta_decoder_pair_rep,
        #     decoder_x_norm,
        #     delta_decoder_pair_rep_norm,
        # ) = self.decoder(encoder_output_embedding, padding_mask=padding_mask, attn_mask=attn_bias)

        # decoder_pair_rep[decoder_pair_rep == float("-inf")] = 0

        # encoder_distance = None
        # encoder_coord = None

        # if not features_only:
        #     if self.args.masked_token_loss > 0:
        #         #logits = self.lm_head(decoder_rep, encoder_masked_tokens) # padding_mask
        #         logits = self.lm_head(decoder_rep, src_tokens.ne(self.padding_idx)) 
        #     if self.args.masked_coord_loss > 0:
        #         coords_emb = src_coord
        #         if padding_mask is not None:
        #             atom_num = (torch.sum(1 - padding_mask.type_as(x), dim=1) - 1).view(
        #                 -1, 1, 1, 1
        #             )
        #         else:
        #             atom_num = src_coord.shape[1] - 1
        #         delta_pos = coords_emb.unsqueeze(1) - coords_emb.unsqueeze(2)
        #         attn_probs = self.pair2coord_proj(delta_decoder_pair_rep)
        #         coord_update = delta_pos / atom_num * attn_probs
        #         coord_update = torch.sum(coord_update, dim=2)
        #         encoder_coord = coords_emb + coord_update
        #     if self.args.masked_dist_loss > 0:
        #         encoder_distance = self.dist_head(decoder_pair_rep)

        # if classification_head_name is not None:
        #     logits = self.classification_heads[classification_head_name](encoder_rep)
        # if self.args.mode == 'infer':
        #     return encoder_rep, encoder_pair_rep
        # else:
        #     return (
        #         logits,
        #         encoder_distance,
        #         encoder_coord,
        #         encoder_x_norm,
        #         decoder_x_norm,
        #         delta_encoder_pair_rep_norm,
        #         delta_decoder_pair_rep_norm,
        #     )            
           

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates



class MaskLMHead(nn.Module):

    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        # if masked_tokens is not None:
        #     features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense1 = nn.Linear(input_dim, inner_dim)
        self.dense2 = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)
        self.batchnorm1 = nn.BatchNorm1d(inner_dim)
        self.batchnorm2 = nn.BatchNorm1d(inner_dim)
        self.layernorm1 = nn.LayerNorm(inner_dim)
        self.layernorm2 = nn.LayerNorm(inner_dim)        
        
    def forward(self, features, **kwargs):
        #x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        #x = x.mean(1)
        x = self.dropout(features)
        
        x = self.dense1(x)
        try:
            x = self.batchnorm1(x)
        except:
            x = x
        
        #x = self.layernorm1(x)
        x = self.activation_fn(x)
        x = self.dropout(x)

        x = self.dense2(x)
        try:
            x = self.batchnorm2(x)
        except:
            x = x
            
        #x = self.layernorm2(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        
        x = self.out_proj(x)
        return x


class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x

class NonLinearHeadPos(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, 3)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x

class DistanceHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        # x[x == float('-inf')] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)



"""Current Version [Encoder-Decoder Structure]"""
@register_model("unimol_MIM_padding")
class UniMolMIMPaddingModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="L", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="A",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--emb-dropout",
            type=float,
            metavar="D",
            help="dropout probability for embeddings",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-seq-len", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--post-ln", type=bool, help="use post layernorm or pre layernorm"
        )
        parser.add_argument(
            "--masked-token-loss",
            type=float,
            metavar="D",
            help="mask loss ratio",
        )
        parser.add_argument(
            "--masked-dist-loss",
            type=float,
            metavar="D",
            help="masked distance loss ratio",
        )
        parser.add_argument(
            "--masked-coord-loss",
            type=float,
            metavar="D",
            help="masked coord loss ratio",
        )
        parser.add_argument(
            "--encoder-x-norm-loss",
            type=float,
            metavar="D",
            help="x norm loss ratio",
        )
        parser.add_argument(
            "--encoder-delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio",
        )
        parser.add_argument(
            "--decoder-x-norm-loss",
            type=float,
            metavar="D",
            help="x norm loss ratio",
        )
        parser.add_argument(
            "--decoder-delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio",
        )
        parser.add_argument(
            "--masked-coord-dist-loss",
            type=float,
            metavar="D",
            help="masked coord dist loss ratio",
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="train",
            choices=["train", "infer"],
        )
        parser.add_argument(
            "--encoder-unmasked-tokens-only", action='store_true', help="only input unmasked tokens into encoder"
        )
        parser.add_argument(
            "--encoder-masked-3d-pe", action='store_true', help="only masked #D PE for encoder"
        )
        parser.add_argument(
            "--encoder-apply-pe", action='store_true', help="apply PE for encoder"
        )
        parser.add_argument(
            "--feed-pair-rep-to-decoder", action='store_true', help="feed the pair representations of encoder to decoder"
        )
        parser.add_argument(
            "--decoder-no-pe", action='store_true', help="Don't apply PE for decoder"
        )
        parser.add_argument(
            "--feed-token-rep-to-decoder", action='store_true', help="feed the token representations of encoder to decoder"
        )
        parser.add_argument(
            "--decoder-noise", action='store_true', help="Feed noise or [mask] to decoder"
        )
        parser.add_argument(
            "--random-order", action='store_true', help="Feed noise or [mask] to decoder"
        )

    def __init__(self, args, dictionary):
        super().__init__()
        print('Using modified MAE')
        base_architecture(args)
        self.args = args
        self.encoder_masked_3d_pe = args.encoder_masked_3d_pe
        self.encoder_apply_pe = args.encoder_apply_pe
        self.feed_pair_rep_to_decoder = args.feed_pair_rep_to_decoder
        self.decoder_no_pe = args.decoder_no_pe
        self.feed_token_rep_to_decoder = args.feed_token_rep_to_decoder
        self.decoder_noise = args.decoder_noise
        self.random_order = args.random_order
        self.interpolant = FlowMatchingInterpolant(device="cpu")
        #self.interpolant = None

        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(dictionary), args.encoder_embed_dim, self.padding_idx
        )

        self.PE = None
        self.index = None
        if self.random_order:
            self.init_state()
        # Original TransformerWrapper backbone (commented for DiT replacement)
        # self.backbone = TransformerWrapper(
        #     attn_layers=Encoder(dim=args.encoder_embed_dim, depth=10),
        #     emb_dropout=0.0)
        
        # Use DiT model as backbone instead of TransformerWrapper
        self.backbone = DiT(
            # d_x=args.encoder_embed_dim,
            # d_model=args.encoder_embed_dim,
            # nhead=args.encoder_attention_heads,
            d_x=8,
            d_model=512,
            nhead=1,
        )
        self._num_updates = None
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.encoder_delta_pair_repr_norm_loss < 0,
        )
        self.decoder = TransformerEncoderWithPair(
            #encoder_layers=args.decoder_layers,
            encoder_layers=3,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.decoder_ffn_embed_dim,
            attention_heads=args.decoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.decoder_delta_pair_repr_norm_loss < 0,
        )
        if args.masked_token_loss > 0:
            self.lm_head = MaskLMHead(
                embed_dim=args.encoder_embed_dim,
                output_dim=len(dictionary),
                activation_fn=args.activation_fn,
                weight=None,
            )
            
        # self.logvar_proj = NonLinearHead(
        #     args.encoder_embed_dim, args.encoder_embed_dim, 
        #     args.activation_fn
        # )
        
        self.quant_mean = NonLinearHead(
            input_dim = args.encoder_embed_dim, 
            out_dim = 8, 
            activation_fn = args.activation_fn, 
            hidden = args.encoder_embed_dim//2
        )
        self.logvar_proj = NonLinearHead(
            input_dim = args.encoder_embed_dim, 
            out_dim = 8, 
            activation_fn = args.activation_fn, 
            hidden = args.encoder_embed_dim//2
        )
        self.quant_expand = NonLinearHead(
            input_dim = 8, 
            out_dim = args.encoder_embed_dim, 
            activation_fn = args.activation_fn, 
            hidden = args.encoder_embed_dim//2
        )
        
        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(
            K, args.encoder_attention_heads, args.activation_fn
        )
        self.gbf_proj2 = NonLinearHead(
            K, args.encoder_attention_heads, args.activation_fn
        )
        self.gbf = GaussianLayer(K, n_edge_type)
        self.gbf2 = GaussianLayer(K, 5)
        
        if args.masked_coord_loss > 0:
            self.pair2coord_proj = NonLinearHead(
                args.decoder_attention_heads, 1, args.activation_fn
            )
        if args.masked_dist_loss > 0:
            self.dist_head = DistanceHead(
                args.decoder_attention_heads, args.activation_fn
            )
        self.init_pos_prog = NonLinearHeadPos(
            512, 1, args.activation_fn
        )
        self.pos_embedder = nn.Sequential(
            nn.Linear(3, args.encoder_embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim),
        )
        self.pos_decoder = nn.Sequential(
            nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(args.encoder_embed_dim, 3),
        )
        self.classification_heads = nn.ModuleDict()
        self.apply(init_bert_params)
        self.encoder_unmasked_tokens_only = args.encoder_unmasked_tokens_only
        self.dictionary = dictionary

        self.embed_positions = SinusoidalPositionalEmbedding(
            embedding_dim = args.max_seq_len,
            padding_idx = dictionary.pad(),
            init_size = args.max_seq_len,
        )

        self.mask_idx = dictionary.index("[MASK]")
        self.encoder_attention_heads = args.encoder_attention_heads
        self.self_condition = True
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)
    
    @classmethod
    def init_state(self):
        original_state = np.random.get_state()
        np.random.seed(0)
        self.numpy_random_state = np.random.get_state()
        np.random.set_state(original_state)
    
    @classmethod
    def Myenter(self):
        self.original_state = np.random.get_state()
        np.random.set_state(self.numpy_random_state)

    @classmethod
    def Myexit(self):
        self.numpy_random_state = np.random.get_state()
        np.random.set_state(self.original_state)

    def enc(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        src_bond_type,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargs        
    ):
        encoder_src_tokens = src_tokens
        encoder_src_coord = src_coord
        encoder_src_distance = src_distance
        encoder_src_edge_type = src_edge_type

        encoder_padding_mask = encoder_src_tokens.eq(self.padding_idx)
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        x = self.embed_tokens(encoder_src_tokens) + self.embed_positions(~padding_mask)
        x = x + self.pos_embedder(encoder_src_coord)
        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias
        def get_bond_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf2(dist, et)
            gbf_result = self.gbf_proj2(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(encoder_src_distance, encoder_src_edge_type) #+ get_bond_features(encoder_src_distance, src_bond_type)

        (
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            encoder_x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0
        encoder_rep_quant = self.quant_mean(encoder_rep)
        encoder_output_embedding = torch.nn.Tanh()(encoder_rep_quant)
        
        # if self.feed_token_rep_to_decoder:
        #     encoder_output_embedding = encoder_rep
        # else:
        #     if encoder_masked_tokens is None:
        #         encoder_output_embedding = encoder_rep
        #     else:
        #         mask_tokens = torch.zeros_like(src_tokens).fill_(self.mask_idx)
        #         masked_embeddings = self.embed_tokens(mask_tokens)
        #         encoder_output_embedding = torch.where(encoder_masked_tokens.unsqueeze(-1).expand_as(encoder_rep), masked_embeddings, encoder_rep)
        latent_logvar = self.logvar_proj(encoder_rep)
        
        return encoder_output_embedding, latent_logvar, encoder_x_norm, delta_encoder_pair_rep_norm
    
    def dec(self, encoder_output_embedding, padding_mask, encoder_x_norm=None, delta_encoder_pair_rep_norm=None, classification_head_name=None):
        encoder_masked_tokens=None
        features_only=False
        #classification_head_name=None   
             
        #if not self.decoder_no_pe:
        #    encoder_output_embedding = encoder_output_embedding + self.embed_positions(~padding_mask)
        encoder_output_embedding = self.quant_expand(encoder_output_embedding) + self.embed_positions(~padding_mask)
        n_node = encoder_output_embedding.size(1)
        # if self.feed_pair_rep_to_decoder:
        #     assert self.decoder_noise is not True
        #     attn_bias = encoder_pair_rep.reshape(-1, n_node, n_node)
        # else:
        #     if not self.decoder_noise:
        #         bsz = padding_mask.size(0)
        #         attn_bias = encoder_output_embedding.new_zeros((bsz*self.encoder_attention_heads, n_node, n_node))
        #     else:
        #         attn_bias = get_dist_features(src_distance, src_edge_type)
        bsz = padding_mask.size(0)
        attn_bias = encoder_output_embedding.new_zeros((bsz*self.encoder_attention_heads, n_node, n_node))
        
        (
            decoder_rep,
            decoder_pair_rep,
            delta_decoder_pair_rep,
            decoder_x_norm,
            delta_decoder_pair_rep_norm,
        ) = self.decoder(encoder_output_embedding, padding_mask=padding_mask, attn_mask=attn_bias)

        decoder_pair_rep[decoder_pair_rep == float("-inf")] = 0

        encoder_distance = None
        encoder_coord = None

        if not features_only:
            if self.args.masked_token_loss > 0:
                #logits = self.lm_head(decoder_rep, encoder_masked_tokens) # padding_mask
                logits = self.lm_head(decoder_rep, ~padding_mask) 
            if self.args.masked_coord_loss > 0:
                # atom_num = (torch.sum(1 - padding_mask.type_as(decoder_rep), dim=1)).view(-1, 1, 1, 1)
                # #coords_emb = torch.zeros((padding_mask.size(0),padding_mask.size(1),3), device = decoder_pair_rep.device)
                # coords_emb = torch.randn((padding_mask.size(0),padding_mask.size(1),3), device = decoder_pair_rep.device)
                
                # #coords_emb = self.init_pos_prog(encoder_output_embedding)
                # for i in range(4):
                #     coords_emb = coords_emb * ~padding_mask.unsqueeze(-1)
                #     delta_pos = coords_emb.unsqueeze(1) - coords_emb.unsqueeze(2)
                #     attn_probs = self.pair2coord_proj(delta_decoder_pair_rep)
                #     coord_update = delta_pos / atom_num * attn_probs
                #     pair_coords_mask = (1 - padding_mask.float()).unsqueeze(-1) * (1 - padding_mask.float()).unsqueeze(1)
                #     coord_update = coord_update * pair_coords_mask.unsqueeze(-1)
                #     coord_update = torch.sum(coord_update, dim=2)
                #     #encoder_coord = coords_emb + coord_update
                #     coords_emb = coords_emb + coord_update
                # encoder_coord = coords_emb
                encoder_coord = self.pos_decoder(decoder_rep)
                
            if self.args.masked_dist_loss > 0:
                encoder_distance = self.dist_head(decoder_pair_rep)

        if classification_head_name is not None:
            finetune_input = torch.sum(encoder_output_embedding * ~padding_mask.unsqueeze(-1), 1) / torch.sum(~padding_mask, 1).unsqueeze(-1)
            logits = self.classification_heads[classification_head_name](finetune_input)
        if self.args.mode == 'infer':
            return encoder_rep, encoder_pair_rep
        else:
            return (
                logits,
                encoder_distance,
                encoder_coord,
                encoder_x_norm,
                decoder_x_norm,
                delta_encoder_pair_rep_norm,
                delta_decoder_pair_rep_norm,
            )            
    def flow_training(
        self,
        model,
        z_1,
        mask
    ):
        """Vanilla Flow"""
        # dtype = torch.float32
        # t = torch.rand((z_1.size(0),), dtype=dtype, device=z_1.device)
        # t = t.view(-1,1,1)
        # z_0 = torch.randn_like(z_1, device=z_1.device)
        # #z_t = (1 - t) * z_0 + (1e-5 + (1 - 1e-5) * t) * z_1
        # z_t = (1 - t) * z_0 + t * z_1
        # #u = (1 - 1e-5) * z_1 - z_0
        # u = (1 - 1e-5) * z_1 - z_0
        # #v = model(z_t, t.squeeze(-1), mask=mask)
        # v = self.flow_infer(model, z_t, t.squeeze(-1), mask=mask)
        # loss = F.mse_loss(v, u, reduction='none')
        
        
        """ OT-CFM"""

        # FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.)
        # x1 = z_1
        # x0 = torch.randn_like(x1)
        # t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        # vt = self.flow_infer(model, xt, t.unsqueeze(-1), mask=mask)
        # loss = F.mse_loss(vt, ut, reduction='none')
        # return loss, vt
    
        """Protein Frame Flow"""
        self.interpolant.device = z_1.device
        dense_encoded_batch = {"x_1": z_1, "token_mask": mask, "diffuse_mask": mask}
        noisy_dense_encoded_batch = self.interpolant.corrupt_batch(dense_encoded_batch)

        #if self.self_condition:
        # Use self-conditioning for ~half training batches
        if (
            self.interpolant.self_condition
            and random.random() < self.interpolant.self_condition_prob
        ):
            
            with torch.no_grad():
                x_sc = model(noisy_dense_encoded_batch["x_t"],
                             noisy_dense_encoded_batch["t"],
                             mask=mask,
                             x_sc=None)
        else:
            x_sc = None
            
        pred_x = model(noisy_dense_encoded_batch["x_t"],
                        noisy_dense_encoded_batch["t"],
                        mask=mask,
                        x_sc=x_sc)

        gt_x_1 = noisy_dense_encoded_batch["x_1"]
        norm_scale = 1 - torch.min(noisy_dense_encoded_batch["t"].unsqueeze(-1), torch.tensor(0.9))
        x_error = (gt_x_1 - pred_x) / norm_scale
        loss_mask = (
            noisy_dense_encoded_batch["token_mask"] * noisy_dense_encoded_batch["diffuse_mask"]
        )
        loss_denom = torch.sum(loss_mask, dim=-1) * pred_x.size(-1)
        x_loss = torch.sum(x_error**2 * loss_mask[..., None], dim=(-1, -2)) / loss_denom
        #loss_dict = {"loss": x_loss.mean(), "x_loss": x_loss}
        loss_dict = {"loss": x_loss.mean()}
         
        num_bins = 4
        #flat_losses = x_loss.detach().cpu().numpy().flatten()
        #flat_losses = (gt_x_1 - pred_x)
        flat_losses = torch.sum((gt_x_1 - pred_x)**2 * loss_mask[..., None], dim=(-1, -2)) / loss_denom
        flat_losses = flat_losses.detach().cpu().numpy().flatten()
        flat_t = noisy_dense_encoded_batch["t"].detach().cpu().numpy().flatten()
        bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins + 1)
        bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
        t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
        t_binned_n = np.bincount(bin_idx)
        for t_bin in np.unique(bin_idx).tolist():
            bin_start = bin_edges[t_bin]
            bin_end = bin_edges[t_bin + 1]
            t_range = f"f_loss t=[{int(bin_start*100)},{int(bin_end*100)})"
            range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
            loss_dict[t_range] = range_loss
        loss_dict["t_avg"] = np.mean(flat_t)
        
        return loss_dict, pred_x
        
    def flow_infer(
        self,
        model,
        input,
        t,
        mask
    ):
        if self.self_condition:
            with torch.no_grad():
                x_sc = model(input, t, mask=mask, x_sc=None)
        else:
            x_sc = None
            
        v = model(input, t, mask=mask, x_sc=x_sc)  
        return v     
    
    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        src_bond_type,
        mode,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargs
    ):
        if mode == "ae_only":
            #with torch.no_grad():
            if self.random_order and self.index is None:
                self.Myenter()
                self.index = np.random.permutation(512)
                self.Myexit()

            if classification_head_name is not None:
                features_only = True
    
            padding_mask = src_tokens.eq(self.padding_idx)
            latent_emb, latent_logvar, encoder_x_norm, delta_encoder_pair_rep_norm = self.enc(
                    src_tokens,
                    src_distance,
                    src_coord,
                    src_edge_type,
                    src_bond_type,
                    encoder_masked_tokens=None,
                    features_only=False,
                    classification_head_name=None,
                    **kwargs
                    )
            LOG_STD_MAX = 15
            LOG_STD_MIN = -15 # 너무 작은 분산을 막기 위함
            latent_logvar = torch.clamp(latent_logvar, LOG_STD_MIN, LOG_STD_MAX)
            
            std = torch.exp(0.5 * latent_logvar)
            q_z_given_x = torch.distributions.Normal(loc=latent_emb, scale=std)
            z = q_z_given_x.rsample()
            p_z = torch.distributions.Normal(
                loc=torch.zeros_like(latent_emb), 
                scale=torch.ones_like(std)
            )
        elif mode == "flow_only":
            self.encoder.eval()
            with torch.no_grad():
                if self.random_order and self.index is None:
                    self.Myenter()
                    self.index = np.random.permutation(512)
                    self.Myexit()

                if classification_head_name is not None:
                    features_only = True
        
                padding_mask = src_tokens.eq(self.padding_idx)
                latent_emb, latent_logvar, encoder_x_norm, delta_encoder_pair_rep_norm = self.enc(
                        src_tokens,
                        src_distance,
                        src_coord,
                        src_edge_type,
                        src_bond_type,
                        encoder_masked_tokens=None,
                        features_only=False,
                        classification_head_name=None,
                        **kwargs
                        )
                LOG_STD_MAX = 15
                LOG_STD_MIN = -15 # 너무 작은 분산을 막기 위함
                latent_logvar = torch.clamp(latent_logvar, LOG_STD_MIN, LOG_STD_MAX)
                
                std = torch.exp(0.5 * latent_logvar)
                q_z_given_x = torch.distributions.Normal(loc=latent_emb, scale=std)
                z = q_z_given_x.rsample()
                p_z = torch.distributions.Normal(
                    loc=torch.zeros_like(latent_emb), 
                    scale=torch.ones_like(std)
                )
        
        
        if mode == "ae_only":
            with torch.no_grad():
                flow_loss_dict, _ = self.flow_training(self.backbone, z.detach(), ~padding_mask)
            (
                logits,
                encoder_distance,
                encoder_coord,
                encoder_x_norm,
                decoder_x_norm,
                delta_encoder_pair_rep_norm,
                delta_decoder_pair_rep_norm,
                
            ) = self.dec(
                z,
                padding_mask,
                encoder_x_norm,
                delta_encoder_pair_rep_norm,
                classification_head_name
            )
        elif mode == "flow_only":
            flow_loss_dict, _ = self.flow_training(self.backbone, z.detach(), ~padding_mask)
            with torch.no_grad():
                (
                    logits,
                    encoder_distance,
                    encoder_coord,
                    encoder_x_norm,
                    decoder_x_norm,
                    delta_encoder_pair_rep_norm,
                    delta_decoder_pair_rep_norm,
                ) = self.dec(
                    z,
                    padding_mask,
                    encoder_x_norm,
                    delta_encoder_pair_rep_norm
                )

        if mode == "dual":
            if self.random_order and self.index is None:
                self.Myenter()
                self.index = np.random.permutation(512)
                self.Myexit()

            if classification_head_name is not None:
                features_only = True
    
            padding_mask = src_tokens.eq(self.padding_idx)
            latent_emb, latent_logvar, encoder_x_norm, delta_encoder_pair_rep_norm = self.enc(
                    src_tokens,
                    src_distance,
                    src_coord,
                    src_edge_type,
                    src_bond_type,
                    encoder_masked_tokens=None,
                    features_only=False,
                    classification_head_name=None,
                    **kwargs
                    )
            LOG_STD_MAX = 15
            LOG_STD_MIN = -15 # 너무 작은 분산을 막기 위함
            latent_logvar = torch.clamp(latent_logvar, LOG_STD_MIN, LOG_STD_MAX)
            
            std = torch.exp(0.5 * latent_logvar)
            q_z_given_x = torch.distributions.Normal(loc=latent_emb, scale=std)
            z = q_z_given_x.rsample()
            p_z = torch.distributions.Normal(
                loc=torch.zeros_like(latent_emb), 
                scale=torch.ones_like(std)
            )    
            
            flow_loss_dict, _ = self.flow_training(self.backbone, z.detach(), ~padding_mask)    
            (
                logits,
                encoder_distance,
                encoder_coord,
                encoder_x_norm,
                decoder_x_norm,
                delta_encoder_pair_rep_norm,
                delta_decoder_pair_rep_norm,
            ) = self.dec(
                z,
                padding_mask,
                encoder_x_norm,
                delta_encoder_pair_rep_norm
            )
        
            
        return (
            logits,
            encoder_distance,
            encoder_coord,
            encoder_x_norm,
            decoder_x_norm,
            delta_encoder_pair_rep_norm,
            delta_decoder_pair_rep_norm,
            (z, q_z_given_x, p_z,latent_emb,std),
            flow_loss_dict 
        ) 
        
    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates
        
        
@register_model("unimol_Optimal_padding")
class UniMolOptimalPaddingModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="L", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="A",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--emb-dropout",
            type=float,
            metavar="D",
            help="dropout probability for embeddings",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-seq-len", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--post-ln", type=bool, help="use post layernorm or pre layernorm"
        )
        parser.add_argument(
            "--masked-token-loss",
            type=float,
            metavar="D",
            help="mask loss ratio",
        )
        parser.add_argument(
            "--masked-dist-loss",
            type=float,
            metavar="D",
            help="masked distance loss ratio",
        )
        parser.add_argument(
            "--masked-coord-loss",
            type=float,
            metavar="D",
            help="masked coord loss ratio",
        )
        parser.add_argument(
            "--encoder-x-norm-loss",
            type=float,
            metavar="D",
            help="x norm loss ratio",
        )
        parser.add_argument(
            "--encoder-delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio",
        )
        parser.add_argument(
            "--decoder-x-norm-loss",
            type=float,
            metavar="D",
            help="x norm loss ratio",
        )
        parser.add_argument(
            "--decoder-delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio",
        )
        parser.add_argument(
            "--masked-coord-dist-loss",
            type=float,
            metavar="D",
            help="masked coord dist loss ratio",
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="train",
            choices=["train", "infer"],
        )
        parser.add_argument(
            "--encoder-unmasked-tokens-only", action='store_true', help="only input unmasked tokens into encoder"
        )
        parser.add_argument(
            "--encoder-masked-3d-pe", action='store_true', help="only masked #D PE for encoder"
        )
        parser.add_argument(
            "--encoder-apply-pe", action='store_true', help="apply PE for encoder"
        )
        parser.add_argument(
            "--feed-pair-rep-to-decoder", action='store_true', help="feed the pair representations of encoder to decoder"
        )
        parser.add_argument(
            "--decoder-no-pe", action='store_true', help="Don't apply PE for decoder"
        )
        parser.add_argument(
            "--feed-token-rep-to-decoder", action='store_true', help="feed the token representations of encoder to decoder"
        )
        parser.add_argument(
            "--decoder-noise", action='store_true', help="Feed noise or [mask] to decoder"
        )
        parser.add_argument(
            "--random-order", action='store_true', help="Feed noise or [mask] to decoder"
        )

    def __init__(self, args, dictionary):
        super().__init__()
        #print('Using modified MAE')
        base_architecture(args)
        self.args = args
        self.encoder_masked_3d_pe = args.encoder_masked_3d_pe
        self.encoder_apply_pe = args.encoder_apply_pe
        self.feed_pair_rep_to_decoder = args.feed_pair_rep_to_decoder
        self.decoder_no_pe = args.decoder_no_pe
        self.feed_token_rep_to_decoder = args.feed_token_rep_to_decoder
        self.decoder_noise = args.decoder_noise
        self.random_order = args.random_order
        self.interpolant = FlowMatchingInterpolant(device="cpu")
        #self.interpolant = None

        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(dictionary), args.encoder_embed_dim, self.padding_idx
        )

        self.PE = None
        self.index = None
        if self.random_order:
            self.init_state()
        # Original TransformerWrapper backbone (commented for DiT replacement)
        # self.backbone = TransformerWrapper(
        #     attn_layers=Encoder(dim=args.encoder_embed_dim, depth=10),
        #     emb_dropout=0.0)
        
        # Use DiT model as backbone instead of TransformerWrapper
        self.backbone = DiT(
            # d_x=args.encoder_embed_dim,
            # d_model=args.encoder_embed_dim,
            # nhead=args.encoder_attention_heads,
            d_x=8,
            d_model=512,
            nhead=8,
        )
        self._num_updates = None
        self.encoder_rot = TransformerEncoderWithPair(
            encoder_layers=3,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.encoder_delta_pair_repr_norm_loss < 0,
        )
        
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.encoder_delta_pair_repr_norm_loss < 0,
        )
        self.decoder = TransformerEncoderWithPair(
            #encoder_layers=args.decoder_layers,
            encoder_layers=3,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.decoder_ffn_embed_dim,
            attention_heads=args.decoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.decoder_delta_pair_repr_norm_loss < 0,
        )
        #if args.masked_token_loss > 0:
        self.lm_head = MaskLMHead(
            embed_dim=args.encoder_embed_dim,
            output_dim=len(dictionary),
            activation_fn=args.activation_fn,
            weight=None,
        )
            
        # self.logvar_proj = NonLinearHead(
        #     args.encoder_embed_dim, args.encoder_embed_dim, 
        #     args.activation_fn
        # )
        self.quant_mean = NonLinearHead(
            input_dim = args.encoder_embed_dim, 
            out_dim = 8, 
            activation_fn = args.activation_fn, 
            hidden = args.encoder_embed_dim//2
        )
               
        self.quant_rot = NonLinearHead(
            input_dim = args.encoder_embed_dim, 
            out_dim = 9, 
            activation_fn = args.activation_fn, 
            hidden = args.encoder_embed_dim//2
        )
        self.logvar_proj = NonLinearHead(
            input_dim = args.encoder_embed_dim, 
            out_dim = 8, 
            activation_fn = args.activation_fn, 
            hidden = args.encoder_embed_dim//2
        )
        self.quant_expand = NonLinearHead(
            input_dim = 8, 
            out_dim = args.encoder_embed_dim, 
            activation_fn = args.activation_fn, 
            hidden = args.encoder_embed_dim//2
        )
        
        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(
            K, args.encoder_attention_heads, args.activation_fn
        )
        self.gbf_proj2 = NonLinearHead(
            K, args.encoder_attention_heads, args.activation_fn
        )
        self.gbf = GaussianLayer(K, n_edge_type)
        self.gbf2 = GaussianLayer(K, 5)
        
        #if args.masked_coord_loss > 0:
        self.pair2coord_proj = NonLinearHead(
            args.decoder_attention_heads, 1, args.activation_fn
        )
        #if args.masked_dist_loss > 0:
        self.dist_head = DistanceHead(
            args.decoder_attention_heads, args.activation_fn
        )
        self.init_pos_prog = NonLinearHeadPos(
            512, 1, args.activation_fn
        )
        self.pos_embedder = nn.Sequential(
            nn.Linear(3, args.encoder_embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim),
        )
        self.pos_decoder = nn.Sequential(
            nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(args.encoder_embed_dim, 3),
        )
        self.classification_heads = nn.ModuleDict()
        self.apply(init_bert_params)
        self.encoder_unmasked_tokens_only = args.encoder_unmasked_tokens_only
        self.dictionary = dictionary

        self.embed_positions = SinusoidalPositionalEmbedding(
            embedding_dim = args.max_seq_len,
            padding_idx = dictionary.pad(),
            init_size = args.max_seq_len,
        )

        self.mask_idx = dictionary.index("[MASK]")
        self.encoder_attention_heads = args.encoder_attention_heads
        self.self_condition = True
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)
    
    @classmethod
    def init_state(self):
        original_state = np.random.get_state()
        np.random.seed(0)
        self.numpy_random_state = np.random.get_state()
        np.random.set_state(original_state)
    
    @classmethod
    def Myenter(self):
        self.original_state = np.random.get_state()
        np.random.set_state(self.numpy_random_state)

    @classmethod
    def Myexit(self):
        self.numpy_random_state = np.random.get_state()
        np.random.set_state(self.original_state)

    def rot(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        src_bond_type,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargs        
    ):
        encoder_src_tokens = src_tokens
        encoder_src_coord = src_coord
        encoder_src_distance = src_distance
        encoder_src_edge_type = src_edge_type

        encoder_padding_mask = encoder_src_tokens.eq(self.padding_idx)
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        x = self.embed_tokens(encoder_src_tokens) + self.embed_positions(~padding_mask)
        x = x + self.pos_embedder(encoder_src_coord)
        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias
        def get_bond_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf2(dist, et)
            gbf_result = self.gbf_proj2(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(encoder_src_distance, encoder_src_edge_type) #+ get_bond_features(encoder_src_distance, src_bond_type)

        (
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            encoder_x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder_rot(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0
        
        rot = torch.sum(encoder_rep * ~padding_mask.unsqueeze(-1), 1) / torch.sum(~padding_mask, 1).unsqueeze(-1)
        encoder_rot = self.quant_rot(rot).reshape(-1, 3, 3)
        rot_mat = roma.special_procrustes(encoder_rot)
        x_rot = torch.bmm(encoder_src_coord, rot_mat)
        #encoder_output_embedding = torch.nn.Tanh()(encoder_rep_quant)
        
        # if self.feed_token_rep_to_decoder:
        #     encoder_output_embedding = encoder_rep
        # else:
        #     if encoder_masked_tokens is None:
        #         encoder_output_embedding = encoder_rep
        #     else:
        #         mask_tokens = torch.zeros_like(src_tokens).fill_(self.mask_idx)
        #         masked_embeddings = self.embed_tokens(mask_tokens)
        #         encoder_output_embedding = torch.where(encoder_masked_tokens.unsqueeze(-1).expand_as(encoder_rep), masked_embeddings, encoder_rep)
        #latent_logvar = self.logvar_proj(encoder_rep)
        
        return x_rot, rot_mat
    
    
    def enc(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        src_bond_type,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargs        
    ):
        encoder_src_tokens = src_tokens
        encoder_src_coord = src_coord
        encoder_src_distance = src_distance
        encoder_src_edge_type = src_edge_type

        encoder_padding_mask = encoder_src_tokens.eq(self.padding_idx)
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        x = self.embed_tokens(encoder_src_tokens) + self.embed_positions(~padding_mask)
        x = x + self.pos_embedder(encoder_src_coord)
        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias
        def get_bond_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf2(dist, et)
            gbf_result = self.gbf_proj2(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(encoder_src_distance, encoder_src_edge_type) #+ get_bond_features(encoder_src_distance, src_bond_type)

        (
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            encoder_x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0
        encoder_rep_quant = self.quant_mean(encoder_rep)
        encoder_output_embedding = torch.nn.Tanh()(encoder_rep_quant)
        
        # if self.feed_token_rep_to_decoder:
        #     encoder_output_embedding = encoder_rep
        # else:
        #     if encoder_masked_tokens is None:
        #         encoder_output_embedding = encoder_rep
        #     else:
        #         mask_tokens = torch.zeros_like(src_tokens).fill_(self.mask_idx)
        #         masked_embeddings = self.embed_tokens(mask_tokens)
        #         encoder_output_embedding = torch.where(encoder_masked_tokens.unsqueeze(-1).expand_as(encoder_rep), masked_embeddings, encoder_rep)
        latent_logvar = self.logvar_proj(encoder_rep)
        
        return encoder_output_embedding, latent_logvar, encoder_x_norm, delta_encoder_pair_rep_norm
    
    def dec(self, encoder_output_embedding, padding_mask, encoder_x_norm=None, delta_encoder_pair_rep_norm=None, classification_head_name=None):
        encoder_masked_tokens=None
        features_only=False
        #classification_head_name=None   
             
        #if not self.decoder_no_pe:
        #    encoder_output_embedding = encoder_output_embedding + self.embed_positions(~padding_mask)
        encoder_output_embedding = self.quant_expand(encoder_output_embedding) + self.embed_positions(~padding_mask)
        n_node = encoder_output_embedding.size(1)
        # if self.feed_pair_rep_to_decoder:
        #     assert self.decoder_noise is not True
        #     attn_bias = encoder_pair_rep.reshape(-1, n_node, n_node)
        # else:
        #     if not self.decoder_noise:
        #         bsz = padding_mask.size(0)
        #         attn_bias = encoder_output_embedding.new_zeros((bsz*self.encoder_attention_heads, n_node, n_node))
        #     else:
        #         attn_bias = get_dist_features(src_distance, src_edge_type)
        bsz = padding_mask.size(0)
        attn_bias = encoder_output_embedding.new_zeros((bsz*self.encoder_attention_heads, n_node, n_node))
        
        (
            decoder_rep,
            decoder_pair_rep,
            delta_decoder_pair_rep,
            decoder_x_norm,
            delta_decoder_pair_rep_norm,
        ) = self.decoder(encoder_output_embedding, padding_mask=padding_mask, attn_mask=attn_bias)

        decoder_pair_rep[decoder_pair_rep == float("-inf")] = 0

        encoder_distance = None
        encoder_coord = None

        if not features_only:
            if self.args.masked_token_loss > 0:
                #logits = self.lm_head(decoder_rep, encoder_masked_tokens) # padding_mask
                logits = self.lm_head(decoder_rep, ~padding_mask) 
            if self.args.masked_coord_loss > 0:
                # atom_num = (torch.sum(1 - padding_mask.type_as(decoder_rep), dim=1)).view(-1, 1, 1, 1)
                # #coords_emb = torch.zeros((padding_mask.size(0),padding_mask.size(1),3), device = decoder_pair_rep.device)
                # coords_emb = torch.randn((padding_mask.size(0),padding_mask.size(1),3), device = decoder_pair_rep.device)
                
                # #coords_emb = self.init_pos_prog(encoder_output_embedding)
                # for i in range(4):
                #     coords_emb = coords_emb * ~padding_mask.unsqueeze(-1)
                #     delta_pos = coords_emb.unsqueeze(1) - coords_emb.unsqueeze(2)
                #     attn_probs = self.pair2coord_proj(delta_decoder_pair_rep)
                #     coord_update = delta_pos / atom_num * attn_probs
                #     pair_coords_mask = (1 - padding_mask.float()).unsqueeze(-1) * (1 - padding_mask.float()).unsqueeze(1)
                #     coord_update = coord_update * pair_coords_mask.unsqueeze(-1)
                #     coord_update = torch.sum(coord_update, dim=2)
                #     #encoder_coord = coords_emb + coord_update
                #     coords_emb = coords_emb + coord_update
                # encoder_coord = coords_emb
                encoder_coord = self.pos_decoder(decoder_rep)
                
            if self.args.masked_dist_loss > 0:
                encoder_distance = self.dist_head(decoder_pair_rep)

        if classification_head_name is not None:
            finetuning_embedding = encoder_output_embedding
            # (
            #     finetuning_embedding,
            #     _,
            #     _,
            #     _,
            #     _,
            # ) = self.finetune_block(encoder_output_embedding.detach(), padding_mask=padding_mask, attn_mask=attn_bias)
            
            finetune_input = torch.sum(finetuning_embedding * ~padding_mask.unsqueeze(-1), 1) / torch.sum(~padding_mask, 1).unsqueeze(-1)
            logits = self.classification_heads[classification_head_name](finetune_input)
        if self.args.mode == 'infer':
            return encoder_rep, encoder_pair_rep
        else:
            return (
                logits,
                encoder_distance,
                encoder_coord,
                encoder_x_norm,
                decoder_x_norm,
                delta_encoder_pair_rep_norm,
                delta_decoder_pair_rep_norm,
            )            
    def flow_training(
        self,
        model,
        z_1,
        mask
    ):
        """Vanilla Flow"""
        # dtype = torch.float32
        # t = torch.rand((z_1.size(0),), dtype=dtype, device=z_1.device)
        # t = t.view(-1,1,1)
        # z_0 = torch.randn_like(z_1, device=z_1.device)
        # #z_t = (1 - t) * z_0 + (1e-5 + (1 - 1e-5) * t) * z_1
        # z_t = (1 - t) * z_0 + t * z_1
        # #u = (1 - 1e-5) * z_1 - z_0
        # u = (1 - 1e-5) * z_1 - z_0
        # #v = model(z_t, t.squeeze(-1), mask=mask)
        # v = self.flow_infer(model, z_t, t.squeeze(-1), mask=mask)
        # loss = F.mse_loss(v, u, reduction='none')
        
        
        """ OT-CFM"""

        # FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.)
        # x1 = z_1
        # x0 = torch.randn_like(x1)
        # t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        # vt = self.flow_infer(model, xt, t.unsqueeze(-1), mask=mask)
        # loss = F.mse_loss(vt, ut, reduction='none')
        # return loss, vt
    
        """Protein Frame Flow"""
        self.interpolant.device = z_1.device
        dense_encoded_batch = {"x_1": z_1, "token_mask": mask, "diffuse_mask": mask}
        noisy_dense_encoded_batch = self.interpolant.corrupt_batch(dense_encoded_batch)

        #if self.self_condition:
        # Use self-conditioning for ~half training batches
        if (
            self.interpolant.self_condition
            and random.random() < self.interpolant.self_condition_prob
        ):
            
            with torch.no_grad():
                x_sc = model(noisy_dense_encoded_batch["x_t"],
                             noisy_dense_encoded_batch["t"],
                             mask=mask,
                             x_sc=None)
        else:
            x_sc = None
            
        pred_x = model(noisy_dense_encoded_batch["x_t"],
                        noisy_dense_encoded_batch["t"],
                        mask=mask,
                        x_sc=x_sc)

        gt_x_1 = noisy_dense_encoded_batch["x_1"]
        "Default normalization scale"
        # norm_scale = 1 - torch.min(noisy_dense_encoded_batch["t"].unsqueeze(-1), torch.tensor(0.9))
        # x_error = (gt_x_1 - pred_x) / norm_scale
        
        """Signal-to-Noise Ratio"""
        snr = noisy_dense_encoded_batch["t"].squeeze() / (1 - noisy_dense_encoded_batch["t"].squeeze() + 1e-8)
        # Min-SNR weighting (방법 1: 직접 clamp)
        weight = torch.clamp(snr, max=5.0)
        # 또는 (방법 2: normalized version)
        weight = torch.clamp(snr, max=5.0) / (snr + 1e-8)  # 이건 t→1일 때 weight→0
        # 또는 (방법 3: standard Min-SNR)
        weight = torch.minimum(snr, torch.tensor(5.0, device=snr.device))
        # 최종 loss
        x_error = (gt_x_1 - pred_x) * weight.unsqueeze(-1).unsqueeze(-1) 
        
        
        loss_mask = (
            noisy_dense_encoded_batch["token_mask"] * noisy_dense_encoded_batch["diffuse_mask"]
        )
        loss_denom = torch.sum(loss_mask, dim=-1) * pred_x.size(-1)
        x_loss = torch.sum(x_error**2 * loss_mask[..., None], dim=(-1, -2)) / loss_denom
        #loss_dict = {"loss": x_loss.mean(), "x_loss": x_loss}
        loss_dict = {"loss": x_loss.mean()}
         
        num_bins = 4
        #flat_losses = x_loss.detach().cpu().numpy().flatten()
        #flat_losses = (gt_x_1 - pred_x)
        flat_losses = torch.sum((gt_x_1 - pred_x)**2 * loss_mask[..., None], dim=(-1, -2)) / loss_denom
        flat_losses = flat_losses.detach().cpu().numpy().flatten()
        flat_t = noisy_dense_encoded_batch["t"].detach().cpu().numpy().flatten()
        bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins + 1)
        bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
        t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
        t_binned_n = np.bincount(bin_idx)
        for t_bin in np.unique(bin_idx).tolist():
            bin_start = bin_edges[t_bin]
            bin_end = bin_edges[t_bin + 1]
            t_range = f"f_loss t=[{int(bin_start*100)},{int(bin_end*100)})"
            range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
            loss_dict[t_range] = range_loss
        loss_dict["t_avg"] = np.mean(flat_t)
        
        return loss_dict, pred_x
        
    def flow_infer(
        self,
        model,
        input,
        t,
        mask
    ):
        if self.self_condition:
            with torch.no_grad():
                x_sc = model(input, t, mask=mask, x_sc=None)
        else:
            x_sc = None
            
        v = model(input, t, mask=mask, x_sc=x_sc)  
        return v     
    
    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        src_bond_type,
        mode,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargs
    ):
        if mode == "ae_only":
            #with torch.no_grad():
            if self.random_order and self.index is None:
                self.Myenter()
                self.index = np.random.permutation(512)
                self.Myexit()
                
            padding_mask = src_tokens.eq(self.padding_idx)
            if classification_head_name is not None:
                features_only = True
                with torch.no_grad():
                    src_coord, rot_mat = self.rot(
                            src_tokens,
                            src_distance,
                            src_coord,
                            src_edge_type,
                            src_bond_type,
                            encoder_masked_tokens=None,
                            features_only=False,
                            classification_head_name=None,
                            **kwargs
                            )
            else:
                src_coord, rot_mat = self.rot(
                        src_tokens,
                        src_distance,
                        src_coord,
                        src_edge_type,
                        src_bond_type,
                        encoder_masked_tokens=None,
                        features_only=False,
                        classification_head_name=None,
                        **kwargs
                        )
                
            latent_emb, latent_logvar, encoder_x_norm, delta_encoder_pair_rep_norm = self.enc(
                    src_tokens,
                    src_distance,
                    src_coord,
                    src_edge_type,
                    src_bond_type,
                    encoder_masked_tokens=None,
                    features_only=False,
                    classification_head_name=None,
                    **kwargs
                    )
            LOG_STD_MAX = 15
            LOG_STD_MIN = -15 # 너무 작은 분산을 막기 위함
            latent_logvar = torch.clamp(latent_logvar, LOG_STD_MIN, LOG_STD_MAX)
            
            std = torch.exp(0.5 * latent_logvar)
            q_z_given_x = torch.distributions.Normal(loc=latent_emb, scale=std)
            z = q_z_given_x.rsample()
            p_z = torch.distributions.Normal(
                loc=torch.zeros_like(latent_emb), 
                scale=torch.ones_like(std)
            )
        elif mode == "flow_only":
            self.encoder.eval()
            with torch.no_grad():
                if self.random_order and self.index is None:
                    self.Myenter()
                    self.index = np.random.permutation(512)
                    self.Myexit()

                if classification_head_name is not None:
                    features_only = True
        
                padding_mask = src_tokens.eq(self.padding_idx)
                src_coord,rot_mat = self.rot(
                        src_tokens,
                        src_distance,
                        src_coord,
                        src_edge_type,
                        src_bond_type,
                        encoder_masked_tokens=None,
                        features_only=False,
                        classification_head_name=None,
                        **kwargs
                        )
                latent_emb, latent_logvar, encoder_x_norm, delta_encoder_pair_rep_norm = self.enc(
                        src_tokens,
                        src_distance,
                        src_coord,
                        src_edge_type,
                        src_bond_type,
                        encoder_masked_tokens=None,
                        features_only=False,
                        classification_head_name=None,
                        **kwargs
                        )
                LOG_STD_MAX = 15
                LOG_STD_MIN = -15 # 너무 작은 분산을 막기 위함
                latent_logvar = torch.clamp(latent_logvar, LOG_STD_MIN, LOG_STD_MAX)
                
                std = torch.exp(0.5 * latent_logvar)
                q_z_given_x = torch.distributions.Normal(loc=latent_emb, scale=std)
                z = q_z_given_x.rsample()
                p_z = torch.distributions.Normal(
                    loc=torch.zeros_like(latent_emb), 
                    scale=torch.ones_like(std)
                )
        
        
        if mode == "ae_only":
            with torch.no_grad():
                flow_loss_dict, _ = self.flow_training(self.backbone, z.detach(), ~padding_mask)
            (
                logits,
                encoder_distance,
                encoder_coord,
                encoder_x_norm,
                decoder_x_norm,
                delta_encoder_pair_rep_norm,
                delta_decoder_pair_rep_norm,
                
            ) = self.dec(
                z,
                padding_mask,
                encoder_x_norm,
                delta_encoder_pair_rep_norm,
                classification_head_name
            )
        elif mode == "flow_only":
            flow_loss_dict, _ = self.flow_training(self.backbone, z.detach(), ~padding_mask)
            with torch.no_grad():
                (
                    logits,
                    encoder_distance,
                    encoder_coord,
                    encoder_x_norm,
                    decoder_x_norm,
                    delta_encoder_pair_rep_norm,
                    delta_decoder_pair_rep_norm,
                ) = self.dec(
                    z,
                    padding_mask,
                    encoder_x_norm,
                    delta_encoder_pair_rep_norm
                )

        if mode == "dual":
            if self.random_order and self.index is None:
                self.Myenter()
                self.index = np.random.permutation(512)
                self.Myexit()

            if classification_head_name is not None:
                features_only = True
    
            padding_mask = src_tokens.eq(self.padding_idx)
            src_coord,rot_mat = self.rot(
                    src_tokens,
                    src_distance,
                    src_coord,
                    src_edge_type,
                    src_bond_type,
                    encoder_masked_tokens=None,
                    features_only=False,
                    classification_head_name=None,
                    **kwargs
                    )
            
            latent_emb, latent_logvar, encoder_x_norm, delta_encoder_pair_rep_norm = self.enc(
                    src_tokens,
                    src_distance,
                    src_coord,
                    src_edge_type,
                    src_bond_type,
                    encoder_masked_tokens=None,
                    features_only=False,
                    classification_head_name=None,
                    **kwargs
                    )
            LOG_STD_MAX = 15
            LOG_STD_MIN = -15 # 너무 작은 분산을 막기 위함
            latent_logvar = torch.clamp(latent_logvar, LOG_STD_MIN, LOG_STD_MAX)
            
            std = torch.exp(0.5 * latent_logvar)
            q_z_given_x = torch.distributions.Normal(loc=latent_emb, scale=std)
            z = q_z_given_x.rsample()
            p_z = torch.distributions.Normal(
                loc=torch.zeros_like(latent_emb), 
                scale=torch.ones_like(std)
            )    
            
            flow_loss_dict, _ = self.flow_training(self.backbone, z.detach(), ~padding_mask)    
            (
                logits,
                encoder_distance,
                encoder_coord,
                encoder_x_norm,
                decoder_x_norm,
                delta_encoder_pair_rep_norm,
                delta_decoder_pair_rep_norm,
            ) = self.dec(
                z,
                padding_mask,
                encoder_x_norm,
                delta_encoder_pair_rep_norm
            )
        
            
        return (
            logits,
            encoder_distance,
            (rot_mat, encoder_coord),
            encoder_x_norm,
            decoder_x_norm,
            delta_encoder_pair_rep_norm,
            delta_decoder_pair_rep_norm,
            (z, q_z_given_x, p_z,latent_emb,std),
            flow_loss_dict 
        ) 
        
    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )
        self.finetune_block = TransformerEncoderWithPair(
            #encoder_layers=args.decoder_layers,
            encoder_layers=1,
            embed_dim=self.args.encoder_embed_dim,
            ffn_embed_dim=self.args.decoder_ffn_embed_dim,
            attention_heads=self.args.decoder_attention_heads,
            emb_dropout=self.args.emb_dropout,
            dropout=self.args.dropout,
            attention_dropout=self.args.attention_dropout,
            activation_dropout=self.args.activation_dropout,
            max_seq_len=self.args.max_seq_len,
            activation_fn=self.args.activation_fn,
            no_final_head_layer_norm=self.args.decoder_delta_pair_repr_norm_loss < 0,
        )
        
    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates
        
        
@register_model("unimol_Optimal_padding2")
class UniMolOptimalPaddingModel2(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="L", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="A",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--emb-dropout",
            type=float,
            metavar="D",
            help="dropout probability for embeddings",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-seq-len", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--post-ln", type=bool, help="use post layernorm or pre layernorm"
        )
        parser.add_argument(
            "--masked-token-loss",
            type=float,
            metavar="D",
            help="mask loss ratio",
        )
        parser.add_argument(
            "--masked-dist-loss",
            type=float,
            metavar="D",
            help="masked distance loss ratio",
        )
        parser.add_argument(
            "--masked-coord-loss",
            type=float,
            metavar="D",
            help="masked coord loss ratio",
        )
        parser.add_argument(
            "--encoder-x-norm-loss",
            type=float,
            metavar="D",
            help="x norm loss ratio",
        )
        parser.add_argument(
            "--encoder-delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio",
        )
        parser.add_argument(
            "--decoder-x-norm-loss",
            type=float,
            metavar="D",
            help="x norm loss ratio",
        )
        parser.add_argument(
            "--decoder-delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio",
        )
        parser.add_argument(
            "--masked-coord-dist-loss",
            type=float,
            metavar="D",
            help="masked coord dist loss ratio",
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="train",
            choices=["train", "infer"],
        )
        parser.add_argument(
            "--encoder-unmasked-tokens-only", action='store_true', help="only input unmasked tokens into encoder"
        )
        parser.add_argument(
            "--encoder-masked-3d-pe", action='store_true', help="only masked #D PE for encoder"
        )
        parser.add_argument(
            "--encoder-apply-pe", action='store_true', help="apply PE for encoder"
        )
        parser.add_argument(
            "--feed-pair-rep-to-decoder", action='store_true', help="feed the pair representations of encoder to decoder"
        )
        parser.add_argument(
            "--decoder-no-pe", action='store_true', help="Don't apply PE for decoder"
        )
        parser.add_argument(
            "--feed-token-rep-to-decoder", action='store_true', help="feed the token representations of encoder to decoder"
        )
        parser.add_argument(
            "--decoder-noise", action='store_true', help="Feed noise or [mask] to decoder"
        )
        parser.add_argument(
            "--random-order", action='store_true', help="Feed noise or [mask] to decoder"
        )

    def __init__(self, args, dictionary):
        super().__init__()
        #print('Using modified MAE')
        base_architecture(args)
        self.args = args
        self.encoder_masked_3d_pe = args.encoder_masked_3d_pe
        self.encoder_apply_pe = args.encoder_apply_pe
        self.feed_pair_rep_to_decoder = args.feed_pair_rep_to_decoder
        self.decoder_no_pe = args.decoder_no_pe
        self.feed_token_rep_to_decoder = args.feed_token_rep_to_decoder
        self.decoder_noise = args.decoder_noise
        self.random_order = args.random_order
        self.interpolant = FlowMatchingInterpolant(device="cpu")
        #self.interpolant = None

        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(dictionary), args.encoder_embed_dim, self.padding_idx
        )

        self.PE = None
        self.index = None
        if self.random_order:
            self.init_state()
        # Original TransformerWrapper backbone (commented for DiT replacement)
        # self.backbone = TransformerWrapper(
        #     attn_layers=Encoder(dim=args.encoder_embed_dim, depth=10),
        #     emb_dropout=0.0)
        
        # Use DiT model as backbone instead of TransformerWrapper
        self.backbone = DiT(
            # d_x=args.encoder_embed_dim,
            # d_model=args.encoder_embed_dim,
            # nhead=args.encoder_attention_heads,
            d_x=8,
            d_model=512,
            nhead=8,
        )
        self._num_updates = None
        self.encoder_rot = TransformerEncoderWithPair(
            encoder_layers=3,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.encoder_delta_pair_repr_norm_loss < 0,
        )
        
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.encoder_delta_pair_repr_norm_loss < 0,
        )
        self.decoder = TransformerEncoderWithPair(
            #encoder_layers=args.decoder_layers,
            encoder_layers=3,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.decoder_ffn_embed_dim,
            attention_heads=args.decoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.decoder_delta_pair_repr_norm_loss < 0,
        )
        
        self.pretraining_property = TransformerEncoderWithPair(
            #encoder_layers=args.decoder_layers,
            encoder_layers=3,
            embed_dim=self.args.encoder_embed_dim,
            ffn_embed_dim=self.args.decoder_ffn_embed_dim,
            attention_heads=self.args.decoder_attention_heads,
            emb_dropout=self.args.emb_dropout,
            dropout=self.args.dropout,
            attention_dropout=self.args.attention_dropout,
            activation_dropout=self.args.activation_dropout,
            max_seq_len=self.args.max_seq_len,
            activation_fn=self.args.activation_fn,
            no_final_head_layer_norm=self.args.decoder_delta_pair_repr_norm_loss < 0,
        )
        
        
        #if args.masked_token_loss > 0:
        self.lm_head = MaskLMHead(
            embed_dim=args.encoder_embed_dim,
            output_dim=len(dictionary),
            activation_fn=args.activation_fn,
            weight=None,
        )
            
        # self.logvar_proj = NonLinearHead(
        #     args.encoder_embed_dim, args.encoder_embed_dim, 
        #     args.activation_fn
        # )
        self.quant_mean = NonLinearHead(
            input_dim = args.encoder_embed_dim, 
            out_dim = 8, 
            activation_fn = args.activation_fn, 
            hidden = args.encoder_embed_dim//2
        )
               
        self.quant_rot = NonLinearHead(
            input_dim = args.encoder_embed_dim, 
            out_dim = 9, 
            activation_fn = args.activation_fn, 
            hidden = args.encoder_embed_dim//2
        )
        self.logvar_proj = NonLinearHead(
            input_dim = args.encoder_embed_dim, 
            out_dim = 8, 
            activation_fn = args.activation_fn, 
            hidden = args.encoder_embed_dim//2
        )
        self.quant_expand = NonLinearHead(
            input_dim = 8, 
            out_dim = args.encoder_embed_dim, 
            activation_fn = args.activation_fn, 
            hidden = args.encoder_embed_dim//2
        )
        
        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(
            K, args.encoder_attention_heads, args.activation_fn
        )
        self.gbf_proj2 = NonLinearHead(
            K, args.encoder_attention_heads, args.activation_fn
        )
        self.gbf = GaussianLayer(K, n_edge_type)
        self.gbf2 = GaussianLayer(K, 5)
        
        #if args.masked_coord_loss > 0:
        self.pair2coord_proj = NonLinearHead(
            args.decoder_attention_heads, 1, args.activation_fn
        )
        #if args.masked_dist_loss > 0:
        self.dist_head = DistanceHead(
            args.decoder_attention_heads, args.activation_fn
        )
        self.init_pos_prog = NonLinearHeadPos(
            512, 1, args.activation_fn
        )
        self.pos_embedder = nn.Sequential(
            nn.Linear(3, args.encoder_embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim),
        )
        self.pos_decoder = nn.Sequential(
            nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(args.encoder_embed_dim, 3),
        )
        self.classification_heads = nn.ModuleDict()
        self.apply(init_bert_params)
        self.encoder_unmasked_tokens_only = args.encoder_unmasked_tokens_only
        self.dictionary = dictionary
        self.prop_decoder = nn.Sequential(
            nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(args.encoder_embed_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 38),
        )

        self.embed_positions = SinusoidalPositionalEmbedding(
            embedding_dim = args.max_seq_len,
            padding_idx = dictionary.pad(),
            init_size = args.max_seq_len,
        )

        self.mask_idx = dictionary.index("[MASK]")
        self.encoder_attention_heads = args.encoder_attention_heads
        self.self_condition = True
        self.gk_bond = BondGaussianLayer()
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)
    
    @classmethod
    def init_state(self):
        original_state = np.random.get_state()
        np.random.seed(0)
        self.numpy_random_state = np.random.get_state()
        np.random.set_state(original_state)
    
    @classmethod
    def Myenter(self):
        self.original_state = np.random.get_state()
        np.random.set_state(self.numpy_random_state)

    @classmethod
    def Myexit(self):
        self.numpy_random_state = np.random.get_state()
        np.random.set_state(self.original_state)

    def rot(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        src_bond_type,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargs        
    ):
        encoder_src_tokens = src_tokens
        encoder_src_coord = src_coord
        encoder_src_distance = src_distance
        encoder_src_edge_type = src_edge_type

        encoder_padding_mask = encoder_src_tokens.eq(self.padding_idx)
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        x = self.embed_tokens(encoder_src_tokens) + self.embed_positions(~padding_mask)
        x = x + self.pos_embedder(encoder_src_coord)
        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias
        def get_bond_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf2(dist, et)
            gbf_result = self.gbf_proj2(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(encoder_src_distance, encoder_src_edge_type) #+ get_bond_features(encoder_src_distance, src_bond_type)

        (
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            encoder_x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder_rot(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0
        
        rot = torch.sum(encoder_rep * ~padding_mask.unsqueeze(-1), 1) / torch.sum(~padding_mask, 1).unsqueeze(-1)
        encoder_rot = self.quant_rot(rot).reshape(-1, 3, 3)
        rot_mat = roma.special_procrustes(encoder_rot)
        x_rot = torch.bmm(encoder_src_coord, rot_mat)
        #encoder_output_embedding = torch.nn.Tanh()(encoder_rep_quant)
        
        # if self.feed_token_rep_to_decoder:
        #     encoder_output_embedding = encoder_rep
        # else:
        #     if encoder_masked_tokens is None:
        #         encoder_output_embedding = encoder_rep
        #     else:
        #         mask_tokens = torch.zeros_like(src_tokens).fill_(self.mask_idx)
        #         masked_embeddings = self.embed_tokens(mask_tokens)
        #         encoder_output_embedding = torch.where(encoder_masked_tokens.unsqueeze(-1).expand_as(encoder_rep), masked_embeddings, encoder_rep)
        #latent_logvar = self.logvar_proj(encoder_rep)
        
        return x_rot, rot_mat
    
    
    def enc(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        src_bond_type,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargs        
    ):
        encoder_src_tokens = src_tokens
        encoder_src_coord = src_coord
        encoder_src_distance = src_distance
        encoder_src_edge_type = src_edge_type

        encoder_padding_mask = encoder_src_tokens.eq(self.padding_idx)
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        x = self.embed_tokens(encoder_src_tokens) + self.embed_positions(~padding_mask)
        x = x + self.pos_embedder(encoder_src_coord)
        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias
        def get_bond_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf2(dist, et)
            gbf_result = self.gbf_proj2(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(encoder_src_distance, encoder_src_edge_type) #+ get_bond_features(encoder_src_distance, src_bond_type)

        (
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            encoder_x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0
        encoder_rep_quant = self.quant_mean(encoder_rep)
        encoder_output_embedding = torch.nn.Tanh()(encoder_rep_quant)
        
        # if self.feed_token_rep_to_decoder:
        #     encoder_output_embedding = encoder_rep
        # else:
        #     if encoder_masked_tokens is None:
        #         encoder_output_embedding = encoder_rep
        #     else:
        #         mask_tokens = torch.zeros_like(src_tokens).fill_(self.mask_idx)
        #         masked_embeddings = self.embed_tokens(mask_tokens)
        #         encoder_output_embedding = torch.where(encoder_masked_tokens.unsqueeze(-1).expand_as(encoder_rep), masked_embeddings, encoder_rep)
        latent_logvar = self.logvar_proj(encoder_rep)
        
        return encoder_output_embedding, latent_logvar, encoder_x_norm, delta_encoder_pair_rep_norm
    
    def dec(self, encoder_output_embedding, padding_mask, encoder_x_norm=None, delta_encoder_pair_rep_norm=None, classification_head_name=None):
        encoder_masked_tokens=None
        features_only=False
        #classification_head_name=None   
             
        #if not self.decoder_no_pe:
        #    encoder_output_embedding = encoder_output_embedding + self.embed_positions(~padding_mask)
        encoder_output_embedding = self.quant_expand(encoder_output_embedding) + self.embed_positions(~padding_mask)
        n_node = encoder_output_embedding.size(1)
        # if self.feed_pair_rep_to_decoder:
        #     assert self.decoder_noise is not True
        #     attn_bias = encoder_pair_rep.reshape(-1, n_node, n_node)
        # else:
        #     if not self.decoder_noise:
        #         bsz = padding_mask.size(0)
        #         attn_bias = encoder_output_embedding.new_zeros((bsz*self.encoder_attention_heads, n_node, n_node))
        #     else:
        #         attn_bias = get_dist_features(src_distance, src_edge_type)
        bsz = padding_mask.size(0)
        attn_bias = encoder_output_embedding.new_zeros((bsz*self.encoder_attention_heads, n_node, n_node))
        
        (
            decoder_rep,
            decoder_pair_rep,
            delta_decoder_pair_rep,
            decoder_x_norm,
            delta_decoder_pair_rep_norm,
        ) = self.decoder(encoder_output_embedding, padding_mask=padding_mask, attn_mask=attn_bias)


        (
            prop_embedding,
            _,
            _,
            _,
            _,
        ) = self.pretraining_property(encoder_output_embedding, padding_mask=padding_mask, attn_mask=attn_bias)

        prop_input = torch.sum(prop_embedding * ~padding_mask.unsqueeze(-1), 1) / torch.sum(~padding_mask, 1).unsqueeze(-1)
        prop_pred = self.prop_decoder(prop_input)

        decoder_pair_rep[decoder_pair_rep == float("-inf")] = 0

        encoder_distance = None
        encoder_coord = None

        if not features_only:
           #if self.args.masked_token_loss > 0:
            logits = self.lm_head(decoder_rep, ~padding_mask) 
            
            #if self.args.masked_coord_loss > 0:
            encoder_coord = self.pos_decoder(decoder_rep)
            encoder_pairwise_dist = torch.cdist(encoder_coord, encoder_coord, p=2)
            encoder_bond = self.gk_bond(encoder_pairwise_dist, logits)
            
            #if self.args.masked_dist_loss > 0:
            encoder_distance = self.dist_head(decoder_pair_rep)

        if classification_head_name is not None:
            finetuning_embedding = encoder_output_embedding
            # (
            #     finetuning_embedding,
            #     _,
            #     _,
            #     _,
            #     _,
            # ) = self.finetune_block(encoder_output_embedding.detach(), padding_mask=padding_mask, attn_mask=attn_bias)
            
            #finetune_input = torch.sum(finetuning_embedding * ~padding_mask.unsqueeze(-1), 1) / torch.sum(~padding_mask, 1).unsqueeze(-1)
            finetune_input = prop_input
            logits = self.classification_heads[classification_head_name](finetune_input)
        if self.args.mode == 'infer':
            return _, _
        else:
            return (
                logits,
                encoder_distance,
                (encoder_coord,encoder_bond),
                encoder_x_norm,
                decoder_x_norm,
                delta_encoder_pair_rep_norm,
                delta_decoder_pair_rep_norm,
                prop_pred
            )            
    def flow_training(
        self,
        model,
        z_1,
        mask
    ):
        """Vanilla Flow"""
        # dtype = torch.float32
        # t = torch.rand((z_1.size(0),), dtype=dtype, device=z_1.device)
        # t = t.view(-1,1,1)
        # z_0 = torch.randn_like(z_1, device=z_1.device)
        # #z_t = (1 - t) * z_0 + (1e-5 + (1 - 1e-5) * t) * z_1
        # z_t = (1 - t) * z_0 + t * z_1
        # #u = (1 - 1e-5) * z_1 - z_0
        # u = (1 - 1e-5) * z_1 - z_0
        # #v = model(z_t, t.squeeze(-1), mask=mask)
        # v = self.flow_infer(model, z_t, t.squeeze(-1), mask=mask)
        # loss = F.mse_loss(v, u, reduction='none')
        
        
        """ OT-CFM"""

        # FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.)
        # x1 = z_1
        # x0 = torch.randn_like(x1)
        # t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        # vt = self.flow_infer(model, xt, t.unsqueeze(-1), mask=mask)
        # loss = F.mse_loss(vt, ut, reduction='none')
        # return loss, vt
    
        """Protein Frame Flow"""
        self.interpolant.device = z_1.device
        dense_encoded_batch = {"x_1": z_1, "token_mask": mask, "diffuse_mask": mask}
        noisy_dense_encoded_batch = self.interpolant.corrupt_batch(dense_encoded_batch)

        #if self.self_condition:
        # Use self-conditioning for ~half training batches
        if (
            self.interpolant.self_condition
            and random.random() < self.interpolant.self_condition_prob
        ):
            
            with torch.no_grad():
                x_sc = model(noisy_dense_encoded_batch["x_t"],
                             noisy_dense_encoded_batch["t"],
                             mask=mask,
                             x_sc=None)
        else:
            x_sc = None
            
        pred_x = model(noisy_dense_encoded_batch["x_t"],
                        noisy_dense_encoded_batch["t"],
                        mask=mask,
                        x_sc=x_sc)

        gt_x_1 = noisy_dense_encoded_batch["x_1"]
        norm_scale = 1 - torch.min(noisy_dense_encoded_batch["t"].unsqueeze(-1), torch.tensor(0.9))
        x_error = (gt_x_1 - pred_x) / norm_scale
        loss_mask = (
            noisy_dense_encoded_batch["token_mask"] * noisy_dense_encoded_batch["diffuse_mask"]
        )
        loss_denom = torch.sum(loss_mask, dim=-1) * pred_x.size(-1)
        x_loss = torch.sum(x_error**2 * loss_mask[..., None], dim=(-1, -2)) / loss_denom
        #loss_dict = {"loss": x_loss.mean(), "x_loss": x_loss}
        loss_dict = {"loss": x_loss.mean()}
         
        num_bins = 4
        #flat_losses = x_loss.detach().cpu().numpy().flatten()
        #flat_losses = (gt_x_1 - pred_x)
        flat_losses = torch.sum((gt_x_1 - pred_x)**2 * loss_mask[..., None], dim=(-1, -2)) / loss_denom
        flat_losses = flat_losses.detach().cpu().numpy().flatten()
        flat_t = noisy_dense_encoded_batch["t"].detach().cpu().numpy().flatten()
        bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins + 1)
        bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
        t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
        t_binned_n = np.bincount(bin_idx)
        for t_bin in np.unique(bin_idx).tolist():
            bin_start = bin_edges[t_bin]
            bin_end = bin_edges[t_bin + 1]
            t_range = f"f_loss t=[{int(bin_start*100)},{int(bin_end*100)})"
            range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
            loss_dict[t_range] = range_loss
        loss_dict["t_avg"] = np.mean(flat_t)
        
        return loss_dict, pred_x
        
    def flow_infer(
        self,
        model,
        input,
        t,
        mask
    ):
        if self.self_condition:
            with torch.no_grad():
                x_sc = model(input, t, mask=mask, x_sc=None)
        else:
            x_sc = None
            
        v = model(input, t, mask=mask, x_sc=x_sc)  
        return v     
    
    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        src_bond_type,
        mode,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargs
    ):
        if mode == "ae_only":
            #with torch.no_grad():
            if self.random_order and self.index is None:
                self.Myenter()
                self.index = np.random.permutation(512)
                self.Myexit()
                
            padding_mask = src_tokens.eq(self.padding_idx)
            if classification_head_name is not None:
                features_only = True
                with torch.no_grad():
                    src_coord, rot_mat = self.rot(
                            src_tokens,
                            src_distance,
                            src_coord,
                            src_edge_type,
                            src_bond_type,
                            encoder_masked_tokens=None,
                            features_only=False,
                            classification_head_name=None,
                            **kwargs
                            )
            else:
                src_coord, rot_mat = self.rot(
                        src_tokens,
                        src_distance,
                        src_coord,
                        src_edge_type,
                        src_bond_type,
                        encoder_masked_tokens=None,
                        features_only=False,
                        classification_head_name=None,
                        **kwargs
                        )
                
            latent_emb, latent_logvar, encoder_x_norm, delta_encoder_pair_rep_norm = self.enc(
                    src_tokens,
                    src_distance,
                    src_coord,
                    src_edge_type,
                    src_bond_type,
                    encoder_masked_tokens=None,
                    features_only=False,
                    classification_head_name=None,
                    **kwargs
                    )
            LOG_STD_MAX = 15
            LOG_STD_MIN = -15 # 너무 작은 분산을 막기 위함
            latent_logvar = torch.clamp(latent_logvar, LOG_STD_MIN, LOG_STD_MAX)
            
            std = torch.exp(0.5 * latent_logvar)
            q_z_given_x = torch.distributions.Normal(loc=latent_emb, scale=std)
            z = q_z_given_x.rsample()
            p_z = torch.distributions.Normal(
                loc=torch.zeros_like(latent_emb), 
                scale=torch.ones_like(std)
            )
        elif mode == "flow_only":
            #self.encoder.eval()
            with torch.no_grad():
                if self.random_order and self.index is None:
                    self.Myenter()
                    self.index = np.random.permutation(512)
                    self.Myexit()

                if classification_head_name is not None:
                    features_only = True
        
                padding_mask = src_tokens.eq(self.padding_idx)
                src_coord,rot_mat = self.rot(
                        src_tokens,
                        src_distance,
                        src_coord,
                        src_edge_type,
                        src_bond_type,
                        encoder_masked_tokens=None,
                        features_only=False,
                        classification_head_name=None,
                        **kwargs
                        )
                latent_emb, latent_logvar, encoder_x_norm, delta_encoder_pair_rep_norm = self.enc(
                        src_tokens,
                        src_distance,
                        src_coord,
                        src_edge_type,
                        src_bond_type,
                        encoder_masked_tokens=None,
                        features_only=False,
                        classification_head_name=None,
                        **kwargs
                        )
                LOG_STD_MAX = 15
                LOG_STD_MIN = -15 # 너무 작은 분산을 막기 위함
                latent_logvar = torch.clamp(latent_logvar, LOG_STD_MIN, LOG_STD_MAX)
                
                std = torch.exp(0.5 * latent_logvar)
                q_z_given_x = torch.distributions.Normal(loc=latent_emb, scale=std)
                z = q_z_given_x.rsample()
                p_z = torch.distributions.Normal(
                    loc=torch.zeros_like(latent_emb), 
                    scale=torch.ones_like(std)
                )
        
        
        if mode == "ae_only":
            with torch.no_grad():
                flow_loss_dict, _ = self.flow_training(self.backbone, z.detach(), ~padding_mask)
            (
                logits,
                encoder_distance,
                encoder_coord,
                encoder_x_norm,
                decoder_x_norm,
                delta_encoder_pair_rep_norm,
                delta_decoder_pair_rep_norm,
                prop_pred
                
            ) = self.dec(
                z,
                padding_mask,
                encoder_x_norm,
                delta_encoder_pair_rep_norm,
                classification_head_name
            )
        elif mode == "flow_only":
            flow_loss_dict, _ = self.flow_training(self.backbone, z.detach(), ~padding_mask)
            with torch.no_grad():
                (
                    logits,
                    encoder_distance,
                    encoder_coord,
                    encoder_x_norm,
                    decoder_x_norm,
                    delta_encoder_pair_rep_norm,
                    delta_decoder_pair_rep_norm,
                    prop_pred
                ) = self.dec(
                    z,
                    padding_mask,
                    encoder_x_norm,
                    delta_encoder_pair_rep_norm
                )

        if mode == "dual":
            if self.random_order and self.index is None:
                self.Myenter()
                self.index = np.random.permutation(512)
                self.Myexit()

            if classification_head_name is not None:
                features_only = True
    
            padding_mask = src_tokens.eq(self.padding_idx)
            src_coord,rot_mat = self.rot(
                    src_tokens,
                    src_distance,
                    src_coord,
                    src_edge_type,
                    src_bond_type,
                    encoder_masked_tokens=None,
                    features_only=False,
                    classification_head_name=None,
                    **kwargs
                    )
            
            latent_emb, latent_logvar, encoder_x_norm, delta_encoder_pair_rep_norm = self.enc(
                    src_tokens,
                    src_distance,
                    src_coord,
                    src_edge_type,
                    src_bond_type,
                    encoder_masked_tokens=None,
                    features_only=False,
                    classification_head_name=None,
                    **kwargs
                    )
            LOG_STD_MAX = 15
            LOG_STD_MIN = -15 # 너무 작은 분산을 막기 위함
            latent_logvar = torch.clamp(latent_logvar, LOG_STD_MIN, LOG_STD_MAX)
            
            std = torch.exp(0.5 * latent_logvar)
            q_z_given_x = torch.distributions.Normal(loc=latent_emb, scale=std)
            z = q_z_given_x.rsample()
            p_z = torch.distributions.Normal(
                loc=torch.zeros_like(latent_emb), 
                scale=torch.ones_like(std)
            )    
            
            flow_loss_dict, _ = self.flow_training(self.backbone, z.detach(), ~padding_mask)    
            (
                logits,
                encoder_distance,
                encoder_coord,
                encoder_x_norm,
                decoder_x_norm,
                delta_encoder_pair_rep_norm,
                delta_decoder_pair_rep_norm,
                prop_pred
            ) = self.dec(
                z,
                padding_mask,
                encoder_x_norm,
                delta_encoder_pair_rep_norm
            )
        
            
        return (
            logits,
            encoder_distance,
            (rot_mat, encoder_coord),
            encoder_x_norm,
            decoder_x_norm,
            delta_encoder_pair_rep_norm,
            delta_decoder_pair_rep_norm,
            (z, q_z_given_x, p_z,latent_emb,std),
            flow_loss_dict,
            prop_pred
        ) 
        
    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )
        
    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates
        
@register_model("unimol_Optimal_padding_Dual")
class UniMolOptimalPaddingModelDual(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="L", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="A",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--emb-dropout",
            type=float,
            metavar="D",
            help="dropout probability for embeddings",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-seq-len", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--post-ln", type=bool, help="use post layernorm or pre layernorm"
        )
        parser.add_argument(
            "--masked-token-loss",
            type=float,
            metavar="D",
            help="mask loss ratio",
        )
        parser.add_argument(
            "--masked-dist-loss",
            type=float,
            metavar="D",
            help="masked distance loss ratio",
        )
        parser.add_argument(
            "--masked-coord-loss",
            type=float,
            metavar="D",
            help="masked coord loss ratio",
        )
        parser.add_argument(
            "--encoder-x-norm-loss",
            type=float,
            metavar="D",
            help="x norm loss ratio",
        )
        parser.add_argument(
            "--encoder-delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio",
        )
        parser.add_argument(
            "--decoder-x-norm-loss",
            type=float,
            metavar="D",
            help="x norm loss ratio",
        )
        parser.add_argument(
            "--decoder-delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio",
        )
        parser.add_argument(
            "--masked-coord-dist-loss",
            type=float,
            metavar="D",
            help="masked coord dist loss ratio",
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="train",
            choices=["train", "infer"],
        )
        parser.add_argument(
            "--encoder-unmasked-tokens-only", action='store_true', help="only input unmasked tokens into encoder"
        )
        parser.add_argument(
            "--encoder-masked-3d-pe", action='store_true', help="only masked #D PE for encoder"
        )
        parser.add_argument(
            "--encoder-apply-pe", action='store_true', help="apply PE for encoder"
        )
        parser.add_argument(
            "--feed-pair-rep-to-decoder", action='store_true', help="feed the pair representations of encoder to decoder"
        )
        parser.add_argument(
            "--decoder-no-pe", action='store_true', help="Don't apply PE for decoder"
        )
        parser.add_argument(
            "--feed-token-rep-to-decoder", action='store_true', help="feed the token representations of encoder to decoder"
        )
        parser.add_argument(
            "--decoder-noise", action='store_true', help="Feed noise or [mask] to decoder"
        )
        parser.add_argument(
            "--random-order", action='store_true', help="Feed noise or [mask] to decoder"
        )

    def __init__(self, args, dictionary):
        super().__init__()
        #print('Using modified MAE')
        base_architecture(args)
        self.args = args
        self.encoder_masked_3d_pe = args.encoder_masked_3d_pe
        self.encoder_apply_pe = args.encoder_apply_pe
        self.feed_pair_rep_to_decoder = args.feed_pair_rep_to_decoder
        self.decoder_no_pe = args.decoder_no_pe
        self.feed_token_rep_to_decoder = args.feed_token_rep_to_decoder
        self.decoder_noise = args.decoder_noise
        self.random_order = args.random_order
        self.interpolant = FlowMatchingInterpolant(device="cpu")
        #self.interpolant = None

        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(dictionary), args.encoder_embed_dim, self.padding_idx
        )

        self.PE = None
        self.index = None
        if self.random_order:
            self.init_state()
        # Original TransformerWrapper backbone (commented for DiT replacement)
        # self.backbone = TransformerWrapper(
        #     attn_layers=Encoder(dim=args.encoder_embed_dim, depth=10),
        #     emb_dropout=0.0)
        
        # Use DiT model as backbone instead of TransformerWrapper
        self.backbone = DiT(
            # d_x=args.encoder_embed_dim,
            # d_model=args.encoder_embed_dim,
            # nhead=args.encoder_attention_heads,
            d_x=8,
            d_model=512,
            nhead=8,
        )
        self._num_updates = None
        self.encoder_rot = TransformerEncoderWithPair(
            encoder_layers=3,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.encoder_delta_pair_repr_norm_loss < 0,
        )
        
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.encoder_delta_pair_repr_norm_loss < 0,
        )
        self.decoder = TransformerEncoderWithPair(
            #encoder_layers=args.decoder_layers,
            encoder_layers=3,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.decoder_ffn_embed_dim,
            attention_heads=args.decoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.decoder_delta_pair_repr_norm_loss < 0,
        )
        
        self.pretraining_property = TransformerEncoderWithPair(
            #encoder_layers=args.decoder_layers,
            encoder_layers=3,
            embed_dim=self.args.encoder_embed_dim,
            ffn_embed_dim=self.args.decoder_ffn_embed_dim,
            attention_heads=self.args.decoder_attention_heads,
            emb_dropout=self.args.emb_dropout,
            dropout=self.args.dropout,
            attention_dropout=self.args.attention_dropout,
            activation_dropout=self.args.activation_dropout,
            max_seq_len=self.args.max_seq_len,
            activation_fn=self.args.activation_fn,
            no_final_head_layer_norm=self.args.decoder_delta_pair_repr_norm_loss < 0,
        )
        
        
        #if args.masked_token_loss > 0:
        self.lm_head = MaskLMHead(
            embed_dim=args.encoder_embed_dim,
            output_dim=len(dictionary),
            activation_fn=args.activation_fn,
            weight=None,
        )
            
        # self.logvar_proj = NonLinearHead(
        #     args.encoder_embed_dim, args.encoder_embed_dim, 
        #     args.activation_fn
        # )
        self.quant_mean = NonLinearHead(
            input_dim = args.encoder_embed_dim, 
            out_dim = 8, 
            activation_fn = args.activation_fn, 
            hidden = args.encoder_embed_dim//2
        )
               
        self.quant_rot = NonLinearHead(
            input_dim = args.encoder_embed_dim, 
            out_dim = 9, 
            activation_fn = args.activation_fn, 
            hidden = args.encoder_embed_dim//2
        )
        self.logvar_proj = NonLinearHead(
            input_dim = args.encoder_embed_dim, 
            out_dim = 8, 
            activation_fn = args.activation_fn, 
            hidden = args.encoder_embed_dim//2
        )
        self.quant_expand = NonLinearHead(
            input_dim = 8, 
            out_dim = args.encoder_embed_dim, 
            activation_fn = args.activation_fn, 
            hidden = args.encoder_embed_dim//2
        )
        
        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(
            K, args.encoder_attention_heads, args.activation_fn
        )
        self.gbf_proj2 = NonLinearHead(
            K, args.encoder_attention_heads, args.activation_fn
        )
        self.gbf = GaussianLayer(K, n_edge_type)
        self.gbf2 = GaussianLayer(K, 5)
        
        #if args.masked_coord_loss > 0:
        self.pair2coord_proj = NonLinearHead(
            args.decoder_attention_heads, 1, args.activation_fn
        )
        #if args.masked_dist_loss > 0:
        self.dist_head = DistanceHead(
            args.decoder_attention_heads, args.activation_fn
        )
        self.init_pos_prog = NonLinearHeadPos(
            512, 1, args.activation_fn
        )
        self.pos_embedder = nn.Sequential(
            nn.Linear(3, args.encoder_embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim),
        )
        self.pos_decoder = nn.Sequential(
            nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(args.encoder_embed_dim, 3),
        )
        self.classification_heads = nn.ModuleDict()
        self.apply(init_bert_params)
        self.encoder_unmasked_tokens_only = args.encoder_unmasked_tokens_only
        self.dictionary = dictionary
        self.prop_decoder = nn.Sequential(
            nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(args.encoder_embed_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 38),
        )

        self.embed_positions = SinusoidalPositionalEmbedding(
            embedding_dim = args.max_seq_len,
            padding_idx = dictionary.pad(),
            init_size = args.max_seq_len,
        )

        self.mask_idx = dictionary.index("[MASK]")
        self.encoder_attention_heads = args.encoder_attention_heads
        self.self_condition = True
        self.gk_bond = BondGaussianLayer()
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)
    
    @classmethod
    def init_state(self):
        original_state = np.random.get_state()
        np.random.seed(0)
        self.numpy_random_state = np.random.get_state()
        np.random.set_state(original_state)
    
    @classmethod
    def Myenter(self):
        self.original_state = np.random.get_state()
        np.random.set_state(self.numpy_random_state)

    @classmethod
    def Myexit(self):
        self.numpy_random_state = np.random.get_state()
        np.random.set_state(self.original_state)

    def rot(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        src_bond_type,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargs        
    ):
        encoder_src_tokens = src_tokens
        encoder_src_coord = src_coord
        encoder_src_distance = src_distance
        encoder_src_edge_type = src_edge_type

        encoder_padding_mask = encoder_src_tokens.eq(self.padding_idx)
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        x = self.embed_tokens(encoder_src_tokens) + self.embed_positions(~padding_mask)
        x = x + self.pos_embedder(encoder_src_coord)
        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias
        def get_bond_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf2(dist, et)
            gbf_result = self.gbf_proj2(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(encoder_src_distance, encoder_src_edge_type) #+ get_bond_features(encoder_src_distance, src_bond_type)

        (
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            encoder_x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder_rot(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0
        
        rot = torch.sum(encoder_rep * ~padding_mask.unsqueeze(-1), 1) / torch.sum(~padding_mask, 1).unsqueeze(-1)
        encoder_rot = self.quant_rot(rot).reshape(-1, 3, 3)
        rot_mat = roma.special_procrustes(encoder_rot)
        x_rot = torch.bmm(encoder_src_coord, rot_mat)
        #encoder_output_embedding = torch.nn.Tanh()(encoder_rep_quant)
        
        # if self.feed_token_rep_to_decoder:
        #     encoder_output_embedding = encoder_rep
        # else:
        #     if encoder_masked_tokens is None:
        #         encoder_output_embedding = encoder_rep
        #     else:
        #         mask_tokens = torch.zeros_like(src_tokens).fill_(self.mask_idx)
        #         masked_embeddings = self.embed_tokens(mask_tokens)
        #         encoder_output_embedding = torch.where(encoder_masked_tokens.unsqueeze(-1).expand_as(encoder_rep), masked_embeddings, encoder_rep)
        #latent_logvar = self.logvar_proj(encoder_rep)
        
        return x_rot, rot_mat
    
    
    def enc(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        src_bond_type,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargs        
    ):
        encoder_src_tokens = src_tokens
        encoder_src_coord = src_coord
        encoder_src_distance = src_distance
        encoder_src_edge_type = src_edge_type

        encoder_padding_mask = encoder_src_tokens.eq(self.padding_idx)
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        x = self.embed_tokens(encoder_src_tokens) + self.embed_positions(~padding_mask)
        x = x + self.pos_embedder(encoder_src_coord)
        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias
        def get_bond_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf2(dist, et)
            gbf_result = self.gbf_proj2(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(encoder_src_distance, encoder_src_edge_type) #+ get_bond_features(encoder_src_distance, src_bond_type)

        (
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            encoder_x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0
        encoder_rep_quant = self.quant_mean(encoder_rep)
        encoder_output_embedding = torch.nn.Tanh()(encoder_rep_quant)
        
        # if self.feed_token_rep_to_decoder:
        #     encoder_output_embedding = encoder_rep
        # else:
        #     if encoder_masked_tokens is None:
        #         encoder_output_embedding = encoder_rep
        #     else:
        #         mask_tokens = torch.zeros_like(src_tokens).fill_(self.mask_idx)
        #         masked_embeddings = self.embed_tokens(mask_tokens)
        #         encoder_output_embedding = torch.where(encoder_masked_tokens.unsqueeze(-1).expand_as(encoder_rep), masked_embeddings, encoder_rep)
        latent_logvar = self.logvar_proj(encoder_rep)

        return encoder_output_embedding, latent_logvar, encoder_x_norm, delta_encoder_pair_rep_norm

    def dec(self, encoder_output_embedding, padding_mask, encoder_x_norm=None, delta_encoder_pair_rep_norm=None, classification_head_name=None):
        encoder_masked_tokens=None
        features_only=False
        #classification_head_name=None   
             
        #if not self.decoder_no_pe:
        #    encoder_output_embedding = encoder_output_embedding + self.embed_positions(~padding_mask)
        encoder_output_embedding = self.quant_expand(encoder_output_embedding) + self.embed_positions(~padding_mask)
        n_node = encoder_output_embedding.size(1)
        # if self.feed_pair_rep_to_decoder:
        #     assert self.decoder_noise is not True
        #     attn_bias = encoder_pair_rep.reshape(-1, n_node, n_node)
        # else:
        #     if not self.decoder_noise:
        #         bsz = padding_mask.size(0)
        #         attn_bias = encoder_output_embedding.new_zeros((bsz*self.encoder_attention_heads, n_node, n_node))
        #     else:
        #         attn_bias = get_dist_features(src_distance, src_edge_type)
        bsz = padding_mask.size(0)
        attn_bias = encoder_output_embedding.new_zeros((bsz*self.encoder_attention_heads, n_node, n_node))
        
        (
            decoder_rep,
            decoder_pair_rep,
            delta_decoder_pair_rep,
            decoder_x_norm,
            delta_decoder_pair_rep_norm,
        ) = self.decoder(encoder_output_embedding, padding_mask=padding_mask, attn_mask=attn_bias)
        decoder_pair_rep[decoder_pair_rep == float("-inf")] = 0

        (
            prop_embedding,
            _,
            _,
            _,
            _,
        ) = self.pretraining_property(encoder_output_embedding, padding_mask=padding_mask, attn_mask=attn_bias)

        prop_input = torch.sum(prop_embedding * ~padding_mask.unsqueeze(-1), 1) / torch.sum(~padding_mask, 1).unsqueeze(-1)
        prop_pred = self.prop_decoder(prop_input)

        decoder_pair_rep[decoder_pair_rep == float("-inf")] = 0

        encoder_distance = None
        encoder_coord = None

        def coord_refine(encoder_coord, padding_mask, delta_decoder_pair_rep): 
            #coords_emb = encoder_coord
            """ Dual SE-Equivariant """
            coords_emb = encoder_coord
            if padding_mask is not None:
                atom_num = (~padding_mask).sum(1).view(-1, 1, 1, 1)
            else:
                atom_num = coords_emb.shape[1]
            delta_pos = coords_emb.unsqueeze(1) - coords_emb.unsqueeze(2)
            attn_probs = self.pair2coord_proj(delta_decoder_pair_rep)
            coord_update = delta_pos / atom_num * attn_probs
            coord_update = torch.sum(coord_update, dim=2)
            encoder_coord_refine = coords_emb + coord_update
            
            return encoder_coord_refine            
        if not features_only:
           #if self.args.masked_token_loss > 0:
            logits = self.lm_head(decoder_rep, ~padding_mask) 
            
            #if self.args.masked_coord_loss > 0:
            encoder_coord = self.pos_decoder(decoder_rep)
            encoder_coord_refine = encoder_coord
            with data_utils.numpy_seed(self.get_num_updates()):
                recycling = np.random.randint(3) + 1
                #encoder_coord_refine = coord_refine(encoder_coord, padding_mask, delta_decoder_pair_rep)
                for i in range(recycling):
                    with torch.no_grad():
                        encoder_coord_refine = coord_refine(encoder_coord_refine, padding_mask, delta_decoder_pair_rep)
                if np.random.randint(2) == 0:        
                    encoder_coord_refine = coord_refine(encoder_coord_refine, padding_mask, delta_decoder_pair_rep)
                else:
                    encoder_coord_refine = coord_refine(encoder_coord_refine.detach(), padding_mask, delta_decoder_pair_rep)
            """"""

            encoder_pairwise_dist = torch.cdist(encoder_coord, encoder_coord, p=2)
            encoder_bond = self.gk_bond(encoder_pairwise_dist, logits)
            
            #if self.args.masked_dist_loss > 0:
            encoder_distance = self.dist_head(decoder_pair_rep)

        if classification_head_name is not None:
            finetuning_embedding = encoder_output_embedding
            # (
            #     finetuning_embedding,
            #     _,
            #     _,
            #     _,
            #     _,
            # ) = self.finetune_block(encoder_output_embedding.detach(), padding_mask=padding_mask, attn_mask=attn_bias)
            
            #finetune_input = torch.sum(finetuning_embedding * ~padding_mask.unsqueeze(-1), 1) / torch.sum(~padding_mask, 1).unsqueeze(-1)
            finetune_input = prop_input
            logits = self.classification_heads[classification_head_name](finetune_input)
        if self.args.mode == 'infer':
            return _, _
        else:
            return (
                logits,
                encoder_distance,
                (encoder_coord_refine, encoder_bond,encoder_coord),
                encoder_x_norm,
                decoder_x_norm,
                delta_encoder_pair_rep_norm,
                delta_decoder_pair_rep_norm,
                prop_pred
            )            
    def flow_training(
        self,
        model,
        z_1,
        mask
    ):
        """Vanilla Flow"""
        # dtype = torch.float32
        # t = torch.rand((z_1.size(0),), dtype=dtype, device=z_1.device)
        # t = t.view(-1,1,1)
        # z_0 = torch.randn_like(z_1, device=z_1.device)
        # #z_t = (1 - t) * z_0 + (1e-5 + (1 - 1e-5) * t) * z_1
        # z_t = (1 - t) * z_0 + t * z_1
        # #u = (1 - 1e-5) * z_1 - z_0
        # u = (1 - 1e-5) * z_1 - z_0
        # #v = model(z_t, t.squeeze(-1), mask=mask)
        # v = self.flow_infer(model, z_t, t.squeeze(-1), mask=mask)
        # loss = F.mse_loss(v, u, reduction='none')
        
        
        """ OT-CFM"""

        # FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.)
        # x1 = z_1
        # x0 = torch.randn_like(x1)
        # t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        # vt = self.flow_infer(model, xt, t.unsqueeze(-1), mask=mask)
        # loss = F.mse_loss(vt, ut, reduction='none')
        # return loss, vt
    
        """Protein Frame Flow"""
        self.interpolant.device = z_1.device
        dense_encoded_batch = {"x_1": z_1, "token_mask": mask, "diffuse_mask": mask}
        noisy_dense_encoded_batch = self.interpolant.corrupt_batch(dense_encoded_batch)

        #if self.self_condition:
        # Use self-conditioning for ~half training batches
        if (
            self.interpolant.self_condition
            and random.random() < self.interpolant.self_condition_prob
        ):
            
            with torch.no_grad():
                x_sc = model(noisy_dense_encoded_batch["x_t"],
                             noisy_dense_encoded_batch["t"],
                             mask=mask,
                             x_sc=None)
        else:
            x_sc = None
            
        pred_x = model(noisy_dense_encoded_batch["x_t"],
                        noisy_dense_encoded_batch["t"],
                        mask=mask,
                        x_sc=x_sc)

        gt_x_1 = noisy_dense_encoded_batch["x_1"]
        norm_scale = 1 - torch.min(noisy_dense_encoded_batch["t"].unsqueeze(-1), torch.tensor(0.9))
        x_error = (gt_x_1 - pred_x) / norm_scale
        loss_mask = (
            noisy_dense_encoded_batch["token_mask"] * noisy_dense_encoded_batch["diffuse_mask"]
        )
        loss_denom = torch.sum(loss_mask, dim=-1) * pred_x.size(-1)
        x_loss = torch.sum(x_error**2 * loss_mask[..., None], dim=(-1, -2)) / loss_denom
        #loss_dict = {"loss": x_loss.mean(), "x_loss": x_loss}
        loss_dict = {"loss": x_loss.mean()}
         
        num_bins = 4
        #flat_losses = x_loss.detach().cpu().numpy().flatten()
        #flat_losses = (gt_x_1 - pred_x)
        flat_losses = torch.sum((gt_x_1 - pred_x)**2 * loss_mask[..., None], dim=(-1, -2)) / loss_denom
        flat_losses = flat_losses.detach().cpu().numpy().flatten()
        flat_t = noisy_dense_encoded_batch["t"].detach().cpu().numpy().flatten()
        bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins + 1)
        bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
        t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
        t_binned_n = np.bincount(bin_idx)
        for t_bin in np.unique(bin_idx).tolist():
            bin_start = bin_edges[t_bin]
            bin_end = bin_edges[t_bin + 1]
            t_range = f"f_loss t=[{int(bin_start*100)},{int(bin_end*100)})"
            range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
            loss_dict[t_range] = range_loss
        loss_dict["t_avg"] = np.mean(flat_t)
        
        return loss_dict, pred_x
        
    def flow_infer(
        self,
        model,
        input,
        t,
        mask
    ):
        if self.self_condition:
            with torch.no_grad():
                x_sc = model(input, t, mask=mask, x_sc=None)
        else:
            x_sc = None
            
        v = model(input, t, mask=mask, x_sc=x_sc)  
        return v     
    
    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        src_bond_type,
        mode,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargs
    ):
        if mode == "ae_only":
            #with torch.no_grad():
            if self.random_order and self.index is None:
                self.Myenter()
                self.index = np.random.permutation(512)
                self.Myexit()
                
            padding_mask = src_tokens.eq(self.padding_idx)
            if classification_head_name is not None:
                features_only = True
                with torch.no_grad():
                    src_coord, rot_mat = self.rot(
                            src_tokens,
                            src_distance,
                            src_coord,
                            src_edge_type,
                            src_bond_type,
                            encoder_masked_tokens=None,
                            features_only=False,
                            classification_head_name=None,
                            **kwargs
                            )
            else:
                src_coord, rot_mat = self.rot(
                        src_tokens,
                        src_distance,
                        src_coord,
                        src_edge_type,
                        src_bond_type,
                        encoder_masked_tokens=None,
                        features_only=False,
                        classification_head_name=None,
                        **kwargs
                        )
                
            latent_emb, latent_logvar, encoder_x_norm, delta_encoder_pair_rep_norm = self.enc(
                    src_tokens,
                    src_distance,
                    src_coord,
                    src_edge_type,
                    src_bond_type,
                    encoder_masked_tokens=None,
                    features_only=False,
                    classification_head_name=None,
                    **kwargs
                    )
            LOG_STD_MAX = 15
            LOG_STD_MIN = -15 # 너무 작은 분산을 막기 위함
            latent_logvar = torch.clamp(latent_logvar, LOG_STD_MIN, LOG_STD_MAX)
            
            std = torch.exp(0.5 * latent_logvar)
            q_z_given_x = torch.distributions.Normal(loc=latent_emb, scale=std)
            z = q_z_given_x.rsample()
            p_z = torch.distributions.Normal(
                loc=torch.zeros_like(latent_emb), 
                scale=torch.ones_like(std)
            )
        elif mode == "flow_only":
            #self.encoder.eval()
            with torch.no_grad():
                if self.random_order and self.index is None:
                    self.Myenter()
                    self.index = np.random.permutation(512)
                    self.Myexit()

                if classification_head_name is not None:
                    features_only = True
        
                padding_mask = src_tokens.eq(self.padding_idx)
                src_coord,rot_mat = self.rot(
                        src_tokens,
                        src_distance,
                        src_coord,
                        src_edge_type,
                        src_bond_type,
                        encoder_masked_tokens=None,
                        features_only=False,
                        classification_head_name=None,
                        **kwargs
                        )
                latent_emb, latent_logvar, encoder_x_norm, delta_encoder_pair_rep_norm = self.enc(
                        src_tokens,
                        src_distance,
                        src_coord,
                        src_edge_type,
                        src_bond_type,
                        encoder_masked_tokens=None,
                        features_only=False,
                        classification_head_name=None,
                        **kwargs
                        )
                LOG_STD_MAX = 15
                LOG_STD_MIN = -15 # 너무 작은 분산을 막기 위함
                latent_logvar = torch.clamp(latent_logvar, LOG_STD_MIN, LOG_STD_MAX)
                
                std = torch.exp(0.5 * latent_logvar)
                q_z_given_x = torch.distributions.Normal(loc=latent_emb, scale=std)
                z = q_z_given_x.rsample()
                p_z = torch.distributions.Normal(
                    loc=torch.zeros_like(latent_emb), 
                    scale=torch.ones_like(std)
                )
        
        
        if mode == "ae_only":
            with torch.no_grad():
                flow_loss_dict, _ = self.flow_training(self.backbone, z.detach(), ~padding_mask)
            (
                logits,
                encoder_distance,
                encoder_coord,
                encoder_x_norm,
                decoder_x_norm,
                delta_encoder_pair_rep_norm,
                delta_decoder_pair_rep_norm,
                prop_pred
                
            ) = self.dec(
                z,
                padding_mask,
                encoder_x_norm,
                delta_encoder_pair_rep_norm,
                classification_head_name
            )
        elif mode == "flow_only":
            flow_loss_dict, _ = self.flow_training(self.backbone, z.detach(), ~padding_mask)
            with torch.no_grad():
                (
                    logits,
                    encoder_distance,
                    encoder_coord,
                    encoder_x_norm,
                    decoder_x_norm,
                    delta_encoder_pair_rep_norm,
                    delta_decoder_pair_rep_norm,
                    prop_pred
                ) = self.dec(
                    z,
                    padding_mask,
                    encoder_x_norm,
                    delta_encoder_pair_rep_norm
                )

        if mode == "dual":
            if self.random_order and self.index is None:
                self.Myenter()
                self.index = np.random.permutation(512)
                self.Myexit()

            if classification_head_name is not None:
                features_only = True
    
            padding_mask = src_tokens.eq(self.padding_idx)
            src_coord,rot_mat = self.rot(
                    src_tokens,
                    src_distance,
                    src_coord,
                    src_edge_type,
                    src_bond_type,
                    encoder_masked_tokens=None,
                    features_only=False,
                    classification_head_name=None,
                    **kwargs
                    )
            
            latent_emb, latent_logvar, encoder_x_norm, delta_encoder_pair_rep_norm = self.enc(
                    src_tokens,
                    src_distance,
                    src_coord,
                    src_edge_type,
                    src_bond_type,
                    encoder_masked_tokens=None,
                    features_only=False,
                    classification_head_name=None,
                    **kwargs
                    )
            LOG_STD_MAX = 15
            LOG_STD_MIN = -15 # 너무 작은 분산을 막기 위함
            latent_logvar = torch.clamp(latent_logvar, LOG_STD_MIN, LOG_STD_MAX)
            
            std = torch.exp(0.5 * latent_logvar)
            q_z_given_x = torch.distributions.Normal(loc=latent_emb, scale=std)
            z = q_z_given_x.rsample()
            p_z = torch.distributions.Normal(
                loc=torch.zeros_like(latent_emb), 
                scale=torch.ones_like(std)
            )    
            
            flow_loss_dict, _ = self.flow_training(self.backbone, z.detach(), ~padding_mask)    
            (
                logits,
                encoder_distance,
                encoder_coord,
                encoder_x_norm,
                decoder_x_norm,
                delta_encoder_pair_rep_norm,
                delta_decoder_pair_rep_norm,
                prop_pred
            ) = self.dec(
                z,
                padding_mask,
                encoder_x_norm,
                delta_encoder_pair_rep_norm
            )
        
            
        return (
            logits,
            encoder_distance,
            (rot_mat, encoder_coord),
            encoder_x_norm,
            decoder_x_norm,
            delta_encoder_pair_rep_norm,
            delta_decoder_pair_rep_norm,
            (z, q_z_given_x, p_z,latent_emb,std),
            flow_loss_dict,
            prop_pred
        ) 
        
    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )
        
    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates

        
@register_model("unimol_Optimal_padding_Type2")
class UniMolOptimalPaddingModelType2(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="L", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="A",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--emb-dropout",
            type=float,
            metavar="D",
            help="dropout probability for embeddings",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-seq-len", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--post-ln", type=bool, help="use post layernorm or pre layernorm"
        )
        parser.add_argument(
            "--masked-token-loss",
            type=float,
            metavar="D",
            help="mask loss ratio",
        )
        parser.add_argument(
            "--masked-dist-loss",
            type=float,
            metavar="D",
            help="masked distance loss ratio",
        )
        parser.add_argument(
            "--masked-coord-loss",
            type=float,
            metavar="D",
            help="masked coord loss ratio",
        )
        parser.add_argument(
            "--encoder-x-norm-loss",
            type=float,
            metavar="D",
            help="x norm loss ratio",
        )
        parser.add_argument(
            "--encoder-delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio",
        )
        parser.add_argument(
            "--decoder-x-norm-loss",
            type=float,
            metavar="D",
            help="x norm loss ratio",
        )
        parser.add_argument(
            "--decoder-delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio",
        )
        parser.add_argument(
            "--masked-coord-dist-loss",
            type=float,
            metavar="D",
            help="masked coord dist loss ratio",
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="train",
            choices=["train", "infer"],
        )
        parser.add_argument(
            "--encoder-unmasked-tokens-only", action='store_true', help="only input unmasked tokens into encoder"
        )
        parser.add_argument(
            "--encoder-masked-3d-pe", action='store_true', help="only masked #D PE for encoder"
        )
        parser.add_argument(
            "--encoder-apply-pe", action='store_true', help="apply PE for encoder"
        )
        parser.add_argument(
            "--feed-pair-rep-to-decoder", action='store_true', help="feed the pair representations of encoder to decoder"
        )
        parser.add_argument(
            "--decoder-no-pe", action='store_true', help="Don't apply PE for decoder"
        )
        parser.add_argument(
            "--feed-token-rep-to-decoder", action='store_true', help="feed the token representations of encoder to decoder"
        )
        parser.add_argument(
            "--decoder-noise", action='store_true', help="Feed noise or [mask] to decoder"
        )
        parser.add_argument(
            "--random-order", action='store_true', help="Feed noise or [mask] to decoder"
        )

    def __init__(self, args, dictionary):
        super().__init__()
        #print('Using modified MAE')
        base_architecture(args)
        self.args = args
        self.encoder_masked_3d_pe = args.encoder_masked_3d_pe
        self.encoder_apply_pe = args.encoder_apply_pe
        self.feed_pair_rep_to_decoder = args.feed_pair_rep_to_decoder
        self.decoder_no_pe = args.decoder_no_pe
        self.feed_token_rep_to_decoder = args.feed_token_rep_to_decoder
        self.decoder_noise = args.decoder_noise
        self.random_order = args.random_order
        self.interpolant = FlowMatchingInterpolant(device="cpu")
        #self.interpolant = None

        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(dictionary), args.encoder_embed_dim, self.padding_idx
        )

        self.PE = None
        self.index = None
        if self.random_order:
            self.init_state()
        # Original TransformerWrapper backbone (commented for DiT replacement)
        # self.backbone = TransformerWrapper(
        #     attn_layers=Encoder(dim=args.encoder_embed_dim, depth=10),
        #     emb_dropout=0.0)
        
        # Use DiT model as backbone instead of TransformerWrapper
        self.backbone = DiT(
            # d_x=args.encoder_embed_dim,
            # d_model=args.encoder_embed_dim,
            # nhead=args.encoder_attention_heads,
            d_x=8,
            d_model=512,
            nhead=8,
        )
        self._num_updates = None
        self.encoder_rot = TransformerEncoderWithPair(
            encoder_layers=3,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.encoder_delta_pair_repr_norm_loss < 0,
        )
        
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.encoder_delta_pair_repr_norm_loss < 0,
        )
        self.decoder = TransformerEncoderWithPair(
            #encoder_layers=args.decoder_layers,
            encoder_layers=3,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.decoder_ffn_embed_dim,
            attention_heads=args.decoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.decoder_delta_pair_repr_norm_loss < 0,
        )
        
        self.pretraining_property = TransformerEncoderWithPair(
            #encoder_layers=args.decoder_layers,
            encoder_layers=3,
            embed_dim=self.args.encoder_embed_dim,
            ffn_embed_dim=self.args.decoder_ffn_embed_dim,
            attention_heads=self.args.decoder_attention_heads,
            emb_dropout=self.args.emb_dropout,
            dropout=self.args.dropout,
            attention_dropout=self.args.attention_dropout,
            activation_dropout=self.args.activation_dropout,
            max_seq_len=self.args.max_seq_len,
            activation_fn=self.args.activation_fn,
            no_final_head_layer_norm=self.args.decoder_delta_pair_repr_norm_loss < 0,
        )
        
        
        #if args.masked_token_loss > 0:
        self.lm_head = MaskLMHead(
            embed_dim=args.encoder_embed_dim,
            output_dim=len(dictionary),
            activation_fn=args.activation_fn,
            weight=None,
        )
            
        # self.logvar_proj = NonLinearHead(
        #     args.encoder_embed_dim, args.encoder_embed_dim, 
        #     args.activation_fn
        # )
        self.quant_mean = NonLinearHead(
            input_dim = args.encoder_embed_dim, 
            out_dim = 8, 
            activation_fn = args.activation_fn, 
            hidden = args.encoder_embed_dim//2
        )
               
        self.quant_rot = NonLinearHead(
            input_dim = args.encoder_embed_dim, 
            out_dim = 9, 
            activation_fn = args.activation_fn, 
            hidden = args.encoder_embed_dim//2
        )
        self.logvar_proj = NonLinearHead(
            input_dim = args.encoder_embed_dim, 
            out_dim = 8, 
            activation_fn = args.activation_fn, 
            hidden = args.encoder_embed_dim//2
        )
        self.quant_expand = NonLinearHead(
            input_dim = 8, 
            out_dim = args.encoder_embed_dim, 
            activation_fn = args.activation_fn, 
            hidden = args.encoder_embed_dim//2
        )
        
        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(
            K, args.encoder_attention_heads, args.activation_fn
        )
        self.gbf_proj2 = NonLinearHead(
            K, args.encoder_attention_heads, args.activation_fn
        )
        self.gbf = GaussianLayer(K, n_edge_type)
        self.gbf2 = GaussianLayer(K, 5)
        
        #if args.masked_coord_loss > 0:
        self.pair2coord_proj = NonLinearHead(
            args.decoder_attention_heads, 1, args.activation_fn
        )
        #if args.masked_dist_loss > 0:
        self.dist_head = DistanceHead(
            args.decoder_attention_heads, args.activation_fn
        )
        self.init_pos_prog = NonLinearHeadPos(
            512, 1, args.activation_fn
        )
        self.pos_embedder = nn.Sequential(
            nn.Linear(3, args.encoder_embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim),
        )
        self.pos_decoder = nn.Sequential(
            nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(args.encoder_embed_dim, 3),
        )
        self.classification_heads = nn.ModuleDict()
        self.apply(init_bert_params)
        self.encoder_unmasked_tokens_only = args.encoder_unmasked_tokens_only
        self.dictionary = dictionary
        self.prop_decoder = nn.Sequential(
            nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(args.encoder_embed_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 38),
        )

        self.embed_positions = SinusoidalPositionalEmbedding(
            embedding_dim = args.max_seq_len,
            padding_idx = dictionary.pad(),
            init_size = args.max_seq_len,
        )

        self.mask_idx = dictionary.index("[MASK]")
        self.encoder_attention_heads = args.encoder_attention_heads
        self.self_condition = True
        self.gk_bond = BondGaussianLayer()
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)
    
    @classmethod
    def init_state(self):
        original_state = np.random.get_state()
        np.random.seed(0)
        self.numpy_random_state = np.random.get_state()
        np.random.set_state(original_state)
    
    @classmethod
    def Myenter(self):
        self.original_state = np.random.get_state()
        np.random.set_state(self.numpy_random_state)

    @classmethod
    def Myexit(self):
        self.numpy_random_state = np.random.get_state()
        np.random.set_state(self.original_state)

    def rot(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        src_bond_type,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargs        
    ):
        encoder_src_tokens = src_tokens
        encoder_src_coord = src_coord
        encoder_src_distance = src_distance
        encoder_src_edge_type = src_edge_type

        encoder_padding_mask = encoder_src_tokens.eq(self.padding_idx)
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        x = self.embed_tokens(encoder_src_tokens) + self.embed_positions(~padding_mask)
        x = x + self.pos_embedder(encoder_src_coord)
        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias
        def get_bond_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf2(dist, et)
            gbf_result = self.gbf_proj2(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(encoder_src_distance, encoder_src_edge_type) #+ get_bond_features(encoder_src_distance, src_bond_type)

        (
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            encoder_x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder_rot(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0
        
        rot = torch.sum(encoder_rep * ~padding_mask.unsqueeze(-1), 1) / torch.sum(~padding_mask, 1).unsqueeze(-1)
        encoder_rot = self.quant_rot(rot).reshape(-1, 3, 3)
        rot_mat = roma.special_procrustes(encoder_rot)
        x_rot = torch.bmm(encoder_src_coord, rot_mat)
        #encoder_output_embedding = torch.nn.Tanh()(encoder_rep_quant)
        
        # if self.feed_token_rep_to_decoder:
        #     encoder_output_embedding = encoder_rep
        # else:
        #     if encoder_masked_tokens is None:
        #         encoder_output_embedding = encoder_rep
        #     else:
        #         mask_tokens = torch.zeros_like(src_tokens).fill_(self.mask_idx)
        #         masked_embeddings = self.embed_tokens(mask_tokens)
        #         encoder_output_embedding = torch.where(encoder_masked_tokens.unsqueeze(-1).expand_as(encoder_rep), masked_embeddings, encoder_rep)
        #latent_logvar = self.logvar_proj(encoder_rep)
        
        return x_rot, rot_mat
    
    
    def enc(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        src_bond_type,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargs        
    ):
        encoder_src_tokens = src_tokens
        encoder_src_coord = src_coord
        encoder_src_distance = src_distance
        encoder_src_edge_type = src_edge_type

        encoder_padding_mask = encoder_src_tokens.eq(self.padding_idx)
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        x = self.embed_tokens(encoder_src_tokens) + self.embed_positions(~padding_mask)
        x = x + self.pos_embedder(encoder_src_coord)
        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias
        def get_bond_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf2(dist, et)
            gbf_result = self.gbf_proj2(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(encoder_src_distance, encoder_src_edge_type) #+ get_bond_features(encoder_src_distance, src_bond_type)

        (
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            encoder_x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0
        encoder_rep_quant = self.quant_mean(encoder_rep)
        encoder_output_embedding = torch.nn.Tanh()(encoder_rep_quant)
        
        # if self.feed_token_rep_to_decoder:
        #     encoder_output_embedding = encoder_rep
        # else:
        #     if encoder_masked_tokens is None:
        #         encoder_output_embedding = encoder_rep
        #     else:
        #         mask_tokens = torch.zeros_like(src_tokens).fill_(self.mask_idx)
        #         masked_embeddings = self.embed_tokens(mask_tokens)
        #         encoder_output_embedding = torch.where(encoder_masked_tokens.unsqueeze(-1).expand_as(encoder_rep), masked_embeddings, encoder_rep)
        latent_logvar = self.logvar_proj(encoder_rep)
        
        return encoder_output_embedding, latent_logvar, encoder_x_norm, delta_encoder_pair_rep_norm
    
    def dec(self, encoder_output_embedding, padding_mask, encoder_x_norm=None, delta_encoder_pair_rep_norm=None, classification_head_name=None):
        encoder_masked_tokens=None
        features_only=False
        #classification_head_name=None   
             
        #if not self.decoder_no_pe:
        #    encoder_output_embedding = encoder_output_embedding + self.embed_positions(~padding_mask)
        encoder_output_embedding = self.quant_expand(encoder_output_embedding) + self.embed_positions(~padding_mask)
        n_node = encoder_output_embedding.size(1)
        # if self.feed_pair_rep_to_decoder:
        #     assert self.decoder_noise is not True
        #     attn_bias = encoder_pair_rep.reshape(-1, n_node, n_node)
        # else:
        #     if not self.decoder_noise:
        #         bsz = padding_mask.size(0)
        #         attn_bias = encoder_output_embedding.new_zeros((bsz*self.encoder_attention_heads, n_node, n_node))
        #     else:
        #         attn_bias = get_dist_features(src_distance, src_edge_type)
        bsz = padding_mask.size(0)
        attn_bias = encoder_output_embedding.new_zeros((bsz*self.encoder_attention_heads, n_node, n_node))
        
        (
            decoder_rep,
            decoder_pair_rep,
            delta_decoder_pair_rep,
            decoder_x_norm,
            delta_decoder_pair_rep_norm,
        ) = self.decoder(encoder_output_embedding, padding_mask=padding_mask, attn_mask=attn_bias)


        (
            prop_embedding,
            _,
            _,
            _,
            _,
        ) = self.pretraining_property(encoder_output_embedding, padding_mask=padding_mask, attn_mask=attn_bias)

        prop_input = torch.sum(prop_embedding * ~padding_mask.unsqueeze(-1), 1) / torch.sum(~padding_mask, 1).unsqueeze(-1)
        prop_pred = self.prop_decoder(prop_input)

        decoder_pair_rep[decoder_pair_rep == float("-inf")] = 0

        encoder_distance = None
        encoder_coord = None

        if not features_only:
           #if self.args.masked_token_loss > 0:
            logits = self.lm_head(decoder_rep, ~padding_mask) 
            
            #if self.args.masked_coord_loss > 0:
            encoder_coord = self.pos_decoder(decoder_rep)
            encoder_pairwise_dist = torch.cdist(encoder_coord, encoder_coord, p=2)
            encoder_bond = self.gk_bond(encoder_pairwise_dist, logits)
            
            #if self.args.masked_dist_loss > 0:
            encoder_distance = self.dist_head(decoder_pair_rep)

        if classification_head_name is not None:
            finetuning_embedding = encoder_output_embedding
            # (
            #     finetuning_embedding,
            #     _,
            #     _,
            #     _,
            #     _,
            # ) = self.finetune_block(encoder_output_embedding.detach(), padding_mask=padding_mask, attn_mask=attn_bias)
            
            #finetune_input = torch.sum(finetuning_embedding * ~padding_mask.unsqueeze(-1), 1) / torch.sum(~padding_mask, 1).unsqueeze(-1)
            finetune_input = prop_input
            logits = self.classification_heads[classification_head_name](finetune_input)
        if self.args.mode == 'infer':
            return _, _
        else:
            return (
                logits,
                encoder_distance,
                (encoder_coord,encoder_bond),
                encoder_x_norm,
                decoder_x_norm,
                delta_encoder_pair_rep_norm,
                delta_decoder_pair_rep_norm,
                prop_pred
            )            
    def flow_training(
        self,
        model,
        z_1,
        mask
    ):
        """Vanilla Flow"""
        # dtype = torch.float32
        # t = torch.rand((z_1.size(0),), dtype=dtype, device=z_1.device)
        # t = t.view(-1,1,1)
        # z_0 = torch.randn_like(z_1, device=z_1.device)
        # #z_t = (1 - t) * z_0 + (1e-5 + (1 - 1e-5) * t) * z_1
        # z_t = (1 - t) * z_0 + t * z_1
        # #u = (1 - 1e-5) * z_1 - z_0
        # u = (1 - 1e-5) * z_1 - z_0
        # #v = model(z_t, t.squeeze(-1), mask=mask)
        # v = self.flow_infer(model, z_t, t.squeeze(-1), mask=mask)
        # loss = F.mse_loss(v, u, reduction='none')
        
        
        """ OT-CFM"""

        # FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.)
        # x1 = z_1
        # x0 = torch.randn_like(x1)
        # t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        # vt = self.flow_infer(model, xt, t.unsqueeze(-1), mask=mask)
        # loss = F.mse_loss(vt, ut, reduction='none')
        # return loss, vt
    
        """Protein Frame Flow"""
        self.interpolant.device = z_1.device
        dense_encoded_batch = {"x_1": z_1, "token_mask": mask, "diffuse_mask": mask}
        noisy_dense_encoded_batch = self.interpolant.corrupt_batch(dense_encoded_batch)

        #if self.self_condition:
        # Use self-conditioning for ~half training batches
        if (
            self.interpolant.self_condition
            and random.random() < self.interpolant.self_condition_prob
        ):
            
            with torch.no_grad():
                x_sc = model(noisy_dense_encoded_batch["x_t"],
                             noisy_dense_encoded_batch["t"],
                             mask=mask,
                             x_sc=None)
        else:
            x_sc = None
            
        pred_x = model(noisy_dense_encoded_batch["x_t"],
                        noisy_dense_encoded_batch["t"],
                        mask=mask,
                        x_sc=x_sc)

        gt_x_1 = noisy_dense_encoded_batch["x_1"]
        norm_scale = 1 - torch.min(noisy_dense_encoded_batch["t"].unsqueeze(-1), torch.tensor(0.9))
        x_error = (gt_x_1 - pred_x) / norm_scale
        loss_mask = (
            noisy_dense_encoded_batch["token_mask"] * noisy_dense_encoded_batch["diffuse_mask"]
        )
        loss_denom = torch.sum(loss_mask, dim=-1) * pred_x.size(-1)
        x_loss = torch.sum(x_error**2 * loss_mask[..., None], dim=(-1, -2)) / loss_denom
        #loss_dict = {"loss": x_loss.mean(), "x_loss": x_loss}
        loss_dict = {"loss": x_loss.mean()}
         
        num_bins = 4
        #flat_losses = x_loss.detach().cpu().numpy().flatten()
        #flat_losses = (gt_x_1 - pred_x)
        flat_losses = torch.sum((gt_x_1 - pred_x)**2 * loss_mask[..., None], dim=(-1, -2)) / loss_denom
        flat_losses = flat_losses.detach().cpu().numpy().flatten()
        flat_t = noisy_dense_encoded_batch["t"].detach().cpu().numpy().flatten()
        bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins + 1)
        bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
        t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
        t_binned_n = np.bincount(bin_idx)
        for t_bin in np.unique(bin_idx).tolist():
            bin_start = bin_edges[t_bin]
            bin_end = bin_edges[t_bin + 1]
            t_range = f"f_loss t=[{int(bin_start*100)},{int(bin_end*100)})"
            range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
            loss_dict[t_range] = range_loss
        loss_dict["t_avg"] = np.mean(flat_t)
        
        return loss_dict, pred_x
        
    def flow_infer(
        self,
        model,
        input,
        t,
        mask
    ):
        if self.self_condition:
            with torch.no_grad():
                x_sc = model(input, t, mask=mask, x_sc=None)
        else:
            x_sc = None
            
        v = model(input, t, mask=mask, x_sc=x_sc)  
        return v     
    
    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        src_bond_type,
        mode,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargs
    ):
        if mode == "ae_only":
            #with torch.no_grad():
            if self.random_order and self.index is None:
                self.Myenter()
                self.index = np.random.permutation(512)
                self.Myexit()
                
            padding_mask = src_tokens.eq(self.padding_idx)

            with torch.no_grad():
                _, rot_mat = self.rot(
                        src_tokens,
                        src_distance,
                        src_coord,
                        src_edge_type,
                        src_bond_type,
                        encoder_masked_tokens=None,
                        features_only=False,
                        classification_head_name=None,
                        **kwargs
                        )

                
            latent_emb, latent_logvar, encoder_x_norm, delta_encoder_pair_rep_norm = self.enc(
                    src_tokens,
                    src_distance,
                    src_coord,
                    src_edge_type,
                    src_bond_type,
                    encoder_masked_tokens=None,
                    features_only=False,
                    classification_head_name=None,
                    **kwargs
                    )
            LOG_STD_MAX = 15
            LOG_STD_MIN = -15 # 너무 작은 분산을 막기 위함
            latent_logvar = torch.clamp(latent_logvar, LOG_STD_MIN, LOG_STD_MAX)
            
            std = torch.exp(0.5 * latent_logvar)
            q_z_given_x = torch.distributions.Normal(loc=latent_emb, scale=std)
            z = q_z_given_x.rsample()
            p_z = torch.distributions.Normal(
                loc=torch.zeros_like(latent_emb), 
                scale=torch.ones_like(std)
            )
        elif mode == "flow_only":
            #self.encoder.eval()
            with torch.no_grad():
                if self.random_order and self.index is None:
                    self.Myenter()
                    self.index = np.random.permutation(512)
                    self.Myexit()

        
                padding_mask = src_tokens.eq(self.padding_idx)
                _, rot_mat = self.rot(
                        src_tokens,
                        src_distance,
                        src_coord,
                        src_edge_type,
                        src_bond_type,
                        encoder_masked_tokens=None,
                        features_only=False,
                        classification_head_name=None,
                        **kwargs
                        )
                
                latent_emb, latent_logvar, encoder_x_norm, delta_encoder_pair_rep_norm = self.enc(
                        src_tokens,
                        src_distance,
                        src_coord,
                        src_edge_type,
                        src_bond_type,
                        encoder_masked_tokens=None,
                        features_only=False,
                        classification_head_name=None,
                        **kwargs
                        )
                LOG_STD_MAX = 15
                LOG_STD_MIN = -15 # 너무 작은 분산을 막기 위함
                latent_logvar = torch.clamp(latent_logvar, LOG_STD_MIN, LOG_STD_MAX)
                
                std = torch.exp(0.5 * latent_logvar)
                q_z_given_x = torch.distributions.Normal(loc=latent_emb, scale=std)
                z = q_z_given_x.rsample()
                p_z = torch.distributions.Normal(
                    loc=torch.zeros_like(latent_emb), 
                    scale=torch.ones_like(std)
                )
        
        
        if mode == "ae_only":
            with torch.no_grad():
                flow_loss_dict, _ = self.flow_training(self.backbone, z.detach(), ~padding_mask)
            (
                logits,
                encoder_distance,
                encoder_coord,
                encoder_x_norm,
                decoder_x_norm,
                delta_encoder_pair_rep_norm,
                delta_decoder_pair_rep_norm,
                prop_pred
                
            ) = self.dec(
                z,
                padding_mask,
                encoder_x_norm,
                delta_encoder_pair_rep_norm,
                classification_head_name
            )
        elif mode == "flow_only":
            flow_loss_dict, _ = self.flow_training(self.backbone, z.detach(), ~padding_mask)
            with torch.no_grad():
                (
                    logits,
                    encoder_distance,
                    encoder_coord,
                    encoder_x_norm,
                    decoder_x_norm,
                    delta_encoder_pair_rep_norm,
                    delta_decoder_pair_rep_norm,
                    prop_pred
                ) = self.dec(
                    z,
                    padding_mask,
                    encoder_x_norm,
                    delta_encoder_pair_rep_norm
                )

        if mode == "dual":
            if self.random_order and self.index is None:
                self.Myenter()
                self.index = np.random.permutation(512)
                self.Myexit()

            if classification_head_name is not None:
                features_only = True
    
            padding_mask = src_tokens.eq(self.padding_idx)
            with torch.no_grad():
                _, rot_mat = self.rot(
                        src_tokens,
                        src_distance,
                        src_coord,
                        src_edge_type,
                        src_bond_type,
                        encoder_masked_tokens=None,
                        features_only=False,
                        classification_head_name=None,
                        **kwargs
                        )
            
            latent_emb, latent_logvar, encoder_x_norm, delta_encoder_pair_rep_norm = self.enc(
                    src_tokens,
                    src_distance,
                    src_coord,
                    src_edge_type,
                    src_bond_type,
                    encoder_masked_tokens=None,
                    features_only=False,
                    classification_head_name=None,
                    **kwargs
                    )
            LOG_STD_MAX = 15
            LOG_STD_MIN = -15 # 너무 작은 분산을 막기 위함
            latent_logvar = torch.clamp(latent_logvar, LOG_STD_MIN, LOG_STD_MAX)
            
            std = torch.exp(0.5 * latent_logvar)
            q_z_given_x = torch.distributions.Normal(loc=latent_emb, scale=std)
            z = q_z_given_x.rsample()
            p_z = torch.distributions.Normal(
                loc=torch.zeros_like(latent_emb), 
                scale=torch.ones_like(std)
            )    
            
            flow_loss_dict, _ = self.flow_training(self.backbone, z.detach(), ~padding_mask)    
            (
                logits,
                encoder_distance,
                encoder_coord,
                encoder_x_norm,
                decoder_x_norm,
                delta_encoder_pair_rep_norm,
                delta_decoder_pair_rep_norm,
                prop_pred
            ) = self.dec(
                z,
                padding_mask,
                encoder_x_norm,
                delta_encoder_pair_rep_norm
            )
        
            
        return (
            logits,
            encoder_distance,
            (rot_mat, encoder_coord),
            encoder_x_norm,
            decoder_x_norm,
            delta_encoder_pair_rep_norm,
            delta_decoder_pair_rep_norm,
            (z, q_z_given_x, p_z,latent_emb,std),
            flow_loss_dict,
            prop_pred
        ) 
        
    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )
        
    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates
 

@register_model_architecture("unimol_MAE_padding", "unimol_MAE_padding")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 15)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 512)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
    args.masked_coord_loss = getattr(args, "masked_coord_loss", -1.0)
    args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)
    args.x_norm_loss = getattr(args, "x_norm_loss", -1.0)
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)

    args.encoder_masked_3d_pe = getattr(args, "encoder_masked_3d_pe", False)
    args.encoder_unmasked_tokens_only = getattr(args, "encoder_unmasked_tokens_only", False)
    args.encoder_apply_pe = getattr(args, "encoder_apply_pe", False)
    args.feed_pair_rep_to_decoder = getattr(args, "feed_pair_rep_to_decoder", False)
    args.decoder_no_pe = getattr(args, "decoder_no_pe", False)
    args.feed_token_rep_to_decoder = getattr(args, "feed_token_rep_to_decoder", False)
    args.decoder_noise = getattr(args, "decoder_noise", False)
    args.random_order = getattr(args, "random_order", False)
    

@register_model_architecture("unimol_MAE_padding", "unimol_MAE_padding_base")
def unimol_base_architecture(args):
    base_architecture(args)

@register_model_architecture("unimol_MAE_padding", "unimol_MAE_padding_150M")
def base_150M_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 30)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 640)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2560)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 20)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 512)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
    args.masked_coord_loss = getattr(args, "masked_coord_loss", -1.0)
    args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)
    args.x_norm_loss = getattr(args, "x_norm_loss", -1.0)
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)

    args.decoder_noise = getattr(args, "decoder_noise", False)
    args.random_order = getattr(args, "random_order", False)

@register_model_architecture("unimol_MIM_padding", "unimol_MIM_padding")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 10)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 128)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 128)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 128)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
    args.masked_coord_loss = getattr(args, "masked_coord_loss", -1.0)
    args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)
    args.x_norm_loss = getattr(args, "x_norm_loss", -1.0)
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)

    args.encoder_masked_3d_pe = getattr(args, "encoder_masked_3d_pe", False)
    args.encoder_unmasked_tokens_only = getattr(args, "encoder_unmasked_tokens_only", False)
    args.encoder_apply_pe = getattr(args, "encoder_apply_pe", False)
    args.feed_pair_rep_to_decoder = getattr(args, "feed_pair_rep_to_decoder", False)
    args.decoder_no_pe = getattr(args, "decoder_no_pe", False)
    args.feed_token_rep_to_decoder = getattr(args, "feed_token_rep_to_decoder", False)
    args.decoder_noise = getattr(args, "decoder_noise", False)
    args.random_order = getattr(args, "random_order", False)
    


@register_model_architecture("unimol_Optimal_padding", "unimol_Optimal_padding")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 10)
    #args.encoder_layers = getattr(args, "rot_layers", 3)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 128)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 128)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 128)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
    args.masked_coord_loss = getattr(args, "masked_coord_loss", -1.0)
    args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)
    args.x_norm_loss = getattr(args, "x_norm_loss", -1.0)
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)

    args.encoder_masked_3d_pe = getattr(args, "encoder_masked_3d_pe", False)
    args.encoder_unmasked_tokens_only = getattr(args, "encoder_unmasked_tokens_only", False)
    args.encoder_apply_pe = getattr(args, "encoder_apply_pe", False)
    args.feed_pair_rep_to_decoder = getattr(args, "feed_pair_rep_to_decoder", False)
    args.decoder_no_pe = getattr(args, "decoder_no_pe", False)
    args.feed_token_rep_to_decoder = getattr(args, "feed_token_rep_to_decoder", False)
    args.decoder_noise = getattr(args, "decoder_noise", False)
    args.random_order = getattr(args, "random_order", False)
    
@register_model_architecture("unimol_Optimal_padding2", "unimol_Optimal_padding2")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 10)
    #args.encoder_layers = getattr(args, "rot_layers", 3)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 128)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 128)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 128)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
    args.masked_coord_loss = getattr(args, "masked_coord_loss", -1.0)
    args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)
    args.x_norm_loss = getattr(args, "x_norm_loss", -1.0)
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)

    args.encoder_masked_3d_pe = getattr(args, "encoder_masked_3d_pe", False)
    args.encoder_unmasked_tokens_only = getattr(args, "encoder_unmasked_tokens_only", False)
    args.encoder_apply_pe = getattr(args, "encoder_apply_pe", False)
    args.feed_pair_rep_to_decoder = getattr(args, "feed_pair_rep_to_decoder", False)
    args.decoder_no_pe = getattr(args, "decoder_no_pe", False)
    args.feed_token_rep_to_decoder = getattr(args, "feed_token_rep_to_decoder", False)
    args.decoder_noise = getattr(args, "decoder_noise", False)
    args.random_order = getattr(args, "random_order", False)
    
# unimol_Optimal_padding_Dual unimol_Optimal_padding_Type2
    
@register_model_architecture("unimol_Optimal_padding_Type2", "unimol_Optimal_padding_Type2")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 10)
    #args.encoder_layers = getattr(args, "rot_layers", 3)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 128)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 128)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 128)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
    args.masked_coord_loss = getattr(args, "masked_coord_loss", -1.0)
    args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)
    args.x_norm_loss = getattr(args, "x_norm_loss", -1.0)
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)

    args.encoder_masked_3d_pe = getattr(args, "encoder_masked_3d_pe", False)
    args.encoder_unmasked_tokens_only = getattr(args, "encoder_unmasked_tokens_only", False)
    args.encoder_apply_pe = getattr(args, "encoder_apply_pe", False)
    args.feed_pair_rep_to_decoder = getattr(args, "feed_pair_rep_to_decoder", False)
    args.decoder_no_pe = getattr(args, "decoder_no_pe", False)
    args.feed_token_rep_to_decoder = getattr(args, "feed_token_rep_to_decoder", False)
    args.decoder_noise = getattr(args, "decoder_noise", False)
    args.random_order = getattr(args, "random_order", False)
    

@register_model_architecture("unimol_Optimal_padding_Dual", "unimol_Optimal_padding_Dual")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 10)
    #args.encoder_layers = getattr(args, "rot_layers", 3)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 128)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 128)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 128)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
    args.masked_coord_loss = getattr(args, "masked_coord_loss", -1.0)
    args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)
    args.x_norm_loss = getattr(args, "x_norm_loss", -1.0)
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)

    args.encoder_masked_3d_pe = getattr(args, "encoder_masked_3d_pe", False)
    args.encoder_unmasked_tokens_only = getattr(args, "encoder_unmasked_tokens_only", False)
    args.encoder_apply_pe = getattr(args, "encoder_apply_pe", False)
    args.feed_pair_rep_to_decoder = getattr(args, "feed_pair_rep_to_decoder", False)
    args.decoder_no_pe = getattr(args, "decoder_no_pe", False)
    args.feed_token_rep_to_decoder = getattr(args, "feed_token_rep_to_decoder", False)
    args.decoder_noise = getattr(args, "decoder_noise", False)
    args.random_order = getattr(args, "random_order", False)